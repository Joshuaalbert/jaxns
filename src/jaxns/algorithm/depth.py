"""Pure-JAX lineage planning, constrained sampling, and depth traversal."""

import dataclasses
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxns.algorithm.allocation import (
    AllocationPlan,
    VolumePath,
    build_allocation_plan,
    expected_volume_path,
)
from jaxns.algorithm.race_tree import (
    BlockState,
    build_block_state,
    initialise_likelihood_order,
)
from jaxns.algorithm.scheduler import (
    ThreadSchedule,
    _insert_seed_reservation,
    _seed_reservation_contains,
)
from jaxns.constrained_sampler import (
    AbstractSampler,
    ConstrainedSampleBatch,
    ConstrainedSampleRequest,
    sample_request,
)
from jaxns.depth_condition import (
    DepthCondition,
)
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree, pytree_ravel
from jaxns.samples import Samples, SeedPoint
from jaxns.sampling.ellipsoid import update_sampler_data
from jaxns.shrinkage.classic import (
    classic_dirichlet_concentrations,
)
from jaxns.state import State
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey

# This reason is shared by local and distributed depth execution so both
# runners expose the same resumable storage-boundary contract.
MAX_SAMPLES_REACHED = 1
# Four root-width windows amortise early seed-index construction while the
# later 25% population-growth rule keeps publication geometric rather than a
# fixed-cadence population operation.
SEED_SOURCE_REFRESH_WINDOWS = 4


@dataclasses.dataclass(slots=True, frozen=True)
class CoreWorkBatch(PureDataclassPytree):
    """Fixed-shape description of one vmapped replacement batch."""

    valid: BoolArray  # [S]
    # This storage index is transient. It is retained only until acceptance
    # increments the selected parent's out-degree, then it is discarded.
    parent_idx: IntArray  # [S]
    log_L_constraint: FloatArray  # [S]
    seed_idx: IntArray  # [S]

    @property
    def num_valid(self) -> IntArray:
        """Number of accepted prefix lanes in this static schedule."""
        return jnp.sum(self.valid, dtype=mp_policy.count_dtype)


CoreWorkBatch.register_pytree()


def _depth_relevant_path(
        log_L_blocks: FloatArray,
        valid: BoolArray,
        volume_path: VolumePath,
        depth_cond: DepthCondition,
) -> BoolArray:
    """Mask blocks before the expectation-based evidence/posterior cutoff."""
    log_shell_mass = jnp.where(
        valid & (volume_path.shell_mass > 0),
        jnp.log(volume_path.shell_mass),
        -jnp.inf,
    )
    log_dZ = log_L_blocks + log_shell_mass
    log_Z_prefix = jax.lax.associative_scan(jnp.logaddexp, log_dZ)
    log_remaining = (
        log_L_blocks
        + jnp.where(
            volume_path.X > 0,
            jnp.log(volume_path.X),
            -jnp.inf,
        )
    )
    remaining_fraction = jnp.exp(
        log_remaining - jnp.logaddexp(log_Z_prefix, log_remaining)
    )
    relevant = valid
    if depth_cond.dlogZ is not None:
        fails_depth = valid & (
            remaining_fraction >= depth_cond.dlogZ
        )
        # L_g X_g may rise again before the posterior peak. The depth domain
        # must therefore be the complete prefix through the last failing
        # contour, not a pointwise mask with holes that strands lineage gaps.
        before_depth = (
            jnp.cumsum(fails_depth[::-1].astype(mp_policy.count_dtype))[::-1]
            > 0
        )
        relevant = relevant & before_depth

    if depth_cond.cummax_XL_frac is not None:
        log_XL = jnp.where(valid, log_remaining, -jnp.inf)
        log_XL_peak = jax.lax.associative_scan(jnp.maximum, log_XL)
        fails_depth = valid & (
            log_XL
            >= log_XL_peak + jnp.log(depth_cond.cummax_XL_frac)
        )
        before_depth = (
            jnp.cumsum(fails_depth[::-1].astype(mp_policy.count_dtype))[::-1]
            > 0
        )
        relevant = relevant & before_depth
    return relevant


def _depth_relevant_blocks(
        plan: AllocationPlan,
        depth_cond: DepthCondition,
) -> BoolArray:
    """Apply expected-depth limits to one already-built allocation plan."""
    return _depth_relevant_path(
        plan.log_L_blocks,
        plan.valid,
        plan.volume_path,
        depth_cond,
    )


def _sample_parent_from_block(
        key: PRNGKey,
        block_state: BlockState,
        block_idx: IntArray,
) -> IntArray:
    """Choose the concrete sample whose out-degree will gain one child.

    Samples tied in one likelihood block define the same strict contour, so
    the scheduler may choose uniformly among them. The returned storage index
    lives only until the acceptance scatter updates that sample's out-degree.
    """
    safe_block_idx = jnp.maximum(block_idx, 0)
    start = block_state.block_start[safe_block_idx]
    size = block_state.block_size[safe_block_idx]
    offset = jax.random.randint(
        key,
        (),
        minval=jnp.asarray(0, mp_policy.index_dtype),
        maxval=jnp.maximum(size, 1),
    )
    parent_idx = block_state.block_sample_indices[start + offset]
    return jnp.where(block_idx >= 0, parent_idx, -1).astype(
        mp_policy.index_dtype
    )


def _decompose_gap(gap: IntArray) -> tuple[
    IntArray,
    IntArray,
    IntArray,
    IntArray,
]:
    """Compress a gap curve into laminar maximal-thread runs in O(G).

    A maximal thread is one connected component of an integer superlevel of
    the gap. A LIFO rise/fall sweep is the canonical laminar decomposition and
    emits at most G `(start, terminal, multiplicity)` runs even when the
    integer gap itself is very large.
    """
    gap = gap.astype(mp_policy.count_dtype)
    size = gap.shape[0]
    stack_start = jnp.zeros((size,), mp_policy.index_dtype)  # [G]
    stack_count = jnp.zeros((size,), mp_policy.count_dtype)  # [G]
    run_start = jnp.zeros((size,), mp_policy.index_dtype)  # [G]
    run_terminal = jnp.zeros((size,), mp_policy.index_dtype)  # [G]
    run_count = jnp.zeros((size,), mp_policy.count_dtype)  # [G]

    def sweep(block_idx, carry):
        (
            starts,
            counts,
            stack_size,
            emitted_start,
            emitted_terminal,
            emitted_count,
            num_runs,
        ) = carry
        previous = jnp.where(
            block_idx > 0,
            gap[jnp.maximum(block_idx - 1, 0)],
            jnp.asarray(0, gap.dtype),
        )
        following = jnp.where(
            block_idx + 1 < size,
            gap[jnp.minimum(block_idx + 1, size - 1)],
            jnp.asarray(0, gap.dtype),
        )
        rise = jnp.maximum(gap[block_idx] - previous, 0)
        fall = jnp.maximum(gap[block_idx] - following, 0)

        def push(values):
            one_starts, one_counts, one_size = values
            one_starts = one_starts.at[one_size].set(block_idx)
            one_counts = one_counts.at[one_size].set(rise)
            return one_starts, one_counts, one_size + 1

        starts, counts, stack_size = jax.lax.cond(
            rise > 0,
            push,
            lambda values: values,
            (starts, counts, stack_size),
        )

        def consume_cond(values):
            remaining, *_ = values
            return remaining > 0

        def consume_body(values):
            (
                remaining,
                one_starts,
                one_counts,
                one_size,
                one_run_start,
                one_run_terminal,
                one_run_count,
                one_num_runs,
            ) = values
            top = one_size - 1
            take = jnp.minimum(remaining, one_counts[top])
            one_run_start = one_run_start.at[one_num_runs].set(
                one_starts[top]
            )
            one_run_terminal = one_run_terminal.at[one_num_runs].set(
                block_idx
            )
            one_run_count = one_run_count.at[one_num_runs].set(take)
            left = one_counts[top] - take
            one_counts = one_counts.at[top].set(left)
            one_size = one_size - (left == 0).astype(one_size.dtype)
            return (
                remaining - take,
                one_starts,
                one_counts,
                one_size,
                one_run_start,
                one_run_terminal,
                one_run_count,
                one_num_runs + 1,
            )

        (
            _,
            starts,
            counts,
            stack_size,
            emitted_start,
            emitted_terminal,
            emitted_count,
            num_runs,
        ) = jax.lax.while_loop(
            consume_cond,
            consume_body,
            (
                fall,
                starts,
                counts,
                stack_size,
                emitted_start,
                emitted_terminal,
                emitted_count,
                num_runs,
            ),
        )
        return (
            starts,
            counts,
            stack_size,
            emitted_start,
            emitted_terminal,
            emitted_count,
            num_runs,
        )

    (
        _,
        _,
        _,
        run_start,
        run_terminal,
        run_count,
        num_runs,
    ) = jax.lax.fori_loop(
        0,
        size,
        sweep,
        (
            stack_start,
            stack_count,
            jnp.asarray(0, mp_policy.index_dtype),
            run_start,
            run_terminal,
            run_count,
            jnp.asarray(0, mp_policy.index_dtype),
        ),
    )
    return run_start, run_terminal, run_count, num_runs


def _continuation_capacity(
        sample_capacity: int,
        reservoir_size: int,
) -> int:
    """Choose a small initial heap that can grow at a Python boundary.

    Heap updates are scalar and logarithmic algorithmically, but an XLA CPU
    scatter may copy its static backing buffer. Coupling that buffer to sample
    capacity therefore made every replacement batch linear in the complete
    race size. Four coordination windows plus one in-flight window cover the
    ordinary breadth-first frontier; exceptional wider frontiers trigger a
    transparent doubling rather than taxing every run pre-emptively.
    """
    del sample_capacity
    return max(
        (SEED_SOURCE_REFRESH_WINDOWS + 1) * reservoir_size,
        1,
    )


def _continuation_storage_full(schedule: ThreadSchedule) -> BoolArray:
    """Return whether one worst-case replacement window can overflow."""
    return (
        schedule.continuation_count + schedule.valid.shape[0]
        > schedule.continuation_parent_idx.shape[0]
    )


def _seed_reservation_capacity(shell_size: int) -> int:
    """Give ordinary groups several batches before logarithmic growth."""
    minimum = max(8 * shell_size, 2)
    return 1 << (minimum - 1).bit_length()


def _start_seed_storage_full(schedule: ThreadSchedule) -> BoolArray:
    """Return whether another full window would exceed half load.

    Keeping the exact hash set below half load bounds expected probes. The
    table doubles at the Python coordination boundary, so this threshold is a
    performance boundary rather than a scientific cap.
    """
    required = (
        schedule.num_start_seeds
        + jnp.asarray(schedule.valid.shape[0], mp_policy.count_dtype)
    )
    return (
        2 * required
        > schedule.start_seed_reservation_idx.shape[0]
    )


def _build_seed_rank_index(
        state: State,
        block_state: BlockState,
) -> tuple[FloatArray, IntArray, IntArray]:
    """Index frozen lineage intervals for exact O(log A) seed selection.

    Samples are ordered by their birth contour, while their values are unique
    ranks in likelihood order. A wavelet matrix over those ranks answers the
    two-dimensional query ``birth <= contour < likelihood`` without scanning
    or rejection. The construction is planning-boundary work and is reused by
    every replacement batch in the frozen generation.
    """
    capacity = state.samples.log_likelihoods.shape[0]
    levels = max(1, (capacity - 1).bit_length())
    sample_idx = jnp.arange(capacity, dtype=mp_policy.index_dtype)  # [A]
    valid = sample_idx < state.num_samples  # [A]
    birth = jnp.where(
        valid,
        state.samples.log_L_constraints,
        jnp.inf,
    )  # [A]
    seed_birth_contours, birth_order = jax.lax.sort(
        (birth, sample_idx),
        dimension=0,
        is_stable=True,
        num_keys=1,
    )

    # Likelihood ranks are unique even inside a plateau because the stable
    # block order retains every concrete sample identity.
    rank = jnp.arange(capacity, dtype=jnp.int32)  # [A]
    ordered_sample = block_state.block_sample_indices  # [A]
    scatter_idx = jnp.where(
        valid,
        ordered_sample,
        jnp.asarray(capacity, mp_policy.index_dtype),
    )
    inverse_rank = jnp.zeros((capacity,), dtype=jnp.int32).at[
        scatter_idx
    ].set(rank, mode="drop")
    values = jnp.where(valid, inverse_rank[birth_order], 0)  # [A]
    prefixes = jnp.zeros((levels, capacity + 1), dtype=jnp.int32)
    zero_counts = jnp.zeros((levels,), dtype=jnp.int32)

    def build_level(level, carry):
        current, current_prefixes, current_zero_counts = carry
        shift = levels - level - 1
        bit = ((current >> shift) & 1).astype(jnp.int32)  # [A]
        prefix = jnp.concatenate(
            (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(bit)),
        )  # [A + 1]
        zero = 1 - bit
        zero_rank = jnp.cumsum(zero) - 1
        one_rank = jnp.cumsum(bit) - 1
        zero_count = jnp.sum(zero, dtype=jnp.int32)
        position = jnp.where(
            bit == 0,
            zero_rank,
            zero_count + one_rank,
        )
        next_values = jnp.zeros_like(current).at[position].set(current)
        return (
            next_values,
            current_prefixes.at[level].set(prefix),
            current_zero_counts.at[level].set(zero_count),
        )

    _, prefixes, zero_counts = jax.lax.fori_loop(
        0,
        levels,
        build_level,
        (values, prefixes, zero_counts),
    )
    return (
        seed_birth_contours,
        prefixes,
        zero_counts,
    )


def _seed_count_below_rank(
        schedule: ThreadSchedule,
        prefix_size: IntArray,
        likelihood_rank: IntArray,
) -> IntArray:
    """Count birth-prefix samples below one likelihood rank."""
    left = jnp.asarray(0, jnp.int32)
    right = prefix_size.astype(jnp.int32)
    count = jnp.asarray(0, jnp.int32)
    levels = schedule.seed_rank_prefix.shape[0]
    def count_level(level, carry):
        current_left, current_right, current_count = carry
        ones_left = schedule.seed_rank_prefix[level, current_left]
        ones_right = schedule.seed_rank_prefix[level, current_right]
        zeros_left = current_left - ones_left
        zeros_right = current_right - ones_right
        bit = (
            likelihood_rank.astype(jnp.int32)
            >> (levels - level - 1)
        ) & 1
        current_count = current_count + jnp.where(
            bit == 1,
            zeros_right - zeros_left,
            0,
        )
        current_left = jnp.where(
            bit == 0,
            zeros_left,
            schedule.seed_zero_count[level] + ones_left,
        )
        current_right = jnp.where(
            bit == 0,
            zeros_right,
            schedule.seed_zero_count[level] + ones_right,
        )
        return current_left, current_right, current_count

    _, _, count = jax.lax.fori_loop(
        0,
        levels,
        count_level,
        (left, right, count),
    )
    return count.astype(mp_policy.index_dtype)


def _seed_rank_quantile(
        schedule: ThreadSchedule,
        prefix_size: IntArray,
        quantile: IntArray,
) -> IntArray:
    """Return one unique likelihood rank from a birth-prefix quantile."""
    left = jnp.asarray(0, jnp.int32)
    right = prefix_size.astype(jnp.int32)
    remaining = quantile.astype(jnp.int32)
    value = jnp.asarray(0, jnp.int32)
    levels = schedule.seed_rank_prefix.shape[0]
    def quantile_level(level, carry):
        current_left, current_right, current_remaining, current_value = carry
        ones_left = schedule.seed_rank_prefix[level, current_left]
        ones_right = schedule.seed_rank_prefix[level, current_right]
        zeros_left = current_left - ones_left
        zeros_right = current_right - ones_right
        zeros = zeros_right - zeros_left
        choose_one = current_remaining >= zeros
        current_remaining = jnp.where(
            choose_one,
            current_remaining - zeros,
            current_remaining,
        )
        current_value = (
            (current_value << 1) | choose_one.astype(jnp.int32)
        )
        current_left = jnp.where(
            choose_one,
            schedule.seed_zero_count[level] + ones_left,
            zeros_left,
        )
        current_right = jnp.where(
            choose_one,
            schedule.seed_zero_count[level] + ones_right,
            zeros_right,
        )
        return (
            current_left,
            current_right,
            current_remaining,
            current_value,
        )

    _, _, _, value = jax.lax.fori_loop(
        0,
        levels,
        quantile_level,
        (left, right, remaining, value),
    )
    return value.astype(mp_policy.index_dtype)


def _sample_frozen_seed_rank(
        schedule: ThreadSchedule,
        log_L_constraint: FloatArray,
        fraction: FloatArray,
) -> IntArray:
    """Select one frozen active lineage by a uniform fractional rank."""
    prefix_size = jnp.searchsorted(
        schedule.seed_birth_contours,
        log_L_constraint,
        side="right",
    ).astype(mp_policy.index_dtype)
    containing_block = (
        jnp.searchsorted(
            schedule.seed_block_state.log_L_blocks,
            log_L_constraint,
            side="right",
        )
        - 1
    ).astype(mp_policy.index_dtype)
    likelihood_rank = jnp.where(
        containing_block >= 0,
        schedule.seed_block_state.block_stop[
            jnp.maximum(containing_block, 0)
        ],
        jnp.asarray(0, mp_policy.index_dtype),
    )
    below = _seed_count_below_rank(
        schedule,
        prefix_size,
        likelihood_rank,
    )
    count = prefix_size - below
    offset = jnp.minimum(
        (fraction * jnp.maximum(count, 1)).astype(mp_policy.index_dtype),
        jnp.maximum(count - 1, 0),
    )
    death_rank = _seed_rank_quantile(
        schedule,
        prefix_size,
        below + offset,
    )
    return schedule.seed_block_state.block_sample_indices[death_rank]


def _frozen_seed_count_at_constraint(
        schedule: ThreadSchedule,
        log_L_constraint: FloatArray,
) -> IntArray:
    """Count concrete frozen edges crossing one strict contour."""
    prefix_size = jnp.searchsorted(
        schedule.seed_birth_contours,
        log_L_constraint,
        side="right",
    ).astype(mp_policy.index_dtype)
    containing_block = (
        jnp.searchsorted(
            schedule.seed_block_state.log_L_blocks,
            log_L_constraint,
            side="right",
        )
        - 1
    ).astype(mp_policy.index_dtype)
    likelihood_rank = jnp.where(
        containing_block >= 0,
        schedule.seed_block_state.block_stop[
            jnp.maximum(containing_block, 0)
        ],
        jnp.asarray(0, mp_policy.index_dtype),
    )
    return prefix_size - _seed_count_below_rank(
        schedule,
        prefix_size,
        likelihood_rank,
    )


class _SeedSourceIndex(NamedTuple):
    """Exact stationary-seed view published independently of thread work."""

    block_state: BlockState
    count: IntArray  # [G]
    previous_seedable: IntArray  # [G]
    birth_contours: FloatArray  # [A]
    rank_prefix: IntArray  # [H, A + 1]
    zero_count: IntArray  # [H]
    root_idx: IntArray  # [A]


def _build_seed_source_index(
        state: State,
        block_state: BlockState,
) -> _SeedSourceIndex:
    """Build seed availability from every sample in one published race."""
    count = jnp.where(
        block_state.valid,
        (
            block_state.incoming_K
            - block_state.block_size
            + block_state.block_out_degree
        ),
        jnp.asarray(0, mp_policy.count_dtype),
    )
    block_idx = jnp.arange(
        block_state.valid.shape[0],
        dtype=mp_policy.index_dtype,
    )
    previous_seedable = jax.lax.associative_scan(
        jnp.maximum,
        jnp.where(count > 0, block_idx, -1),
    )
    sample_idx = jnp.arange(
        state.samples.log_likelihoods.shape[0],
        dtype=mp_policy.index_dtype,
    )
    root_idx = jnp.nonzero(
        (sample_idx < state.num_samples)
        & jnp.isneginf(state.samples.log_L_constraints),
        size=sample_idx.shape[0],
        fill_value=0,
    )[0]  # [A]
    birth_contours, rank_prefix, zero_count = _build_seed_rank_index(
        state,
        block_state,
    )
    return _SeedSourceIndex(
        block_state=block_state,
        count=count,
        previous_seedable=previous_seedable,
        birth_contours=birth_contours,
        rank_prefix=rank_prefix,
        zero_count=zero_count,
        root_idx=root_idx,
    )


def _new_thread_schedule(
        state: State,
        block_state: BlockState,
        plan: AllocationPlan,
        relevant: BoolArray,
        shell_size: int,
        tail_K: IntArray,
        seed_reservoir_size: int | None = None,
) -> ThreadSchedule:
    """Freeze one gap curve and its compressed maximal-thread starts."""
    if seed_reservoir_size is None:
        seed_reservoir_size = shell_size
    # Begin with four ordinary frontier windows plus one window that may be in
    # flight when the boundary is observed. An unusually wide continuation
    # frontier returns to Python and doubles this heap transparently instead of
    # making every ordinary device scatter pay for sample-capacity storage.
    continuation_size = _continuation_capacity(
        state.samples.log_likelihoods.shape[0],
        seed_reservoir_size,
    )
    gap = jnp.where(
        relevant,
        plan.allocation_gap(),
        jnp.asarray(0, mp_policy.count_dtype),
    )
    start_block, terminal_block, multiplicity, num_runs = _decompose_gap(gap)
    # The laminar stack emits deep rises first when a tail gap closes. Execute
    # shallower starts first so a new allocation broadens the constrained
    # population before reinforcing modes that happened to survive at the
    # tail. Sorting only compressed runs occurs once per frozen source; the
    # repeated batch path still advances fixed-width heads without sorting.
    run_idx = jnp.arange(
        start_block.shape[0],
        dtype=mp_policy.index_dtype,
    )  # [G]
    valid_run = run_idx < num_runs  # [G]
    invalid_key = jnp.asarray(start_block.shape[0], mp_policy.index_dtype)
    (
        _,
        _,
        start_block,
        terminal_block,
        multiplicity,
    ) = jax.lax.sort(
        (
            jnp.where(valid_run, start_block, invalid_key),
            jnp.where(valid_run, terminal_block, invalid_key),
            start_block,
            terminal_block,
            multiplicity,
        ),
        dimension=0,
        is_stable=True,
        num_keys=2,
    )
    num_threads = jnp.sum(multiplicity, dtype=mp_policy.count_dtype)

    seed_source = _build_seed_source_index(state, block_state)
    reservation_size = _seed_reservation_capacity(shell_size)
    return ThreadSchedule(
        block_state=block_state,
        seed_block_state=seed_source.block_state,
        start_block=start_block,
        terminal_block=terminal_block,
        multiplicity=multiplicity,
        num_runs=num_runs,
        target_K=plan.target_K,
        tail_K=tail_K,
        seed_count=seed_source.count,
        previous_seedable=seed_source.previous_seedable,
        seed_birth_contours=seed_source.birth_contours,
        seed_rank_prefix=seed_source.rank_prefix,
        seed_zero_count=seed_source.zero_count,
        root_seed_idx=seed_source.root_idx,
        seed_reservoir_idx=jnp.full(
            (seed_reservoir_size,),
            -1,
            dtype=mp_policy.index_dtype,
        ),
        seed_reservoir_priority=jnp.full(
            (seed_reservoir_size,),
            -jnp.inf,
            dtype=mp_policy.measure_dtype,
        ),
        seed_reservoir_valid=jnp.zeros(
            (seed_reservoir_size,),
            dtype=mp_policy.bool_dtype,
        ),
        seed_reservoir_key=jax.random.fold_in(
            state.random_key,
            state.allocation_loop_iter,
        ),
        start_seed_reservation_idx=jnp.full(
            (reservation_size,),
            -1,
            dtype=mp_policy.index_dtype,
        ),
        start_seed_reservation_group=jnp.zeros(
            (reservation_size,),
            dtype=mp_policy.index_dtype,
        ),
        # Zero slot generations mean empty; the first active group is one.
        current_start_group=jnp.asarray(1, mp_policy.index_dtype),
        start_seed_log_L_constraint=jnp.asarray(
            -jnp.inf,
            dtype=mp_policy.measure_dtype,
        ),
        num_start_seeds=jnp.asarray(0, mp_policy.count_dtype),
        parent_idx=jnp.full(
            (shell_size,),
            -1,
            dtype=mp_policy.index_dtype,
        ),
        thread_id=jnp.full(
            (shell_size,),
            -1,
            dtype=mp_policy.index_dtype,
        ),
        log_L_constraint=jnp.full(
            (shell_size,),
            -jnp.inf,
            dtype=mp_policy.measure_dtype,
        ),
        terminal_log_L=jnp.full(
            (shell_size,),
            -jnp.inf,
            dtype=mp_policy.measure_dtype,
        ),
        new_start=jnp.zeros((shell_size,), mp_policy.bool_dtype),
        valid=jnp.zeros((shell_size,), mp_policy.bool_dtype),
        continuation_parent_idx=jnp.full(
            (continuation_size,),
            -1,
            dtype=mp_policy.index_dtype,
        ),
        continuation_thread_id=jnp.full(
            (continuation_size,),
            -1,
            dtype=mp_policy.index_dtype,
        ),
        continuation_log_L_constraint=jnp.full(
            (continuation_size,),
            -jnp.inf,
            dtype=mp_policy.measure_dtype,
        ),
        continuation_frontier=jnp.full(
            (continuation_size,),
            jnp.inf,
            dtype=mp_policy.measure_dtype,
        ),
        continuation_terminal_log_L=jnp.full(
            (continuation_size,),
            -jnp.inf,
            dtype=mp_policy.measure_dtype,
        ),
        continuation_count=jnp.asarray(0, mp_policy.count_dtype),
        next_run=jnp.asarray(0, mp_policy.index_dtype),
        remaining_in_run=jnp.where(
            num_runs > 0,
            multiplicity[0],
            jnp.asarray(0, mp_policy.count_dtype),
        ),
        next_thread_id=jnp.asarray(0, mp_policy.index_dtype),
        num_threads=num_threads.astype(mp_policy.count_dtype),
        source_num_samples=state.num_samples,
        active=num_threads > 0,
    )


def _resize_depth_state(state: State, sample_capacity: int) -> State:
    """Grow all depth carry shapes once at a physical capacity boundary."""
    resized = state.resize(sample_capacity)
    schedule = resized.scheduler_data
    if schedule is None:
        return resized
    continuation_size = _continuation_capacity(
        sample_capacity,
        schedule.seed_reservoir_idx.shape[0],
    )
    continuation_size = max(
        continuation_size,
        schedule.continuation_parent_idx.shape[0],
    )
    return dataclasses.replace(
        resized,
        scheduler_data=schedule.resize_threads(
            schedule.valid.shape[0],
            continuation_size=continuation_size,
        ),
    )


def _seed_count_at_constraint(
        schedule: ThreadSchedule,
        log_L_constraint: FloatArray,
) -> tuple[IntArray, IntArray]:
    """Return frozen seed count and preceding block for one contour."""
    block_state = schedule.seed_block_state
    block_idx = (
        jnp.searchsorted(
            block_state.log_L_blocks,
            log_L_constraint,
            side="right",
        )
        - 1
    ).astype(mp_policy.index_dtype)
    safe_block_idx = jnp.maximum(block_idx, 0)
    # Below the first observed contour exactly the root children cross the
    # requested level. Their frozen count is K_1, before the first block races.
    count = jnp.where(
        block_idx >= 0,
        schedule.seed_count[safe_block_idx],
        block_state.incoming_K[0],
    )
    return count.astype(mp_policy.count_dtype), block_idx


def _effective_parent_contour(
        schedule: ThreadSchedule,
        log_L_constraint: FloatArray,
) -> tuple[BoolArray, IntArray, FloatArray]:
    """Resolve a requested contour to its closest seedable predecessor."""
    seed_count, containing_block = _seed_count_at_constraint(
        schedule,
        log_L_constraint,
    )
    requested_has_seed = seed_count > 0
    safe_containing = jnp.maximum(containing_block, 0)
    fallback_block = jnp.where(
        containing_block >= 0,
        schedule.previous_seedable[safe_containing],
        jnp.asarray(-1, mp_policy.index_dtype),
    )
    effective_block = jnp.where(
        requested_has_seed,
        jnp.asarray(-2, mp_policy.index_dtype),
        fallback_block,
    )
    effective_constraint = jnp.where(
        requested_has_seed,
        log_L_constraint,
        jnp.where(
            effective_block >= 0,
                schedule.seed_block_state.log_L_blocks[
                jnp.maximum(effective_block, 0)
            ],
            jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        ),
    )
    return requested_has_seed, effective_block, effective_constraint


def _update_seed_reservoir(
        schedule: ThreadSchedule,
        sample_idx: IntArray,
        valid: BoolArray,
) -> ThreadSchedule:
    """Retain a bounded value-independent sample of post-freeze rows.

    Random priorities make membership independent of likelihood, contour, and
    append order. Keeping only the largest bounded set makes seed coordination
    independent of schedule length, while the next planning boundary promotes
    every accepted row into the exact frozen race population.
    """
    num_candidates = sample_idx.shape[0]

    def add_one(candidate_idx, current):
        reservoir_idx, reservoir_priority, reservoir_valid = current
        candidate_valid = valid[candidate_idx]
        row_idx = sample_idx[candidate_idx].astype(mp_policy.index_dtype)
        priority = jax.random.uniform(
            jax.random.fold_in(schedule.seed_reservoir_key, row_idx),
            (),
            dtype=mp_policy.measure_dtype,
        )
        has_empty = jnp.any(jnp.logical_not(reservoir_valid))
        empty_slot = jnp.argmin(reservoir_valid).astype(mp_policy.index_dtype)
        minimum_slot = jnp.argmin(reservoir_priority).astype(
            mp_policy.index_dtype
        )
        slot = jnp.where(has_empty, empty_slot, minimum_slot)
        retain = candidate_valid & (
            has_empty | (priority > reservoir_priority[minimum_slot])
        )
        reservoir_idx = reservoir_idx.at[slot].set(jnp.where(
            retain,
            row_idx,
            reservoir_idx[slot],
        ))
        reservoir_priority = reservoir_priority.at[slot].set(jnp.where(
            retain,
            priority,
            reservoir_priority[slot],
        ))
        reservoir_valid = reservoir_valid.at[slot].set(
            reservoir_valid[slot] | retain
        )
        return reservoir_idx, reservoir_priority, reservoir_valid

    reservoir_idx, reservoir_priority, reservoir_valid = jax.lax.fori_loop(
        0,
        num_candidates,
        add_one,
        (
            schedule.seed_reservoir_idx,
            schedule.seed_reservoir_priority,
            schedule.seed_reservoir_valid,
        ),
    )
    return dataclasses.replace(
        schedule,
        seed_reservoir_idx=reservoir_idx,
        seed_reservoir_priority=reservoir_priority,
        seed_reservoir_valid=reservoir_valid,
    )


def _seed_source_refresh_due(
        state: State,
        schedule: ThreadSchedule,
) -> BoolArray:
    """Return whether the next geometric seed generation should be frozen."""
    minimum_rows = jnp.asarray(
        SEED_SOURCE_REFRESH_WINDOWS
        * schedule.seed_reservoir_idx.shape[0],
        mp_policy.count_dtype,
    )
    # A fixed publication interval would rebuild a population-scale race and
    # seed index after every O(1) rows, producing at least quadratic total
    # coordination. Requiring 25% population growth makes generations
    # geometric: the cumulative rows rebuilt are linear in final population,
    # while the rank-index sorting factor remains logarithmic.
    growth_rows = (
        schedule.source_num_samples
        + jnp.asarray(3, mp_policy.count_dtype)
    ) // jnp.asarray(4, mp_policy.count_dtype)
    refresh_rows = jnp.maximum(
        minimum_rows,
        growth_rows,
    )
    return (
        state.num_samples - schedule.source_num_samples
        >= refresh_rows
    )


def _sample_stationary_seeds(
        key: PRNGKey,
        state: State,
        schedule: ThreadSchedule,
        log_L_constraint: FloatArray,
        from_root: BoolArray,
        valid: BoolArray,
        reserved_seed_idx: IntArray,
        reserved_log_L_constraint: FloatArray,
        reserved_valid: BoolArray,
) -> IntArray:
    """Draw same-contour seeds without replacement whenever possible.

    The frozen race population is queried exactly through its birth/likelihood
    rank index. A bounded value-independent reservoir adds recent accepted
    rows without making query cost depend on total population. Every proposed
    identity therefore already crosses the requested contour; rejection is
    needed only for simultaneous no-replacement coordination, never to search
    for a scientifically valid seed.
    """
    shell_size = valid.shape[0]
    rows = jnp.arange(shell_size, dtype=mp_policy.index_dtype)  # [S]
    assignment_key, strata_key, rejection_key = jax.random.split(key, 3)
    strata = jax.random.permutation(assignment_key, rows)  # [S]
    rotation = jax.random.uniform(
        strata_key,
        (),
        dtype=mp_policy.measure_dtype,
    )
    stratified_fraction = jnp.mod(
        rotation
        + strata.astype(mp_policy.measure_dtype)
        / jnp.asarray(shell_size, mp_policy.measure_dtype),
        1.0,
    )  # [S]
    same_group = (
        valid[:, None]
        & valid[None, :]
        & (log_L_constraint[:, None] == log_L_constraint[None, :])
    )  # [S, S]
    group_position = jnp.sum(
        same_group & (rows[None, :] < rows[:, None]),
        axis=1,
        dtype=mp_policy.index_dtype,
    )  # [S]
    reserved_same_group = (
        valid[:, None]
        & reserved_valid[None, :]
        & (
            log_L_constraint[:, None]
            == reserved_log_L_constraint[None, :]
        )
    )  # [S, S]
    reserved_count = jnp.sum(
        reserved_same_group,
        axis=1,
        dtype=mp_policy.index_dtype,
    )  # [S]
    retained_start_group = (
        valid
        & schedule.new_start
        & (
            log_L_constraint
            == schedule.start_seed_log_L_constraint
        )
    )  # [S]

    frozen_count = jax.vmap(
        lambda constraint: _frozen_seed_count_at_constraint(
            schedule,
            constraint,
        )
    )(log_L_constraint)  # [S]

    # The frozen rank index handles root and non-root contours uniformly. The
    # recent-row reservoir still distinguishes sentinel births from ordinary
    # birth contours when checking whether a retained row crosses the request.
    use_root = valid & from_root  # [S]
    reservoir_sample_idx = schedule.seed_reservoir_idx  # [R]
    reservoir_safe_idx = jnp.maximum(reservoir_sample_idx, 0)  # [R]
    reservoir_birth = state.samples.log_L_constraints[
        reservoir_safe_idx
    ]  # [R]
    reservoir_log_L = state.samples.log_likelihoods[
        reservoir_safe_idx
    ]  # [R]
    # The frozen index is exact. A value-independent bounded reservoir adds
    # recent stationary rows until geometric publication promotes the complete
    # generation; frontier ordering still prevents descendants from running
    # ahead of shallower allocation work.
    reservoir_eligible = (
        valid[:, None]
        & schedule.seed_reservoir_valid[None, :]
        & jnp.where(
            use_root[:, None],
            jnp.isneginf(reservoir_birth)[None, :],
            (
                reservoir_birth[None, :] <= log_L_constraint[:, None]
            )
            & (
                reservoir_log_L[None, :] > log_L_constraint[:, None]
            ),
        )
    )  # [S, R]
    reservoir_count = jnp.sum(
        reservoir_eligible,
        axis=1,
        dtype=mp_policy.index_dtype,
    )  # [S]
    # The recent reservoir evolves while a wide start group spans batches.
    # Count only retained identities that remain in the current eligible
    # frozen+reservoir population; evicted rows must not make an unseen recent
    # row look exhausted. Publication makes every older identity frozen, so
    # the same compact calculation remains exact across source generations.
    active_reservation = (
        schedule.start_seed_reservation_group
        == schedule.current_start_group
    )  # [V]
    published_reserved = jnp.sum(
        active_reservation
        & (
            schedule.start_seed_reservation_idx
            < schedule.source_num_samples
        ),
        dtype=mp_policy.index_dtype,
    )
    reservoir_was_reserved = jax.lax.cond(
        jnp.any(retained_start_group),
        lambda unused: jax.vmap(
            lambda one_seed: _seed_reservation_contains(
                schedule.start_seed_reservation_idx,
                schedule.start_seed_reservation_group,
                schedule.current_start_group,
                one_seed,
            )
        )(reservoir_safe_idx),
        lambda unused: jnp.zeros(
            reservoir_safe_idx.shape,
            dtype=mp_policy.bool_dtype,
        ),
        operand=None,
    )  # [R]
    retained_reserved = (
        published_reserved
        + jnp.sum(
            reservoir_eligible & reservoir_was_reserved[None, :],
            axis=1,
            dtype=mp_policy.index_dtype,
        )
    )  # [S]
    reserved_count = reserved_count + jnp.where(
        retained_start_group,
        retained_reserved,
        jnp.asarray(0, mp_policy.index_dtype),
    )
    proposal_width = 64
    selected_seed_idx = jnp.full(
        (shell_size,),
        -1,
        dtype=mp_policy.index_dtype,
    )  # [S]

    def select_lane(lane_idx, selected):
        constraint = log_L_constraint[lane_idx]
        lane_valid = valid[lane_idx]
        lane_retains_start_group = retained_start_group[lane_idx]

        # Earlier local lanes and in-flight distributed tasks are the complete
        # set of simultaneous reservations. Only equal-contour identities are
        # excluded; different constrained priors do not compete for seeds.
        prior_valid = valid & (rows < lane_idx)  # [S]
        forbidden_idx = jnp.concatenate(
            (reserved_seed_idx, selected),
            axis=0,
        )  # [2S]
        forbidden_constraint = jnp.concatenate(
            (reserved_log_L_constraint, log_L_constraint),
            axis=0,
        )  # [2S]
        forbidden_valid = jnp.concatenate(
            (reserved_valid, prior_valid),
            axis=0,
        )  # [2S]
        forbidden_same_contour = (
            forbidden_valid
            & (forbidden_idx >= 0)
            & (forbidden_constraint == constraint)
        )  # [2S]
        distinct_count = frozen_count[lane_idx].astype(
            mp_policy.index_dtype
        ) + reservoir_count[lane_idx]
        remaining_distinct = jnp.maximum(
            distinct_count - reserved_count[lane_idx],
            0,
        )
        require_unused = group_position[lane_idx] < remaining_distinct

        frozen_seed_count = frozen_count[lane_idx].astype(
            mp_policy.index_dtype
        )
        total_count = frozen_seed_count + reservoir_count[lane_idx]
        reservoir_cumulative = jnp.cumsum(
            reservoir_eligible[lane_idx].astype(mp_policy.index_dtype)
        )  # [R]

        def rejection_cond(carry):
            _, _, accepted = carry
            return lane_valid & jnp.logical_not(accepted)

        def rejection_body(carry):
            attempt, current_seed, _ = carry
            proposal_fraction = jax.random.uniform(
                jax.random.fold_in(
                    jax.random.fold_in(rejection_key, lane_idx),
                    attempt,
                ),
                (proposal_width,),
                dtype=mp_policy.measure_dtype,
            )  # [C]
            # The first proposal gives the batch systematic coverage of the
            # stationary lineage population. Random rotation and assignment
            # keep every lane marginally uniform without coupling seed rank to
            # the shallow-to-deep ordering of heterogeneous thread contours.
            # Independent proposals handle any no-replacement collision.
            proposal_fraction = proposal_fraction.at[0].set(jnp.where(
                attempt == 0,
                stratified_fraction[lane_idx],
                proposal_fraction[0],
            ))
            proposal_rank = (
                proposal_fraction * jnp.maximum(total_count, 1)
            ).astype(mp_policy.index_dtype)  # [C]
            frozen_offset = jnp.minimum(
                proposal_rank,
                jnp.maximum(frozen_seed_count - 1, 0),
            )
            frozen_fraction = (
                frozen_offset.astype(mp_policy.measure_dtype) + 0.5
            ) / jnp.maximum(
                frozen_seed_count,
                1,
            ).astype(mp_policy.measure_dtype)
            frozen_candidate = jax.vmap(
                lambda fraction: _sample_frozen_seed_rank(
                    schedule,
                    constraint,
                    fraction,
                )
            )(frozen_fraction)  # [C]
            reservoir_rank = jnp.maximum(
                proposal_rank - frozen_seed_count,
                0,
            )
            reservoir_slot = jnp.argmax(
                reservoir_cumulative[None, :] > reservoir_rank[:, None],
                axis=1,
            ).astype(mp_policy.index_dtype)  # [C]
            reservoir_candidate = reservoir_sample_idx[
                reservoir_slot
            ]  # [C]
            candidate = jnp.where(
                proposal_rank < frozen_seed_count,
                frozen_candidate,
                reservoir_candidate,
            )  # [C]
            safe_candidate = jnp.maximum(candidate, 0)
            candidate_forbidden = jnp.any(
                (candidate[:, None] == forbidden_idx[None, :])
                & forbidden_same_contour[None, :],
                axis=1,
            )  # [C]
            candidate_was_start_seed = jax.lax.cond(
                lane_retains_start_group,
                lambda unused: jax.vmap(
                    lambda one_seed: _seed_reservation_contains(
                        schedule.start_seed_reservation_idx,
                        schedule.start_seed_reservation_group,
                        schedule.current_start_group,
                        one_seed,
                    )
                )(safe_candidate),
                lambda unused: jnp.zeros(
                    candidate.shape,
                    dtype=mp_policy.bool_dtype,
                ),
                operand=None,
            )  # [C]
            eligible = jnp.logical_not(
                require_unused
                & (candidate_forbidden | candidate_was_start_seed)
            )
            found = jnp.any(eligible)
            first = jnp.argmax(eligible).astype(mp_policy.index_dtype)
            return (
                attempt + jnp.asarray(1, attempt.dtype),
                jnp.where(found, candidate[first], current_seed),
                found,
            )

        _, seed, _ = jax.lax.while_loop(
            rejection_cond,
            rejection_body,
            (
                jnp.asarray(0, mp_policy.index_dtype),
                jnp.asarray(0, mp_policy.index_dtype),
                jnp.logical_not(lane_valid),
            ),
        )
        return selected.at[lane_idx].set(jnp.where(lane_valid, seed, -1))

    return jax.lax.fori_loop(
        0,
        shell_size,
        select_lane,
        selected_seed_idx,
    )


class _ContinuationHeap(NamedTuple):
    """Small mutable view used while updating the continuation frontier."""

    parent_idx: IntArray  # [Q]
    thread_id: IntArray  # [Q]
    log_L_constraint: FloatArray  # [Q] requested contours
    frontier: FloatArray  # [Q] effective ordering contours
    terminal_log_L: FloatArray  # [Q]
    count: IntArray  # []


def _continuation_heap(schedule: ThreadSchedule) -> _ContinuationHeap:
    """Extract only fields that a heap update is allowed to mutate."""
    return _ContinuationHeap(
        parent_idx=schedule.continuation_parent_idx,
        thread_id=schedule.continuation_thread_id,
        log_L_constraint=schedule.continuation_log_L_constraint,
        frontier=schedule.continuation_frontier,
        terminal_log_L=schedule.continuation_terminal_log_L,
        count=schedule.continuation_count,
    )


def _replace_continuation_heap(
        schedule: ThreadSchedule,
        heap: _ContinuationHeap,
) -> ThreadSchedule:
    """Write one compact heap result back into its owning schedule."""
    return dataclasses.replace(
        schedule,
        continuation_parent_idx=heap.parent_idx,
        continuation_thread_id=heap.thread_id,
        continuation_log_L_constraint=heap.log_L_constraint,
        continuation_frontier=heap.frontier,
        continuation_terminal_log_L=heap.terminal_log_L,
        continuation_count=heap.count,
    )


def _push_thread_continuation(
        heap: _ContinuationHeap,
        parent_idx: IntArray,
        thread_id: IntArray,
        log_L_constraint: FloatArray,
        frontier: FloatArray,
        terminal_log_L: FloatArray,
        valid: BoolArray,
) -> _ContinuationHeap:
    """Insert one valid payload and restore the minimum-frontier invariant."""
    queue_size = heap.parent_idx.shape[0]
    append = valid & (heap.count < queue_size)
    position = jnp.minimum(
        heap.count,
        jnp.asarray(queue_size - 1, mp_policy.count_dtype),
    ).astype(mp_policy.index_dtype)
    inserted = _ContinuationHeap(
        parent_idx=heap.parent_idx.at[position].set(jnp.where(
            append,
            parent_idx,
            heap.parent_idx[position],
        )),
        thread_id=heap.thread_id.at[position].set(jnp.where(
            append,
            thread_id,
            heap.thread_id[position],
        )),
        log_L_constraint=heap.log_L_constraint.at[position].set(jnp.where(
            append,
            log_L_constraint,
            heap.log_L_constraint[position],
        )),
        frontier=heap.frontier.at[position].set(jnp.where(
            append,
            frontier,
            heap.frontier[position],
        )),
        terminal_log_L=heap.terminal_log_L.at[position].set(jnp.where(
            append,
            terminal_log_L,
            heap.terminal_log_L[position],
        )),
        count=heap.count + append.astype(heap.count.dtype),
    )

    def bubble_cond(carry):
        child, current = carry
        parent = (child - 1) // 2
        return (
            append
            & (child > 0)
            & (current.frontier[child] < current.frontier[parent])
        )

    def bubble_body(carry):
        child, current = carry
        parent = (child - 1) // 2
        current = _ContinuationHeap(
            parent_idx=(
                current.parent_idx
                .at[child].set(current.parent_idx[parent])
                .at[parent].set(current.parent_idx[child])
            ),
            thread_id=(
                current.thread_id
                .at[child].set(current.thread_id[parent])
                .at[parent].set(current.thread_id[child])
            ),
            log_L_constraint=(
                current.log_L_constraint
                .at[child].set(current.log_L_constraint[parent])
                .at[parent].set(current.log_L_constraint[child])
            ),
            frontier=(
                current.frontier
                .at[child].set(current.frontier[parent])
                .at[parent].set(current.frontier[child])
            ),
            terminal_log_L=(
                current.terminal_log_L
                .at[child].set(current.terminal_log_L[parent])
                .at[parent].set(current.terminal_log_L[child])
            ),
            count=current.count,
        )
        return parent, current

    _, inserted = jax.lax.while_loop(
        bubble_cond,
        bubble_body,
        (position, inserted),
    )
    return inserted


def _enqueue_thread_continuations(
        schedule: ThreadSchedule,
        parent_idx: IntArray,
        thread_id: IntArray,
        log_L_constraint: FloatArray,
        terminal_log_L: FloatArray,
        valid: BoolArray,
) -> ThreadSchedule:
    """Insert continuing heads into the bounded minimum-contour heap."""
    frontiers = jax.vmap(
        lambda constraint: _effective_parent_contour(
            schedule,
            constraint,
        )[2]
    )(log_L_constraint)  # [S]

    def enqueue(lane_idx, heap: _ContinuationHeap) -> _ContinuationHeap:
        return _push_thread_continuation(
            heap,
            parent_idx[lane_idx],
            thread_id[lane_idx],
            log_L_constraint[lane_idx],
            frontiers[lane_idx],
            terminal_log_L[lane_idx],
            valid[lane_idx],
        )

    heap = jax.lax.fori_loop(
        0,
        valid.shape[0],
        enqueue,
        _continuation_heap(schedule),
    )
    return _replace_continuation_heap(schedule, heap)


def _pop_continuation_heap(
        heap: _ContinuationHeap,
) -> _ContinuationHeap:
    """Remove the shallowest payload and restore the heap invariant."""
    count = heap.count - jnp.asarray(
        1,
        heap.count.dtype,
    )
    last = count.astype(mp_policy.index_dtype)
    parent_idx = (
        heap.parent_idx
        .at[0].set(heap.parent_idx[last])
        .at[last].set(-1)
    )
    thread_id = (
        heap.thread_id
        .at[0].set(heap.thread_id[last])
        .at[last].set(-1)
    )
    log_L_constraint = (
        heap.log_L_constraint
        .at[0].set(heap.log_L_constraint[last])
        .at[last].set(-jnp.inf)
    )
    frontier = (
        heap.frontier
        .at[0].set(heap.frontier[last])
        .at[last].set(jnp.inf)
    )
    terminal_log_L = (
        heap.terminal_log_L
        .at[0].set(heap.terminal_log_L[last])
        .at[last].set(-jnp.inf)
    )
    heap = _ContinuationHeap(
        parent_idx=parent_idx,
        thread_id=thread_id,
        log_L_constraint=log_L_constraint,
        frontier=frontier,
        terminal_log_L=terminal_log_L,
        count=count,
    )

    def bubble_cond(carry):
        parent, current, moving = carry
        left = 2 * parent + 1
        right = left + 1
        safe_left = jnp.minimum(left, jnp.maximum(count - 1, 0))
        safe_right = jnp.minimum(right, jnp.maximum(count - 1, 0))
        choose_right = (
            (right < count)
            & (
                current.frontier[safe_right]
                < current.frontier[safe_left]
            )
        )
        child = jnp.where(choose_right, right, left)
        safe_child = jnp.minimum(child, jnp.maximum(count - 1, 0))
        keep_moving = (
            (left < count)
            & (
                current.frontier[safe_child]
                < current.frontier[parent]
            )
        )
        return moving & keep_moving

    def bubble_body(carry):
        parent, current, _ = carry
        left = 2 * parent + 1
        right = left + 1
        safe_right = jnp.minimum(right, jnp.maximum(count - 1, 0))
        choose_right = (
            (right < count)
            & (
                current.frontier[safe_right]
                < current.frontier[left]
            )
        )
        child = jnp.where(choose_right, right, left)
        current = _ContinuationHeap(
            parent_idx=(
                current.parent_idx
                .at[parent].set(current.parent_idx[child])
                .at[child].set(current.parent_idx[parent])
            ),
            thread_id=(
                current.thread_id
                .at[parent].set(current.thread_id[child])
                .at[child].set(current.thread_id[parent])
            ),
            log_L_constraint=(
                current.log_L_constraint
                .at[parent].set(current.log_L_constraint[child])
                .at[child].set(current.log_L_constraint[parent])
            ),
            frontier=(
                current.frontier
                .at[parent].set(current.frontier[child])
                .at[child].set(current.frontier[parent])
            ),
            terminal_log_L=(
                current.terminal_log_L
                .at[parent].set(current.terminal_log_L[child])
                .at[child].set(current.terminal_log_L[parent])
            ),
            count=current.count,
        )
        return child, current, jnp.asarray(True, mp_policy.bool_dtype)

    _, heap, _ = jax.lax.while_loop(
        bubble_cond,
        bubble_body,
        (
            jnp.asarray(0, mp_policy.index_dtype),
            heap,
            count > 1,
        ),
    )
    return heap


def _pop_thread_continuation(schedule: ThreadSchedule) -> ThreadSchedule:
    """Remove the shallowest continuation from a complete schedule."""
    return _replace_continuation_heap(
        schedule,
        _pop_continuation_heap(_continuation_heap(schedule)),
    )


class _ThreadHeadCarry(NamedTuple):
    """Only the fields mutated while merging starts and continuations."""

    parent_idx: IntArray  # [S]
    thread_id: IntArray  # [S]
    log_L_constraint: FloatArray  # [S]
    terminal_log_L: FloatArray  # [S]
    new_start: BoolArray  # [S]
    valid: BoolArray  # [S]
    heap: _ContinuationHeap
    next_run: IntArray  # []
    remaining_in_run: IntArray  # []
    next_thread_id: IntArray  # []


def _fill_thread_heads(
        key: PRNGKey,
        state: State,
        schedule: ThreadSchedule,
) -> ThreadSchedule:
    """Fill lanes from the shallowest frozen or continuing thread frontier."""
    del state
    shell_size = schedule.valid.shape[0]
    parent_keys = jax.random.split(key, shell_size)  # [S, 2]

    def fill_lane(
            lane_idx,
            current: _ThreadHeadCarry,
    ) -> _ThreadHeadCarry:
        empty = jnp.logical_not(current.valid[lane_idx])
        has_start = current.next_run < schedule.num_runs
        run_idx = jnp.minimum(
            current.next_run,
            jnp.maximum(schedule.num_runs - 1, 0),
        ).astype(mp_policy.index_dtype)
        start_block = schedule.start_block[run_idx]
        terminal_block = schedule.terminal_block[run_idx]
        parent_block = start_block - 1
        parent_idx = _sample_parent_from_block(
            parent_keys[lane_idx],
            schedule.block_state,
            parent_block,
        )
        log_L_constraint = jnp.where(
            parent_block >= 0,
            schedule.block_state.log_L_blocks[jnp.maximum(parent_block, 0)],
            jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        )
        terminal_log_L = schedule.block_state.log_L_blocks[terminal_block]
        _, _, start_frontier_constraint = _effective_parent_contour(
            schedule,
            log_L_constraint,
        )

        # A child that has not reached its terminal contour creates a live
        # frontier at its own likelihood. Merge those frontiers with frozen
        # starts by contour, just as rebuilding the allocation gap would do,
        # without reconstructing or sorting the complete race tree.
        has_continuation = current.heap.count > 0
        continuation_frontier = current.heap.frontier[0]
        start_new = (
            empty
            & has_start
            & (
                jnp.logical_not(has_continuation)
                | (start_frontier_constraint <= continuation_frontier)
            )
        )
        resume = empty & has_continuation & jnp.logical_not(start_new)
        queued_parent_idx = current.heap.parent_idx[0]
        queued_log_L_constraint = current.heap.log_L_constraint[0]
        queued_thread_id = current.heap.thread_id[0]
        queued_terminal_log_L = current.heap.terminal_log_L[0]
        selected_parent_idx = jnp.where(
            resume,
            queued_parent_idx,
            parent_idx,
        )
        selected_thread_id = jnp.where(
            resume,
            queued_thread_id,
            current.next_thread_id,
        )
        selected_log_L_constraint = jnp.where(
            resume,
            queued_log_L_constraint,
            log_L_constraint,
        )
        selected_terminal_log_L = jnp.where(
            resume,
            queued_terminal_log_L,
            terminal_log_L,
        )
        fill = start_new | resume

        remaining = current.remaining_in_run - start_new.astype(
            current.remaining_in_run.dtype
        )
        run_finished = start_new & (remaining == 0)
        next_run = current.next_run + run_finished.astype(
            current.next_run.dtype
        )
        safe_next_run = jnp.minimum(
            next_run,
            jnp.maximum(schedule.num_runs - 1, 0),
        ).astype(mp_policy.index_dtype)
        remaining = jnp.where(
            run_finished & (next_run < schedule.num_runs),
            schedule.multiplicity[safe_next_run],
            remaining,
        )
        heap = jax.lax.cond(
            resume,
            _pop_continuation_heap,
            lambda value: value,
            current.heap,
        )
        return _ThreadHeadCarry(
            parent_idx=current.parent_idx.at[lane_idx].set(jnp.where(
                fill,
                selected_parent_idx,
                current.parent_idx[lane_idx],
            )),
            thread_id=current.thread_id.at[lane_idx].set(jnp.where(
                fill,
                selected_thread_id,
                current.thread_id[lane_idx],
            )),
            log_L_constraint=current.log_L_constraint.at[lane_idx].set(
                jnp.where(
                    fill,
                    selected_log_L_constraint,
                    current.log_L_constraint[lane_idx],
                )
            ),
            terminal_log_L=current.terminal_log_L.at[lane_idx].set(
                jnp.where(
                    fill,
                    selected_terminal_log_L,
                    current.terminal_log_L[lane_idx],
                )
            ),
            new_start=current.new_start.at[lane_idx].set(jnp.where(
                fill,
                start_new,
                current.new_start[lane_idx],
            )),
            valid=current.valid.at[lane_idx].set(
                current.valid[lane_idx] | fill
            ),
            heap=heap,
            next_run=next_run,
            remaining_in_run=remaining,
            next_thread_id=(
                current.next_thread_id
                + start_new.astype(current.next_thread_id.dtype)
            ),
        )

    heads = jax.lax.fori_loop(
        0,
        shell_size,
        fill_lane,
        _ThreadHeadCarry(
            parent_idx=schedule.parent_idx,
            thread_id=schedule.thread_id,
            log_L_constraint=schedule.log_L_constraint,
            terminal_log_L=schedule.terminal_log_L,
            new_start=schedule.new_start,
            valid=schedule.valid,
            heap=_continuation_heap(schedule),
            next_run=schedule.next_run,
            remaining_in_run=schedule.remaining_in_run,
            next_thread_id=schedule.next_thread_id,
        ),
    )
    schedule = dataclasses.replace(
        schedule,
        parent_idx=heads.parent_idx,
        thread_id=heads.thread_id,
        log_L_constraint=heads.log_L_constraint,
        terminal_log_L=heads.terminal_log_L,
        new_start=heads.new_start,
        valid=heads.valid,
        next_run=heads.next_run,
        remaining_in_run=heads.remaining_in_run,
        next_thread_id=heads.next_thread_id,
    )
    return _replace_continuation_heap(schedule, heads.heap)


def _retain_start_seed_reservations(
        schedule: ThreadSchedule,
        seed_idx: IntArray,
        effective_constraint: FloatArray,
        dispatched: BoolArray,
) -> ThreadSchedule:
    """Carry no-replacement seeds only while one start contour is unfinished.

    Frozen runs are ordered by start contour, so only the final start contour
    in a planning window can remain for a later batch. Retaining that one group
    bounds coordination state by sample capacity rather than by the number of
    logical threads. Reservations are discarded once the group is fully
    dispatched; continuations are independent Markov chains and do not
    participate in this same-contour start assignment.
    """
    # A physical-capacity boundary may leave already-materialised starts in
    # the window. They precede compressed runs that have not yet been opened.
    undispatched = (
        schedule.valid
        & schedule.new_start
        & jnp.logical_not(dispatched)
    )  # [S]
    has_window_start = jnp.any(undispatched)
    window_start = jnp.argmax(undispatched).astype(mp_policy.index_dtype)
    window_constraint = schedule.log_L_constraint[window_start]

    has_run_start = schedule.next_run < schedule.num_runs
    safe_run = jnp.minimum(
        schedule.next_run,
        jnp.maximum(schedule.num_runs - 1, 0),
    ).astype(mp_policy.index_dtype)
    start_block = schedule.start_block[safe_run]
    parent_block = start_block - 1
    run_constraint = jnp.where(
        parent_block >= 0,
        schedule.block_state.log_L_blocks[jnp.maximum(parent_block, 0)],
        jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
    )
    requested_constraint = jnp.where(
        has_window_start,
        window_constraint,
        run_constraint,
    )
    has_future_start = has_window_start | has_run_start
    _, _, future_constraint = _effective_parent_contour(
        schedule,
        requested_constraint,
    )

    keep_existing = (
        has_future_start
        & (
            schedule.start_seed_log_L_constraint
            == future_constraint
        )
    )
    keep_selected = (
        has_future_start
        & dispatched
        & schedule.new_start
        & (effective_constraint == future_constraint)
    )  # [S]
    # Changing the scalar group identity logically clears every reservation.
    # The exact hash set grows only with this unfinished contour's used seeds,
    # rather than carrying and updating one tag for every scientific sample.
    group = jnp.where(
        keep_existing,
        schedule.current_start_group,
        schedule.current_start_group
        + jnp.asarray(1, mp_policy.index_dtype),
    )
    reservation_idx = schedule.start_seed_reservation_idx  # [V]
    reservation_group = schedule.start_seed_reservation_group  # [V]
    num_used = jnp.where(
        keep_existing,
        schedule.num_start_seeds,
        jnp.asarray(0, mp_policy.count_dtype),
    )

    selected_idx = seed_idx.astype(mp_policy.index_dtype)  # [S]
    safe_idx = jnp.maximum(selected_idx, 0)  # [S]
    retain = (
        keep_selected
        & (selected_idx >= 0)
        & (selected_idx < schedule.seed_birth_contours.shape[0])
    )  # [S]

    def retain_lane(lane_idx, carry):
        current_idx, current_group, current_count = carry
        already_used = _seed_reservation_contains(
            current_idx,
            current_group,
            group,
            safe_idx[lane_idx],
        )

        def insert(unused):
            del unused
            next_idx, next_group = _insert_seed_reservation(
                selected_idx[lane_idx],
                group,
                current_idx,
                current_group,
            )
            return (
                next_idx,
                next_group,
                current_count + jnp.asarray(1, current_count.dtype),
            )

        return jax.lax.cond(
            retain[lane_idx] & jnp.logical_not(already_used),
            insert,
            lambda unused: carry,
            operand=None,
        )

    reservation_idx, reservation_group, num_used = jax.lax.fori_loop(
        0,
        seed_idx.shape[0],
        retain_lane,
        (reservation_idx, reservation_group, num_used),
    )
    return dataclasses.replace(
        schedule,
        start_seed_reservation_idx=reservation_idx,
        start_seed_reservation_group=reservation_group,
        current_start_group=group,
        start_seed_log_L_constraint=jnp.where(
            has_future_start,
            future_constraint,
            -jnp.inf,
        ),
        num_start_seeds=num_used,
    )


def _plan_scheduled_work_batch(
        key: PRNGKey,
        state: State,
        schedule: ThreadSchedule,
        max_valid_lanes: IntArray,
        reserved_seed_idx: IntArray | None = None,
        reserved_log_L_constraint: FloatArray | None = None,
        reserved_valid: BoolArray | None = None,
) -> tuple[ThreadSchedule, CoreWorkBatch]:
    """Resolve one fixed-width batch from already materialised threads."""
    parent_key, seed_key = jax.random.split(key)
    schedule = _fill_thread_heads(parent_key, state, schedule)
    shell_size = schedule.valid.shape[0]
    if reserved_seed_idx is None:
        reserved_seed_idx = jnp.full(
            (shell_size,),
            -1,
            dtype=mp_policy.index_dtype,
        )
    if reserved_log_L_constraint is None:
        reserved_log_L_constraint = jnp.full(
            (shell_size,),
            -jnp.inf,
            dtype=mp_policy.measure_dtype,
        )
    if reserved_valid is None:
        reserved_valid = jnp.zeros(
            (shell_size,),
            dtype=mp_policy.bool_dtype,
        )
    slots = jnp.arange(shell_size, dtype=mp_policy.index_dtype)
    num_valid = jnp.minimum(
        jnp.sum(schedule.valid, dtype=mp_policy.index_dtype),
        jnp.maximum(max_valid_lanes, 0),
    )
    valid = slots < num_valid

    requested_has_seed, effective_block, effective_constraint = jax.vmap(
        lambda constraint: _effective_parent_contour(schedule, constraint)
    )(schedule.log_L_constraint)
    fallback_keys = jax.random.split(parent_key, shell_size)
    fallback_parent = jax.vmap(
        lambda one_key, block_idx: _sample_parent_from_block(
            one_key,
            schedule.seed_block_state,
            block_idx,
        )
    )(fallback_keys, effective_block)
    parent_idx = jnp.where(
        requested_has_seed,
        schedule.parent_idx,
        fallback_parent,
    )

    # A distributed start remains pending after its identity is recorded in
    # the schedule mask. Exclude that exact duplicate from the simultaneous
    # task count; pending continuations remain independent reservations.
    safe_reserved_idx = jnp.maximum(reserved_seed_idx, 0)  # [S]
    matches_retained_identity = jax.lax.cond(
        jnp.any(reserved_valid),
        lambda unused: jax.vmap(
            lambda one_seed: _seed_reservation_contains(
                schedule.start_seed_reservation_idx,
                schedule.start_seed_reservation_group,
                schedule.current_start_group,
                one_seed,
            )
        )(safe_reserved_idx),
        lambda unused: jnp.zeros(
            reserved_valid.shape,
            dtype=mp_policy.bool_dtype,
        ),
        operand=None,
    )  # [S]
    pending_matches_start = (
        reserved_valid
        & (reserved_seed_idx >= 0)
        & (reserved_seed_idx < schedule.seed_birth_contours.shape[0])
        & (
            reserved_log_L_constraint
            == schedule.start_seed_log_L_constraint
        )
        & matches_retained_identity
    )  # [S]
    unique_pending_valid = (
        reserved_valid & jnp.logical_not(pending_matches_start)
    )  # [S]
    seed_idx = _sample_stationary_seeds(
        seed_key,
        state,
        schedule,
        effective_constraint,
        jnp.isneginf(effective_constraint),
        valid,
        reserved_seed_idx,
        reserved_log_L_constraint,
        unique_pending_valid,
    )
    schedule = _retain_start_seed_reservations(
        schedule,
        seed_idx,
        effective_constraint,
        valid,
    )
    # Fixed-width padding can be traced through both sides of vmapped sampler
    # control flow even though invalid lanes never enter scientific state.
    # Give every padded lane the first valid request so those speculative
    # calculations still start inside a real contour. This is also required
    # by the complete-chain reference path, which has no validity mask.
    first_valid = jnp.argmax(valid).astype(mp_policy.index_dtype)
    parent_idx = jnp.where(valid, parent_idx, parent_idx[first_valid])
    effective_constraint = jnp.where(
        valid,
        effective_constraint,
        effective_constraint[first_valid],
    )
    seed_idx = jnp.where(valid, seed_idx, seed_idx[first_valid])
    return schedule, CoreWorkBatch(
        valid=valid,
        parent_idx=parent_idx,
        log_L_constraint=effective_constraint,
        seed_idx=jnp.maximum(seed_idx, 0),
    )


def _advance_thread_schedule(
        schedule: ThreadSchedule,
        work: CoreWorkBatch,
        batch: ConstrainedSampleBatch,
        insert_idx: IntArray,
) -> ThreadSchedule:
    """Queue sampled continuations and compact only unsampled heads."""
    shell_size = schedule.valid.shape[0]
    slots = jnp.arange(shell_size, dtype=mp_policy.index_dtype)
    continuing = work.valid & (
        batch.log_likelihoods < schedule.terminal_log_L
    )
    schedule = _enqueue_thread_continuations(
        schedule,
        insert_idx.astype(mp_policy.index_dtype) + slots,
        schedule.thread_id,
        batch.log_likelihoods,
        schedule.terminal_log_L,
        continuing,
    )
    # Both masks are valid prefixes. An offset gather preserves the unsampled
    # suffix without sorting the dispatch window on every replacement batch.
    num_sampled = jnp.sum(work.valid, dtype=mp_policy.index_dtype)
    num_active = jnp.sum(schedule.valid, dtype=mp_policy.index_dtype)
    num_remaining = num_active - num_sampled
    source = jnp.minimum(
        slots + num_sampled,
        jnp.asarray(shell_size - 1, mp_policy.index_dtype),
    )
    valid = slots < num_remaining
    active = (
        jnp.any(valid)
        | (schedule.next_run < schedule.num_runs)
        | (schedule.continuation_count > 0)
    )
    return dataclasses.replace(
        schedule,
        parent_idx=schedule.parent_idx[source],
        thread_id=schedule.thread_id[source],
        log_L_constraint=schedule.log_L_constraint[source],
        terminal_log_L=schedule.terminal_log_L[source],
        new_start=schedule.new_start[source],
        valid=valid,
        active=active,
    )


def _release_thread_heads(
        schedule: ThreadSchedule,
        dispatched: BoolArray,
) -> ThreadSchedule:
    """Transfer dispatched heads out of a distributed schedule.

    A local call keeps heads in the fixed-width schedule until sampling
    returns. Distributed calls instead move each dispatched head into a
    retry-stable ``PendingTask``. Compacting only undispatched heads leaves
    exactly the vacated slots needed when those tasks later continue.
    """
    shell_size = schedule.valid.shape[0]
    slots = jnp.arange(shell_size, dtype=mp_policy.index_dtype)
    num_dispatched = jnp.sum(dispatched, dtype=mp_policy.index_dtype)
    num_active = jnp.sum(schedule.valid, dtype=mp_policy.index_dtype)
    num_remaining = num_active - num_dispatched
    source = jnp.minimum(
        slots + num_dispatched,
        jnp.asarray(shell_size - 1, mp_policy.index_dtype),
    )
    valid = slots < num_remaining
    return dataclasses.replace(
        schedule,
        parent_idx=schedule.parent_idx[source],
        thread_id=schedule.thread_id[source],
        log_L_constraint=schedule.log_L_constraint[source],
        terminal_log_L=schedule.terminal_log_L[source],
        new_start=schedule.new_start[source],
        valid=valid,
        active=(
            jnp.any(valid)
            | (schedule.next_run < schedule.num_runs)
            | (schedule.continuation_count > 0)
        ),
    )


def _insert_thread_head(
        schedule: ThreadSchedule,
        thread_id: IntArray,
        parent_idx: IntArray,
        log_L_constraint: FloatArray,
        terminal_log_L: FloatArray,
        continuing: BoolArray,
) -> ThreadSchedule:
    """Return one completed distributed edge to the continuation heap."""
    updated = _enqueue_thread_continuations(
        schedule,
        jnp.reshape(parent_idx, (1,)),
        jnp.reshape(thread_id, (1,)),
        jnp.reshape(log_L_constraint, (1,)),
        jnp.reshape(terminal_log_L, (1,)),
        jnp.reshape(continuing, (1,)),
    )
    return dataclasses.replace(
        updated,
        active=(
            jnp.any(updated.valid)
            | (updated.next_run < updated.num_runs)
            | (updated.continuation_count > 0)
        ),
    )


def _sample_work_batch(
        key: PRNGKey,
        state: State,
        sampler: AbstractSampler,
        work: CoreWorkBatch,
) -> ConstrainedSampleBatch:
    """Execute the planned batch through the sampler-owned batch strategy."""
    keys = jax.random.split(key, work.seed_idx.shape[0])
    seed_points = SeedPoint(
        U0=jax.tree.map(
            lambda values: values[work.seed_idx],
            state.samples.U_samples,
        ),
        log_L0=state.samples.log_likelihoods[work.seed_idx],
    )
    # Parent planning and race accounting remain outside constrained sampling.
    # A self-contained request is shared by the local depth loop and workers,
    # so both execution modes use exactly the same continuation law.
    request = ConstrainedSampleRequest(
        keys=keys,
        valid=work.valid,
        log_L_constraints=work.log_L_constraint,
        seed_points=seed_points,
        sampler_data=state.sampler_data,
    )
    return sample_request(
        sampler,
        request,
        args=state.args,
        params=state.params,
    )


def _prepare_sampler_data(
        key: PRNGKey,
        state: State,
        sampler: AbstractSampler,
        work: CoreWorkBatch,
        schedule: ThreadSchedule,
) -> State:
    """Update geometry from one internally consistent frozen population."""
    block_state = schedule.seed_block_state
    population_num_samples = schedule.source_num_samples
    direction = sampler.direction_config()
    if direction is None:
        raise ValueError(
            "Adaptive direction sampling requires a direction configuration."
        )
    if state.sampler_data is None:
        raise ValueError("Ellipsoidal sampling requires sampler_data on State.")

    dimension = state.sampler_data.centres.shape[1]
    min_effective_samples = direction.min_effective_samples
    if min_effective_samples is None:
        # A full covariance needs at least D+1 independent points per
        # component. Four times that algebraic minimum is a conservative
        # initial gate and remains configurable for benchmark sweeps.
        min_effective_samples = (
            4 * direction.num_components * (dimension + 1)
        )
    retry_increment = max(1, min_effective_samples // 2)

    ready = jnp.any(state.sampler_data.valid)
    lane_has_geometry = jax.vmap(
        lambda constraint: jnp.any(
            state.sampler_data.valid
            & (state.sampler_data.log_L_max > constraint)
        )
    )(work.log_L_constraint)
    stale = jnp.any(work.valid & jnp.logical_not(lane_has_geometry))
    first_attempt = state.sampler_data.num_attempted == 0
    enough_rows = population_num_samples >= min_effective_samples
    enough_new_rows = (
        population_num_samples
        >= state.sampler_data.num_attempted + retry_increment
    )
    initialise = jnp.where(first_attempt, enough_rows, enough_new_rows)
    last_attempt_succeeded = (
        state.sampler_data.num_attempted
        == state.sampler_data.num_samples
    )
    population_advanced = (
        population_num_samples > state.sampler_data.num_attempted
    )
    # A newly stale successful model is refreshed immediately. If that refresh
    # fails, wait for a materially larger population before retrying so a
    # singular late contour cannot make every replacement batch refit it.
    refresh = (
        stale
        & population_advanced
        & (last_attempt_succeeded | enough_new_rows)
    )
    should_update = jnp.where(ready, refresh, initialise)

    def update(_):
        concentrations = classic_dirichlet_concentrations(block_state)
        alpha0 = (
            concentrations.alpha_gt
            + concentrations.alpha_eq
            + concentrations.alpha_lt
        )
        p_gt = jnp.where(
            alpha0 > 0.0,
            concentrations.alpha_gt / alpha0,
            1.0,
        )
        p_eq = jnp.where(
            alpha0 > 0.0,
            concentrations.alpha_eq / alpha0,
            0.0,
        )

        # Compute the expected volume prefix with a dynamic sequential loop.
        # The relevant operations then have identical order before and after a
        # physical buffer resize, instead of allowing a capacity-dependent
        # parallel reduction to perturb the direction kernel's last bits.
        def volume_step(block_idx, carry):
            log_X, log_X_prev = carry
            log_X_prev = log_X_prev.at[block_idx].set(log_X)
            log_X = log_X + jnp.log(jnp.clip(
                p_gt[block_idx],
                1e-300,
                1.0,
            ))
            return log_X, log_X_prev

        _, log_X_prev = jax.lax.fori_loop(
            0,
            block_state.num_blocks,
            volume_step,
            (
                jnp.asarray(0.0, mp_policy.measure_dtype),
                jnp.zeros_like(block_state.log_L_blocks),
            ),
        )
        plateau_weight = (
            block_state.log_L_blocks
            + log_X_prev
            + jnp.log(jnp.clip(p_eq, 1e-300, 1.0))
            - jnp.log(jnp.maximum(
                block_state.block_size,
                1,
            ).astype(mp_policy.measure_dtype))
        )
        ordinary_weight = (
            block_state.log_L_blocks
            + log_X_prev
            + jnp.log(jnp.clip(1.0 - p_gt, 1e-300, 1.0))
        )
        block_log_weights = jnp.where(
            block_state.block_size > 1,
            plateau_weight,
            ordinary_weight,
        )

        # A fixed likelihood-ordered population makes the fit's HLO and
        # arithmetic independent of State's physical storage capacity. Use all
        # rows while they fit, then an evenly spaced deterministic subset.
        population_size = direction.population_size
        slots = jnp.arange(population_size, dtype=mp_policy.index_dtype)
        use_all = population_num_samples <= population_size
        spread = jnp.floor(
            slots.astype(mp_policy.measure_dtype)
            * population_num_samples.astype(mp_policy.measure_dtype)
            / jnp.asarray(population_size, mp_policy.measure_dtype)
        ).astype(mp_policy.index_dtype)
        positions = jnp.where(use_all, slots, spread)
        mask = jnp.where(
            use_all,
            slots < population_num_samples,
            jnp.ones((population_size,), mp_policy.bool_dtype),
        )
        positions = jnp.minimum(
            positions,
            jnp.maximum(population_num_samples - 1, 0),
        )
        sample_indices = block_state.block_sample_indices[positions]
        sample_indices = jnp.where(mask, sample_indices, 0)
        selected_U = jax.tree.map(
            lambda values: values[sample_indices],
            state.samples.U_samples,
        )
        points = jax.vmap(lambda sample: pytree_ravel(sample)[0])(
            selected_U
        )
        selected_log_L = state.samples.log_likelihoods[sample_indices]
        selected_blocks = jnp.searchsorted(
            block_state.log_L_blocks,
            selected_log_L,
            side="left",
        )
        selected_blocks = jnp.clip(
            selected_blocks,
            0,
            block_state.log_L_blocks.shape[0] - 1,
        )
        log_weights = jnp.where(
            mask,
            block_log_weights[selected_blocks],
            -jnp.inf,
        )
        data = update_sampler_data(
            key,
            state.sampler_data,
            points,
            selected_log_L,
            log_weights,
            mask,
            population_num_samples,
            n_iters=direction.num_iterations,
            min_effective_samples=min_effective_samples,
            regularisation=direction.regularisation,
        )
        return dataclasses.replace(state, sampler_data=data)

    return jax.lax.cond(
        should_update,
        update,
        lambda unused: state,
        operand=None,
    )


def _accept_work_batch(
        state: State,
        work: CoreWorkBatch,
        batch: ConstrainedSampleBatch,
        *,
        update_likelihood_order: bool = True,
) -> State:
    shell_size = batch.log_likelihoods.shape[0]
    # Phantom shrinkage needs likelihoods and cluster identity, not phantom
    # coordinates. Discarding coordinates keeps the persistent state compact.
    # `NestedSampler.store_phantom_samples` remains a constructor-compatibility
    # field, but is intentionally absent from this compiled hot path.
    stored_phantoms = dataclasses.replace(
        batch.phantom_samples,
        U_samples=None,
    )
    new_samples = Samples(
        log_L_constraints=work.log_L_constraint,
        log_likelihoods=batch.log_likelihoods,
        U_samples=batch.U_samples,
        out_degree=jnp.zeros((shell_size,), mp_policy.count_dtype),
        num_likelihood_evaluations=(
            batch.num_likelihood_evaluations.astype(
                mp_policy.count_dtype
            )
        ),
        phantom_samples=stored_phantoms,
    )
    # The static batch always has S rows, but valid lanes form the visible
    # prefix. Invalid tail rows remain outside num_samples and are overwritten
    # by a later batch. Only valid lanes can alter the race-tree degrees.
    parent_slots = jnp.maximum(work.parent_idx, 0)
    parent_delta = (work.valid & (work.parent_idx >= 0)).astype(
        state.samples.out_degree.dtype
    )
    appended = state.samples.append_samples(
        state.num_samples,
        parent_slots,
        new_samples,
        parent_delta,
    )

    classic_idx = jnp.argmax(
        jnp.where(work.valid, batch.log_likelihoods, -jnp.inf)
    )
    candidate_log_L = batch.log_likelihoods[classic_idx]
    candidate_U = jax.tree.map(
        lambda u: u[classic_idx],
        batch.U_samples,
    )
    improves = candidate_log_L > state.log_L_supremum
    sampler_data = state.sampler_data
    if sampler_data is not None:
        sampler_data = dataclasses.replace(
            sampler_data,
            num_directions=(
                sampler_data.num_directions
                + jnp.sum(
                    jnp.where(work.valid, batch.num_directions, 0)
                ).astype(mp_policy.count_dtype)
            ),
            num_isotropic=(
                sampler_data.num_isotropic
                + jnp.sum(
                    jnp.where(work.valid, batch.num_isotropic, 0)
                ).astype(mp_policy.count_dtype)
            ),
        )
    return dataclasses.replace(
        state,
        root_out_degree=(
            state.root_out_degree
            + jnp.sum(work.valid & (work.parent_idx < 0)).astype(
                state.root_out_degree.dtype
            )
        ),
        samples=appended,
        num_samples=(
            state.num_samples
            + work.num_valid.astype(state.num_samples.dtype)
        ),
        log_L_supremum=jnp.where(
            improves,
            candidate_log_L,
            state.log_L_supremum,
        ),
        U_supremum=jax.tree.map(
            lambda u_new, u_old: jnp.where(improves, u_new, u_old),
            candidate_U,
            state.U_supremum,
        ),
        depth_loop_iter=state.depth_loop_iter + jnp.asarray(
            1,
            state.depth_loop_iter.dtype,
        ),
        likelihood_order=(
            None
            if state.likelihood_order is None
            else state.likelihood_order.insert(
                appended.log_likelihoods,
                state.num_samples,
                work.valid,
            )
            if update_likelihood_order
            else state.likelihood_order
        ),
        sampler_data=sampler_data,
    )


class _DepthCarry(NamedTuple):
    key: PRNGKey  # [2]
    state: State
    schedule: ThreadSchedule
    depth_done: BoolArray  # []


@partial(jax.jit, inline=True)
def _refresh_likelihood_order(state: State) -> State:
    """Publish one generation's post-freeze rows into the sorted index."""
    if state.likelihood_order is None or state.scheduler_data is None:
        # Compatibility states without the persistent index have no merge
        # source. This one-time sort is not used by normally initialised runs.
        return dataclasses.replace(
            state,
            likelihood_order=initialise_likelihood_order(
                state.samples.log_likelihoods,
                state.num_samples,
            ),
        )

    order = state.likelihood_order
    published = jnp.sum(
        order.sample_indices >= 0,
        dtype=mp_policy.index_dtype,
    )
    # Publication is independent of the much smaller continuation heap. A
    # generation stops after at most 25% capacity growth plus one dispatch
    # window, so this static merge width covers every accepted identity without
    # coupling per-batch heap scatters back to total sample capacity.
    capacity = state.samples.log_likelihoods.shape[0]
    merge_size = min(
        capacity,
        (capacity + 3) // 4 + state.scheduler_data.valid.shape[0],
    )
    offsets = jnp.arange(merge_size, dtype=mp_policy.index_dtype)  # [P]
    valid_new = published + offsets < state.num_samples  # [P]
    order = jax.lax.cond(
        jnp.any(valid_new),
        lambda current: current.insert(
            state.samples.log_likelihoods,
            published,
            valid_new,
        ),
        lambda current: current,
        order,
    )
    return dataclasses.replace(
        state,
        likelihood_order=order,
    )


@partial(jax.jit, inline=True)
def _publish_seed_source(state: State) -> State:
    """Promote a complete generation without changing its logical threads.

    The allocation target, compressed runs, active heads, and continuation
    heap define scientific work and remain frozen. Only the exact stationary
    seed view advances to include newly accepted samples; the bounded recent
    reservoir can then restart empty because all of its rows are now indexed.
    """
    state = _refresh_likelihood_order(state)
    schedule = state.scheduler_data
    if schedule is None:
        raise ValueError("Seed publication requires an active schedule.")
    block_state = build_block_state(
        state.samples,
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
        likelihood_order=state.likelihood_order,
    )
    source = _build_seed_source_index(state, block_state)
    schedule = dataclasses.replace(
        schedule,
        seed_block_state=source.block_state,
        seed_count=source.count,
        previous_seedable=source.previous_seedable,
        seed_birth_contours=source.birth_contours,
        seed_rank_prefix=source.rank_prefix,
        seed_zero_count=source.zero_count,
        root_seed_idx=source.root_idx,
        seed_reservoir_idx=jnp.full_like(
            schedule.seed_reservoir_idx,
            -1,
        ),
        seed_reservoir_priority=jnp.full_like(
            schedule.seed_reservoir_priority,
            -jnp.inf,
        ),
        seed_reservoir_valid=jnp.zeros_like(
            schedule.seed_reservoir_valid,
        ),
        source_num_samples=state.num_samples,
    )
    return dataclasses.replace(state, scheduler_data=schedule)


def _build_depth_view(
        state: State,
        depth_cond: DepthCondition,
        *,
        allocation_target: str,
        root_degree: int,
        delta_K: int,
) -> tuple[BlockState, AllocationPlan, BoolArray, IntArray]:
    """Build the compact allocation and stopping view of append-order state.

    This is a planning-round calculation, not replacement-batch work. The
    returned arrays remain frozen until every maximal thread in the round has
    reached its terminal contour.
    """
    block_state = build_block_state(
        state.samples,
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
        likelihood_order=state.likelihood_order,
    )
    plan = build_allocation_plan(
        state=state,
        allocation_target=allocation_target,
        depth_iteration=(
            state.allocation_loop_iter
            + jnp.asarray(1, state.allocation_loop_iter.dtype)
        ),
        delta_K=jnp.asarray(delta_K, mp_policy.count_dtype),
        # d_0 is the fixed initial allocation. The sentinel's current
        # out-degree grows when later epochs start root threads; using it here
        # would make the target chase every accepted root child.
        root_out_degree=jnp.asarray(root_degree, mp_policy.count_dtype),
        block_state=block_state,
    )
    relevant = _depth_relevant_blocks(plan, depth_cond)
    if allocation_target == "uniform":
        tail_K = (
            jnp.asarray(root_degree, mp_policy.count_dtype)
            * jnp.asarray(delta_K, mp_policy.count_dtype)
            * (
                state.allocation_loop_iter
                + jnp.asarray(1, state.allocation_loop_iter.dtype)
            )
        )
    else:
        # Utility is defined from downstream evidence/posterior value at the
        # frozen race. There is no utility beyond its final observed contour;
        # an overshooting edge already overfills that zero target.
        tail_K = jnp.asarray(0, mp_policy.count_dtype)
    return block_state, plan, relevant, tail_K


def _project_allocation_target(
        schedule: ThreadSchedule,
        block_state: BlockState,
) -> IntArray:
    """Project one frozen absolute target onto a refined contour grid.

    A new block inherits the first old target at or above its likelihood,
    matching the half-open edge coverage ``L_parent < L <= L_child``.
    """
    successor = jnp.searchsorted(
        schedule.block_state.log_L_blocks,
        block_state.log_L_blocks,
        side="left",
    ).astype(mp_policy.index_dtype)  # [G]
    old_num_blocks = schedule.block_state.num_blocks
    within_old_grid = successor < old_num_blocks  # [G]
    safe_successor = jnp.minimum(
        successor,
        jnp.maximum(old_num_blocks - 1, 0),
    )
    projected = jnp.where(
        within_old_grid,
        schedule.target_K[safe_successor],
        schedule.tail_K,
    )  # [G]
    return jnp.where(block_state.valid, projected, 0)


@partial(jax.jit, inline=True)
def _depth_condition_reached(
        state: State,
        depth_cond: DepthCondition,
) -> BoolArray:
    """Evaluate the expected-depth policy at a Python goal boundary."""
    if depth_cond.dlogZ is None and depth_cond.cummax_XL_frac is None:
        return jnp.asarray(True, mp_policy.bool_dtype)
    block_state = build_block_state(
        state.samples,
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
        likelihood_order=state.likelihood_order,
    )
    concentrations = classic_dirichlet_concentrations(block_state)
    relevant = _depth_relevant_path(
        block_state.log_L_blocks,
        block_state.valid,
        expected_volume_path(concentrations, valid=block_state.valid),
        depth_cond,
    )
    last_block = jnp.maximum(
        block_state.num_blocks - 1,
        jnp.asarray(0, mp_policy.index_dtype),
    )
    return (block_state.num_blocks == 0) | jnp.logical_not(
        relevant[last_block]
    )


@partial(
    jax.jit,
    inline=True,
    static_argnames=(
        "shell_size",
        "allocation_target",
        "root_degree",
        "delta_K",
    ),
)
def _start_schedule_round(
        state: State,
        depth_cond: DepthCondition,
        *,
        shell_size: int,
        allocation_target: str,
        root_degree: int,
        delta_K: int,
) -> tuple[State, ThreadSchedule, BoolArray]:
    """Build one frozen schedule from an already-published likelihood order."""
    block_state, plan, relevant, tail_K = _build_depth_view(
        state,
        depth_cond,
        allocation_target=allocation_target,
        root_degree=root_degree,
        delta_K=delta_K,
    )
    schedule = _new_thread_schedule(
        state,
        block_state,
        plan,
        relevant,
        shell_size,
        tail_K,
        seed_reservoir_size=max(shell_size, root_degree),
    )
    state = dataclasses.replace(state, scheduler_data=schedule)
    return state, schedule, jnp.logical_not(schedule.active)


@partial(
    jax.jit,
    inline=True,
    static_argnames=("shell_size",),
)
def _continue_schedule_round(
        state: State,
        previous: ThreadSchedule,
        depth_cond: DepthCondition,
        *,
        shell_size: int,
) -> tuple[State, ThreadSchedule, BoolArray]:
    """Fill newly exposed gaps without changing the allocation target."""
    block_state = build_block_state(
        state.samples,
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
        likelihood_order=state.likelihood_order,
    )
    concentrations = classic_dirichlet_concentrations(block_state)
    target_K = _project_allocation_target(previous, block_state)
    plan = AllocationPlan(
        target_K=target_K,
        current_K=block_state.incoming_K,
        unit_peak_utility=jnp.zeros_like(
            block_state.log_L_blocks,
            dtype=mp_policy.measure_dtype,
        ),
        log_L_blocks=block_state.log_L_blocks,
        valid=block_state.valid,
        volume_path=expected_volume_path(
            concentrations,
            valid=block_state.valid,
        ),
    )
    relevant = _depth_relevant_blocks(plan, depth_cond)
    schedule = _new_thread_schedule(
        state,
        block_state,
        plan,
        relevant,
        shell_size,
        previous.tail_K,
        seed_reservoir_size=previous.seed_reservoir_idx.shape[0],
    )
    state = dataclasses.replace(state, scheduler_data=schedule)
    return state, schedule, jnp.logical_not(schedule.active)


@partial(
    jax.jit,
    inline=True,
    static_argnames=(
        "shell_size",
        "allocation_target",
        "root_degree",
        "delta_K",
        "max_samples",
    ),
)
def _run_depth(
        state: State,
        sampler: AbstractSampler,
        depth_cond: DepthCondition,
        *,
        shell_size: int,
        allocation_target: str,
        root_degree: int,
        delta_K: int,
        max_samples: int | None,
) -> State:
    """Run one allocation depth epoch entirely in compiled JAX.

    The returned state atomically carries the continuation key and exactly one
    exit outcome. A physical-capacity return may therefore be resized and
    resumed without changing the logical allocation epoch or random stream.
    """

    def cond(carry: _DepthCarry):
        # One planning round owns exactly one frozen maximal-thread schedule.
        # Draining it returns to Python so the newly exposed tail can be
        # inspected cheaply. If expected depth is not yet reached, Python
        # constructs another round at the same outer allocation target.
        has_buffer = (
            carry.state.num_samples
            < carry.state.samples.log_likelihoods.shape[0]
        )
        sample_limit = jnp.asarray(
            carry.state.samples.log_likelihoods.shape[0],
            mp_policy.count_dtype,
        )
        if max_samples is not None:
            sample_limit = jnp.asarray(max_samples, mp_policy.count_dtype)
        below_global_limit = carry.state.num_samples < sample_limit
        return (
            carry.schedule.active
            & has_buffer
            & below_global_limit
            & jnp.logical_not(_continuation_storage_full(carry.schedule))
            & jnp.logical_not(_start_seed_storage_full(carry.schedule))
            & jnp.logical_not(_seed_source_refresh_due(
                carry.state,
                carry.schedule,
            ))
            & jnp.logical_not(carry.depth_done)
            & (carry.state.termination_reason == 0)
        )

    def body(carry: _DepthCarry):
        if sampler.uses_adaptive_directions():
            plan_key, fit_key, sample_key, next_key = jax.random.split(
                carry.key,
                4,
            )
        else:
            # Preserve the established isotropic random stream exactly. The
            # opt-in geometry key must not perturb reference runs that never
            # request an ellipsoidal update.
            plan_key, sample_key, next_key = jax.random.split(carry.key, 3)
            fit_key = sample_key
        sample_limit = jnp.asarray(
            carry.state.samples.log_likelihoods.shape[0],
            mp_policy.count_dtype,
        )
        if max_samples is not None:
            sample_limit = jnp.asarray(max_samples, mp_policy.count_dtype)
        max_valid_lanes = (
            jnp.minimum(
                sample_limit,
                jnp.asarray(
                    carry.state.samples.log_likelihoods.shape[0],
                    mp_policy.count_dtype,
                ),
            ).astype(mp_policy.index_dtype)
            - carry.state.num_samples.astype(mp_policy.index_dtype)
        )
        schedule, work = _plan_scheduled_work_batch(
            plan_key,
            carry.state,
            carry.schedule,
            max_valid_lanes,
        )
        sampling_state = carry.state
        if sampler.uses_adaptive_directions():
            # This scalar conditional is outside the replacement vmap. The
            # current geometry and each lane's parent contour then remain
            # fixed for the complete Markov chain, as required for a common
            # symmetric direction law across all transitions in that chain.
            sampling_state = _prepare_sampler_data(
                fit_key,
                carry.state,
                sampler,
                work,
                schedule,
            )
        sampled_batch = _sample_work_batch(
            sample_key,
            sampling_state,
            sampler,
            work,
        )
        insert_idx = sampling_state.num_samples
        next_state = _accept_work_batch(
            sampling_state,
            work,
            sampled_batch,
            # The frozen schedule never consumes likelihood order. Publishing
            # one merge per accepted batch was the measured quadratic path;
            # merge once at a geometric source boundary or final drain.
            update_likelihood_order=False,
        )
        slots = jnp.arange(
            work.valid.shape[0],
            dtype=mp_policy.index_dtype,
        )  # [S]
        schedule = _update_seed_reservoir(
            schedule,
            insert_idx.astype(mp_policy.index_dtype) + slots,
            work.valid,
        )
        schedule = _advance_thread_schedule(
            schedule,
            work,
            sampled_batch,
            insert_idx,
        )
        return _DepthCarry(
            key=next_key,
            state=next_state,
            schedule=schedule,
            depth_done=jnp.logical_not(schedule.active),
        )

    # A normal depth boundary starts the same per-depth split used before
    # transparent growth existed. A growth resume instead uses `random_key`
    # as the exact inner continuation after the last completed batch, while
    # `goal_key` retains the already-derived key for the next logical epoch.
    new_depth_key, next_goal_key = jax.random.split(state.random_key)
    depth_key = jnp.where(
        state.depth_reached,
        new_depth_key,
        state.random_key,
    )
    goal_key = jnp.where(
        state.depth_reached,
        next_goal_key,
        state.goal_key,
    )
    initial_state = dataclasses.replace(
        state,
        random_key=depth_key,
        goal_key=goal_key,
        needs_growth=jnp.asarray(False, mp_policy.bool_dtype),
        depth_reached=jnp.asarray(False, mp_policy.bool_dtype),
    )
    schedule = state.scheduler_data
    if schedule is None:
        raise ValueError("Compiled depth requires a materialised schedule.")
    # Planning precedes the per-depth key split so every large `_run_depth`
    # call has one stable Pytree signature. Restore the established reservoir
    # key here when this is a newly materialised round; physical resumes retain
    # the exact key and published seed generation already in the schedule.
    schedule = dataclasses.replace(
        schedule,
        seed_reservoir_key=jnp.where(
            state.depth_reached,
            jax.random.fold_in(depth_key, state.allocation_loop_iter),
            schedule.seed_reservoir_key,
        ),
    )
    depth_done = jnp.logical_not(schedule.active)
    # ThreadSchedule contains the frozen block view and exact seed rank index,
    # so carrying it both here and inside State would duplicate the largest
    # planning buffers across every XLA while iteration. The scientific State
    # does not consume scheduler_data inside the loop; attach the one canonical
    # schedule only at the Python return boundary below.
    initial_state = dataclasses.replace(initial_state, scheduler_data=None)
    initial_carry = _DepthCarry(
        key=depth_key,
        state=initial_state,
        schedule=schedule,
        depth_done=depth_done,
    )
    final_carry = jax.lax.while_loop(cond, body, initial_carry)

    hard_limit_reached = jnp.asarray(False, mp_policy.bool_dtype)
    if max_samples is not None:
        hard_limit_reached = final_carry.state.num_samples >= max_samples
    termination_reason = jnp.where(
        final_carry.state.termination_reason != 0,
        final_carry.state.termination_reason,
        jnp.where(
            hard_limit_reached,
            jnp.asarray(MAX_SAMPLES_REACHED, mp_policy.count_dtype),
            jnp.asarray(0, mp_policy.count_dtype),
        ),
    )
    terminal = termination_reason != 0
    storage_full = (
        final_carry.state.num_samples
        >= final_carry.state.samples.log_likelihoods.shape[0]
    )
    needs_growth = jnp.logical_and(
        jnp.logical_not(terminal),
        final_carry.schedule.active & storage_full,
    )
    schedule_drained = (
        final_carry.depth_done
        & jnp.logical_not(terminal | needs_growth)
    )
    # The active schedule owns its frozen likelihood and seed views. Publishing
    # is a separate Python coordination boundary so the large merge/index build
    # is not embedded in every specialised replacement-loop executable.
    final_state = dataclasses.replace(
        final_carry.state,
        scheduler_data=final_carry.schedule,
    )
    # This compiled flag reports only a genuinely drained schedule. Python
    # replaces it with the actual expected-depth result using the separately
    # compiled boundary check, keeping that block reconstruction out of this
    # large program materially reduces lowering and compilation time.
    depth_reached = schedule_drained
    continuation_key = jnp.where(
        terminal,
        goal_key,
        final_carry.key,
    )
    return dataclasses.replace(
        final_state,
        termination_reason=termination_reason,
        needs_growth=needs_growth,
        depth_reached=depth_reached,
        random_key=continuation_key,
        goal_key=goal_key,
        scheduler_data=dataclasses.replace(
            final_carry.schedule,
            active=jnp.where(
                terminal,
                jnp.asarray(False, mp_policy.bool_dtype),
                final_carry.schedule.active,
            ),
        ),
    )
