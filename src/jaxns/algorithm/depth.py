"""Pure-JAX lineage planning, constrained sampling, and depth traversal."""

import dataclasses
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxns.algorithm.allocation import AllocationPlan, build_allocation_plan
from jaxns.algorithm.race_tree import (
    BlockState,
    build_block_state,
)
from jaxns.constrained_sampler import (
    AbstractSampler,
    ConstrainedSampleBatch,
    ConstrainedSampleRequest,
    sample_request,
)
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree, pytree_ravel
from jaxns.samples import Samples, SeedPoint
from jaxns.sampling.ellipsoid import update_sampler_data
from jaxns.shrinkage.classic import (
    classic_dirichlet_concentrations,
)
from jaxns.state import State, termination_register_from_volume_path
from jaxns.termination_condition import (
    TerminationCondition,
    TerminationRegister,
)
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey

# This reason is shared by local and distributed depth execution so both
# runners expose the same resumable storage-boundary contract.
MAX_SAMPLES_REACHED = 1


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


def _first_masked_index(mask: BoolArray) -> IntArray:
    indices = jnp.arange(mask.shape[0], dtype=mp_policy.index_dtype)
    return jnp.min(
        jnp.where(mask, indices, jnp.asarray(mask.shape[0], indices.dtype))
    )


def _uniform_ranked_masked(
        unit_draw: FloatArray,
        mask: BoolArray,
) -> IntArray:
    """Map a uniform draw to a uniform rank among masked entries."""
    cumulative = jnp.cumsum(mask.astype(mp_policy.index_dtype))
    count = cumulative[-1]
    rank = jnp.minimum(
        jnp.floor(unit_draw * count).astype(mp_policy.index_dtype),
        jnp.maximum(count - 1, 0),
    )
    return _first_masked_index(mask & (cumulative > rank))


def _depth_relevant_blocks(
        plan: AllocationPlan,
        depth_cond: TerminationCondition,
) -> BoolArray:
    """Mask blocks before the expectation-based evidence/posterior cutoff."""
    # This runs inside every compiled depth iteration. Use the deterministic
    # expected volume path here; the more expensive MC shrinkage ensemble is a
    # final-result calculation and may also be used by the Python goal loop.
    valid = plan.valid
    log_shell_mass = jnp.where(
        valid & (plan.volume_path.shell_mass > 0),
        jnp.log(plan.volume_path.shell_mass),
        -jnp.inf,
    )
    log_dZ = plan.log_L_blocks + log_shell_mass
    log_Z_prefix = jax.lax.associative_scan(jnp.logaddexp, log_dZ)
    log_remaining = (
        plan.log_L_blocks
        + jnp.where(
            plan.volume_path.X > 0,
            jnp.log(plan.volume_path.X),
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


def _stationary_seed_mask(
        state: State,
        log_L_constraint: FloatArray,
        from_root: BoolArray,
) -> BoolArray:
    """Return exact stationary seeds for a requested parent contour.

    A classic sample is stationary at lambda exactly when its own generation
    interval contains lambda: parent_constraint <= lambda < likelihood. Root
    fallback is deliberately narrower and may only use sentinel children.
    """
    valid = (
        jnp.arange(state.samples.log_likelihoods.shape[0])
        < state.num_samples
    )
    interval_contains = (
        (state.samples.log_L_constraints <= log_L_constraint)
        & (state.samples.log_likelihoods > log_L_constraint)
    )
    root_child = jnp.isneginf(state.samples.log_L_constraints)
    return valid & jnp.where(from_root, root_child, interval_contains)


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


def _closest_seedable_parent_block(
        state: State,
        block_state: BlockState,
        requested_block_idx: IntArray,
) -> IntArray:
    """Resolve a requested contour to itself or its closest seedable ancestor."""
    requested_constraint = jnp.where(
        requested_block_idx >= 0,
        block_state.log_L_blocks[jnp.maximum(requested_block_idx, 0)],
        jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
    )
    requested_has_seed = jnp.any(
        _stationary_seed_mask(
            state,
            requested_constraint,
            requested_block_idx < 0,
        )
    )

    # The fallback is rarely taken. Scan ancestors one at a time so the common
    # scheduler path never constructs a blocks-by-samples eligibility matrix.
    def fallback_cond(fallback_carry):
        candidate_block, found = fallback_carry
        return (candidate_block >= 0) & jnp.logical_not(found)

    def fallback_body(fallback_carry):
        candidate_block, _ = fallback_carry
        candidate_constraint = block_state.log_L_blocks[candidate_block]
        found = jnp.any(
            _stationary_seed_mask(
                state,
                candidate_constraint,
                jnp.asarray(False, mp_policy.bool_dtype),
            )
        )
        return jnp.where(found, candidate_block, candidate_block - 1), found

    fallback_block_idx, fallback_found = jax.lax.while_loop(
        fallback_cond,
        fallback_body,
        (
            requested_block_idx - 1,
            jnp.asarray(False, mp_policy.bool_dtype),
        ),
    )
    return jnp.where(
        requested_has_seed,
        requested_block_idx,
        jnp.where(fallback_found, fallback_block_idx, -1),
    )


def _plan_work_batch(
        key: PRNGKey,
        state: State,
        block_state: BlockState,
        plan: AllocationPlan,
        relevant: BoolArray,
        shell_size: int,
        max_valid_lanes: IntArray | None = None,
) -> CoreWorkBatch:
    """Schedule the next edges of maximal allocation-gap threads."""
    slots = jnp.arange(shell_size, dtype=mp_policy.index_dtype)
    plan_key, seed_draw_key = jax.random.split(key)

    if max_valid_lanes is None:
        max_valid_lanes = jnp.asarray(shell_size, mp_policy.index_dtype)
    max_valid_lanes = jnp.maximum(max_valid_lanes, 0)

    # A positive rise in G_g starts that many maximal threads at block g.
    # If an edge falls short, the remaining gap rises after its new child, so
    # a later depth iteration continues the logical thread without persisting
    # either a thread identity or a parent index in scientific state.
    gap = jnp.where(
        relevant,
        jnp.maximum(plan.target_K - plan.current_K, 0),
        jnp.asarray(0, dtype=plan.current_K.dtype),
    )
    previous_gap = jnp.concatenate(
        [jnp.zeros((1,), dtype=gap.dtype), gap[:-1]],
    )
    start_count = jnp.maximum(gap - previous_gap, 0)
    cumulative_starts = jnp.cumsum(start_count)
    num_starts = cumulative_starts[-1]
    lane_valid = (slots < num_starts) & (slots < max_valid_lanes)
    start_block_idx = jnp.searchsorted(
        cumulative_starts,
        slots,
        side="right",
    ).astype(mp_policy.index_dtype)
    start_block_idx = jnp.minimum(
        start_block_idx,
        jnp.asarray(plan.valid.shape[0] - 1, mp_policy.index_dtype),
    )
    requested_block_idx = jnp.where(
        lane_valid,
        start_block_idx - 1,
        jnp.asarray(-1, mp_policy.index_dtype),
    )

    # Reparent both the constraint and degree update together when the
    # immediate contour has no seed. A shallower seed under the original
    # constraint would not start a stationary Markov chain.
    effective_block_idx = jax.vmap(
        lambda block_idx: _closest_seedable_parent_block(
            state,
            block_state,
            block_idx,
        )
    )(requested_block_idx)
    effective_constraint = jnp.where(
        effective_block_idx >= 0,
        plan.log_L_blocks[jnp.maximum(effective_block_idx, 0)],
        jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
    )
    parent_keys = jax.random.split(plan_key, shell_size)
    parent_idx = jax.vmap(
        lambda parent_key, block_idx: _sample_parent_from_block(
            parent_key,
            block_state,
            block_idx,
        )
    )(parent_keys, effective_block_idx)

    # Apply one random rotation to evenly spaced lane quantiles. Every lane's
    # quantile remains Uniform[0, 1), hence its selected seed is exactly
    # uniform on that lane's stationary set even when the contours differ.
    # The stratification spreads a parallel batch over the lineage population
    # without persistent seed-use state or likelihood-dependent exclusion.
    rotation = jax.random.uniform(
        seed_draw_key,
        (),
        dtype=mp_policy.measure_dtype,
    )
    seed_fraction = jnp.mod(
        rotation
        + slots.astype(mp_policy.measure_dtype)
        / jnp.asarray(shell_size, mp_policy.measure_dtype),
        1.0,
    )

    def choose_seed(lane_idx, seed_indices):
        lane_block = effective_block_idx[lane_idx]
        seed_mask = _stationary_seed_mask(
            state,
            effective_constraint[lane_idx],
            lane_block < 0,
        )
        seed_idx = _uniform_ranked_masked(
            seed_fraction[lane_idx],
            seed_mask,
        )
        return seed_indices.at[lane_idx].set(seed_idx)

    seed_idx = jax.lax.fori_loop(
        0,
        shell_size,
        choose_seed,
        jnp.zeros((shell_size,), dtype=mp_policy.index_dtype),
    )
    return CoreWorkBatch(
        valid=lane_valid,
        parent_idx=parent_idx,
        log_L_constraint=effective_constraint,
        seed_idx=seed_idx,
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
        block_state: BlockState,
) -> State:
    """Update opt-in direction geometry once before the replacement vmap."""
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
    enough_rows = state.num_samples >= min_effective_samples
    enough_new_rows = (
        state.num_samples
        >= state.sampler_data.num_attempted + retry_increment
    )
    initialise = jnp.where(first_attempt, enough_rows, enough_new_rows)
    last_attempt_succeeded = (
        state.sampler_data.num_attempted
        == state.sampler_data.num_samples
    )
    # A newly stale successful model is refreshed immediately. If that refresh
    # fails, wait for a materially larger population before retrying so a
    # singular late contour cannot make every replacement batch refit it.
    refresh = stale & (last_attempt_succeeded | enough_new_rows)
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
        use_all = state.num_samples <= population_size
        spread = jnp.floor(
            slots.astype(mp_policy.measure_dtype)
            * state.num_samples.astype(mp_policy.measure_dtype)
            / jnp.asarray(population_size, mp_policy.measure_dtype)
        ).astype(mp_policy.index_dtype)
        positions = jnp.where(use_all, slots, spread)
        mask = jnp.where(
            use_all,
            slots < state.num_samples,
            jnp.ones((population_size,), mp_policy.bool_dtype),
        )
        positions = jnp.minimum(
            positions,
            jnp.maximum(state.num_samples - 1, 0),
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
            state.num_samples,
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
        ),
        sampler_data=sampler_data,
    )


class _DepthCarry(NamedTuple):
    key: PRNGKey  # [2]
    state: State
    block_state: BlockState
    plan: AllocationPlan
    relevant: BoolArray  # [G]
    register: TerminationRegister


def _build_depth_view(
        state: State,
        depth_cond: TerminationCondition,
        *,
        allocation_target: str,
        root_degree: int,
        delta_K: int,
) -> tuple[BlockState, AllocationPlan, BoolArray, TerminationRegister]:
    """Build the compact allocation and stopping view of append-order state.

    Local and distributed depth execution differ only at the sampling
    boundary. Sharing this calculation ensures they assign lineage gaps and
    expectation-based depth relevance with exactly the same scientific model.
    """
    # Out-degree changes alter K_g, the expected volume path, and the depth
    # stopping estimate. Rebuild this compact block view after each accepted
    # batch while leaving the scientific sample pytrees in append order.
    block_state = build_block_state(
        state.samples,
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
        likelihood_order=state.likelihood_order,
    )
    plan = build_allocation_plan(
        state=state,
        allocation_target=allocation_target,
        iteration=state.goal_loop_iter,
        delta_K=jnp.asarray(delta_K, mp_policy.count_dtype),
        # d_0 is the fixed initial allocation. The sentinel's current
        # out-degree grows when later epochs start root threads; using it here
        # would make the target chase every accepted root child.
        root_out_degree=jnp.asarray(root_degree, mp_policy.count_dtype),
        block_state=block_state,
    )
    relevant = _depth_relevant_blocks(plan, depth_cond)
    register = termination_register_from_volume_path(
        state,
        block_state,
        plan.volume_path.X,
        plan.volume_path.shell_mass,
    )
    return block_state, plan, relevant, register


@partial(
    jax.jit,
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
        depth_cond: TerminationCondition,
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
        # The compiled depth epoch stops at the first of: filled allocation,
        # exhausted storage/sample budget, or a scalar state/budget condition.
        # The user goal condition remains outside this JAX loop.
        has_gap = jnp.any(carry.plan.under_allocated(carry.relevant))
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
        if depth_cond.max_samples is not None:
            sample_limit = jnp.minimum(
                sample_limit,
                depth_cond.max_samples.astype(mp_policy.count_dtype),
            )
        below_global_limit = carry.state.num_samples < sample_limit
        # dlogZ and posterior-tail thresholds are contour-local depth cuts:
        # `_depth_relevant_blocks` already excludes work beyond them. Treating
        # them again as scalar termination would permanently prevent a later
        # outer epoch from filling a larger target on the still-relevant
        # contours. Hard budgets and state failures remain scalar stops.
        scalar_cond = dataclasses.replace(
            depth_cond,
            dlogZ=None,
            cummax_XL_frac=None,
        )
        depth_done, _ = carry.register.is_done(scalar_cond)
        return (
            has_gap
            & has_buffer
            & below_global_limit
            & jnp.logical_not(depth_done)
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
        if depth_cond.max_samples is not None:
            sample_limit = jnp.minimum(
                sample_limit,
                depth_cond.max_samples.astype(mp_policy.count_dtype),
            )
        work = _plan_work_batch(
            plan_key,
            carry.state,
            carry.block_state,
            carry.plan,
            carry.relevant,
            shell_size,
            max_valid_lanes=(
                jnp.minimum(
                    sample_limit,
                    jnp.asarray(
                        carry.state.samples.log_likelihoods.shape[0],
                        mp_policy.count_dtype,
                    ),
                ).astype(mp_policy.index_dtype)
                - carry.state.num_samples.astype(mp_policy.index_dtype)
            ),
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
                carry.block_state,
            )
        sampled_batch = _sample_work_batch(
            sample_key,
            sampling_state,
            sampler,
            work,
        )
        next_state = _accept_work_batch(
            sampling_state,
            work,
            sampled_batch,
        )
        next_block_state, next_plan, next_relevant, next_register = (
            _build_depth_view(
                next_state,
                depth_cond,
                allocation_target=allocation_target,
                root_degree=root_degree,
                delta_K=delta_K,
            )
        )
        return _DepthCarry(
            key=next_key,
            state=next_state,
            block_state=next_block_state,
            plan=next_plan,
            relevant=next_relevant,
            register=next_register,
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
    block_state, plan, relevant, register = _build_depth_view(
        state,
        depth_cond,
        allocation_target=allocation_target,
        root_degree=root_degree,
        delta_K=delta_K,
    )
    initial_state = dataclasses.replace(
        state,
        random_key=depth_key,
        goal_key=goal_key,
        needs_growth=jnp.asarray(False, mp_policy.bool_dtype),
        depth_reached=jnp.asarray(False, mp_policy.bool_dtype),
    )
    initial_carry = _DepthCarry(
        key=depth_key,
        state=initial_state,
        block_state=block_state,
        plan=plan,
        relevant=relevant,
        register=register,
    )
    final_carry = jax.lax.while_loop(cond, body, initial_carry)

    # Classify the single reason the compiled loop returned. Terminal reasons
    # take precedence over a full physical buffer, while a growable buffer
    # takes precedence over ordinary completion of the allocation/depth work.
    scalar_cond = dataclasses.replace(
        depth_cond,
        dlogZ=None,
        cummax_XL_frac=None,
    )
    scalar_done, scalar_reason = final_carry.register.is_done(scalar_cond)
    hard_limit_reached = jnp.asarray(False, mp_policy.bool_dtype)
    if max_samples is not None:
        hard_limit_reached = final_carry.state.num_samples >= max_samples
    termination_reason = jnp.where(
        final_carry.state.termination_reason != 0,
        final_carry.state.termination_reason,
        jnp.where(
            scalar_done,
            scalar_reason,
            jnp.where(
                hard_limit_reached,
                jnp.asarray(MAX_SAMPLES_REACHED, mp_policy.count_dtype),
                jnp.asarray(0, mp_policy.count_dtype),
            ),
        ),
    )
    terminal = termination_reason != 0
    has_gap = jnp.any(
        final_carry.plan.under_allocated(final_carry.relevant)
    )
    storage_full = (
        final_carry.state.num_samples
        >= final_carry.state.samples.log_likelihoods.shape[0]
    )
    needs_growth = jnp.logical_and(
        jnp.logical_not(terminal),
        has_gap & storage_full,
    )
    depth_reached = jnp.logical_not(terminal | needs_growth)
    continuation_key = jnp.where(
        needs_growth,
        final_carry.key,
        goal_key,
    )
    return dataclasses.replace(
        final_carry.state,
        termination_reason=termination_reason,
        needs_growth=needs_growth,
        depth_reached=depth_reached,
        random_key=continuation_key,
        goal_key=goal_key,
    )
