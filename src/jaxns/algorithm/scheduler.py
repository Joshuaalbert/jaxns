"""Immutable continuation data for one frozen lineage schedule."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from jaxns.algorithm.race_tree import BlockState
from jaxns.pytree import PureDataclassPytree
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey


def _resize_vector(
        values: jax.Array,
        size: int,
        fill_value: float | bool,
) -> jax.Array:
    """Resize one frozen scheduler vector without changing its prefix."""
    current = values.shape[0]
    if size <= current:
        return values[:size]
    return jnp.pad(
        values,
        ((0, size - current),),
        constant_values=fill_value,
    )


def _resize_seed_rank_index(
        prefixes: IntArray,
        zero_count: IntArray,
        size: int,
) -> tuple[IntArray, IntArray]:
    """Match future publication shape without changing old rank queries.

    A grown sample buffer may need another most-significant rank bit. Leading
    zero levels leave every frozen rank unchanged. Their zero partition uses
    the old logical matrix width, because padded columns are storage only and
    are not members of the frozen seed source.
    """
    current_columns = prefixes.shape[1]
    current_capacity = current_columns - 1
    target_levels = max(1, (size - 2).bit_length())
    extra_levels = max(target_levels - prefixes.shape[0], 0)
    if extra_levels:
        prefixes = jnp.pad(
            prefixes,
            ((extra_levels, 0), (0, 0)),
        )
        zero_count = jnp.concatenate((
            jnp.full(
                (extra_levels,),
                current_capacity,
                dtype=zero_count.dtype,
            ),
            zero_count,
        ))
    elif target_levels < prefixes.shape[0]:
        # Active-state trimming can remove only leading zero rank bits. Keep
        # the minimal height so a later publication has this same Pytree shape.
        prefixes = prefixes[-target_levels:, :]
        zero_count = zero_count[-target_levels:]
    if size <= current_columns:
        return prefixes[:, :size], zero_count
    return (
        jnp.pad(prefixes, ((0, 0), (0, size - current_columns))),
        zero_count,
    )


def _resize_block_state(block_state: BlockState, size: int) -> BlockState:
    """Pad frozen contour lookups only when storage growth recompiles JAX."""
    return BlockState(
        log_L_blocks=_resize_vector(block_state.log_L_blocks, size, jnp.inf),
        block_first_idx=_resize_vector(block_state.block_first_idx, size, -1),
        block_size=_resize_vector(block_state.block_size, size, 0),
        incoming_K=_resize_vector(block_state.incoming_K, size, 0),
        block_out_degree=_resize_vector(
            block_state.block_out_degree,
            size,
            0,
        ),
        valid=_resize_vector(block_state.valid, size, False),
        block_start=_resize_vector(block_state.block_start, size, 0),
        block_stop=_resize_vector(block_state.block_stop, size, 0),
        block_sample_indices=_resize_vector(
            block_state.block_sample_indices,
            size,
            -1,
        ),
    )


def _seed_reservation_slot(seed_idx: IntArray, size: int) -> IntArray:
    """Hash one non-negative sample identity into a power-of-two table."""
    bits = size.bit_length() - 1
    hashed = seed_idx.astype(jnp.uint32) * jnp.asarray(
        2654435761,
        dtype=jnp.uint32,
    )
    return (hashed >> (32 - bits)).astype(jnp.int32)


def _insert_seed_reservation(
        seed_idx: IntArray,
        group: IntArray,
        reservation_idx: IntArray,
        reservation_group: IntArray,
) -> tuple[IntArray, IntArray]:
    """Insert one exact identity using linear probing within its group.

    The caller reserves an empty slot by growing before this function runs.
    """
    size = reservation_idx.shape[0]
    first_slot = _seed_reservation_slot(seed_idx, size)

    def cond(carry):
        attempt, _, found = carry
        return (attempt < size) & jnp.logical_not(found)

    def body(carry):
        attempt, selected_slot, _ = carry
        slot = (first_slot + attempt) & jnp.asarray(size - 1, jnp.int32)
        occupied = reservation_group[slot] == group
        same_seed = occupied & (reservation_idx[slot] == seed_idx)
        available = jnp.logical_not(occupied) | same_seed
        return (
            attempt + jnp.asarray(1, jnp.int32),
            jnp.where(available, slot, selected_slot),
            available,
        )

    _, slot, found = jax.lax.while_loop(
        cond,
        body,
        (
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(False),
        ),
    )
    reservation_idx = reservation_idx.at[slot].set(
        jnp.where(found, seed_idx, reservation_idx[slot])
    )
    reservation_group = reservation_group.at[slot].set(
        jnp.where(found, group, reservation_group[slot])
    )
    return reservation_idx, reservation_group


def _seed_reservation_contains(
        reservation_idx: IntArray,
        reservation_group: IntArray,
        group: IntArray,
        seed_idx: IntArray,
) -> BoolArray:
    """Return exact membership without scanning unused reservation storage."""
    size = reservation_idx.shape[0]
    first_slot = _seed_reservation_slot(seed_idx, size)

    def cond(carry):
        attempt, found, stopped = carry
        return (
            (attempt < size)
            & jnp.logical_not(found)
            & jnp.logical_not(stopped)
        )

    def body(carry):
        attempt, _, _ = carry
        slot = (first_slot + attempt) & jnp.asarray(size - 1, jnp.int32)
        occupied = reservation_group[slot] == group
        found = occupied & (reservation_idx[slot] == seed_idx)
        return (
            attempt + jnp.asarray(1, jnp.int32),
            found,
            jnp.logical_not(occupied),
        )

    _, found, _ = jax.lax.while_loop(
        cond,
        body,
        (
            jnp.asarray(0, jnp.int32),
            jnp.asarray(False),
            jnp.asarray(False),
        ),
    )
    return found


@dataclasses.dataclass(slots=True, frozen=True)
class ThreadSchedule(PureDataclassPytree):
    """Frozen gap plus the fixed-width thread heads advancing through it.

    Parent identities here are transient continuation data. They are never
    inserted into scientific samples or results, and disappear when the depth
    iteration completes.
    """

    block_state: BlockState
    seed_block_state: BlockState
    start_block: IntArray  # [G] compressed thread-run starts
    terminal_block: IntArray  # [G] compressed thread-run terminals
    multiplicity: IntArray  # [G] number of identical T(a, b) threads
    num_runs: IntArray  # []
    target_K: IntArray  # [G] absolute frozen lineage target
    relevant: BoolArray  # [G] blocks included in this frozen schedule
    tail_K: IntArray  # [] target beyond the last frozen contour
    log_mean_X: FloatArray  # [G] log of planning-time mean volume path
    seed_count: IntArray  # [G]
    previous_seedable: IntArray  # [G]
    seed_birth_contours: FloatArray  # [A] birth-sorted frozen contours
    seed_rank_prefix: IntArray  # [H, A + 1] wavelet rank prefixes
    seed_zero_count: IntArray  # [H] zero partition sizes
    # These replacement maps exist only when phantom seeding changes the
    # representative intervals. ``None`` leaves the classic schedule at its
    # established memory footprint and likelihood-order lookup.
    seed_reservoir_idx: IntArray  # [R] bounded coordination sample indices
    seed_reservoir_priority: FloatArray  # [R] value-independent priorities
    seed_reservoir_valid: BoolArray  # [R]
    seed_reservoir_key: PRNGKey  # [2]
    phantom_slot_miss_probability: FloatArray  # [R] predicted unfilled mass
    start_seed_reservation_idx: IntArray  # [V] exact reserved identities
    start_seed_reservation_group: IntArray  # [V] logical-clear generations
    current_start_group: IntArray  # [] retained same-contour group identity
    start_seed_log_L_constraint: FloatArray  # [] retained effective contour
    num_start_seeds: IntArray  # [] unique seeds used at retained contour
    num_published_start_seeds: IntArray  # [] reserved seeds in frozen source
    parent_idx: IntArray  # [S]
    thread_id: IntArray  # [S]
    log_L_constraint: FloatArray  # [S]
    terminal_log_L: FloatArray  # [S]
    new_start: BoolArray  # [S] heads beginning a logical maximal thread
    valid: BoolArray  # [S]
    continuation_parent_idx: IntArray  # [Q] minimum-contour heap parents
    continuation_thread_id: IntArray  # [Q] minimum-contour heap identities
    continuation_log_L_constraint: FloatArray  # [Q] requested contours
    continuation_frontier: FloatArray  # [Q] effective heap-order contours
    continuation_terminal_log_L: FloatArray  # [Q] heap terminals
    continuation_count: IntArray  # [] occupied heap prefix
    next_run: IntArray  # []
    remaining_in_run: IntArray  # []
    next_thread_id: IntArray  # []
    num_threads: IntArray  # []
    source_num_samples: IntArray  # []
    active: BoolArray  # []

    def resize(self, size: int) -> ThreadSchedule:
        """Pad frozen lookups so a resumed compiled carry has one shape.

        Padding never adds a valid block or thread run. It exists solely
        because storage growth recompiles and then resumes this same planning
        round with a larger sample-indexed shape.
        """
        seed_rank_prefix, seed_zero_count = _resize_seed_rank_index(
            self.seed_rank_prefix,
            self.seed_zero_count,
            size + 1,
        )
        return dataclasses.replace(
            self,
            block_state=_resize_block_state(self.block_state, size),
            seed_block_state=_resize_block_state(
                self.seed_block_state,
                size,
            ),
            start_block=_resize_vector(self.start_block, size, 0),
            terminal_block=_resize_vector(self.terminal_block, size, 0),
            multiplicity=_resize_vector(self.multiplicity, size, 0),
            target_K=_resize_vector(self.target_K, size, 0),
            relevant=(
                self.relevant
                if self.relevant.shape[0] == 0
                else _resize_vector(self.relevant, size, False)
            ),
            log_mean_X=(
                self.log_mean_X
                if self.log_mean_X.shape[0] == 0
                else _resize_vector(self.log_mean_X, size, -jnp.inf)
            ),
            seed_count=_resize_vector(self.seed_count, size, 0),
            previous_seedable=_resize_vector(
                self.previous_seedable,
                size,
                -1,
            ),
            seed_birth_contours=_resize_vector(
                self.seed_birth_contours,
                size,
                jnp.inf,
            ),
            seed_rank_prefix=seed_rank_prefix,
            seed_zero_count=seed_zero_count,
        )

    def resize_start_seed_reservations(self, size: int) -> ThreadSchedule:
        """Grow and rehash the active exact reservation set.

        ``size`` must be a power of two. Old generations are logically empty,
        so only the current group's compact identities are copied.
        """
        current = self.start_seed_reservation_idx.shape[0]
        if size <= current:
            return self
        if size & (size - 1):
            raise ValueError("Seed reservation size must be a power of two.")
        reservation_idx = jnp.full(
            (size,),
            -1,
            dtype=self.start_seed_reservation_idx.dtype,
        )  # [V]
        reservation_group = jnp.zeros(
            (size,),
            dtype=self.start_seed_reservation_group.dtype,
        )  # [V]

        def rehash(slot, carry):
            current_idx, current_group = carry
            valid = (
                self.start_seed_reservation_group[slot]
                == self.current_start_group
            )

            def insert(unused):
                del unused
                return _insert_seed_reservation(
                    self.start_seed_reservation_idx[slot],
                    self.current_start_group,
                    current_idx,
                    current_group,
                )

            return jax.lax.cond(
                valid,
                insert,
                lambda unused: carry,
                operand=None,
            )

        reservation_idx, reservation_group = jax.lax.fori_loop(
            0,
            current,
            rehash,
            (reservation_idx, reservation_group),
        )
        return dataclasses.replace(
            self,
            start_seed_reservation_idx=reservation_idx,
            start_seed_reservation_group=reservation_group,
        )

    def resize_threads(
            self,
            size: int,
            continuation_size: int | None = None,
    ) -> ThreadSchedule:
        """Grow the in-flight head window without changing logical work.

        Distributed worker capacity may increase after a planning round has
        started. Extra invalid slots admit more independent thread heads and,
        crucially, reserve heap space for every in-flight edge that may return
        as a continuing thread. Padding a binary heap preserves its ordering.
        """
        current = self.valid.shape[0]
        size = max(size, current)
        reservoir_size = max(size, self.seed_reservoir_idx.shape[0])
        current_continuation_size = self.continuation_parent_idx.shape[0]
        if continuation_size is None:
            continuation_size = current_continuation_size
        else:
            # Head and heap dimensions grow independently. A worker-capacity
            # change must not undo an earlier heap doubling and truncate live
            # continuations merely because the nominal width is now smaller.
            continuation_size = max(
                continuation_size,
                current_continuation_size,
            )
        if (
            size == current
            and continuation_size == current_continuation_size
        ):
            return self

        return dataclasses.replace(
            self,
            seed_reservoir_idx=_resize_vector(
                self.seed_reservoir_idx,
                reservoir_size,
                -1,
            ),
            seed_reservoir_priority=_resize_vector(
                self.seed_reservoir_priority,
                reservoir_size,
                -jnp.inf,
            ),
            seed_reservoir_valid=_resize_vector(
                self.seed_reservoir_valid,
                reservoir_size,
                False,
            ),
            parent_idx=_resize_vector(self.parent_idx, size, -1),
            thread_id=_resize_vector(self.thread_id, size, -1),
            log_L_constraint=_resize_vector(
                self.log_L_constraint,
                size,
                -jnp.inf,
            ),
            terminal_log_L=_resize_vector(
                self.terminal_log_L,
                size,
                -jnp.inf,
            ),
            new_start=_resize_vector(self.new_start, size, False),
            valid=_resize_vector(self.valid, size, False),
            continuation_parent_idx=_resize_vector(
                self.continuation_parent_idx,
                continuation_size,
                -1,
            ),
            continuation_thread_id=_resize_vector(
                self.continuation_thread_id,
                continuation_size,
                -1,
            ),
            continuation_log_L_constraint=_resize_vector(
                self.continuation_log_L_constraint,
                continuation_size,
                -jnp.inf,
            ),
            continuation_frontier=_resize_vector(
                self.continuation_frontier,
                continuation_size,
                jnp.inf,
            ),
            continuation_terminal_log_L=_resize_vector(
                self.continuation_terminal_log_L,
                continuation_size,
                -jnp.inf,
            ),
        )


ThreadSchedule.register_pytree()


def has_thread_work(schedule: ThreadSchedule) -> BoolArray:
    """Return whether an active head or undispatched thread remains."""
    return (
        jnp.any(schedule.valid)
        | (schedule.next_run < schedule.num_runs)
        | (schedule.continuation_count > 0)
    )


def decompose_gap_python(
        gap: np.ndarray | list[int] | tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference maximal-thread decomposition for tests and diagnostics."""
    values = np.asarray(gap, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("gap must be one-dimensional.")
    if np.any(values < 0):
        raise ValueError("gap must be nonnegative.")
    stack: list[list[int]] = []
    runs: list[tuple[int, int, int]] = []
    for block_idx in range(values.size):
        previous = 0 if block_idx == 0 else int(values[block_idx - 1])
        following = (
            0
            if block_idx + 1 == values.size
            else int(values[block_idx + 1])
        )
        rise = max(int(values[block_idx]) - previous, 0)
        fall = max(int(values[block_idx]) - following, 0)
        if rise:
            stack.append([block_idx, rise])
        while fall:
            start, count = stack[-1]
            take = min(fall, count)
            runs.append((start, block_idx, take))
            fall -= take
            count -= take
            if count:
                stack[-1][1] = count
            else:
                stack.pop()
    if stack:
        raise ValueError("gap decomposition did not close at its sentinel.")
    if not runs:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, empty
    starts, terminals, multiplicities = zip(*runs, strict=True)
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(terminals, dtype=np.int64),
        np.asarray(multiplicities, dtype=np.int64),
    )
