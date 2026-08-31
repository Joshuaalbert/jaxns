"""Immutable continuation data for one frozen lineage schedule."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from jaxns.algorithm.race_tree import BlockState
from jaxns.mixed_precision import mp_policy
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


@dataclasses.dataclass(slots=True, frozen=True)
class ThreadSchedule(PureDataclassPytree):
    """Frozen gap plus the fixed-width thread heads advancing through it.

    Parent identities here are transient continuation data. They are never
    inserted into scientific samples or results, and disappear when the depth
    iteration completes.
    """

    block_state: BlockState
    start_block: IntArray  # [G] compressed thread-run starts
    terminal_block: IntArray  # [G] compressed thread-run terminals
    multiplicity: IntArray  # [G] number of identical T(a, b) threads
    num_runs: IntArray  # []
    target_K: IntArray  # [G] absolute frozen lineage target
    tail_K: IntArray  # [] target beyond the last frozen contour
    seed_count: IntArray  # [G]
    previous_seedable: IntArray  # [G]
    root_seed_idx: IntArray  # [G] compact sentinel-child identities
    seed_reservoir_idx: IntArray  # [R] bounded post-freeze candidate indices
    seed_reservoir_priority: FloatArray  # [R] value-independent priorities
    seed_reservoir_valid: BoolArray  # [R]
    seed_reservoir_key: PRNGKey  # [2]
    parent_idx: IntArray  # [S]
    thread_id: IntArray  # [S]
    log_L_constraint: FloatArray  # [S]
    terminal_log_L: FloatArray  # [S]
    valid: BoolArray  # [S]
    continuation_parent_idx: IntArray  # [Q] FIFO continuation parents
    continuation_thread_id: IntArray  # [Q] FIFO logical identities
    continuation_terminal_log_L: FloatArray  # [Q] FIFO terminals
    continuation_head: IntArray  # [] oldest FIFO position
    continuation_count: IntArray  # [] occupied FIFO positions
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
        return dataclasses.replace(
            self,
            block_state=_resize_block_state(self.block_state, size),
            start_block=_resize_vector(self.start_block, size, 0),
            terminal_block=_resize_vector(self.terminal_block, size, 0),
            multiplicity=_resize_vector(self.multiplicity, size, 0),
            target_K=_resize_vector(self.target_K, size, 0),
            seed_count=_resize_vector(self.seed_count, size, 0),
            previous_seedable=_resize_vector(
                self.previous_seedable,
                size,
                -1,
            ),
            root_seed_idx=_resize_vector(self.root_seed_idx, size, 0),
        )

    def resize_threads(
            self,
            size: int,
            continuation_size: int | None = None,
    ) -> ThreadSchedule:
        """Grow the in-flight head window without changing logical work.

        Distributed worker capacity may increase after a planning round has
        started. Extra invalid slots admit more independent thread heads and,
        crucially, reserve space for every in-flight edge that may return as
        a continuing thread.
        """
        current = self.valid.shape[0]
        if size <= current:
            return self
        reservoir_size = max(size, self.seed_reservoir_idx.shape[0])
        if continuation_size is None:
            continuation_size = self.continuation_parent_idx.shape[0]

        # A ring cannot be padded in physical order when its head has wrapped.
        # Growth is a Python/recompile boundary, so linearise the logical FIFO
        # once here and keep repeated device batches at O(S).
        queue_size = self.continuation_parent_idx.shape[0]
        positions = (
            self.continuation_head
            + jnp.arange(queue_size, dtype=mp_policy.index_dtype)
        ) % queue_size
        continuation_parent_idx = self.continuation_parent_idx[positions]
        continuation_thread_id = self.continuation_thread_id[positions]
        continuation_terminal_log_L = (
            self.continuation_terminal_log_L[positions]
        )
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
            valid=_resize_vector(self.valid, size, False),
            continuation_parent_idx=_resize_vector(
                continuation_parent_idx,
                continuation_size,
                -1,
            ),
            continuation_thread_id=_resize_vector(
                continuation_thread_id,
                continuation_size,
                -1,
            ),
            continuation_terminal_log_L=_resize_vector(
                continuation_terminal_log_L,
                continuation_size,
                -jnp.inf,
            ),
            continuation_head=jnp.asarray(0, mp_policy.index_dtype),
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
