"""Host-side growth of transient scheduling storage shared by runners."""

import dataclasses

import jax.numpy as jnp

from jaxns.mixed_precision import mp_policy
from jaxns.state import State


def _grow_continuation_storage(state: State, replacement_width: int) -> State:
    """Double the transient thread heap without advancing logical work."""
    schedule = state.scheduler_data
    if schedule is None:
        raise ValueError("Continuation growth requires an active schedule.")
    current_size = schedule.continuation_parent_idx.shape[0]
    required_size = int(schedule.continuation_count) + replacement_width
    new_size = max(2 * current_size, required_size)
    return dataclasses.replace(
        state,
        scheduler_data=schedule.resize_threads(
            schedule.valid.shape[0],
            continuation_size=new_size,
        ),
        # This is a physical recompilation boundary, not completion of the
        # frozen target or expected-depth traversal.
        depth_reached=jnp.asarray(False, mp_policy.bool_dtype),
    )


def _grow_start_seed_storage(state: State, replacement_width: int) -> State:
    """Double exact no-replacement storage without advancing the schedule."""
    schedule = state.scheduler_data
    if schedule is None:
        raise ValueError("Seed reservation growth requires an active schedule.")
    current_size = schedule.start_seed_reservation_idx.shape[0]
    required_size = 2 * (int(schedule.num_start_seeds) + replacement_width)
    new_size = max(2 * current_size, required_size)
    new_size = 1 << (new_size - 1).bit_length()
    return dataclasses.replace(
        state,
        scheduler_data=schedule.resize_start_seed_reservations(new_size),
        depth_reached=jnp.asarray(False, mp_policy.bool_dtype),
    )
