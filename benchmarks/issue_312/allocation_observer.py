"""Read-only batch observations for the offline allocation comparison.

Use the existing interruptible continuation, then replay only the deterministic
seed planner to recover actual seed identities. No likelihood is replayed, no
host callback enters compiled sampling, and no production interface changes.
"""

import dataclasses
from collections.abc import Callable
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp

import jaxns.core as core_module
from jaxns.algorithm.depth import CoreWorkBatch, _plan_scheduled_work_batch
from jaxns.core import NestedSampler
from jaxns.state import State


@partial(jax.jit, static_argnames=("max_samples",))
def replay_work(state: State, max_samples: int | None) -> CoreWorkBatch:
    """Reproduce the request for one batch, including the new-depth key split."""
    new_depth_key, _ = jax.random.split(state.random_key)
    depth_key = jnp.where(state.depth_reached, new_depth_key, state.random_key)
    plan_key, _, _ = jax.random.split(depth_key, 3)
    schedule = dataclasses.replace(
        state.scheduler_data,
        seed_reservoir_key=jnp.where(
            state.depth_reached,
            jax.random.fold_in(depth_key, state.allocation_loop_iter),
            state.scheduler_data.seed_reservoir_key,
        ),
    )
    capacity = state.samples.log_likelihoods.shape[0]
    limit = capacity if max_samples is None else min(capacity, max_samples)
    _, work = _plan_scheduled_work_batch(
        plan_key, dataclasses.replace(state, scheduler_data=None), schedule,
        limit - state.num_samples,
    )
    return work


class _ObserverStop(Exception):
    """Return a committed batch prefix without claiming a scientific goal."""

    def __init__(self, state: State):
        super().__init__("The benchmark observation budget was reached.")
        self.state = state


def run_with_observer(
    runner: NestedSampler,
    state: State,
    observer: Callable[[State, State, CoreWorkBatch], bool],
    goal: Callable[[State], bool],
) -> State:
    """Observe completed batches; return immediately if the observer says stop.

    Physical pauses do not become goal boundaries. The production driver still
    owns allocation, seed publication, growth and completed-depth bookkeeping.
    This single-process experiment deliberately supplies no checkpoint manager.
    """
    original_depth = core_module._run_depth

    def observed_depth(current, sampler, depth_cond, *, max_samples, max_batches=None):
        initial_batch = int(current.depth_loop_iter)
        while True:
            before = current
            start = int(before.num_samples)
            current = original_depth(
                before, sampler, depth_cond, max_samples=max_samples, max_batches=1,
            )
            stop = int(current.num_samples)
            if stop > start:
                work = replay_work(before, max_samples)
                if int(work.num_valid) != stop - start:
                    raise RuntimeError("Replayed request has a different valid prefix.")
                np.testing.assert_array_equal(
                    current.samples.log_L_constraints[start:stop],
                    work.log_L_constraint[:stop - start],
                )
                if observer(before, current, work):
                    raise _ObserverStop(current)
            if (
                stop == start or bool(current.depth_reached) or bool(current.needs_growth)
                or int(current.termination_reason) != 0
            ):
                return current
            if (max_batches is not None
                    and int(current.depth_loop_iter) - initial_batch >= max_batches):
                return current

    core_module._run_depth = observed_depth
    try:
        return runner.resume_until_goal(state, goal)
    except _ObserverStop as stopped:
        return stopped.state
    finally:
        core_module._run_depth = original_depth
