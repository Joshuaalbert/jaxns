"""Benchmark batch observations and an explicit optional sampler-adaptation seam.

The default path is read-only and replays only deterministic seed planning.
The optional supplier deliberately changes the benchmark kernel between batches
using immutable data. Neither path replays likelihoods, changes production
interfaces, or inserts a host callback into compiled sampling.
"""

import dataclasses
from collections.abc import Callable
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp

import jaxns.core as core_module
from jaxns.algorithm.depth import CoreWorkBatch, _plan_scheduled_work_batch
from jaxns.constrained_sampler import AbstractSampler
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
    *,
    prepare_sampler: Callable[
        [State, CoreWorkBatch, AbstractSampler], AbstractSampler,
    ] | None = None,
) -> State:
    """Observe completed batches, optionally preparing a frozen sampler beforehand.

    With no supplier, observations cannot change the sampling kernel. With a
    supplier, adaptation is explicit benchmark behavior between complete batches.

    Physical pauses do not become goal boundaries. The production driver still
    owns allocation, seed publication, growth and completed-depth bookkeeping.
    This single-process experiment deliberately supplies no checkpoint manager.
    A benchmark may supply immutable sampler data before dispatch; its observer
    owns all adaptation and must checkpoint it together with the returned State.
    """
    original_depth = core_module._run_depth

    def observed_depth(current, sampler, depth_cond, *, max_samples, max_batches=None):
        initial_batch = int(current.depth_loop_iter)
        while True:
            before = current
            start = int(before.num_samples)
            active_sampler = sampler
            planned_work = None
            if prepare_sampler is not None:
                # The optional experiment observes existing planner output; it
                # cannot change scheduling or manufacture a scientific goal.
                planned_work = replay_work(before, max_samples)
                active_sampler = prepare_sampler(before, planned_work, sampler)
            current = original_depth(
                before, active_sampler, depth_cond, max_samples=max_samples, max_batches=1,
            )
            stop = int(current.num_samples)
            if stop > start:
                work = (replay_work(before, max_samples) if planned_work is None
                        else planned_work)
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
