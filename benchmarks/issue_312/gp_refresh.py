"""Host-side GP refits between existing one-batch depth continuations.

This isolated benchmark hook leaves the production runner, allocation and
seed planner intact. It is single-process and intentionally has no checkpoint
support: a production adaptation API would need to own the fit in State.
"""

from collections.abc import Callable

import jax
from jax import numpy as jnp

import jaxns.core as core_module
from jaxns.algorithm.race_tree import build_block_state
from jaxns.constrained_sampler import AbstractSampler
from jaxns.core import NestedSampler
from jaxns.shrinkage.classic import classic_dirichlet_concentrations
from jaxns.state import State


@jax.jit
def batch_frontier(state: State, start: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Shallowest parent of the last batch and its expected classic volume."""
    indices = jnp.arange(state.samples.log_likelihoods.shape[0])
    contours = state.samples.log_L_constraints
    selected = (indices >= start) & (indices < state.num_samples) & jnp.isfinite(contours)
    contour = jnp.min(jnp.where(selected, contours, jnp.inf))
    blocks = build_block_state(
        state.samples, root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
    )
    concentrations = classic_dirichlet_concentrations(blocks)
    total = concentrations.alpha_gt + concentrations.alpha_eq + concentrations.alpha_lt
    log_ratio = jnp.log(concentrations.alpha_gt) - jnp.log(total)
    log_volume = jnp.sum(jnp.where(
        blocks.valid & (blocks.log_L_blocks <= contour), log_ratio, 0.0,
    ))
    return contour, log_volume


def run_with_refresh(
    runner: NestedSampler,
    state: State,
    sampler: AbstractSampler,
    refresh: Callable[[State, float, float], AbstractSampler],
    refresh_efolds: float = 1.0,
) -> State:
    """Run one ordinary depth epoch, refitting after each completed e-fold.

    A refit uses the just-completed batch's frontier and takes effect on the
    next batch. The first fit must be supplied by the caller. Neither a fit
    nor a physical batch return is promoted to an allocation or goal boundary.
    """
    if refresh_efolds <= 0:
        raise ValueError("Refresh interval must be positive.")
    original_depth = core_module._run_depth
    active_sampler = sampler
    last_log_volume = 0.0

    def batched_depth(current, unused_sampler, depth_cond, *, max_samples, max_batches=None):
        nonlocal active_sampler, last_log_volume
        initial_batch = int(current.depth_loop_iter)
        while True:
            start = int(current.num_samples)
            current = original_depth(
                current, active_sampler, depth_cond,
                max_samples=max_samples, max_batches=1,
            )
            # Return every original lifecycle event to the production driver.
            # A zero-progress return exposes seed publication/storage guards.
            if (
                int(current.num_samples) == start or bool(current.depth_reached)
                or bool(current.needs_growth) or int(current.termination_reason) != 0
            ):
                return current
            contour, log_volume = map(float, batch_frontier(current, jnp.asarray(start)))
            if jnp.isfinite(contour) and abs(log_volume - last_log_volume) >= refresh_efolds:
                active_sampler = refresh(current, contour, log_volume)
                last_log_volume = log_volume
            if (max_batches is not None
                    and int(current.depth_loop_iter) - initial_batch >= max_batches):
                return current

    # Scope the hook to this one offline run and restore it even on failure.
    # Copying the large runner lifecycle here would risk changing race or
    # stopping semantics; exact frozen-kernel replay checks cover this seam.
    core_module._run_depth = batched_depth
    try:
        return runner.resume_until_goal(state, lambda s: int(s.goal_loop_iter) > 0)
    finally:
        core_module._run_depth = original_depth
