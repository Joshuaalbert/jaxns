from typing import Tuple

from jax.lax import while_loop, dynamic_update_slice
from jax import numpy as jnp, tree_map

from jaxns.internals.log_semiring import LogSpace
from jaxns.nested_sampler.evidence_calculation import _update_evidence_calculation
from jaxns.nested_sampler.reservoir_refiller import ReservoirRefiller
from jaxns.types import NestedSamplerState, SampleCollection
from jaxns.nested_sampler.termination import termination_condition


def _collect_sample(state):
    # get the next dead point, which will be used if lower than the next point in main samples.
    idx_min_thread = jnp.argmin(jnp.where(state.reservoir.available, state.reservoir.log_L, jnp.inf))
    next_num_live_points = jnp.sum(state.reservoir.available)
    next_log_L = state.reservoir.log_L[idx_min_thread]
    next_point_X = tree_map(lambda y: y[idx_min_thread], state.reservoir.points_X)
    next_point_U = state.reservoir.points_U[idx_min_thread]
    next_num_likelihood_evaluations = state.reservoir.num_likelihood_evaluations[idx_min_thread]

    # Get log_dZ_mean, and log_X_mean
    next_evidence_calculation = _update_evidence_calculation(num_live_points=next_num_live_points,
                                                             log_L=state.log_L_contour,
                                                             next_log_L_contour=next_log_L,
                                                             evidence_calculation=state.evidence_calculation)
    dZ_mean = (LogSpace(next_evidence_calculation.log_Z_mean)
               - LogSpace(state.evidence_calculation.log_Z_mean)).abs()

    def _update(operand, update):
        start_indices = [state.sample_idx] + [jnp.asarray(0, jnp.int_)] * len(update.shape)
        return dynamic_update_slice(operand, update[None].astype(operand.dtype), start_indices)

    next_sample_collection = \
        tree_map(_update,
                 state.sample_collection,
                 SampleCollection(
                     # The XUvalued sample
                     points_U=next_point_U,
                     # The X-valued sample
                     points_X=next_point_X,
                     # The log-likelihood of sample.
                     log_L=next_log_L,
                     # How many likelihood evaluations were required to obtain sample
                     num_likelihood_evaluations=next_num_likelihood_evaluations,
                     # The log mean evidence difference of the sample
                     log_dZ_mean=dZ_mean.log_abs_val,
                     # The log mean enclosed prior volume of sample
                     log_X_mean=next_evidence_calculation.log_X_mean,
                     # How many live points were taken for the sample.
                     num_live_points=next_num_live_points
                 ))
    next_reservoir = state.reservoir._replace(
        available=dynamic_update_slice(state.reservoir.available, jnp.asarray([False]), idx_min_thread[None])
    )
    state = state._replace(sample_collection=next_sample_collection,
                           evidence_calculation=next_evidence_calculation,
                           reservoir=next_reservoir,
                           log_L_contour=next_log_L,
                           sample_idx=state.sample_idx + jnp.ones_like(state.sample_idx),
                           step_idx=state.step_idx + jnp.ones_like(state.step_idx))
    return state


def _empty_reservoir(state: NestedSamplerState) -> NestedSamplerState:
    def thread_cond(state: NestedSamplerState):
        return jnp.bitwise_not(state.done)

    def thread_body(state: NestedSamplerState) -> NestedSamplerState:
        next_state = _collect_sample(state)
        next_reservoir_empty = jnp.bitwise_not(jnp.any(next_state.reservoir.available))
        next_out_of_space = next_state.sample_idx == state.sample_collection.log_L.size
        done = next_reservoir_empty | next_out_of_space
        next_state = next_state._replace(done=done)
        return next_state

    # on each thread loop, we require that we collect some new point, or else we can get in an infinite loop.
    reservoir_empty = jnp.bitwise_not(jnp.any(state.reservoir.available))
    state = state._replace(done=reservoir_empty)
    final_state = while_loop(thread_cond,
                             thread_body,
                             state)

    return final_state


def _static_run(init_state: NestedSamplerState,
                reservoir_refiller: ReservoirRefiller,
                termination_evidence_frac: jnp.ndarray) -> NestedSamplerState:
    def thread_cond(thread_state: Tuple[NestedSamplerState, jnp.ndarray]):
        (state, _) = thread_state
        return jnp.bitwise_not(state.done)

    def thread_body(thread_state: Tuple[NestedSamplerState, jnp.ndarray]) -> Tuple[NestedSamplerState, jnp.ndarray]:
        (state, on_plateau) = thread_state

        # We refill if we are not on a plateau
        refill_thread = jnp.bitwise_not(on_plateau)
        # We accept points on contour if there are not any satisfying points (pure plateau) and we demand refill.
        reservoir_has_any_satisfying = jnp.any(state.reservoir.log_L > state.log_L_contour)
        strict_contour = reservoir_has_any_satisfying
        # draw samples to at least goal_num_live_points, else no (or not much) work.
        # The number of parallel workers should be set appropriately.
        state = reservoir_refiller(
            state=state,
            refill_thread=refill_thread,
            strict_contour=strict_contour)
        # collect the next sample from the reservoir.
        next_state = _collect_sample(state)
        # before replacing contour
        next_on_plateau = state.log_L_contour == next_state.log_L_contour
        # We are done when thread goal reach and reservoir is empty or reservoir is empty.
        thread_reached_goal, terminination_reason = termination_condition(next_state,
                                                                               termination_evidence_frac=termination_evidence_frac,
                                                                               termination_likelihood_contour=None)
        reservoir_empty = jnp.bitwise_not(jnp.any(next_state.reservoir.available))
        done = thread_reached_goal | reservoir_empty
        # set this, so that when we are done it's set correctly
        next_state = next_state._replace(done=done, termination_reason=terminination_reason)
        return (next_state, next_on_plateau)

    # Collect until done condition met
    (final_state, _) = while_loop(thread_cond,
                                        thread_body,
                                        (init_state, jnp.asarray(False, jnp.bool_)))
    # empty remaining points
    final_state = _empty_reservoir(final_state)
    final_state = final_state._replace(thread_idx=final_state.thread_idx + jnp.ones_like(final_state.thread_idx))

    return final_state
