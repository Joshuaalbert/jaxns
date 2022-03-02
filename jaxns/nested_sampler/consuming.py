from typing import NamedTuple

from jax import numpy as jnp, tree_map
from jax.lax import dynamic_update_slice, while_loop

from jaxns.log_math import LogSpace
from jaxns.types import NestedSamplerState, SampleCollection, EvidenceCalculation


def _update_evidence_calculation(num_live_points: jnp.ndarray,
                                 log_L_contour: jnp.ndarray,
                                 next_log_L_contour: jnp.ndarray,
                                 evidence_calculation: EvidenceCalculation):
    next_L_contour = LogSpace(next_log_L_contour)
    # TODO: bring MIN_FLOAT to initialisation of log_contour rather than check here each time.
    L_contour = LogSpace(jnp.maximum(jnp.finfo(log_L_contour.dtype).min, log_L_contour))
    midL = LogSpace(jnp.log(0.5)) * (next_L_contour + L_contour)
    X_mean = LogSpace(evidence_calculation.log_X_mean)
    X2_mean = LogSpace(evidence_calculation.log_X2_mean)
    Z_mean = LogSpace(evidence_calculation.log_Z_mean)
    ZX_mean = LogSpace(evidence_calculation.log_ZX_mean)
    Z2_mean = LogSpace(evidence_calculation.log_Z2_mean)

    # T_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.))
    T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
    t_mean = LogSpace(- jnp.log(num_live_points + 1.))
    T2_mean = LogSpace(- jnp.logaddexp(jnp.log(2.), -jnp.log(num_live_points)))
    t2_mean = LogSpace(jnp.log(2.) - jnp.log(num_live_points + 1.) - jnp.log(num_live_points + 2.))
    tT_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)) - jnp.log(num_live_points + 2.))

    next_X_mean = X_mean * T_mean
    next_X2_mean = X2_mean * T2_mean
    next_Z_mean = Z_mean + X_mean * t_mean * midL
    next_ZX_mean = ZX_mean * T_mean + X2_mean * tT_mean * midL
    next_Z2_mean = Z2_mean + LogSpace(jnp.log(2.)) * ZX_mean * t_mean * midL + X2_mean * t2_mean * midL ** 2

    next_evidence_calculation = evidence_calculation._replace(log_X_mean=next_X_mean.log_abs_val,
                                                              log_X2_mean=next_X2_mean.log_abs_val,
                                                              log_Z_mean=next_Z_mean.log_abs_val,
                                                              log_Z2_mean=next_Z2_mean.log_abs_val,
                                                              log_ZX_mean=next_ZX_mean.log_abs_val
                                                              )
    # TODO: determine where NaN is coming from and stop it.
    next_evidence_calculation = tree_map(lambda old, new: jnp.where(jnp.isnan(new), old, new),
                                         evidence_calculation, next_evidence_calculation)

    return next_evidence_calculation


def consume_reservoir(state: NestedSamplerState, goal_num_live_points) -> NestedSamplerState:
    class MergeState(NamedTuple):
        i_thread: jnp.ndarray
        i_main: jnp.ndarray

    def consume_cond(consume_state):
        (state, idx_min, num_live_points_thread, _) = consume_state
        on_plateau = state.reservoir.log_L[idx_min] == state.log_L_contour
        not_done = jnp.greater_equal(num_live_points_thread, goal_num_live_points) | on_plateau
        return not_done

    sample_collection_main = state.sample_collection

    def consume_body(consume_state):
        (state, idx_min, num_live_points_thread, merge_state) = consume_state

        num_live_points_main = sample_collection_main.num_live_points[merge_state.i_main]
        num_live_points = num_live_points_main + num_live_points_thread
        log_L_thread = state.reservoir.log_L[idx_min]
        log_L_main = sample_collection_main.log_L[merge_state.i_main]
        # If select_main, then we increment sample collection with a sample from the main sample collection, and
        # increment evidence accordingly. Otherwise, we do the same with sample from reservoir (the thread).
        # We move main index marker forward if yes, otherwise we pick the next reservoir point.
        select_main = jnp.less(log_L_main, log_L_thread)
        next_log_L_contour = jnp.where(select_main,
                                       log_L_main,
                                       log_L_thread)
        merge_state = merge_state._replace(i_thread=jnp.where(select_main,
                                                              merge_state.i_thread,
                                                              merge_state.i_thread + 1),
                                           i_main=jnp.where(select_main,
                                                            merge_state.i_main + 1,
                                                            merge_state.i_main))

        # increment evidence

        next_evidence_calculation = _update_evidence_calculation(num_live_points=num_live_points,
                                                                 log_L_contour=state.log_L_contour,
                                                                 next_log_L_contour=next_log_L_contour,
                                                                 evidence_calculation=state.evidence_calculation)

        dZ_mean = (LogSpace(next_evidence_calculation.log_Z_mean)
                   - LogSpace(state.evidence_calculation.log_Z_mean)).abs()

        # save dead point

        def _update(operand, update):
            start_indices = jnp.concatenate([
                state.sample_idx[None],
                jnp.zeros((len(update.shape),), state.sample_idx.dtype)])
            return dynamic_update_slice(operand, update[None], start_indices)

        next_sample_collection = \
            tree_map(_update,
                     state.sample_collection,
                     SampleCollection(
                         points_U=state.reservoir.points_U[idx_min],
                         points_X=tree_map(lambda x: x[idx_min],
                                           state.reservoir.points_X),
                         # The X-valued sample
                         log_L=state.reservoir.log_L[idx_min],
                         # The log-likelihood of sample.
                         num_likelihood_evaluations=state.reservoir.num_likelihood_evaluations[idx_min],
                         # How many likelihood evaluations were required to obtain sample
                         log_dZ_mean=dZ_mean.log_abs_val,
                         # The log mean evidence difference of the sample
                         log_X_mean=next_evidence_calculation.log_X_mean,
                         # The log mean enclosed prior volume of sample
                         num_live_points=num_live_points
                         # How many live points were taken for the sample.
                     ))

        next_reservoir = state.reservoir._replace(
            available=dynamic_update_slice(state.reservoir.available, jnp.asarray([False]), idx_min[None]))

        state = state._replace(sample_collection=next_sample_collection,
                               evidence_calculation=next_evidence_calculation,
                               reservoir=next_reservoir,
                               log_L_contour=next_log_L_contour,
                               sample_idx=state.sample_idx + jnp.ones_like(state.sample_idx))

        next_idx_min = jnp.argmin(jnp.where(state.reservoir.available, state.reservoir.log_L, jnp.inf))
        next_num_live_points = jnp.sum(state.reservoir.available)

        return (state, next_idx_min, next_num_live_points)

    init_idx_min = jnp.argmin(jnp.where(state.reservoir.available, state.reservoir.log_L, jnp.inf))
    init_num_live_points = jnp.sum(state.reservoir.available)
    (state, _, _) = while_loop(consume_cond,
                               consume_body,
                               (state, init_idx_min, init_num_live_points))
    return state
