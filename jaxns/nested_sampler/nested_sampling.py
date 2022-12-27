from typing import Tuple

from jax import numpy as jnp, random
from jax.lax import dynamic_update_slice
from jax.lax import while_loop

from jaxns.internals.log_semiring import LogSpace, normalise_log_space
from jaxns.internals.maps import replace_index
from jaxns.internals.stats import linear_to_log_stats
from jaxns.internals.types import NestedSamplerState, int_type


def sample_goal_distribution(key, log_goal_weights, S: int, *, replace: bool = True):
    """
    Sample indices that match unnormalised log_probabilities.

    Args:
        key: PRNG key
        log_goal_weights: unnormalised log probabilities
        S: number of samples
        replace: bool, whether to sample with replacement

    Returns:
        indices that draw from target density
    """
    if replace:
        p_cuml = LogSpace(log_goal_weights).cumsum()
        # 1 - U in (0,1] instead of [0,1)
        r = p_cuml[-1] * LogSpace(jnp.log(1 - random.uniform(key, (S,))))
        idx = jnp.searchsorted(p_cuml.log_abs_val, r.log_abs_val)
    else:
        assert S <= log_goal_weights.size
        g = -random.gumbel(key, shape=log_goal_weights.shape) - log_goal_weights
        idx = jnp.argsort(g)[:S]
    return idx


def compute_remaining_evidence(sample_idx, log_dZ_mean):
    # Z_remaining = dZ_mean.cumsum(reverse=True)

    def logsumexp_cumsum_body(_state):
        (log_abs_val, idx) = _state
        next_idx = idx - jnp.ones_like(idx)
        next_val = LogSpace(log_abs_val[idx]) + LogSpace(log_abs_val[next_idx])
        next_log_abs_val = dynamic_update_slice(log_abs_val, next_val.log_abs_val[None], [next_idx])
        return (next_log_abs_val, next_idx)

    # Calculate remaining evidence, doing only the minimal amount of work necessary.

    (log_Z_remaining, _) = while_loop(lambda _state: _state[1] > 0,
                                      logsumexp_cumsum_body,
                                      (log_dZ_mean,
                                       sample_idx - jnp.ones_like(sample_idx)))
    return log_Z_remaining


def evidence_goal(state: NestedSamplerState):
    """
    Estimates the impact of adding a sample at a certain likelihood contour by computing the impact of removing a point.

    Args:
        state:

    Returns:

    """
    # evidence uncertainty minimising goal.
    # remove points and see what increases uncertainty the most.

    _, log_Z_var0 = linear_to_log_stats(log_f_mean=state.evidence_calculation.log_Z_mean,
                                        log_f2_mean=state.evidence_calculation.log_Z2_mean)
    num_shrinkages = -state.evidence_calculation.log_X_mean
    delta_idx = state.sample_idx / (2. * num_shrinkages)

    def body(body_state):
        (remove_idx, inf_max_dvar, inf_max_dvar_idx, sup_max_dvar, sup_max_dvar_idx) = body_state
        # removing a sample is equivalent to setting that n=inf at that point
        perturbed_num_live_points = replace_index(state.sample_collection.num_live_points, jnp.inf,
                                                  remove_idx.astype(int_type))
        perturbed_sample_collection = state.sample_collection._replace(num_live_points=perturbed_num_live_points)
        (evidence_calculation, _, _, _, _) = \
            compute_evidence(num_samples=state.sample_idx, sample_collection=perturbed_sample_collection)
        _, log_Z_var = linear_to_log_stats(log_f_mean=evidence_calculation.log_Z_mean,
                                           log_f2_mean=evidence_calculation.log_Z2_mean)
        dvar = log_Z_var - log_Z_var0

        return remove_idx + delta_idx, dvar


def _get_dynamic_goal(state: NestedSamplerState, G: jnp.ndarray):
    """
    Get contiguous contours that we'd like to reinforce.

    We have two objectives, which can be mixed by setting `G`.
    G=0: choose contours that decrease evidence uncertainty the most.
    G=1: choose contours that increase ESS the most.

    Note: This slightly departs from the Dynamic Nested Sampling paper.
    """

    n_i = state.sample_collection.num_live_points
    dZ_mean = LogSpace(state.sample_collection.log_dZ_mean)
    # Calculate remaining evidence, doing only the amount of work necessary.
    Z_remaining = LogSpace(compute_remaining_evidence(state.sample_idx, state.sample_collection.log_dZ_mean))
    # TODO: numerically compute goal using custome norm
    I_evidence = ((LogSpace(jnp.log(n_i + 1.)) * Z_remaining + LogSpace(jnp.log(n_i)) * dZ_mean) / (
            LogSpace(jnp.log(n_i + 1.)).sqrt() * LogSpace(jnp.log(n_i + 2.)) ** (1.5)))
    I_evidence = normalise_log_space(I_evidence)

    I_posterior = dZ_mean
    I_posterior = normalise_log_space(I_posterior)

    I_goal = LogSpace(jnp.log(1. - G)) * I_evidence + LogSpace(jnp.log(G)) * I_posterior
    # I_goal = normalise_log_space(I_goal) # unnecessary for sampling

    mask = jnp.arange(I_goal.size) >= state.sample_idx
    I_goal = LogSpace(jnp.where(mask, -jnp.inf, I_goal.log_abs_val))

    return I_goal.log_abs_val


def get_dynamic_goal(key, state: NestedSamplerState, num_samples: int, G: jnp.ndarray) -> Tuple[
    jnp.ndarray, jnp.ndarray]:
    """
    Determines what seed points to sample above.
    """
    contours = jnp.concatenate([state.sample_collection.log_L_constraint[0:1],
                                state.sample_collection.log_L_samples])
    if G is None:
        raise ValueError(f"G should be a float in [0,1].")
    log_goal_weights = _get_dynamic_goal(state, G)
    # Probabilistically sample the contours according to goal distribution
    indices_constraint_reinforce = sample_goal_distribution(key, log_goal_weights, num_samples, replace=True)
    start_idx = indices_constraint_reinforce.min()
    end_idx = indices_constraint_reinforce.max()
    log_L_constraint_start = contours[start_idx]
    log_L_constraint_end = contours[end_idx]

    return log_L_constraint_start, log_L_constraint_end
