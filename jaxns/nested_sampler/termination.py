from jax import numpy as jnp

from jaxns.types import NestedSamplerState
from jaxns.internals.stats import linear_to_log_stats
from jaxns.internals.log_semiring import LogSpace

def termination_condition(state: NestedSamplerState, *,
                          termination_evidence_frac=None,
                          termination_likelihood_frac=None,
                          termination_ess=None,
                          termination_likelihood_contour=None,
                          termination_evidence_uncert=None,
                          termination_max_num_threads=None):
    """
    Decide whether to terminate the sampler.

    Current termination criteria:

    1. all remaining points are a single plateau.
    2. maximum number of iterations taken.

    (Optional termination conditions)
    3. small amount of evidence in reservoir (with at least num live points points to compute that from)
    4. small improvement in likelihood in latest dead points
    5. Effective sample-size reached.

    The first condition that is true causes termination.

    Args:
        state: NestedSamplerState
        *
        termination_evidence_frac: float, what frac of evidence should live points hold.
        termination_likelihood_frac: float, what is the minimum likelihood improvement.
        termination_ess: float, what is the desired ESS.
        termination_likelihood_contour: float, what is the contour to reach.
        termination_evidence_uncert: float, threshold of StdDev(log(Z))

    Returns:
        done: bool
        termination_condition: int
    """

    termination_condition = jnp.zeros_like(state.termination_reason)
    # used all points
    reached_max_samples = state.sample_idx >= state.sample_collection.log_L.size
    done = reached_max_samples
    termination_condition += jnp.where(reached_max_samples,
                                       jnp.asarray(2 ** 0, jnp.int_),
                                       jnp.asarray(0, jnp.int_))

    if termination_evidence_uncert is not None:
        log_Z_mean, log_Z_var = linear_to_log_stats(log_f_mean=state.evidence_calculation.log_Z_mean,
                                                    log_f2_mean=state.evidence_calculation.log_Z2_mean)
        evidence_uncert_low_enough = log_Z_var <= jnp.square(termination_evidence_uncert)
        done = done | evidence_uncert_low_enough
        termination_condition += jnp.where(evidence_uncert_low_enough,
                                           jnp.asarray(2 ** 1, jnp.int_),
                                           jnp.asarray(0, jnp.int_))
    if termination_likelihood_contour is not None:
        likelihood_contour_reached = state.log_L_contour >= termination_likelihood_contour
        done = done | likelihood_contour_reached
        termination_condition += jnp.where(likelihood_contour_reached,
                                           jnp.asarray(2 ** 2, jnp.int_),
                                           jnp.asarray(0, jnp.int_))
    if termination_evidence_frac is not None:
        # Z_live = <L> X_i = exp(logsumexp(log_L_live) - log(N) + log(X))
        log_Z_mean = state.evidence_calculation.log_Z_mean
        Z_live = LogSpace(jnp.where(state.reservoir.available, state.reservoir.log_L, -jnp.inf)).sum() / LogSpace(
            jnp.log(jnp.sum(state.reservoir.available))) \
                 * LogSpace(state.evidence_calculation.log_X_mean)
        # Z_live < f * Z => logZ_live < log(f) + logZ
        small_remaining_evidence = Z_live.log_abs_val < jnp.log(termination_evidence_frac) + log_Z_mean
        done = done | small_remaining_evidence
        termination_condition += jnp.where(small_remaining_evidence,
                                           jnp.asarray(2 ** 3, jnp.int_),
                                           jnp.asarray(0, jnp.int_))
    if termination_likelihood_frac is not None:
        # small likelihood increase available, L_[argmax]/L_[contour] - 1 < frac
        likelihood_peak_reached = jnp.max(state.reservoir.log_L) - state.log_L_contour <= jnp.log1p(
            termination_likelihood_frac)
        done = done | likelihood_peak_reached
        termination_condition += jnp.where(likelihood_peak_reached,
                                           jnp.asarray(2 ** 4, jnp.int_),
                                           jnp.asarray(0, jnp.int_))
    if termination_ess is not None:
        # Kish's ESS = [sum weights]^2 / [sum weights^2]
        ess = LogSpace(state.evidence_calculation.log_Z_mean).square() \
              / LogSpace(state.evidence_calculation.log_dZ2_mean)
        ess_reached = ess.log_abs_val >= jnp.log(termination_ess)
        done = done | ess_reached
        termination_condition += jnp.where(ess_reached,
                                           jnp.asarray(2 ** 5, jnp.int_),
                                           jnp.asarray(0, jnp.int_))
    if termination_max_num_threads is not None:
        max_threads_reached = state.thread_idx >= termination_max_num_threads
        done = done | max_threads_reached
        termination_condition += jnp.where(max_threads_reached,
                                           jnp.asarray(2 ** 6, jnp.int_),
                                           jnp.asarray(0, jnp.int_))
    return done, termination_condition
