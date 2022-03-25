from jax import numpy as jnp

from jaxns.types import TerminationStats
from jaxns.internals.stats import linear_to_log_stats
from jaxns.internals.log_semiring import LogSpace

def termination_condition(termination_stats: TerminationStats, *,
                          termination_live_evidence_frac=None,
                          termination_ess=None,
                          termination_likelihood_contour=None,
                          termination_evidence_uncert=None,
                          termination_max_num_steps=None,
                          termination_max_samples=None):
    """
    Decide whether to terminate the nested sampler.

    Args:
        termination_stats: TerminationStats
        termination_live_evidence_frac: Stop when mean log evidence changes by less than this amount
        termination_ess: Stop when this many effective samples acquired.
        termination_likelihood_contour: Stop when this likelihood contour reached.
        termination_evidence_uncert: Stop when uncertainty in log evidence is below this.
        termination_max_num_steps: Stop when this many steps taken.
        termination_max_samples: Stop when this many samples taken.

    Returns:
        done: bool, whether to Stop
        termination_condition: int, binary flag recording the reasons for stopping (if more than one).
    """
    termination_condition = jnp.asarray(0, jnp.int_)
    done = jnp.asarray(False)

    if termination_max_samples is not None:
        # used all points
        reached_max_samples = termination_stats.num_samples >= termination_max_samples
        done = done | reached_max_samples
        termination_condition += jnp.where(reached_max_samples,
                                           jnp.asarray(2 ** 0, jnp.int_),
                                           jnp.asarray(0, jnp.int_))
    if termination_evidence_uncert is not None:
        # dynamic stopping condition, for G=0
        log_Z_mean, log_Z_var = linear_to_log_stats(log_f_mean=termination_stats.current_evidence_calculation.log_Z_mean,
                                                    log_f2_mean=termination_stats.current_evidence_calculation.log_Z2_mean)
        evidence_uncert_low_enough = log_Z_var <= jnp.square(termination_evidence_uncert)
        done = done | evidence_uncert_low_enough
        termination_condition += jnp.where(evidence_uncert_low_enough,
                                           jnp.asarray(2 ** 1, jnp.int_),
                                           jnp.asarray(0, jnp.int_))
    if termination_likelihood_contour is not None:
        # dynamic stopping condition, controlling thread termination
        likelihood_contour_reached = termination_stats.log_L_contour_max >= termination_likelihood_contour
        done = done | likelihood_contour_reached
        termination_condition += jnp.where(likelihood_contour_reached,
                                           jnp.asarray(2 ** 2, jnp.int_),
                                           jnp.asarray(0, jnp.int_))
    if termination_live_evidence_frac is not None:
        # dynamic stopping condition, for stopping static run
        # Z_live = <L> X_i = exp(logsumexp(log_L_live) - log(N) + log(X))
        # dynamic stopping condition, for G=0
        prev_log_Z_mean, _ = linear_to_log_stats(
            log_f_mean=termination_stats.previous_evidence_calculation.log_Z_mean,
            log_f2_mean=termination_stats.previous_evidence_calculation.log_Z2_mean)
        log_Z_mean, _ = linear_to_log_stats(
            log_f_mean=termination_stats.current_evidence_calculation.log_Z_mean,
            log_f2_mean=termination_stats.current_evidence_calculation.log_Z2_mean)
       # Z_live < f * Z_prev <=> |Z-Z_prev| < f * Z_prev <=> Z < Z_prev * (1 + f) => log(Z) < log(Z_prev) + log(1+f)
        small_remaining_evidence = jnp.abs(log_Z_mean - prev_log_Z_mean) < jnp.log(1. + termination_live_evidence_frac)
        done = done | small_remaining_evidence
        termination_condition += jnp.where(small_remaining_evidence,
                                           jnp.asarray(2 ** 3, jnp.int_),
                                           jnp.asarray(0, jnp.int_))
    if termination_ess is not None:
        # Kish's ESS = [sum weights]^2 / [sum weights^2]
        ess = LogSpace(termination_stats.current_evidence_calculation.log_Z_mean).square() \
              / LogSpace(termination_stats.current_evidence_calculation.log_dZ2_mean)
        ess_reached = ess.log_abs_val >= jnp.log(termination_ess)
        done = done | ess_reached
        termination_condition += jnp.where(ess_reached,
                                           jnp.asarray(2 ** 4, jnp.int_),
                                           jnp.asarray(0, jnp.int_))
    if termination_max_num_steps is not None:
        max_steps_reached = termination_stats.num_steps >= termination_max_num_steps
        done = done | max_steps_reached
        termination_condition += jnp.where(max_steps_reached,
                                           jnp.asarray(2 ** 5, jnp.int_),
                                           jnp.asarray(0, jnp.int_))
    return done, termination_condition
