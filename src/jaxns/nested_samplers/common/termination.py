import warnings
from typing import Union, Tuple

from jax import numpy as jnp

from jaxns.internals.mixed_precision import mp_policy
from jaxns.internals.stats import linear_to_log_stats, effective_sample_size_kish
from jaxns.internals.types import BoolArray, IntArray
from jaxns.nested_samplers.common.types import TerminationConditionDisjunction, TerminationConditionConjunction, \
    TerminationRegister, TerminationCondition


def determine_termination(
        term_cond: Union[TerminationConditionDisjunction, TerminationConditionConjunction, TerminationCondition],
        termination_register: TerminationRegister) -> Tuple[BoolArray, IntArray]:
    """
    Determine if termination should happen. Termination Flags are bits:
        0-bit -> 1: used maximum allowed number of samples
        1-bit -> 2: evidence uncert below threshold
        2-bit -> 4: live points evidence below threshold
        3-bit -> 8: effective sample size big enough
        4-bit -> 16: used maxmimum allowed number of likelihood evaluations
        5-bit -> 32: maximum log-likelihood contour reached
        6-bit -> 64: sampler efficiency too low
        7-bit -> 128: entire live-points set is a single plateau
        8-bit -> 256: relative spread of live points < rtol
        9-bit -> 512: absolute spread of live points < atol
        10-bit -> 1024: no seed points left
        11-bit -> 2048: XL < max(XL) * peak_XL_frac

    Multiple flags are summed together

    Args:
        term_cond: termination condition
        termination_register: register of termination variables to check against termination condition

    Returns:
        boolean done signal, and termination reason
    """

    termination_reason = jnp.asarray(0, mp_policy.count_dtype)
    done = jnp.asarray(False, jnp.bool_)

    def _set_done_bit(bit_done, bit_reason, done, termination_reason):
        if bit_done.size > 1:
            raise RuntimeError("bit_done must be a scalar.")
        done = jnp.bitwise_or(bit_done, done)
        termination_reason += jnp.where(bit_done,
                                        jnp.asarray(2 ** bit_reason, mp_policy.count_dtype),
                                        jnp.asarray(0, mp_policy.count_dtype))
        return done, termination_reason

    if isinstance(term_cond, TerminationConditionConjunction):
        for c in term_cond.conds:
            _done, _reason = determine_termination(term_cond=c, termination_register=termination_register)
            done = jnp.bitwise_and(_done, done)
            termination_reason = jnp.bitwise_and(_reason, termination_reason)
        return done, termination_reason

    if isinstance(term_cond, TerminationConditionDisjunction):
        for c in term_cond.conds:
            _done, _reason = determine_termination(term_cond=c, termination_register=termination_register)
            done = jnp.bitwise_or(_done, done)
            termination_reason = jnp.bitwise_or(_reason, termination_reason)
        return done, termination_reason

    if term_cond.live_evidence_frac is not None:
        warnings.warn("live_evidence_frac is deprecated, use dlogZ instead.")

    if term_cond.max_samples is not None:
        # used all points
        reached_max_samples = termination_register.num_samples_used >= term_cond.max_samples
        done, termination_reason = _set_done_bit(reached_max_samples, 0,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.evidence_uncert is not None:
        _, log_Z_var = linear_to_log_stats(
            log_f_mean=termination_register.evidence_calc_with_remaining.log_Z_mean,
            log_f2_mean=termination_register.evidence_calc_with_remaining.log_Z2_mean)
        evidence_uncert_low_enough = log_Z_var <= jnp.square(term_cond.evidence_uncert)
        done, termination_reason = _set_done_bit(evidence_uncert_low_enough, 1,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.dlogZ is not None:
        # (Z_remaining + Z_current) / Z_remaining < exp(dlogZ)
        log_Z_mean_1, log_Z_var_1 = linear_to_log_stats(
            log_f_mean=termination_register.evidence_calc_with_remaining.log_Z_mean,
            log_f2_mean=termination_register.evidence_calc_with_remaining.log_Z2_mean)

        log_Z_mean_0, log_Z_var_0 = linear_to_log_stats(
            log_f_mean=termination_register.evidence_calc.log_Z_mean,
            log_f2_mean=termination_register.evidence_calc.log_Z2_mean)

        small_remaining_evidence = jnp.less(
            log_Z_mean_1 - log_Z_mean_0, term_cond.dlogZ
        )
        done, termination_reason = _set_done_bit(small_remaining_evidence, 2,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.ess is not None:
        # Kish's ESS = [sum weights]^2 / [sum weights^2]
        ess = effective_sample_size_kish(termination_register.evidence_calc_with_remaining.log_Z_mean,
                                         termination_register.evidence_calc_with_remaining.log_dZ2_mean)
        ess_reached = ess >= term_cond.ess
        done, termination_reason = _set_done_bit(ess_reached, 3,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.max_num_likelihood_evaluations is not None:
        num_likelihood_evaluations = jnp.sum(termination_register.num_likelihood_evaluations)
        too_max_likelihood_evaluations = num_likelihood_evaluations >= term_cond.max_num_likelihood_evaluations
        done, termination_reason = _set_done_bit(too_max_likelihood_evaluations, 4,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.log_L_contour is not None:
        likelihood_contour_reached = termination_register.log_L_contour >= term_cond.log_L_contour
        done, termination_reason = _set_done_bit(likelihood_contour_reached, 5,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.efficiency_threshold is not None:
        efficiency_too_low = termination_register.efficiency < term_cond.efficiency_threshold
        done, termination_reason = _set_done_bit(efficiency_too_low, 6,
                                                 done=done, termination_reason=termination_reason)

    done, termination_reason = _set_done_bit(termination_register.plateau, 7,
                                             done=done, termination_reason=termination_reason)

    if term_cond.rtol is not None:
        relative_spread_low = termination_register.relative_spread < term_cond.rtol
        done, termination_reason = _set_done_bit(relative_spread_low, 8,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.atol is not None:
        absolute_spread_low = termination_register.absolute_spread < term_cond.atol
        done, termination_reason = _set_done_bit(absolute_spread_low, 9,
                                                 done=done, termination_reason=termination_reason)

    done, termination_reason = _set_done_bit(termination_register.no_seed_points, 10,
                                             done=done, termination_reason=termination_reason)

    if term_cond.peak_XL_frac is not None:
        log_XL = termination_register.evidence_calc.log_X_mean + termination_register.evidence_calc.log_L
        peak_log_XL = termination_register.peak_log_XL
        XL_reduction_reached = log_XL < peak_log_XL + jnp.log(term_cond.peak_XL_frac)
        done, termination_reason = _set_done_bit(XL_reduction_reached, 11,
                                                 done=done, termination_reason=termination_reason)

    return done, termination_reason
