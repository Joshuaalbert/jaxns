import dataclasses
from functools import partial
from typing import Tuple

import jax
from jax import numpy as jnp

from jaxns.nested_samplers.mixed_precision import mp_policy
from jaxns.nested_samplers.pytree import PureDataclassPytree
from jaxns.internals.stats import linear_to_log_stats, effective_sample_size_kish
from jaxns.nested_samplers.evidence_calculation import EvidenceCalculation
from jaxns.nested_samplers.types import FloatArray, IntArray, BoolArray


@dataclasses.dataclass(slots=True)
class TerminationCondition(PureDataclassPytree):
    """
    Contains the termination conditions for the nested sampling run.

    Args:
        ess: The effective sample size, if the ESS (Kish's estimate) is greater than this the run will terminate.
        evidence_uncert: The uncertainty in the evidence, if the uncertainty is less than this the run will terminate.
        dlogZ: Terminate if log(Z_current + Z_remaining) - log(Z_current) < dlogZ. Default log(1 + 1e-2)
        max_samples: Terminate if the number of samples exceeds this.
        max_num_likelihood_evaluations: Terminate if the number of likelihood evaluations exceeds this.
        log_L_contour: Terminate if this log(L) contour is reached. A contour is reached if any dead point
            has log(L) > log_L_contour. Uncollected live points are not considered.
        efficiency_threshold: Terminate if the efficiency (num_samples / num_likelihood_evaluations) is less than this,
            for the last shrinkage iteration.
        rtol: finish when the relative value 2*|log_L_max - log_L_min|/|log_L_max + log_L_min| < rol
        atol: finish when the absolute |log_L_max - log_L_min| < atol
        cummax_XL_frac: Terminate when XL < cummax(XL) * cummax_XL_frac
    """
    ess: FloatArray | None = None
    evidence_uncert: FloatArray | None = None
    dlogZ: FloatArray | None = None  #
    max_samples: IntArray | None = None
    max_num_likelihood_evaluations: IntArray | None = None
    log_L_contour: FloatArray | None = None
    efficiency_threshold: FloatArray | None = None
    rtol: FloatArray | None = None
    atol: FloatArray | None = None
    cummax_XL_frac: FloatArray | None = None


TerminationCondition.register_pytree()


@dataclasses.dataclass(slots=True)
class TerminationRegister(PureDataclassPytree):
    num_samples_used: IntArray
    evidence_calc: EvidenceCalculation
    evidence_calc_with_remaining: EvidenceCalculation
    num_likelihood_evaluations: IntArray
    log_L_contour: FloatArray
    efficiency: FloatArray
    plateau: BoolArray
    no_seed_points: BoolArray
    relative_spread: FloatArray
    absolute_spread: FloatArray
    cummax_log_XL: FloatArray

    @staticmethod
    def initialise():
        zero_count = jnp.asarray(0, mp_policy.count_dtype)
        init_evidence_calc = EvidenceCalculation.initialise()
        return TerminationRegister(
            num_samples_used=zero_count,
            evidence_calc=init_evidence_calc,
            evidence_calc_with_remaining=init_evidence_calc,
            num_likelihood_evaluations=zero_count,
            log_L_contour=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
            efficiency=jnp.asarray(0., mp_policy.measure_dtype),
            plateau=jnp.asarray(False, mp_policy.bool_dtype),
            no_seed_points=jnp.asarray(False, mp_policy.bool_dtype),
            relative_spread=jnp.asarray(jnp.inf, mp_policy.measure_dtype),
            absolute_spread=jnp.asarray(jnp.inf, mp_policy.measure_dtype),
            cummax_log_XL=jnp.asarray(-jnp.inf, mp_policy.measure_dtype)
        )

    def is_done(self, term_cond: TerminationCondition) -> tuple[BoolArray, IntArray]:
        """
        Determine if termination should happen. Termination Flags are bits:
            0-bit -> 1: used maximum allowed number of samples
            1-bit -> 2: evidence uncert below threshold
            2-bit -> 4: live points evidence below threshold
            3-bit -> 8: effective sample size big enough
            4-bit -> 16: used maxmimum allowed number of likelihood evaluations
            5-bit -> 32: maximum log-likelihood contour reached
            6-bit -> 64: sampler efficiency too low
            7-bit -> 128: all lineages on a single plateau
            8-bit -> 256: relative spread of live points < rtol
            9-bit -> 512: absolute spread of live points < atol
            10-bit -> 1024: no seed points left
            11-bit -> 2048: XL < cummax(XL) * peak_XL_frac

        Multiple flags are summed together

        Args:
            term_cond: termination condition

        Returns:
            boolean done signal, and termination reason
        """
        return _is_done(self, term_cond)


TerminationRegister.register_pytree()


@partial(jax.jit, inline=True)
def _is_done(self, term_cond: TerminationCondition) -> tuple[BoolArray, IntArray]:
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

    if term_cond.max_samples is not None:
        # used all points
        reached_max_samples = self.num_samples_used >= term_cond.max_samples
        done, termination_reason = _set_done_bit(reached_max_samples, 0,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.evidence_uncert is not None:
        _, log_Z_var = linear_to_log_stats(
            log_f_mean=self.evidence_calc_with_remaining.Z_mean.log_abs_val,
            log_f2_mean=self.evidence_calc_with_remaining.Z2_mean.log_abs_val)
        evidence_uncert_low_enough = log_Z_var <= jnp.square(term_cond.evidence_uncert)
        done, termination_reason = _set_done_bit(evidence_uncert_low_enough, 1,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.dlogZ is not None:
        # (Z_remaining + Z_current) / Z_remaining < exp(dlogZ)
        log_Z_mean_1, log_Z_var_1 = linear_to_log_stats(
            log_f_mean=self.evidence_calc_with_remaining.Z_mean.log_abs_val,
            log_f2_mean=self.evidence_calc_with_remaining.Z2_mean.log_abs_val)

        log_Z_mean_0, log_Z_var_0 = linear_to_log_stats(
            log_f_mean=self.evidence_calc.Z_mean.log_abs_val,
            log_f2_mean=self.evidence_calc.Z2_mean.log_abs_val)

        small_remaining_evidence = jnp.less(
            log_Z_mean_1 - log_Z_mean_0, term_cond.dlogZ
        )
        done, termination_reason = _set_done_bit(small_remaining_evidence, 2,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.ess is not None:
        # Kish's ESS = [sum weights]^2 / [sum weights^2]
        ess = effective_sample_size_kish(self.evidence_calc_with_remaining.Z_mean.log_abs_val,
                                         self.evidence_calc_with_remaining.dZ2_mean.log_abs_val)
        ess_reached = ess >= term_cond.ess
        done, termination_reason = _set_done_bit(ess_reached, 3,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.max_num_likelihood_evaluations is not None:
        num_likelihood_evaluations = jnp.sum(self.num_likelihood_evaluations)
        too_max_likelihood_evaluations = num_likelihood_evaluations >= term_cond.max_num_likelihood_evaluations
        done, termination_reason = _set_done_bit(too_max_likelihood_evaluations, 4,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.log_L_contour is not None:
        likelihood_contour_reached = self.log_L_contour >= term_cond.log_L_contour
        done, termination_reason = _set_done_bit(likelihood_contour_reached, 5,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.efficiency_threshold is not None:
        efficiency_too_low = self.efficiency < term_cond.efficiency_threshold
        done, termination_reason = _set_done_bit(efficiency_too_low, 6,
                                                 done=done, termination_reason=termination_reason)

    done, termination_reason = _set_done_bit(self.plateau, 7,
                                             done=done, termination_reason=termination_reason)

    if term_cond.rtol is not None:
        relative_spread_low = self.relative_spread < term_cond.rtol
        done, termination_reason = _set_done_bit(relative_spread_low, 8,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.atol is not None:
        absolute_spread_low = self.absolute_spread < term_cond.atol
        done, termination_reason = _set_done_bit(absolute_spread_low, 9,
                                                 done=done, termination_reason=termination_reason)

    # Sampler relies upon this termination condition to stop when no seed points are left.
    done, termination_reason = _set_done_bit(self.no_seed_points, 10,
                                             done=done, termination_reason=termination_reason)

    if term_cond.cummax_XL_frac is not None:
        log_XL = self.evidence_calc.X_mean.log_abs_val + self.evidence_calc.L.log_abs_val
        peak_log_XL = self.cummax_log_XL
        XL_reduction_reached = log_XL < peak_log_XL + jnp.log(term_cond.cummax_XL_frac)
        done, termination_reason = _set_done_bit(XL_reduction_reached, 11,
                                                 done=done, termination_reason=termination_reason)

    return done, termination_reason
