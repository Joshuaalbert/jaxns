import dataclasses
from functools import partial

import jax
from jax import numpy as jnp

from jaxns.evidence_calculation import EvidenceCalculation
from jaxns.log_semiring import LogSpace
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.stats_utils import linear_to_log_stats, effective_sample_size_kish
from jaxns.types import FloatArray, IntArray, BoolArray


@dataclasses.dataclass(slots=True)
class DepthCondition(PureDataclassPytree):
    """
    Contains the termination conditions for the nested sampling run.

    Args:
        ess: The effective sample size, or an estimate, is greater than this the run will terminate.
        evidence_uncert: The uncertainty in the evidence, if the uncertainty is less than this the run will terminate.
        dlogZ: Terminate if dZ_last_shrinkage/Z < dlogZ.
        max_samples: Terminate if the number of samples exceeds this.
        max_num_likelihood_evaluations: Terminate if the number of likelihood evaluations exceeds this.
        log_L_target: Terminate if any likelihood passes this peak (not only iid samples).
        log_L_contour_target: Terminate if the likelihood contour passes this peak (i.e. all samples with log_L < log_L_contour_max have desired live point count)
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
    log_L_target: FloatArray | None = None
    log_L_contour_target: FloatArray | None = None
    efficiency_threshold: FloatArray | None = None
    rtol: FloatArray | None = None
    atol: FloatArray | None = None
    cummax_XL_frac: FloatArray | None = None


DepthCondition.register_pytree()


@dataclasses.dataclass(slots=True)
class TerminationRegister(PureDataclassPytree):
    num_samples_used: IntArray
    evidence_calc: EvidenceCalculation
    dZ_shrinkage: LogSpace
    num_likelihood_evaluations: IntArray
    log_L_max: FloatArray
    log_L_contour_max: FloatArray
    efficiency_shrinkage: FloatArray
    plateau: BoolArray
    no_seed_points: BoolArray
    relative_spread: FloatArray  # 2*|log_L_max - log_L_min|/|log_L_max + log_L_min|
    absolute_spread: FloatArray  # |log_L_max - log_L_min|
    cummax_XL: LogSpace

    def is_done(self, depth_cond: DepthCondition) -> tuple[BoolArray, IntArray]:
        """
        Determine if termination should happen. Termination Flags are bits:
            0-bit -> 1: used maximum allowed number of samples
            1-bit -> 2: evidence uncert below threshold
            2-bit -> 4: evidence from last shrinkage cycle below threshold (dZ/Z < dlogZ)
            3-bit -> 8: effective sample size big enough
            4-bit -> 16: used maxmimum allowed number of likelihood evaluations
            5-bit -> 32: maximum log-likelihood reached
            6-bit -> 64: maximum log-likelihood contour reached
            7-bit -> 128: sampler efficiency too low
            8-bit -> 256: all lineages on a single plateau
            9-bit -> 512: relative spread of last shrinkage < rtol
            10-bit -> 1024: absolute spread of last shrinkage < atol
            11-bit -> 2048: no seed points left
            12-bit -> 4096: XL < cummax(XL) * peak_XL_frac

        Multiple flags are summed together

        Args:
            depth_cond: termination condition

        Returns:
            boolean done signal, and termination reason
        """
        return _is_done(self, depth_cond)


TerminationRegister.register_pytree()


@partial(jax.jit, inline=True)
def _is_done(self: TerminationRegister, depth_cond: DepthCondition) -> tuple[BoolArray, IntArray]:
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

    if depth_cond.max_samples is not None:
        # used all points
        reached_max_samples = self.num_samples_used >= depth_cond.max_samples
        done, termination_reason = _set_done_bit(reached_max_samples, 0,
                                                 done=done, termination_reason=termination_reason)
    log_Z_mu, log_Z_var = linear_to_log_stats(
        log_f_mean=self.evidence_calc.Z_mean.log_abs_val,
        log_f2_mean=self.evidence_calc.Z2_mean.log_abs_val
    )

    if depth_cond.evidence_uncert is not None:
        evidence_uncert_low_enough = log_Z_var <= jnp.square(depth_cond.evidence_uncert)
        done, termination_reason = _set_done_bit(evidence_uncert_low_enough, 1,
                                                 done=done, termination_reason=termination_reason)

    if depth_cond.dlogZ is not None:
        dZ_over_Z = (self.dZ_shrinkage / LogSpace(log_Z_mu)).value
        small_remaining_evidence = jnp.less(dZ_over_Z, depth_cond.dlogZ)
        done, termination_reason = _set_done_bit(small_remaining_evidence, 2,
                                                 done=done, termination_reason=termination_reason)

    if depth_cond.ess is not None:
        # Kish's ESS = [sum weights]^2 / [sum weights^2]
        ess = effective_sample_size_kish(
            self.evidence_calc.Z_mean.log_abs_val,
            self.evidence_calc.dZ2_mean.log_abs_val
        )
        ess_reached = ess >= depth_cond.ess
        done, termination_reason = _set_done_bit(ess_reached, 3,
                                                 done=done, termination_reason=termination_reason)

    if depth_cond.max_num_likelihood_evaluations is not None:
        num_likelihood_evaluations = self.num_likelihood_evaluations
        too_max_likelihood_evaluations = num_likelihood_evaluations >= depth_cond.max_num_likelihood_evaluations
        done, termination_reason = _set_done_bit(too_max_likelihood_evaluations, 4,
                                                 done=done, termination_reason=termination_reason)

    if depth_cond.log_L_target is not None:
        likelihood_reached = self.log_L_max >= depth_cond.log_L_target
        done, termination_reason = _set_done_bit(likelihood_reached, 5,
                                                 done=done, termination_reason=termination_reason)

    if depth_cond.log_L_contour_target is not None:
        likelihood_contour_reached = self.log_L_contour_max >= depth_cond.log_L_contour_target
        done, termination_reason = _set_done_bit(likelihood_contour_reached, 6,
                                                 done=done, termination_reason=termination_reason)

    if depth_cond.efficiency_threshold is not None:
        efficiency_too_low = self.efficiency_shrinkage < depth_cond.efficiency_threshold
        done, termination_reason = _set_done_bit(efficiency_too_low, 7,
                                                 done=done, termination_reason=termination_reason)

    done, termination_reason = _set_done_bit(self.plateau, 8,
                                             done=done, termination_reason=termination_reason)

    if depth_cond.rtol is not None:
        relative_spread_low = self.relative_spread < depth_cond.rtol
        done, termination_reason = _set_done_bit(relative_spread_low, 9,
                                                 done=done, termination_reason=termination_reason)

    if depth_cond.atol is not None:
        absolute_spread_low = self.absolute_spread < depth_cond.atol
        done, termination_reason = _set_done_bit(absolute_spread_low, 10,
                                                 done=done, termination_reason=termination_reason)

    # Sampler relies upon this termination condition to stop when no seed points are left.
    done, termination_reason = _set_done_bit(self.no_seed_points, 11,
                                             done=done, termination_reason=termination_reason)

    if depth_cond.cummax_XL_frac is not None:
        XL = self.evidence_calc.X_mean * self.evidence_calc.L
        XL_reduction_reached = XL.log_abs_val < self.cummax_XL.log_abs_val + jnp.log(depth_cond.cummax_XL_frac)
        done, termination_reason = _set_done_bit(XL_reduction_reached, 12,
                                                 done=done, termination_reason=termination_reason)

    return done, termination_reason
