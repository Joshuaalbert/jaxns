from typing import Tuple, Optional, NamedTuple

import jax.numpy as jnp

from jaxns.internals.cumulative_ops import cumulative_op_dynamic, scan_associative_cumulative_op, cumulative_op_static
from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.mixed_precision import mp_policy
from jaxns.internals.tree_structure import SampleTreeGraph, count_crossed_edges
from jaxns.internals.types import MeasureType, IntArray, FloatArray
from jaxns.nested_samplers.common.types import EvidenceCalculation


def compute_enclosed_prior_volume(sample_tree: SampleTreeGraph) -> MeasureType:
    """
    Compute the enclosed prior volume of the likelihood constraint.

    Args:
        sample_tree: The sample tree graph.

    Returns:
        The log enclosed prior volume.
    """
    live_point_counts = count_crossed_edges(sample_tree=sample_tree)

    def op(log_X, num_live_points):
        X_mean = LogSpace(log_X)
        # T_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.))
        # T_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points))
        T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
        next_X_mean = X_mean * T_mean
        return next_X_mean.log_abs_val

    _, log_X = scan_associative_cumulative_op(op=op, init=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
                                              xs=live_point_counts.num_live_points)
    return log_X


class EvidenceUpdateVariables(NamedTuple):
    num_live_points: FloatArray
    log_L_next: FloatArray


def _update_evidence_calc_op(carry: EvidenceCalculation, y: EvidenceUpdateVariables) -> EvidenceCalculation:
    # num_live_points = num_live_points.astype(float_type)
    next_L = LogSpace(y.log_L_next)
    L_contour = LogSpace(carry.log_L)
    midL = LogSpace(jnp.log(0.5)) * (next_L + L_contour)
    X_mean = LogSpace(carry.log_X_mean)
    X2_mean = LogSpace(carry.log_X2_mean)
    Z_mean = LogSpace(carry.log_Z_mean)
    ZX_mean = LogSpace(carry.log_ZX_mean)
    Z2_mean = LogSpace(carry.log_Z2_mean)
    dZ2_mean = LogSpace(carry.log_dZ2_mean)
    # num_live_points = jnp.maximum(y.num_live_points, jnp.zeros_like(y.num_live_points))
    num_live_points = mp_policy.cast_to_measure(y.num_live_points, quiet=True)
    log_num_live_points = jnp.log(num_live_points)
    log_num_live_points_p1 = jnp.log(num_live_points + jnp.asarray(1., num_live_points.dtype))
    log_num_live_points_p2 = jnp.log(num_live_points + jnp.asarray(2., num_live_points.dtype))

    # T_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.))
    # T_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points))
    T_mean = LogSpace(- jnp.logaddexp(0., -log_num_live_points))
    # T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
    t_mean = LogSpace(- log_num_live_points_p1)
    # T2_mean = LogSpace(jnp.log(num_live_points) - jnp.log( num_live_points + 2.))
    # T2_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 2./num_live_points))
    T2_mean = LogSpace(- jnp.logaddexp((0.), jnp.log(2.) - log_num_live_points))
    # T2_mean = LogSpace(- jnp.logaddexp(jnp.log(2.), -jnp.log(num_live_points)))
    t2_mean = LogSpace(jnp.log(2.) - log_num_live_points_p1 - log_num_live_points_p2)
    # tT_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.) - jnp.log(num_live_points + 2.))
    # tT_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points) - jnp.log(num_live_points + 2.))
    tT_mean = LogSpace(- jnp.logaddexp(0., -log_num_live_points) - log_num_live_points_p2)
    # tT_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)) - jnp.log(num_live_points + 2.))

    dZ_mean = X_mean * t_mean * midL
    next_X_mean = X_mean * T_mean
    next_X2_mean = X2_mean * T2_mean
    next_Z_mean = Z_mean + dZ_mean
    next_ZX_mean = ZX_mean * T_mean + X2_mean * tT_mean * midL
    next_Z2_mean = Z2_mean + LogSpace(jnp.log(2.)) * ZX_mean * t_mean * midL + (X2_mean * t2_mean * midL ** 2)
    next_dZ2_mean = dZ2_mean + (X2_mean * t2_mean * midL ** 2)

    next_evidence_calculation = EvidenceCalculation(
        log_L=y.log_L_next,
        log_X_mean=next_X_mean.log_abs_val,
        log_X2_mean=next_X2_mean.log_abs_val,
        log_Z_mean=next_Z_mean.log_abs_val,
        log_Z2_mean=next_Z2_mean.log_abs_val,
        log_ZX_mean=next_ZX_mean.log_abs_val,
        log_dZ_mean=dZ_mean.log_abs_val,
        log_dZ2_mean=next_dZ2_mean.log_abs_val
    )

    return mp_policy.cast_to_measure(next_evidence_calculation)


def update_evicence_calculation(evidence_calculation: EvidenceCalculation,
                                update: EvidenceUpdateVariables) -> EvidenceCalculation:
    """
    Update the evidence statistics with a new sample.

    Args:
        evidence_calculation: The current evidence statistics.
        update: The update variables.

    Returns:
        The updated evidence statistics.
    """
    return _update_evidence_calc_op(evidence_calculation, update)


def create_init_evidence_calc() -> EvidenceCalculation:
    """
    Initialise the evidence statistics.

    Returns:
        The initial evidence statistics.
    """
    return EvidenceCalculation(
        log_L=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        log_X_mean=jnp.asarray(0., mp_policy.measure_dtype),
        log_X2_mean=jnp.asarray(0., mp_policy.measure_dtype),
        log_Z_mean=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        log_ZX_mean=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        log_Z2_mean=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        log_dZ_mean=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        log_dZ2_mean=jnp.asarray(-jnp.inf, mp_policy.measure_dtype)
    )


def compute_evidence_stats(log_L: MeasureType, num_live_points: FloatArray, num_samples: Optional[IntArray] = None) -> \
        Tuple[EvidenceCalculation, EvidenceCalculation]:
    """
    Compute the evidence statistics along the shrinkage process.

    Args:
        log_L: The log likelihoods of the samples.
        num_live_points: The number of live points at each sample.
        num_samples: The number of samples to use. If None, all samples are used.

    Returns:
        The final evidence statistics, and the evidence statistics for each sample.
    """
    init_evidence_calc = create_init_evidence_calc()
    xs = EvidenceUpdateVariables(
        num_live_points=num_live_points.astype(mp_policy.measure_dtype),
        log_L_next=log_L
    )
    if num_samples is not None:
        stop_idx = num_samples
        final_accumulate, result = cumulative_op_dynamic(op=_update_evidence_calc_op, init=init_evidence_calc, xs=xs,
                                                         stop_idx=stop_idx)
    else:
        final_accumulate, result = cumulative_op_static(op=_update_evidence_calc_op, init=init_evidence_calc, xs=xs)
    final_evidence_calculation = final_accumulate
    per_sample_evidence_calculation = result
    return final_evidence_calculation, per_sample_evidence_calculation
