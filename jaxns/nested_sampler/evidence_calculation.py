from jax import numpy as jnp

from jaxns.internals.log_semiring import LogSpace
from jaxns.types import EvidenceCalculation


def _update_evidence_calculation(num_live_points: jnp.ndarray,
                                 log_L: jnp.ndarray,
                                 next_log_L_contour: jnp.ndarray,
                                 evidence_calculation: EvidenceCalculation):
    next_L = LogSpace(next_log_L_contour)
    L_contour = LogSpace(log_L)
    midL = LogSpace(jnp.log(0.5)) * (next_L + L_contour)
    X_mean = LogSpace(evidence_calculation.log_X_mean)
    X2_mean = LogSpace(evidence_calculation.log_X2_mean)
    Z_mean = LogSpace(evidence_calculation.log_Z_mean)
    ZX_mean = LogSpace(evidence_calculation.log_ZX_mean)
    Z2_mean = LogSpace(evidence_calculation.log_Z2_mean)
    dZ2_mean = LogSpace(evidence_calculation.log_dZ2_mean)

    # T_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.))
    # T_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points))
    T_mean = LogSpace(- jnp.logaddexp(0, -jnp.log(num_live_points)))
    # T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
    t_mean = LogSpace(- jnp.log(num_live_points + 1.))
    # T2_mean = LogSpace(jnp.log(num_live_points) - jnp.log( num_live_points + 2.))
    # T2_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 2./num_live_points))
    T2_mean = LogSpace(- jnp.logaddexp(0., jnp.log(2.) - jnp.log(num_live_points)))
    # T2_mean = LogSpace(- jnp.logaddexp(jnp.log(2.), -jnp.log(num_live_points)))
    t2_mean = LogSpace(jnp.log(2.) - jnp.log(num_live_points + 1.) - jnp.log(num_live_points + 2.))
    # tT_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.) - jnp.log(num_live_points + 2.))
    # tT_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points) - jnp.log(num_live_points + 2.))
    tT_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)) - jnp.log(num_live_points + 2.))
    # tT_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)) - jnp.log(num_live_points + 2.))

    next_X_mean = X_mean * T_mean
    next_X2_mean = X2_mean * T2_mean
    next_Z_mean = Z_mean + X_mean * t_mean * midL
    next_ZX_mean = ZX_mean * T_mean + X2_mean * tT_mean * midL
    next_Z2_mean = Z2_mean + LogSpace(jnp.log(2.)) * ZX_mean * t_mean * midL + (X2_mean * t2_mean * midL ** 2)
    next_dZ2_mean = dZ2_mean + (X2_mean * t2_mean * midL ** 2)

    next_evidence_calculation = evidence_calculation._replace(log_X_mean=next_X_mean.log_abs_val,
                                                              log_X2_mean=next_X2_mean.log_abs_val,
                                                              log_Z_mean=next_Z_mean.log_abs_val,
                                                              log_Z2_mean=next_Z2_mean.log_abs_val,
                                                              log_ZX_mean=next_ZX_mean.log_abs_val,
                                                              log_dZ2_mean=next_dZ2_mean.log_abs_val
                                                              )
    # next_evidence_calculation = tree_map(lambda old, new: jnp.where(jnp.isnan(new), old, new),
    #                                      evidence_calculation, next_evidence_calculation)

    return next_evidence_calculation