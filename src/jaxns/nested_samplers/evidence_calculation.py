import dataclasses
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp

from jaxns.nested_samplers.log_semiring import LogSpace
from jaxns.nested_samplers.mixed_precision import mp_policy
from jaxns.nested_samplers.pytree import PureDataclassPytree
from jaxns.nested_samplers.types import IntArray, FloatArray


@dataclasses.dataclass(slots=True, frozen=True)
class EvidenceCalculation(PureDataclassPytree):
    """
    Contains a running estimate of evidence and related quantities.
    """
    L: LogSpace
    X_mean: LogSpace
    X2_mean: LogSpace
    Z_mean: LogSpace
    ZX_mean: LogSpace
    Z2_mean: LogSpace
    dZ_mean: LogSpace
    dZ2_mean: LogSpace

    @staticmethod
    def initialise() -> 'EvidenceCalculation':
        return EvidenceCalculation(
            L=LogSpace(jnp.array(-jnp.inf, dtype=mp_policy.measure_dtype)),
            X_mean=LogSpace(jnp.array(0., dtype=mp_policy.measure_dtype)),
            X2_mean=LogSpace(jnp.array(0., dtype=mp_policy.measure_dtype)),
            Z_mean=LogSpace(jnp.array(-jnp.inf, dtype=mp_policy.measure_dtype)),
            ZX_mean=LogSpace(jnp.array(-jnp.inf, dtype=mp_policy.measure_dtype)),
            Z2_mean=LogSpace(jnp.array(-jnp.inf, dtype=mp_policy.measure_dtype)),
            dZ_mean=LogSpace(jnp.array(-jnp.inf, dtype=mp_policy.measure_dtype)),
            dZ2_mean=LogSpace(jnp.array(-jnp.inf, dtype=mp_policy.measure_dtype))
        )

    def update_evidence(self, K_total: IntArray, log_L_next: FloatArray) -> 'EvidenceCalculation':
        return _update_evidence(self, K_total, log_L_next)
EvidenceCalculation.register_pytree()


@partial(jax.jit, inline=True)
def _update_evidence(self, K_total: IntArray, log_L_next: FloatArray) -> 'EvidenceCalculation':
    next_L = LogSpace(log_L_next)
    K_total = K_total.astype(mp_policy.measure_dtype)

    zero = jnp.asarray(0, mp_policy.measure_dtype)
    one = jnp.asarray(1, mp_policy.measure_dtype)
    two = jnp.asarray(2, mp_policy.measure_dtype)
    # num_live_points = jnp.maximum(y.num_live_points, jnp.zeros_like(y.num_live_points))
    log_num_live_points = jnp.log(K_total)
    log_num_live_points_p1 = jnp.log(K_total + one)
    log_num_live_points_p2 = jnp.log(K_total + two)

    # T_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.))
    # T_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points))
    T_mean = LogSpace(- jnp.logaddexp(zero, -log_num_live_points))
    # T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
    t_mean = LogSpace(- log_num_live_points_p1)
    # T2_mean = LogSpace(jnp.log(num_live_points) - jnp.log( num_live_points + 2.))
    # T2_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 2./num_live_points))
    T2_mean = LogSpace(- jnp.logaddexp(zero, np.log(two) - log_num_live_points))
    # T2_mean = LogSpace(- jnp.logaddexp(jnp.log(2.), -jnp.log(num_live_points)))
    t2_mean = LogSpace(jnp.log(two) - log_num_live_points_p1 - log_num_live_points_p2)
    # tT_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.) - jnp.log(num_live_points + 2.))
    # tT_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points) - jnp.log(num_live_points + 2.))
    tT_mean = LogSpace(- jnp.logaddexp(zero, -log_num_live_points) - log_num_live_points_p2)
    # tT_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)) - jnp.log(num_live_points + 2.))

    midL = LogSpace(-jnp.log(two)) * (next_L + self.L)
    dZ_mean = self.X_mean * t_mean * midL
    next_X_mean = self.X_mean * T_mean
    next_X2_mean = self.X2_mean * T2_mean
    next_Z_mean = self.Z_mean + dZ_mean
    next_ZX_mean = self.ZX_mean * T_mean + self.X2_mean * tT_mean * midL
    next_Z2_mean = self.Z2_mean + LogSpace(jnp.log(two)) * self.ZX_mean * t_mean * midL + (
            self.X2_mean * t2_mean * midL ** 2)
    next_dZ2_mean = self.dZ2_mean + (self.X2_mean * t2_mean * midL ** 2)

    return EvidenceCalculation(
        L=next_L,
        X_mean=next_X_mean,
        X2_mean=next_X2_mean,
        Z_mean=next_Z_mean,
        ZX_mean=next_ZX_mean,
        Z2_mean=next_Z2_mean,
        dZ_mean=dZ_mean,
        dZ2_mean=next_dZ2_mean
    )
