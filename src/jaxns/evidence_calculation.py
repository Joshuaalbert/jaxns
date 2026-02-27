import dataclasses
from functools import partial

import jax
from jax import numpy as jnp

from jaxns.cumulative_ops import scan_or_while_loop
from jaxns.log_semiring import LogSpace
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.samples import Samples
from jaxns.types import IntArray, FloatArray


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
        """
        Update the evidence calculation with a new likelihood value and the total number of live points at this iteration.
        This is the core update step of the evidence calculation.

        Args:
            K_total: the number of live points before shrinkage.
            log_L_next: the next likelihood level.

        Returns:
            evidence calculation updated with the new likelihood and number of live points.
        """
        return _update_evidence(self, K_total, log_L_next)

    def update_from_samples(self, samples: Samples, root_out_degree: IntArray, num_samples: IntArray | None = None) -> tuple[
        'EvidenceCalculation', 'EvidenceCalculation']:
        """
        Update the evidence calculation from a set of samples. The samples should be ordered in the order they were generated, and the root_out_degree should
        be the out degree of the root node (i.e. the number of live points at the start of the nested sampling run).

        Args:
            samples: sorted samples.
            root_out_degree: the out degree of root node
            num_samples: how many samples to accumulate.

        Returns:
            evidence calcuation updated with the samples, and the sumulative calculation per sample
        """
        return _update_from_samples(self, samples, root_out_degree, num_samples)


EvidenceCalculation.register_pytree()


@partial(jax.jit, inline=True)
def _update_from_samples(self: EvidenceCalculation, samples: Samples, root_out_degree: IntArray,
                         num_samples: IntArray | None = None) -> tuple[EvidenceCalculation, EvidenceCalculation]:
    # Cumulatively apply K[i+1] = K[i] - 1 + d(i)
    def scan_fn(carry, x):
        K_i, self = carry
        out_degree_i, log_L_i = x
        K_ip1 = K_i - jnp.ones((), K_i.dtype) + out_degree_i
        self = self.update_evidence(K_i, log_L_i)
        return (K_ip1, self), self

    (_, self), per_sample_state = scan_or_while_loop(scan_fn,
                                                     (root_out_degree, self),
                                                     (samples.out_degree.astype(root_out_degree.dtype), samples.log_likelihoods),
                                                     length=num_samples,
                                                     unroll=1)
    return self, per_sample_state


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
    T2_mean = LogSpace(- jnp.logaddexp(zero, jnp.log(two) - log_num_live_points))
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
