import dataclasses
import warnings
from typing import NamedTuple, Tuple, Any

import jax
from jax import random, numpy as jnp, lax

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.mixed_precision import mp_policy
from jaxns.internals.types import IntArray, UType
from jaxns.internals.types import PRNGKey, FloatArray
from jaxns.nested_samplers.common.types import Sample
from jaxns.samplers.abc import EphemeralState
from jaxns.samplers.bases import BaseAbstractRejectionSampler
from jaxns.samplers.multi_ellipsoid.multi_ellipsoid_utils import ellipsoid_clustering, MultEllipsoidState
from jaxns.samplers.multi_ellipsoid.multi_ellipsoid_utils import sample_multi_ellipsoid

__all__ = [
    'MultiEllipsoidalSampler'
]


@dataclasses.dataclass(eq=False)
class MultiEllipsoidalSampler(BaseAbstractRejectionSampler[MultEllipsoidState]):
    """
    Uses a multi-ellipsoidal decomposition of the live points to create a bound around regions to sample from.

    Inefficient for high dimensional problems, but can be very efficient for low dimensional problems.
    """
    model: BaseAbstractModel
    depth: int
    expansion_factor: float

    def __post_init__(self):
        warnings.warn(
            "MultiEllipsoidalSampler does not give consistent results/performance. Consider `UniDimSliceSampler`.")
        if self.depth < 0:
            raise ValueError(f"depth {self.depth} must be >= 0")
        if self.expansion_factor <= 0.:
            raise ValueError(f"expansion_factor {self.expansion_factor} must be > 0")

    def num_phantom(self) -> int:
        return 0

    def _pre_process(self, ephemeral_state: EphemeralState) -> Any:
        return ellipsoid_clustering(
            key=ephemeral_state.key,
            points=ephemeral_state.live_points_collection.U_sample,
            log_VS=ephemeral_state.termination_register.evidence_calc_with_remaining.log_X_mean,
            max_num_ellipsoids=self.max_num_ellipsoids,
            method='em_gmm'
        )

    def _post_process(self, ephemeral_state: EphemeralState,
                      sampler_state: Any) -> Any:
        return sampler_state

    @property
    def max_num_ellipsoids(self):
        return 2 ** self.depth

    def _get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, sampler_state: MultEllipsoidState) -> Tuple[
        Sample, Sample]:
        def _sample_multi_ellipsoid(key: PRNGKey) -> UType:
            _, U = sample_multi_ellipsoid(
                key=key,
                mu=sampler_state.params.mu,
                radii=sampler_state.params.radii * jnp.asarray(self.expansion_factor, mp_policy.measure_dtype),
                rotation=sampler_state.params.rotation,
                unit_cube_constraint=True
            )
            return U

        class CarryState(NamedTuple):
            key: PRNGKey
            U: FloatArray
            log_L: FloatArray
            num_likelihood_evals: IntArray

        def cond(carry: CarryState):
            return carry.log_L <= log_L_constraint

        def body(carry: CarryState):
            key, sample_key = random.split(carry.key, 2)
            point_U = _sample_multi_ellipsoid(key=sample_key)
            log_L = self.model.forward(U=point_U)
            num_likelihood_evals = carry.num_likelihood_evals + jnp.ones_like(carry.num_likelihood_evals)
            # backoff by one e-fold per attempt after efficiency threshold reached
            return CarryState(
                key=key,
                U=point_U,
                log_L=log_L,
                num_likelihood_evals=num_likelihood_evals
            )

        key, sample_key = random.split(key, 2)
        point_U = _sample_multi_ellipsoid(key=sample_key)
        init_carry_state = CarryState(
            key=key,
            U=point_U,
            log_L=self.model.forward(point_U),
            num_likelihood_evals=jnp.asarray(1, mp_policy.count_dtype)
        )

        final_carry = lax.while_loop(
            cond_fun=cond,
            body_fun=body,
            init_val=init_carry_state
        )

        sample = Sample(
            U_sample=final_carry.U,
            log_L_constraint=log_L_constraint,
            log_L=final_carry.log_L,
            num_likelihood_evaluations=final_carry.num_likelihood_evals
        )
        # TODO: could use rejected samples, not as phantom because they don't satisfy constraint, but for ML apps
        phantom_samples = jax.tree.map(lambda x: jnp.zeros((0,) + x.shape, x.dtype), sample)  # [k, D]
        return sample, phantom_samples
