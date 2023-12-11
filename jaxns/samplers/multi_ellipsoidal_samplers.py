from typing import NamedTuple, Tuple

from jax import random, numpy as jnp, lax, tree_map

from jaxns.samplers.abc import SamplerState
from jaxns.samplers.bases import BaseAbstractRejectionSampler
from jaxns.samplers.multi_ellipsoid.multi_ellipsoid_utils import ellipsoid_clustering, MultEllipsoidState
from jaxns.samplers.multi_ellipsoid.multi_ellipsoid_utils import sample_multi_ellipsoid
from jaxns.internals.shrinkage_statistics import compute_evidence_stats
from jaxns.internals.tree_structure import SampleTreeGraph, count_crossed_edges
from jaxns.internals.types import IntArray, StaticStandardNestedSamplerState, UType
from jaxns.internals.types import PRNGKey, FloatArray
from jaxns.internals.types import Sample, int_type

__all__ = [
    'MultiEllipsoidalSampler'
]


class MultiEllipsoidalSampler(BaseAbstractRejectionSampler):
    """
    Uses a multi-ellipsoidal decomposition of the live points to create a bound around regions to sample from.

    Inefficient for high dimensional problems, but can be very efficient for low dimensional problems.
    """

    def __init__(self, depth: int, *args, **kwargs):
        self._depth = depth
        super().__init__(*args, **kwargs)

    def num_phantom(self) -> int:
        return 0

    def pre_process(self, state: StaticStandardNestedSamplerState) -> SamplerState:
        key, sampler_key = random.split(state.key)

        sample_tree = SampleTreeGraph(
            sender_node_idx=state.sample_collection.sender_node_idx,
            log_L=state.sample_collection.log_L
        )

        live_point_counts = count_crossed_edges(sample_tree=sample_tree, num_samples=state.next_sample_idx)
        log_L = sample_tree.log_L[live_point_counts.samples_indices]
        num_live_points = live_point_counts.num_live_points

        final_evidence_stats, _ = compute_evidence_stats(
            log_L=log_L,
            num_live_points=num_live_points,
            num_samples=state.next_sample_idx
        )

        points = state.sample_collection.U_samples[state.front_idx]
        return ellipsoid_clustering(
            key=sampler_key,
            points=points,
            log_VS=final_evidence_stats.log_X_mean,
            max_num_ellipsoids=self.max_num_ellipsoids,
            method='em_gmm'
        )

    def post_process(self, state: StaticStandardNestedSamplerState, sampler_state: SamplerState) -> SamplerState:
        return sampler_state

    @property
    def max_num_ellipsoids(self):
        return 2 ** self._depth

    def get_sample(self, key: PRNGKey, log_L_constraint: FloatArray,
                   sampler_state: MultEllipsoidState) -> Tuple[Sample, Sample]:
        def _sample_multi_ellipsoid(key: PRNGKey) -> UType:
            _, U = sample_multi_ellipsoid(
                key=key,
                mu=sampler_state.params.mu,
                radii=sampler_state.params.radii,
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
            num_likelihood_evals=jnp.asarray(1, int_type)
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

        phantom_samples = tree_map(lambda x: jnp.zeros((0,) + x.shape, x.dtype), sample)
        return sample, phantom_samples
