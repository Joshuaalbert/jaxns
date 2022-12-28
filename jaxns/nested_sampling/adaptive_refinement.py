from functools import partial
from typing import Tuple

from etils.array_types import PRNGKey, FloatArray, IntArray, BoolArray
from jax import core, numpy as jnp, tree_map, random, jit, disable_jit
from jax._src.lax.control_flow import while_loop

from jaxns.internals.maps import chunked_pmap
from jaxns.internals.stats import linear_to_log_stats
from jaxns.internals.types import int_type
from jaxns.nested_sampling.model import Model
from jaxns.nested_sampling.slice_sampler import SliceSampler, SeedPoint, PreprocessType
from jaxns.nested_sampling.statistics import analyse_sample_collection
from jaxns.nested_sampling.types import NestedSamplerState, Reservoir
from jaxns.nested_sampling.utils import sort_samples

__all__ = ['AdaptiveRefinement']

class AdaptiveRefinement:
    def __init__(self, model: Model, uncert_improvement_patient: int, num_slices: int, num_parallel_samplers: int = 1):
        if uncert_improvement_patient <= 0:
            raise ValueError(f"uncert_improvement_patient should be > 0, got {uncert_improvement_patient}.")
        self.uncert_improvement_patient = uncert_improvement_patient
        self.num_parallel_samplers = num_parallel_samplers
        self.slice_sampler = SliceSampler(model=model,
                                          midpoint_shrink=True,
                                          destructive_shrink=False,
                                          gradient_boost=False,
                                          multi_ellipse_bound=False)
        self.num_slices = num_slices

    def split_state(self, state: NestedSamplerState) -> Tuple[NestedSamplerState, Reservoir]:
        """
        Splice a state into iid samples and non-iid samples.

        Note: must not be run in a JIT-context, as the nonzero operation non-static.

        Args:
            state: state to split

        Returns:
            a state of iid samples, and reservoir of non-iid samples
        """
        if isinstance(state.sample_collection.reservoir.iid, core.Tracer):
            raise RuntimeError("Tracer detected, but expected imperative context.")
        num_samples = state.sample_collection.reservoir.log_L.size
        (update_indicies,) = jnp.nonzero(
            jnp.bitwise_not(state.sample_collection.reservoir.iid) & (
                    jnp.arange(num_samples) < state.sample_collection.sample_idx
            )
        )

        (keep_indicies,) = jnp.nonzero(state.sample_collection.reservoir.iid)

        update_reservoir = tree_map(lambda x: x[update_indicies], state.sample_collection.reservoir)
        keep_reservoir = tree_map(lambda x: x[keep_indicies], state.sample_collection.reservoir)
        state = state._replace(
            sample_collection=state.sample_collection._replace(
                sample_idx=keep_indicies.size,
                reservoir=keep_reservoir
            )
        )
        return state, update_reservoir

    def combine_state(self, state: NestedSamplerState, update_reservoir: Reservoir) -> NestedSamplerState:
        """
        Concatenates a state and reservoir and sorts the state.

        Args:
            state: state
            update_reservoir: reservoir

        Returns:
            merged and sorted state
        """
        # concatenate reservoirs
        reservoir = tree_map(lambda x, y: jnp.concatenate([x, y], axis=0),
                             state.sample_collection.reservoir, update_reservoir)

        sample_collection = state.sample_collection._replace(
            sample_idx=state.sample_collection.sample_idx + update_reservoir.iid.size,
            reservoir=reservoir
        )
        # Sort
        sample_collection = sort_samples(sample_collection)
        state = state._replace(sample_collection=sample_collection)
        return state

    def _single_improvement(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray,
                            preprocess_data: PreprocessType) -> Reservoir:
        """
        Perform perfect slice sampling, thereby improving the sample.

        Args:
            key: PRNGKey
            seed_point: seed point to sample from
            log_L_constraint: the constraint to sample inside
            preprocess_data: preprocessing data

        Returns:
            a sample from the seed point sampled within the given constraint
        """
        sample = self.slice_sampler.get_sample(key=key,
                                               seed_point=seed_point,
                                               log_L_constraint=log_L_constraint,
                                               num_slices=self.num_slices,
                                               preprocess_data=preprocess_data)
        return Reservoir(*sample)

    @partial(jit, static_argnums=0)
    def improve(self, state: NestedSamplerState, iid_state: NestedSamplerState, non_iid_reservoir: Reservoir):
        """
        Performs perfect slice sampling on the set of samples that are not considered i.i.d. sampled within their
        respective likelihood constraint.

        Args:
            state: state to improve

        Returns:
            state where all samples are considered i.i.d. sampled within their respective likelihood constraints
        """
        # update each sample one slice at a time until the stopping condition

        CarryType = Tuple[PRNGKey, Reservoir, FloatArray, IntArray, PreprocessType]

        def cond(body_state: CarryType) -> BoolArray:
            (_, _, _, patience, _) = body_state
            done = jnp.greater_equal(patience, self.uncert_improvement_patient)
            return jnp.bitwise_not(done)

        def body(body_state: CarryType) -> CarryType:
            (key, old_reservoir, last_log_Z_mean, patience, preprocess_data) = body_state

            batch_size = old_reservoir.iid.size

            key, combine_key, improve_key = random.split(key, 3)

            parallel_improvement = chunked_pmap(
                lambda key, seed_point, log_L_constraint: self._single_improvement(key=key, seed_point=seed_point,
                                                                                   log_L_constraint=log_L_constraint,
                                                                                   preprocess_data=preprocess_data),
                chunksize=self.num_parallel_samplers,
                batch_size=batch_size)
            seed_points = SeedPoint(
                U0=old_reservoir.point_U,
                log_L0=old_reservoir.log_L
            )
            update_reservoir = parallel_improvement(random.split(improve_key, batch_size),
                                                    seed_points,
                                                    old_reservoir.log_L_constraint)

            update_reservoir = update_reservoir._replace(
                num_slices=update_reservoir.num_slices + old_reservoir.num_slices,
                num_likelihood_evaluations=update_reservoir.num_likelihood_evaluations + old_reservoir.num_likelihood_evaluations
            )

            updated_state = self.combine_state(state=iid_state, update_reservoir=update_reservoir)
            evidence_calculation, sample_stats = analyse_sample_collection(
                sample_collection=updated_state.sample_collection,
                sorted_collection=True
            )
            log_Z_mean, log_Z_var = linear_to_log_stats(
                log_f_mean=evidence_calculation.log_Z_mean,
                log_f2_mean=evidence_calculation.log_Z2_mean
            )
            nan_last_log_Z_mean = jnp.isnan(last_log_Z_mean)
            nan_log_Z_var = jnp.isnan(log_Z_var)
            nan_log_Z_mean = jnp.isnan(log_Z_mean)

            diff_log_Z_mean = jnp.abs(last_log_Z_mean - log_Z_mean)
            improvement = jnp.square(2. * diff_log_Z_mean) >= log_Z_var
            patience = jnp.where(improvement | (nan_log_Z_mean | nan_last_log_Z_mean | nan_log_Z_var),
                                 jnp.zeros_like(patience), patience + jnp.ones_like(patience))
            preprocess_data = self.slice_sampler.preprocess(updated_state)
            return (key, update_reservoir, log_Z_mean, patience, preprocess_data)

        preprocess_data = self.slice_sampler.preprocess(state)

        key, improve_key = random.split(state.key, 2)
        init_evidence_calculation, _ = analyse_sample_collection(
            sample_collection=state.sample_collection,
            sorted_collection=True
        )
        init_log_Z_mean, _ = linear_to_log_stats(
            log_f_mean=init_evidence_calculation.log_Z_mean,
            log_f2_mean=init_evidence_calculation.log_Z2_mean
        )

        init_body_state: CarryType = (
            improve_key, non_iid_reservoir, init_log_Z_mean, jnp.asarray(0, int_type), preprocess_data)
        (key, update_reservoir, _, _, _) = while_loop(cond,
                                                      body,
                                                      init_body_state)

        # mark as iid now <==> the evidence stopped changing
        update_reservoir = update_reservoir._replace(iid=jnp.ones_like(update_reservoir.iid))

        updated_state = self.combine_state(state=iid_state, update_reservoir=update_reservoir)

        updated_state = updated_state._replace(key=key)
        return updated_state

    def __call__(self, state: NestedSamplerState) -> NestedSamplerState:
        if isinstance(state.sample_collection.reservoir.iid, core.Tracer):
            raise RuntimeError("Tracer detected, but expected imperative context.")
        iid_state, non_iid_reservoir = self.split_state(state)
        return self.improve(state=state, iid_state=iid_state, non_iid_reservoir=non_iid_reservoir)
