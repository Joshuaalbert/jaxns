from typing import Tuple, NamedTuple

from etils.array_types import PRNGKey, FloatArray, IntArray, BoolArray
from jax import core, numpy as jnp, tree_map, random, pmap
from jax._src.lax.control_flow import while_loop, scan
from jax._src.lax.parallel import all_gather

from jaxns.internals.stats import linear_to_log_stats
from jaxns.model import Model
from jaxns.static_nested_sampler import PreProcessType, add_chunk_dim, remove_chunk_dim, SeedPoint
from jaxns.static_slice import UniDimSliceSampler
from jaxns.statistics import analyse_sample_collection
from jaxns.types import NestedSamplerState, Reservoir, int_type, float_type
from jaxns.utils import sort_samples

__all__ = ['AdaptiveRefinement']


class AdaptiveRefinement:
    def __init__(self, model: Model, patience: int = 1, num_parallel_samplers: int = 1):
        if patience < 1:
            raise ValueError(f"patience should be >= 1, got {patience}.")
        self.patience = patience
        self.num_parallel_samplers = num_parallel_samplers
        self.sampler = UniDimSliceSampler(model=model, num_slices=model.U_ndims, midpoint_shrink=True, perfect=True)

    def _split_state(self, state: NestedSamplerState) -> Tuple[NestedSamplerState, Reservoir]:
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

    def _combine_state(self, state: NestedSamplerState, update_reservoir: Reservoir) -> NestedSamplerState:
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
                            preprocess_data: PreProcessType) -> Reservoir:
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
        sample = self.sampler.get_sample_from_seed(key=key,
                                                   seed_point=seed_point,
                                                   log_L_constraint=log_L_constraint,
                                                   preprocess_data=preprocess_data)
        return Reservoir(*sample)

    def _single_improvement_batch(self,
                                  key: PRNGKey,
                                  non_iid_reservoir: Reservoir,
                                  preprocess_data: PreProcessType) -> Reservoir:
        """
        Run nested sampling to replace an entire live point reservoir via shrinkage.

        Args:
            key: PRNGKey
            non_iid_reservoir: reservoir to improve
            preprocess_data: any data needed to shrink the replace the live point reservoir with i.i.d. samples.

        Returns:
            improved reservoir
        """

        class CarryType(NamedTuple):
            key: PRNGKey

        class XType(NamedTuple):
            reservoir_point: Reservoir

        ResultType = Reservoir

        def body(carry: CarryType, X: XType) -> Tuple[CarryType, ResultType]:
            key, improve_key = random.split(carry.key, 2)
            seed_point = SeedPoint(U0=X.reservoir_point.point_U, log_L0=X.reservoir_point.log_L)
            improved_point = self._single_improvement(
                key=improve_key,
                seed_point=seed_point,
                log_L_constraint=X.reservoir_point.log_L_constraint,
                preprocess_data=preprocess_data
            )
            # Update stats
            improved_point = improved_point._replace(
                num_slices=improved_point.num_slices + X.reservoir_point.num_slices,
                num_likelihood_evaluations=(improved_point.num_likelihood_evaluations
                                            + X.reservoir_point.num_likelihood_evaluations)
            )

            return CarryType(key=key), improved_point

        _, improved_reservoir = scan(body,
                                     CarryType(key),
                                     XType(reservoir_point=non_iid_reservoir))
        return improved_reservoir

    def _single_improve_thread(self, state: NestedSamplerState, iid_state: NestedSamplerState,
                               non_iid_reservoir: Reservoir) -> [NestedSamplerState, Reservoir]:
        """
        Performs perfect slice sampling on the set of samples that are not considered i.i.d. sampled within their
        respective likelihood constraint.

        Args:
            state: state to improve
            iid_state: state with only the i.i.d. samples
            non_iid_reservoir: samples that are not deemed i.i.d.

        Returns:
            reservoir where all samples are considered i.i.d. sampled within their respective likelihood constraints
        """

        # update each sample one slice at a time until the stopping condition

        class CarryType(NamedTuple):
            key: PRNGKey
            reservoir: Reservoir
            last_log_Z_mean: FloatArray
            last_diff_log_Z_mean: FloatArray
            converged_step_count: IntArray
            state: NestedSamplerState

        def cond(body_state: CarryType) -> BoolArray:
            done = jnp.greater_equal(body_state.converged_step_count, self.patience)
            return jnp.bitwise_not(done)

        def body(body_state: CarryType) -> CarryType:
            key, improve_key = random.split(body_state.state.key, 2)
            state = body_state.state._replace(key=key)

            preprocess_data = self.sampler.preprocess(state)

            update_reservoir = self._single_improvement_batch(
                key=improve_key,
                non_iid_reservoir=body_state.reservoir,
                preprocess_data=preprocess_data
            )

            all_update_reservoir: Reservoir = remove_chunk_dim(all_gather(update_reservoir, 'i'))

            all_state = self._combine_state(state=iid_state, update_reservoir=all_update_reservoir)

            evidence_calculation, sample_stats = analyse_sample_collection(
                sample_collection=all_state.sample_collection,
                sorted_collection=True
            )
            log_Z_mean, log_Z_var = linear_to_log_stats(
                log_f_mean=evidence_calculation.log_Z_mean,
                log_f2_mean=evidence_calculation.log_Z2_mean
            )
            nan_last_log_Z_mean = jnp.isnan(body_state.last_log_Z_mean)
            nan_log_Z_var = jnp.isnan(log_Z_var)
            nan_log_Z_mean = jnp.isnan(log_Z_mean)

            diff_log_Z_mean = jnp.abs(body_state.last_log_Z_mean - log_Z_mean)
            small_change = jnp.square(2. * diff_log_Z_mean) < log_Z_var
            decreasing_change = diff_log_Z_mean < body_state.last_diff_log_Z_mean
            stable = decreasing_change | small_change
            reset_patience = jnp.bitwise_not(stable) | (nan_log_Z_mean | nan_last_log_Z_mean | nan_log_Z_var)
            converged_step_count = jnp.where(reset_patience, jnp.asarray(0, int_type),
                                             body_state.converged_step_count + jnp.asarray(1, int_type))
            return CarryType(key, update_reservoir, log_Z_mean, diff_log_Z_mean, converged_step_count, all_state)

        key, improve_key = random.split(state.key, 2)
        init_evidence_calculation, _ = analyse_sample_collection(
            sample_collection=state.sample_collection,
            sorted_collection=True
        )
        init_log_Z_mean, _ = linear_to_log_stats(
            log_f_mean=init_evidence_calculation.log_Z_mean,
            log_f2_mean=init_evidence_calculation.log_Z2_mean
        )

        init_body_state: CarryType = CarryType(
            improve_key, non_iid_reservoir, init_log_Z_mean, jnp.asarray(0., float_type), jnp.asarray(0, int_type),
            state)
        output_state = while_loop(cond,
                                  body,
                                  init_body_state)

        return output_state.state

    def __call__(self, state: NestedSamplerState) -> NestedSamplerState:
        """
        Adaptively refines the state until all samples are i.i.d. (according to a given stopping condition).

        Args:
            state: nested sampler state

        Returns:
            state after refinement
        """
        # Split state will give an error if this is a tracer context so don't JIT compile.
        iid_state, non_iid_reservoir = self._split_state(state)

        chunked_non_iid_reservoir = add_chunk_dim(non_iid_reservoir, self.num_parallel_samplers)

        parallel_ar = pmap(lambda non_iid_reservoir: self._single_improve_thread(
            state=state,
            iid_state=iid_state,
            non_iid_reservoir=non_iid_reservoir
        ), axis_name='i')

        chunked_updated_state = parallel_ar(chunked_non_iid_reservoir)

        # Is there a better way to get only one out, since they are identical on each device?
        updated_state = tree_map(lambda x: x[0], chunked_updated_state)

        return updated_state
