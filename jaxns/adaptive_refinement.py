import logging
from typing import Tuple, NamedTuple

from jaxns.types import PRNGKey, FloatArray, IntArray, BoolArray
from jax import core, numpy as jnp, tree_map, random, pmap
from jax._src.lax.control_flow import while_loop, scan
from jax._src.lax.parallel import all_gather

from jaxns.initial_state import sort_sample_collection, find_first_true_indices
from jaxns.internals.stats import linear_to_log_stats
from jaxns.model import Model
from jaxns.likelihood_samplers.slice_samplers import UniDimSliceSampler
from jaxns.common import remove_chunk_dim, add_chunk_dim
from jaxns.abc import PreProcessType, SeedPoint
from jaxns.statistics import analyse_sample_collection
from jaxns.types import NestedSamplerState, Reservoir, int_type, float_type, SampleCollection
from jaxns.utils import sort_samples

__all__ = ['AdaptiveRefinement']

logger = logging.getLogger('jaxns')


def revise_state(state: NestedSamplerState, sorted_collection: bool = True) -> NestedSamplerState:
    sample_collection = state.sample_collection
    if not sorted_collection:
        sample_collection = sort_sample_collection(sample_collection)

    # Pick the first num_live_points samples from the mask. The order of mask, determines which ones
    # are selected
    class CarryType(NamedTuple):
        available: jnp.ndarray
        sample_collection: SampleCollection
        log_L_constraint: jnp.ndarray

    def outer_body(outer_carry: CarryType) -> CarryType:
        def body(carry: CarryType) -> CarryType:
            sample_mask = jnp.bitwise_and(carry.available,
                                          jnp.bitwise_and(
                                              carry.sample_collection.reservoir.log_L_constraint <= carry.log_L_constraint,
                                              carry.sample_collection.reservoir.log_L > carry.log_L_constraint)
                                          )
            sample_index = find_first_true_indices(mask=sample_mask, N=1)[0]
            reservoir = carry.sample_collection.reservoir._replace(
                log_L_constraint=carry.sample_collection.reservoir.log_L_constraint.at[sample_index].set(
                    carry.log_L_constraint)
            )
            next_log_L_constraint = carry.sample_collection.reservoir.log_L[sample_index]
            sample_collection = carry.sample_collection._replace(reservoir=reservoir)
            available = carry.available.at[sample_index].set(jnp.asarray(False))
            return CarryType(available=available,
                             sample_collection=sample_collection,
                             log_L_constraint=next_log_L_constraint)

        def cond(carry: CarryType) -> jnp.ndarray:
            sample_mask = jnp.bitwise_and(carry.available,
                                          jnp.bitwise_and(
                                              carry.sample_collection.reservoir.log_L_constraint <= carry.log_L_constraint,
                                              carry.sample_collection.reservoir.log_L > carry.log_L_constraint)
                                          )
            return jnp.any(sample_mask)

        init_carry = outer_carry._replace(
            log_L_constraint=-jnp.inf
        )
        output_carry = while_loop(cond,
                                  body,
                                  init_carry)
        return output_carry

    def outer_cond(outer_carry: CarryType) -> jnp.ndarray:
        done = jnp.any(outer_carry.available)
        return jnp.bitwise_not(done)

    init_carry = CarryType(
        available=jnp.ones(sample_collection.reservoir.log_L.size, jnp.bool_),
        sample_collection=sample_collection,
        log_L_constraint=-jnp.inf
    )
    output_carry = while_loop(outer_cond,
                              outer_body,
                              init_carry)
    state = state._replace(sample_collection=output_carry.sample_collection)
    return state


class AdaptiveRefinement:
    """
    Class for adaptive refinement.
    """

    def __init__(self, model: Model, patience: int = 1, num_parallel_samplers: int = 1):
        """
        Initialised adaptive refinement.

        Args:
            model: Model
            patience: how many steps with stopping condition met before stopping
            num_parallel_samplers: how many parallel samplers to use.
        """
        if patience < 1:
            raise ValueError(f"patience should be >= 1, got {patience}.")
        self.patience = patience
        self.num_parallel_samplers = num_parallel_samplers
        self.sampler = UniDimSliceSampler(
            model=model,
            num_slices=model.U_ndims,
            midpoint_shrink=True,
            perfect=True
        )

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
            (jnp.bitwise_not(state.sample_collection.reservoir.iid) &
             (jnp.arange(num_samples) < state.sample_collection.sample_idx)
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

        class XsType(NamedTuple):
            reservoir_point: Reservoir

        ResultType = Reservoir

        def body(carry: CarryType, X: XsType) -> Tuple[CarryType, ResultType]:
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
                                     XsType(reservoir_point=non_iid_reservoir))
        return improved_reservoir

    def _single_improve_thread(self, key: PRNGKey, iid_state: NestedSamplerState,
                               non_iid_reservoir: Reservoir) -> [NestedSamplerState, Reservoir]:
        """
        Performs perfect slice sampling on the set of samples that are not considered i.i.d. sampled within their
        respective likelihood constraint.

        Args:
            key: PRNGKey
            iid_state: state with only the i.i.d. samples
            non_iid_reservoir: samples that are not deemed i.i.d.

        Returns:
            reservoir where all samples are considered i.i.d. sampled within their respective likelihood constraints
        """

        iid_state = iid_state._replace(key=key)

        # update each sample one slice at a time until the stopping condition

        class CarryType(NamedTuple):
            key: PRNGKey
            non_iid_reservoir: Reservoir
            last_log_Z_mean: FloatArray
            last_diff_log_Z_mean: FloatArray
            converged_step_count: IntArray
            state: NestedSamplerState
            output_log_Z: FloatArray
            iter_j: IntArray

        def cond(body_state: CarryType) -> BoolArray:
            done = jnp.greater_equal(body_state.converged_step_count, self.patience)
            done = body_state.iter_j >= 20  # TODO: remove
            return jnp.bitwise_not(done)

        def body(body_state: CarryType) -> CarryType:
            key, improve_key = random.split(body_state.state.key, 2)
            state = body_state.state._replace(key=key)

            preprocess_data = self.sampler.preprocess(
                state=state,
                live_points=None
            )

            update_reservoir = self._single_improvement_batch(
                key=improve_key,
                non_iid_reservoir=body_state.non_iid_reservoir,
                preprocess_data=preprocess_data
            )

            all_update_reservoir: Reservoir = remove_chunk_dim(all_gather(update_reservoir, 'i'))

            all_state = self._combine_state(state=iid_state, update_reservoir=all_update_reservoir)

            evidence_calculation, sample_stats = analyse_sample_collection(
                sample_collection=all_state.sample_collection,
                sorted_collection=True,
                dual=False
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

            # output_log_Z = body_state.output_log_Z.at[body_state.iter_j].set(jnp.sum(sample_stats.num_live_points))
            output_log_Z = body_state.output_log_Z.at[body_state.iter_j].set(log_Z_mean)  # TODO: remove
            return CarryType(key=key, non_iid_reservoir=update_reservoir, last_log_Z_mean=log_Z_mean,
                             last_diff_log_Z_mean=diff_log_Z_mean, converged_step_count=converged_step_count,
                             state=all_state,
                             output_log_Z=output_log_Z, iter_j=body_state.iter_j + 1)

        key, improve_key = random.split(iid_state.key, 2)
        init_state = self._combine_state(state=iid_state, update_reservoir=non_iid_reservoir)
        init_evidence_calculation, _ = analyse_sample_collection(
            sample_collection=init_state.sample_collection,
            sorted_collection=True
        )
        init_log_Z_mean, _ = linear_to_log_stats(
            log_f_mean=init_evidence_calculation.log_Z_mean,
            log_f2_mean=init_evidence_calculation.log_Z2_mean
        )
        output_log_Z = jnp.full((20,), jnp.nan, float_type)
        output_log_Z = output_log_Z.at[0].set(init_log_Z_mean)
        init_body_state: CarryType = CarryType(
            key=improve_key, non_iid_reservoir=non_iid_reservoir, last_log_Z_mean=init_log_Z_mean,
            last_diff_log_Z_mean=jnp.asarray(0., float_type), converged_step_count=jnp.asarray(0, int_type),
            state=init_state, output_log_Z=output_log_Z, iter_j=jnp.asarray(1, int_type))
        output_state = while_loop(cond,
                                  body,
                                  init_body_state)

        return output_state.state, output_state.output_log_Z

    def __call__(self, state: NestedSamplerState) -> NestedSamplerState:
        """
        Adaptively refines the state until all samples are i.i.d. (according to a given stopping condition).

        Args:
            state: nested sampler state

        Returns:
            state after refinement
        """
        # Current = (l*1, l1)
        # Next = (l*2, l2)
        # Case: l*2 <= l1
        # If i.i.d. accept, else new sample
        # Case: l*2 > l1 => l2 > l1 and new sample
        # Split state will give an error if this is a tracer context so don't JIT compile.
        iid_state, non_iid_reservoir = self._split_state(state)

        keys = random.split(state.key, self.num_parallel_samplers)

        non_iid_count = non_iid_reservoir.log_L.size
        remainder = int(non_iid_count) % self.num_parallel_samplers
        extra = (self.num_parallel_samplers - remainder) % self.num_parallel_samplers
        if extra > 0:
            logger.warning(
                f"Increasing non_iid_reservoir ({non_iid_count}) by {extra} to closest multiple of num_parallel_samplers.")
            non_iid_reservoir = tree_map(lambda x: jnp.concatenate([x] + [x[0:1]] * extra, axis=0), non_iid_reservoir)

        chunked_non_iid_reservoir = add_chunk_dim(non_iid_reservoir, self.num_parallel_samplers)

        parallel_ar = pmap(
            lambda key, non_iid_reservoir: self._single_improve_thread(
                key=key,
                iid_state=iid_state,
                non_iid_reservoir=non_iid_reservoir
            ),
            axis_name='i')

        chunked_updated_state, chunked_output_log_Z = parallel_ar(keys, chunked_non_iid_reservoir)

        # Is there a better way to get only one out, since they are identical on each device?
        (updated_state, output_log_Z) = tree_map(lambda x: x[0], (chunked_updated_state, chunked_output_log_Z))

        import pylab as plt
        plt.plot(output_log_Z)
        plt.show()

        updated_state = revise_state(updated_state)
        return updated_state
