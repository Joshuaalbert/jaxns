import logging
from typing import Tuple

from etils.array_types import PRNGKey, IntArray, BoolArray
from jax import tree_map, numpy as jnp, random, pmap
from jax._src.lax.control_flow import scan, while_loop

from jaxns.model import Model
from jaxns.slice_sampler import PreprocessType, AbstractSliceSampler
from jaxns.statistics import analyse_sample_collection
from jaxns.termination import determine_termination
from jaxns.types import NestedSamplerState, Reservoir, LivePoints, TerminationCondition, int_type
from jaxns.utils import collect_samples

logger = logging.getLogger('jaxns')

__all__ = ['StaticSlice']


class StaticSlice:
    """
    Performs parallel nested sampling with a static number of samples.
    This uses the fact that the union of samples from N independent static nested samplers with M live points is
    equivalent to a single nested sampler with N*M live points.
    """

    def __init__(self, model: Model, slice_sampler: AbstractSliceSampler, num_live_points: int,
                 num_parallel_samplers: int = 1):
        # ensure we can split up request into equal parallel batches of work.
        remainder = num_live_points % num_parallel_samplers
        extra = (num_parallel_samplers - remainder) % num_parallel_samplers
        if extra > 0:
            logger.warning(
                f"Increasing max_samples ({num_live_points}) by {extra} to closest multiple of num_parallel_samplers.")
        self.num_live_points = num_live_points + extra
        self.num_parallel_samplers = num_parallel_samplers
        self.model = model
        self.slice_sampler = slice_sampler

    def _single_thread_ns(self,
                          key: PRNGKey,
                          live_points: LivePoints,
                          num_slices: IntArray,
                          preprocess_data: PreprocessType) -> Tuple[Reservoir, LivePoints]:
        """
        Run nested sampling to replace an entire live point reservoir via shrinkage.

        Args:
            key: PRNGKey
            live_points: live points
            num_slices: slice sampling `num_slices` parameter

        Returns:
            dead point reservoir, live points, slice stats
        """
        CarryType = Tuple[PRNGKey, LivePoints]
        ResultType = Reservoir

        def body(carry: CarryType, unused_X: IntArray) -> Tuple[CarryType, ResultType]:
            del unused_X
            (key, live_points) = carry
            idx_min = jnp.argmin(live_points.reservoir.log_L)
            dead_point: Reservoir = tree_map(lambda x: x[idx_min], live_points.reservoir)
            log_L_dead = dead_point.log_L

            # contour becomes log_L_dead if log_L_dead is not supremum of live-points, else we choose the original
            # constraint of dead point. Note: we are at liberty to choose any log_L level as a contour so long as we
            # can sample within it uniformly.
            on_supremum = jnp.equal(log_L_dead, jnp.max(live_points.reservoir.log_L))
            log_L_contour = jnp.where(on_supremum, dead_point.log_L_constraint, log_L_dead)

            # replace dead point with a new sample about contour
            key, seed_key, sample_key = random.split(key, 3)

            seed_point = self.slice_sampler.get_seed_point(key=seed_key,
                                                           live_points=live_points,
                                                           log_L_constraint=log_L_contour)
            sample = self.slice_sampler.get_sample(key=sample_key,
                                                   seed_point=seed_point,
                                                   log_L_constraint=log_L_contour,
                                                   num_slices=num_slices,
                                                   preprocess_data=preprocess_data)

            live_points_reservoir = tree_map(lambda old, update: old.at[idx_min].set(update),
                                             live_points.reservoir, Reservoir(*sample))
            live_points = live_points._replace(reservoir=live_points_reservoir)

            return (key, live_points), dead_point

        (_, live_points), dead_reservoir = scan(body,
                                                (key, live_points),
                                                live_points.reservoir.log_L)
        return (dead_reservoir, live_points)

    def __call__(self, state: NestedSamplerState,
                 live_points: LivePoints,
                 num_slices: IntArray,
                 termination_cond: TerminationCondition
                 ) -> Tuple[IntArray, NestedSamplerState, LivePoints]:
        """
        Performs nested sampling from an initial state with a static number of live points at each shrinkage step, and
        terminating upon ANY of the terminiation criteria being met.

        Args:
            state: the current state of sampler
            live_points: the live points to start sampling from
            num_slices: the number of slices to take per prior dimension
            termination_cond: the conditions for termination

        Returns:
            termination_reason, final state, and live points
        """
        if live_points.reservoir.log_L.size != self.num_live_points:
            raise ValueError(
                f"live points reservoir is the wrong size. "
                f"Got {live_points.reservoir.log_L.size} by expected {self.num_live_points}.")

        single_thread_sampler = lambda key, live_points, preprocess_data: self._single_thread_ns(key=key,
                                                                                                 live_points=live_points,
                                                                                                 num_slices=num_slices,
                                                                                                 preprocess_data=preprocess_data)

        CarryType = Tuple[BoolArray, IntArray, NestedSamplerState, LivePoints]

        def body(body_state: CarryType) -> CarryType:
            (_, _, state, live_points) = body_state

            key, sample_key = random.split(state.key, 2)
            state = state._replace(key=key)

            preprocess_data = self.slice_sampler.preprocess(state)

            if self.num_parallel_samplers > 1:  # prepare live points to split up

                # TODO(Joshuaalbert): rewrite using pgather to compute preprocess data locally on each device,
                #  getting rid of outer loop.

                parallel_single_thread_sampler = pmap(lambda key, live_points: single_thread_sampler(
                    key=key, live_points=live_points, preprocess_data=preprocess_data))

                def _add_chunk_dim(a):
                    shape = list(a.shape)
                    shape = [self.num_parallel_samplers, shape[0] // self.num_parallel_samplers] + shape[1:]
                    return jnp.reshape(a, shape)

                live_points = tree_map(_add_chunk_dim, live_points)

                keys = random.split(sample_key, self.num_parallel_samplers)
                # parallel sampling
                dead_reservoir, live_points = parallel_single_thread_sampler(keys, live_points)

                # concatenate reservoir samples
                def _remove_chunk_dim(a):
                    shape = list(a.shape)
                    shape = [shape[0] * shape[1]] + shape[2:]
                    return jnp.reshape(a, shape)

                (dead_reservoir, live_points) = tree_map(_remove_chunk_dim, (dead_reservoir, live_points))
            else:
                (dead_reservoir, live_points) = single_thread_sampler(key=sample_key, live_points=live_points,
                                                                      preprocess_data=preprocess_data)

            # update the state with sampled points from last round
            new_state = collect_samples(state, dead_reservoir)
            evidence_calculation, sample_stats = analyse_sample_collection(
                sample_collection=new_state.sample_collection,
                sorted_collection=True
            )
            done, termination_reason = determine_termination(
                term_cond=termination_cond,
                sample_collection=new_state.sample_collection,
                evidence_calculation=evidence_calculation,
                live_points=live_points
            )

            return (done, termination_reason, new_state, live_points)

        (_, termination_reason, state, live_points) = while_loop(
            lambda body_state: jnp.bitwise_not(body_state[0]),
            body,
            (jnp.asarray(False, jnp.bool_),
             jnp.asarray(0, int_type), state, live_points)
        )

        return termination_reason, state, live_points
