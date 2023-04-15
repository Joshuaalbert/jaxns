import logging
from typing import Tuple, NamedTuple, TypeVar

from etils.array_types import PRNGKey, IntArray, BoolArray
from jax import tree_map, numpy as jnp, random, pmap
from jax._src.lax.control_flow import scan, while_loop
from jax._src.lax.parallel import all_gather

from jaxns.model import Model
from jaxns.slice_sampler import PreprocessType, AbstractSliceSampler
from jaxns.statistics import analyse_sample_collection
from jaxns.termination import determine_termination
from jaxns.types import NestedSamplerState, Reservoir, LivePoints, TerminationCondition, int_type
from jaxns.utils import collect_samples

logger = logging.getLogger('jaxns')

__all__ = ['StaticSlice']

T = TypeVar('T')


def remove_chunk_dim(py_tree: T) -> T:
    def _remove_chunk_dim(a):
        shape = list(a.shape)
        shape = [shape[0] * shape[1]] + shape[2:]
        return jnp.reshape(a, shape)

    return tree_map(_remove_chunk_dim, py_tree)


def add_chunk_dim(py_tree: T, chunk_size: int) -> T:
    def _add_chunk_dim(a):
        shape = list(a.shape)
        shape = [chunk_size, shape[0] // chunk_size] + shape[1:]
        return jnp.reshape(a, shape)

    return tree_map(_add_chunk_dim, py_tree)


class StaticSlice:
    """
    Performs parallel nested sampling with a static number of samples.
    This uses the fact that the union of samples from N independent perfect static nested samplers with M live points is
    equivalent to a single nested sampler with N*M live points.
    """

    def __init__(self, model: Model, slice_sampler: AbstractSliceSampler, num_live_points: int, num_slices: int,
                 num_parallel_samplers: int = 1):
        if num_live_points % num_parallel_samplers != 0:
            raise ValueError(
                f"num_live_points {num_live_points} must divide num_parallel_samplers {num_parallel_samplers}"
            )
        self.num_live_points = num_live_points
        self.num_parallel_samplers = num_parallel_samplers
        self.model = model
        self.slice_sampler = slice_sampler
        if num_slices < 1:
            raise ValueError(f"num_slices should be > 0, got {num_slices}.")
        self.num_slices = num_slices

    def _single_live_point_shrink(self,
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

        class CarryType(NamedTuple):
            key: PRNGKey
            live_points: LivePoints

        ResultType = Reservoir

        def body(carry: CarryType, unused_X: IntArray) -> Tuple[CarryType, ResultType]:
            del unused_X
            idx_min = jnp.argmin(carry.live_points.reservoir.log_L)
            dead_point: Reservoir = tree_map(lambda x: x[idx_min], carry.live_points.reservoir)
            log_L_dead = dead_point.log_L

            # contour becomes log_L_dead if log_L_dead is not supremum of live-points, else we choose the original
            # constraint of dead point. Note: we are at liberty to choose any log_L level as a contour so long as we
            # can sample within it uniformly.
            on_supremum = jnp.equal(log_L_dead, jnp.max(carry.live_points.reservoir.log_L))
            log_L_contour = jnp.where(on_supremum, dead_point.log_L_constraint, log_L_dead)

            # replace dead point with a new sample about contour
            key, seed_key, sample_key = random.split(carry.key, 3)

            seed_point = self.slice_sampler.get_seed_point(key=seed_key,
                                                           live_points=carry.live_points,
                                                           log_L_constraint=log_L_contour)
            sample = self.slice_sampler.get_sample(key=sample_key,
                                                   seed_point=seed_point,
                                                   log_L_constraint=log_L_contour,
                                                   num_slices=num_slices,
                                                   preprocess_data=preprocess_data)

            live_points_reservoir = tree_map(lambda old, update: old.at[idx_min].set(update),
                                             carry.live_points.reservoir, Reservoir(*sample))
            live_points = carry.live_points._replace(reservoir=live_points_reservoir)

            return CarryType(key, live_points), dead_point

        (_, live_points), dead_reservoir = scan(body,
                                                CarryType(key, live_points),
                                                live_points.reservoir.log_L)
        return (dead_reservoir, live_points)

    def _single_thread_ns(self, state: NestedSamplerState, live_points: LivePoints,
                          termination_cond: TerminationCondition):

        class CarryType(NamedTuple):
            done: BoolArray
            termination_reason: IntArray
            state: NestedSamplerState
            live_points: LivePoints

        def body(body_state: CarryType) -> CarryType:
            key, sample_key = random.split(body_state.state.key, 2)
            state = body_state.state._replace(key=key)

            preprocess_data = self.slice_sampler.preprocess(state)

            (dead_reservoir, live_points) = self._single_live_point_shrink(
                key=sample_key,
                live_points=body_state.live_points,
                num_slices=self.num_slices,
                preprocess_data=preprocess_data
            )

            all_dead_reservoir: Reservoir = remove_chunk_dim(all_gather(dead_reservoir, 'i'))

            # update the state with sampled points from last round
            new_state = collect_samples(state, all_dead_reservoir)

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

            return CarryType(done=done, termination_reason=termination_reason, state=new_state,
                             live_points=live_points)

        output_carry = while_loop(
            lambda body_state: jnp.bitwise_not(body_state[0]),
            body,
            CarryType(done=jnp.asarray(False, jnp.bool_),
                      termination_reason=jnp.asarray(0, int_type), state=state, live_points=live_points)
        )

        return output_carry.termination_reason, output_carry.state, output_carry.live_points

    def __call__(self, state: NestedSamplerState,
                 live_points: LivePoints,
                 termination_cond: TerminationCondition
                 ) -> Tuple[IntArray, NestedSamplerState, LivePoints]:
        """
        Performs nested sampling from an initial state with a static number of live points at each shrinkage step, and
        terminating upon ANY of the terminiation criteria being met.

        Args:
            state: the current state of sampler
            live_points: the live points to start sampling from
            termination_cond: the conditions for termination

        Returns:
            termination_reason, final state, and live points
        """
        if live_points.reservoir.log_L.size != self.num_live_points:
            raise ValueError(
                f"live points reservoir is the wrong size. "
                f"Got {live_points.reservoir.log_L.size} but expected {self.num_live_points}.")

        chunked_live_points = add_chunk_dim(live_points, self.num_parallel_samplers)

        parallel_ns = pmap(lambda live_points: self._single_thread_ns(
            state=state,
            live_points=live_points,
            termination_cond=termination_cond
        ), axis_name='i')

        chunked_termination_reason, chunked_state, chunked_live_points = parallel_ns(chunked_live_points)

        termination_reason, state = tree_map(lambda x: x[0], (chunked_termination_reason, chunked_state))
        live_points = remove_chunk_dim(chunked_live_points)

        return termination_reason, state, live_points
