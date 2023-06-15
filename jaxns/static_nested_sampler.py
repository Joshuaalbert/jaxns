import logging
from typing import Tuple, NamedTuple, TypeVar, Optional, List

from etils.array_types import PRNGKey, IntArray, BoolArray, FloatArray
from jax import tree_map, numpy as jnp, random, pmap
from jax._src.lax.control_flow import while_loop, scan
from jax._src.lax.parallel import all_gather

from jaxns.model import Model
from jaxns.statistics import analyse_sample_collection
from jaxns.termination import determine_termination
from jaxns.types import NestedSamplerState, Reservoir, LivePoints, TerminationCondition, int_type, Sample, float_type
from jaxns.utils import collect_samples

logger = logging.getLogger('jaxns')

__all__ = ['AbstractSampler', 'MarkovSampler', 'AbstractNestedSampler', 'StaticNestedSampler']

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


PreProcessType = TypeVar('PreProcessType')


class SeedPoint(NamedTuple):
    U0: FloatArray
    log_L0: FloatArray


class AbstractSampler:
    def __init__(self, model: Model, efficiency_threshold: Optional[FloatArray] = None):
        self.model = model
        if efficiency_threshold is None:
            efficiency_threshold = 0.
        if efficiency_threshold < 0. or efficiency_threshold >= 1.:
            raise ValueError(f"{efficiency_threshold} must be in [0., 1.), got {efficiency_threshold}.")
        efficiency_threshold = jnp.asarray(efficiency_threshold, float_type)
        self.efficiency_threshold = efficiency_threshold

    def __repr__(self):
        return self.__class__.__name__

    def preprocess(self, state: NestedSamplerState) -> PreProcessType:
        """
        Produces a data structure that is necessary for sampling to run.
        Typically this is where clustering happens.

        Args:
            state: nested sampler state

        Returns:
            any valid pytree
        """
        raise NotImplementedError()

    def get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, live_points: LivePoints,
                   preprocess_data: PreProcessType) -> Sample:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            log_L_constraint: the constraint to sample within
            live_points: the current live points reservoir
            preprocess_data: the data pytree needed and produced by the sampler

        Returns:
            an i.i.d. sample
        """
        raise NotImplementedError()


class RejectionSampler(AbstractSampler):
    """
    Samplers that are based on rejection sampling. They usually first-lines of attack, and are stopped once efficiency
    gets too low.
    """
    pass


class MarkovSampler(AbstractSampler):
    """
    A sampler that conditions off a known satisfying point, e.g. a seed point.
    """

    def get_seed_point(self, key: PRNGKey, live_points: LivePoints, log_L_constraint: FloatArray) -> SeedPoint:
        """
        Samples a seed point from the live points.

        Args:
            key: PRNGKey
            live_points: the current live point set. All points satisfy the log-L constraint
            log_L_constraint: a log-L constraint to sample within. Note: Currently, redundant because we assume live
                points satisfies the constraint, but in the future, some points may not and this will be used.

        Returns:
            a seed point
        """
        sample_idx = random.randint(key, (), minval=0, maxval=live_points.reservoir.log_L.size)
        return SeedPoint(
            U0=live_points.reservoir.point_U[sample_idx],
            log_L0=live_points.reservoir.log_L[sample_idx]
        )

    def get_sample_from_seed(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray,
                             preprocess_data: PreProcessType) -> Sample:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            seed_point: function that gets the next sample from a seed point
            log_L_constraint: the constraint to sample within
            preprocess_data: the data pytree needed and produced by the sampler

        Returns:
            an i.i.d. sample
        """
        raise NotImplementedError()

    def get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, live_points: LivePoints,
                   preprocess_data: PreProcessType) -> Sample:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            log_L_constraint: the constraint to sample within
            live_points: the current live points reservoir
            preprocess_data: the data pytree needed and produced by the sampler

        Returns:
            an i.i.d. sample
        """
        key, seed_key = random.split(key, 2)
        seed_point = self.get_seed_point(key=seed_key, live_points=live_points, log_L_constraint=log_L_constraint)
        return self.get_sample_from_seed(key=key, seed_point=seed_point, log_L_constraint=log_L_constraint,
                                         preprocess_data=preprocess_data)


class AbstractNestedSampler:
    def __call__(self, state: NestedSamplerState, live_points: LivePoints, termination_cond: TerminationCondition
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
        raise NotImplementedError()


class StaticNestedSampler(AbstractNestedSampler):
    """
    Performs parallel static sampling up an underlying model
    """

    def __init__(self, samplers: List[AbstractSampler], num_live_points: int, num_parallel_samplers: int = 1):
        if num_live_points % num_parallel_samplers != 0:
            raise ValueError(
                f"num_live_points {num_live_points} must divide num_parallel_samplers {num_parallel_samplers}"
            )
        self.num_live_points = num_live_points
        self.num_parallel_samplers = num_parallel_samplers
        self.samplers = samplers

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.num_live_points} live points over {self.num_parallel_samplers} devices) using {'->'.join(map(repr, self.samplers))}"

    def _single_live_point_shrink(self,
                                  key: PRNGKey,
                                  live_points: LivePoints,
                                  log_L_contour: FloatArray,
                                  preprocess_data: PreProcessType,
                                  sampler: AbstractSampler) -> Tuple[Reservoir, LivePoints, FloatArray]:
        """
        Run nested sampling to replace an entire live point reservoir via shrinkage.

        Args:
            key: PRNGKey
            live_points: live points
            log_L_contour: the initial log-L contour of live-points.
                Note: this could be computed from the live_points reservoir but would be inefficient to do so.
            preprocess_data: any data needed to shrink the replace the live point reservoir with i.i.d. samples.
            sampler: sampler to use

        Returns:
            dead point reservoir, live points, the final log-L contour of live points
        """

        class CarryType(NamedTuple):
            key: PRNGKey
            live_points: LivePoints
            log_L_contour: FloatArray

        ResultType = Reservoir

        def body(carry: CarryType, unused_X: IntArray) -> Tuple[CarryType, ResultType]:
            del unused_X
            idx_min = jnp.argmin(carry.live_points.reservoir.log_L)
            dead_point: Reservoir = tree_map(lambda x: x[idx_min], carry.live_points.reservoir)
            log_L_dead = dead_point.log_L

            # contour becomes log_L_dead if log_L_dead is not supremum of live-points, else we choose the original
            # constraint of dead point. Note: we are at liberty to choose any log_L level as a contour so long as we
            # can sample within it uniformly.

            next_dead_contour = jnp.min(carry.live_points.reservoir.log_L.at[idx_min].set(jnp.inf))
            on_supremum = jnp.equal(log_L_dead, next_dead_contour)
            log_L_contour = jnp.where(on_supremum, carry.log_L_contour, log_L_dead)

            # replace dead point with a new sample about contour
            key, sample_key = random.split(carry.key, 2)

            sample = sampler.get_sample(key=sample_key,
                                        log_L_constraint=log_L_contour,
                                        live_points=carry.live_points,
                                        preprocess_data=preprocess_data)

            live_points_reservoir = tree_map(lambda old, update: old.at[idx_min].set(update),
                                             carry.live_points.reservoir, Reservoir(*sample))
            live_points = carry.live_points._replace(reservoir=live_points_reservoir)

            return CarryType(key=key, live_points=live_points, log_L_contour=log_L_contour), dead_point

        init_carry = CarryType(key=key, live_points=live_points, log_L_contour=log_L_contour)
        init_X = live_points.reservoir.log_L
        (_, live_points, log_L_contour), dead_reservoir = scan(body, init_carry, init_X)
        return (dead_reservoir, live_points, log_L_contour)

    def _single_thread_ns(self, key: PRNGKey, state: NestedSamplerState, live_points: LivePoints,
                          termination_cond: TerminationCondition) -> Tuple[IntArray, NestedSamplerState, LivePoints]:
        """
        Runs a single thread of static nested sampling, using all-gather to compute stopping after each live-point
        set shrinkage, which is approximately equivalent to one e-fold decrease in enclosed prior volume. This
        continues until a stopping condition is reached. Due to the all-gather this stopping condition is based on
        all the data across all devices, and is the same on all devices.

        Args:
            key: PRNGKey
            state: the initial state
            live_points: the initial live points
            termination_cond: the termination condition

        Returns:
            termination_reason, state, live_points
        """
        state = state._replace(key=key)

        class CarryType(NamedTuple):
            done: BoolArray
            termination_reason: IntArray
            state: NestedSamplerState
            live_points: LivePoints
            log_L_contour: FloatArray

        def stopping_cond(state: NestedSamplerState, live_points: LivePoints, term_cond: TerminationCondition) -> Tuple[
            BoolArray, IntArray]:
            # Collect all live points from all devices to compute stopping condition
            all_live_points: LivePoints = remove_chunk_dim(all_gather(live_points, 'i'))
            evidence_calculation, sample_stats = analyse_sample_collection(
                sample_collection=state.sample_collection,
                sorted_collection=True
            )
            done, termination_reason = determine_termination(
                term_cond=term_cond,
                sample_collection=state.sample_collection,
                evidence_calculation=evidence_calculation,
                live_points=all_live_points
            )
            return done, termination_reason

        def build_body(sampler: AbstractSampler, term_cond: TerminationCondition):
            def body(body_state: CarryType) -> CarryType:
                key, sample_key = random.split(body_state.state.key, 2)
                state = body_state.state._replace(key=key)

                preprocess_data = sampler.preprocess(state)

                (dead_reservoir, live_points, log_L_contour) = self._single_live_point_shrink(
                    key=sample_key,
                    live_points=body_state.live_points,
                    log_L_contour=body_state.log_L_contour,
                    preprocess_data=preprocess_data,
                    sampler=sampler
                )
                # Collect dead reservoirs from all devices, and merge into state
                all_dead_reservoir: Reservoir = remove_chunk_dim(all_gather(dead_reservoir, 'i'))
                all_live_points: LivePoints = remove_chunk_dim(all_gather(live_points, 'i'))
                new_state = collect_samples(state, all_dead_reservoir)

                done, termination_reason = stopping_cond(state=new_state, live_points=all_live_points, term_cond=term_cond)

                return CarryType(done=done, termination_reason=termination_reason, state=new_state,
                                 live_points=live_points, log_L_contour=log_L_contour)

            return body

        # We start with live points sorted. The starting contour is the constraint of the minimal sample.
        sort_idx = jnp.lexsort((live_points.reservoir.log_L_constraint, live_points.reservoir.log_L))
        live_points: LivePoints = tree_map(lambda x: x[sort_idx], live_points)
        log_L_contour = live_points.reservoir.log_L_constraint[0]

        carry_state = CarryType(done=jnp.asarray(False, jnp.bool_),
                                termination_reason=jnp.asarray(0, int_type),
                                state=state, live_points=live_points, log_L_contour=log_L_contour)
        for sampler in self.samplers:
            # Set efficiency termination point
            term_cond = termination_cond._replace(efficiency_threshold=sampler.efficiency_threshold)
            done, termination_reason = stopping_cond(
                state=carry_state.state,
                live_points=carry_state.live_points,
                term_cond=term_cond
            )

            carry_state = carry_state._replace(
                done=done
            )

            carry_state = while_loop(
                lambda body_state: jnp.bitwise_not(body_state.done),
                build_body(sampler=sampler, term_cond=term_cond),
                carry_state
            )

        return carry_state.termination_reason, carry_state.state, carry_state.live_points

    def __call__(self, state: NestedSamplerState, live_points: LivePoints, termination_cond: TerminationCondition
                 ) -> Tuple[IntArray, NestedSamplerState, LivePoints]:
        if live_points.reservoir.log_L.size != self.num_live_points:
            raise ValueError(
                f"live points reservoir is the wrong size. "
                f"Got {live_points.reservoir.log_L.size} but expected {self.num_live_points}.")

        # live_points should be randomly shuffled between devices
        chunked_live_points = add_chunk_dim(live_points, self.num_parallel_samplers)

        keys = random.split(state.key, self.num_parallel_samplers)

        parallel_ns = pmap(lambda key, live_points: self._single_thread_ns(
            key=key,
            state=state,
            live_points=live_points,
            termination_cond=termination_cond
        ), axis_name='i')

        chunked_termination_reason, chunked_state, chunked_live_points = parallel_ns(keys, chunked_live_points)

        termination_reason, state = tree_map(lambda x: x[0], (chunked_termination_reason, chunked_state))
        live_points = remove_chunk_dim(chunked_live_points)

        return termination_reason, state, live_points
