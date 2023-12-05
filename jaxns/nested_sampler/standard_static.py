import logging
from typing import Tuple, NamedTuple, Union, Any

from jax import random, pmap, tree_map, numpy as jnp, lax, tree_leaves, core, vmap
from jax._src.lax import parallel

from jaxns.common import remove_chunk_dim
from jaxns.internals.log_semiring import LogSpace, normalise_log_space
from jaxns.internals.stats import linear_to_log_stats, effective_sample_size
from jaxns.samplers.uniform_samplers import UniformSampler
from jaxns.model.bases import BaseAbstractModel
from jaxns.nested_sampler.bases import BaseAbstractNestedSampler
from jaxns.samplers.abc import SamplerState
from jaxns.samplers.bases import BaseAbstractSampler
from jaxns.shrinkage_statistics import compute_evidence_stats
from jaxns.tree_structure import SampleTreeGraph, count_crossed_edges
from jaxns.types import TerminationCondition, IntArray, PRNGKey, FloatArray, \
    BoolArray, int_type, UType, MeasureType, float_type, TerminationConditionDisjunction, \
    TerminationConditionConjunction, EvidenceCalculation, Sample, StaticStandardSampleCollection, \
    StaticStandardNestedSamplerState, NestedSamplerResults

logger = logging.getLogger('jaxns')


def _inter_sync_shrinkage_process(
        init_state: StaticStandardNestedSamplerState,
        sampler: BaseAbstractSampler,
        num_samples: int) -> StaticStandardNestedSamplerState:
    """
    Run nested sampling to replace an entire live point reservoir via shrinkage.

    Args:
        init_state: the state of the nested sampler at the start
        sampler: sampler to use
        num_samples: number of samples to take, i.e. work to do

    Returns:
        dead point reservoir, live points, the final log-L contour of live points
    """

    class CarryType(NamedTuple):
        state: StaticStandardNestedSamplerState
        sampler_state: SamplerState

    def body(carry: CarryType, unused_X: IntArray) -> Tuple[CarryType, Any]:
        state = carry.state

        front_loc = jnp.argmin(state.sample_collection.log_L[state.front_idx])
        dead_idx = state.front_idx[front_loc]

        # Node index is based on root of 0, so sample-nodes are 1-indexed
        dead_node_idx = dead_idx + 1

        log_L_contour = state.sample_collection.log_L[dead_idx]

        key, sample_key = random.split(state.key, 2)

        sample, phantom_samples = sampler.get_sample(
            key=sample_key,
            log_L_constraint=log_L_contour,
            sampler_state=carry.sampler_state
        )
        # Set (non-phantom) sample as the next sample

        sample_collection = state.sample_collection
        sample_collection = sample_collection._replace(
            sender_node_idx=sample_collection.sender_node_idx.at[state.next_sample_idx].set(dead_node_idx),
            log_L=sample_collection.log_L.at[state.next_sample_idx].set(sample.log_L),
            U_samples=sample_collection.U_samples.at[state.next_sample_idx].set(sample.U_sample),
            num_likelihood_evaluations=sample_collection.num_likelihood_evaluations.at[state.next_sample_idx].set(
                sample.num_likelihood_evaluations)
            # phantom stays default of False
        )
        front_idx = state.front_idx.at[front_loc].set(state.next_sample_idx)
        next_sample_idx = state.next_sample_idx + 1
        state = state._replace(
            key=key,
            next_sample_idx=next_sample_idx,
            sample_collection=sample_collection,
            front_idx=front_idx
        )

        # Set phantom samples, whose sender nodes are all the dead point. These do not get set on the front.
        num_phantom = phantom_samples.log_L.size
        sample_collection = sample_collection._replace(
            sender_node_idx=sample_collection.sender_node_idx.at[next_sample_idx:next_sample_idx + num_phantom].set(
                dead_node_idx),
            log_L=sample_collection.log_L.at[next_sample_idx:next_sample_idx + num_phantom].set(
                phantom_samples.log_L),
            U_samples=sample_collection.U_samples.at[next_sample_idx:next_sample_idx + num_phantom].set(
                phantom_samples.U_sample),
            num_likelihood_evaluations=sample_collection.num_likelihood_evaluations.at[
                                       next_sample_idx:next_sample_idx + num_phantom].set(
                phantom_samples.num_likelihood_evaluations),
            phantom=sample_collection.phantom.at[next_sample_idx:next_sample_idx + num_phantom].set(
                jnp.ones(num_phantom, dtype=jnp.bool_))
        )
        next_sample_idx = state.next_sample_idx + num_phantom
        state = state._replace(
            next_sample_idx=next_sample_idx,
            sample_collection=sample_collection
        )

        # Fast update of sampler state given this state
        state, sampler_state = sampler.post_process(state, carry.sampler_state)
        return CarryType(state=state, sampler_state=sampler_state), ()

    # Sampler state is created before all this work. Quickly updated during shrinkage.
    init_sampler_state = sampler.pre_process(state=init_state)
    init_carry = CarryType(state=init_state, sampler_state=init_sampler_state)
    out_carry, _ = lax.scan(body, init_carry, jnp.arange(num_samples))
    return out_carry.state


def _single_thread_ns(init_state: StaticStandardNestedSamplerState,
                      termination_cond: TerminationCondition,
                      sampler: BaseAbstractSampler,
                      num_samples_per_sync: int) -> Tuple[IntArray, StaticStandardNestedSamplerState]:
    """
    Runs a single thread of static nested sampling, using all-gather to compute stopping after each live-point
    set shrinkage, which is approximately equivalent to one e-fold decrease in enclosed prior volume. This
    continues until a stopping condition is reached. Due to the all-gather this stopping condition is based on
    all the data across all devices, and is the same on all devices.

    Args:
        init_state: the state of the nested sampler at the start
        termination_cond: the termination condition
        sampler: the sampler to use
        num_samples_per_sync: number of samples to take per all-gather

    Returns:
        termination_reason, final sampler state
    """

    class CarryType(NamedTuple):
        done: BoolArray
        termination_reason: IntArray
        state: StaticStandardNestedSamplerState

    def cond(carry: CarryType) -> BoolArray:
        return jnp.bitwise_not(carry.done)

    def body(carry: CarryType) -> CarryType:
        # Devices are independent, i.e. expect no communication between them in sampler.
        state = _inter_sync_shrinkage_process(
            init_state=carry.state,
            sampler=sampler,
            num_samples=num_samples_per_sync
        )

        # Synchronise
        batched_all_state: StaticStandardNestedSamplerState = parallel.all_gather(state, 'i')
        all_state = unbatch_state(state=batched_all_state)

        # Use synchronised state to determine termination ==> same stopping time on all devices
        done, termination_reason = compute_termination(state=all_state, termination_cond=termination_cond)

        return CarryType(done=done, termination_reason=termination_reason, state=state)

    init_carry_state = CarryType(
        done=jnp.asarray(False, jnp.bool_),
        termination_reason=jnp.asarray(-1, int_type),
        state=init_state
    )

    carry_state: CarryType = lax.while_loop(
        cond_fun=cond,
        body_fun=body,
        init_val=init_carry_state
    )

    # TODO: perhaps we need to do a final sampling run to make all the chains consistent,
    #  e.g. to at least a likelihood contour (i.e. standardise on L(X)). Would mean that some workers are idle.
    target_log_L_contour = jnp.max(
        parallel.all_gather(jnp.max(carry_state.state.sample_collection.log_L), 'i')
    )

    return carry_state.termination_reason, carry_state.state


def compute_termination(state: StaticStandardNestedSamplerState, termination_cond: TerminationCondition) -> Tuple[
    BoolArray, IntArray]:
    """
    Compute termination condition based on state.

    Args:
        state: the state of the nested sampler at the start
        termination_cond: the termination condition

    Returns:
        done bool, termination reason
    """
    # Use synchronised state to determine termination
    sample_tree = SampleTreeGraph(
        sender_node_idx=state.sample_collection.sender_node_idx,
        log_L=state.sample_collection.log_L
    )
    live_point_counts = count_crossed_edges(sample_tree=sample_tree, num_samples=state.next_sample_idx)
    log_L = sample_tree.log_L[live_point_counts.samples_indices]
    num_live_points = live_point_counts.num_live_points
    final_evidence_stats, per_sample_evidence_stats = compute_evidence_stats(
        log_L=log_L,
        num_live_points=num_live_points,
        num_samples=state.next_sample_idx
    )

    # Check rate of change of likelihood
    dL = LogSpace(per_sample_evidence_stats.log_L[-1]) - LogSpace(per_sample_evidence_stats.log_L[-num_live_points])
    dX = LogSpace(per_sample_evidence_stats.log_X_mean[-num_live_points]) - LogSpace(
        per_sample_evidence_stats.log_X_mean[-1])
    log_L_slope = dL.log_abs_val - dX.log_abs_val

    # Check efficiency
    num_likelihood_evals_front_mean = jnp.mean(
        state.sample_collection.num_likelihood_evaluations[state.front_idx]
    )
    efficiency = jnp.reciprocal(num_likelihood_evals_front_mean)

    # Check if on plateau
    front_log_L = state.sample_collection.log_L[state.front_idx]
    plateau = jnp.all(jnp.equal(front_log_L, front_log_L[0]))

    return determine_termination(
        term_cond=termination_cond,
        sample_collection=state.sample_collection,
        num_samples=state.next_sample_idx,
        evidence_calculation=final_evidence_stats,
        log_L=final_evidence_stats.log_L,
        log_L_slope=log_L_slope,
        efficiency=efficiency,
        plateau=plateau
    )


def determine_termination(
        term_cond: Union[TerminationConditionDisjunction, TerminationConditionConjunction, TerminationCondition],
        sample_collection: StaticStandardSampleCollection,
        num_samples: IntArray,
        evidence_calculation: EvidenceCalculation,
        log_L: FloatArray,
        log_L_slope: FloatArray,
        efficiency: FloatArray,
        plateau: BoolArray
):
    """
    Determine if termination should happen. Termination Flags are bits:
        0-bit -> 1: used maximum allowed number of samples
        1-bit -> 2: evidence uncert below threshold
        2-bit -> 4: live points evidence below threshold
        3-bit -> 8: effective sample size big enough
        4-bit -> 16: used maxmimum allowed number of likelihood evaluations
        5-bit -> 32: maximum log-likelihood contour reached
        6-bit -> 64: sampler efficiency too low
        7-bit -> 128: entire live-points set is a single plateau

    Multiple flags are summed together

    Args:
        term_cond: termination condition

    Returns:
        boolean done signal, and termination reason
    """

    termination_reason = jnp.asarray(0, int_type)
    done = jnp.asarray(False, jnp.bool_)

    def _set_done_bit(bit_done, bit_reason, done, termination_reason):
        done = jnp.bitwise_or(bit_done, done)
        termination_reason += jnp.where(bit_done,
                                        jnp.asarray(2 ** bit_reason, int_type),
                                        jnp.asarray(0, int_type))
        return done, termination_reason

    if isinstance(term_cond, TerminationConditionConjunction):
        for c in term_cond.conds:
            _done, _reason = determine_termination(
                term_cond=c,
                sample_collection=sample_collection,
                num_samples=num_samples,
                evidence_calculation=evidence_calculation,
                log_L=log_L,
                log_L_slope=log_L_slope,
                efficiency=efficiency,
                plateau=plateau
            )
            done = jnp.bitwise_and(_done, done)
            termination_reason = jnp.bitwise_and(_reason, termination_reason)
        return done, termination_reason

    if isinstance(term_cond, TerminationConditionDisjunction):
        for c in term_cond.conds:
            _done, _reason = determine_termination(
                term_cond=c,
                sample_collection=sample_collection,
                num_samples=num_samples,
                evidence_calculation=evidence_calculation,
                log_L=log_L,
                log_L_slope=log_L_slope,
                efficiency=efficiency,
                plateau=plateau
            )
            done = jnp.bitwise_or(_done, done)
            termination_reason = jnp.bitwise_or(_reason, termination_reason)
        return done, termination_reason

    if term_cond.max_samples is not None:
        # used all points
        reached_max_samples = num_samples >= term_cond.max_samples
        done, termination_reason = _set_done_bit(reached_max_samples, 0,
                                                 done=done, termination_reason=termination_reason)
    if term_cond.evidence_uncert is not None:
        _, log_Z_var = linear_to_log_stats(
            log_f_mean=evidence_calculation.log_Z_mean,
            log_f2_mean=evidence_calculation.log_Z2_mean)
        evidence_uncert_low_enough = log_Z_var <= jnp.square(term_cond.evidence_uncert)
        done, termination_reason = _set_done_bit(evidence_uncert_low_enough, 1,
                                                 done=done, termination_reason=termination_reason)
    if term_cond.live_evidence_frac is not None:
        # Z_remaining/(Z_remaining + Z_current) < delta
        L_max_estimate = LogSpace(log_L) + LogSpace(log_L_slope) * LogSpace(evidence_calculation.log_X_mean)
        Z_remaining_upper_bound_estimate = L_max_estimate * LogSpace(evidence_calculation.log_X_mean)
        Z_upper = LogSpace(evidence_calculation.log_Z_mean) + Z_remaining_upper_bound_estimate
        delta = LogSpace(term_cond.live_evidence_frac)
        small_remaining_evidence = (
                Z_remaining_upper_bound_estimate.log_abs_val - Z_upper.log_abs_val < delta.log_abs_val
        )
        done, termination_reason = _set_done_bit(small_remaining_evidence, 2,
                                                 done=done, termination_reason=termination_reason)
    if term_cond.ess is not None:
        # Kish's ESS = [sum weights]^2 / [sum weights^2]
        ess = effective_sample_size(evidence_calculation.log_Z_mean,
                                    evidence_calculation.log_dZ2_mean)
        ess_reached = ess >= term_cond.ess
        done, termination_reason = _set_done_bit(ess_reached, 3,
                                                 done=done, termination_reason=termination_reason)
    if term_cond.max_num_likelihood_evaluations is not None:
        num_likelihood_evaluations = jnp.sum(sample_collection.num_likelihood_evaluations)
        too_max_likelihood_evaluations = num_likelihood_evaluations >= term_cond.max_num_likelihood_evaluations
        done, termination_reason = _set_done_bit(too_max_likelihood_evaluations, 4,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.log_L_contour is not None:
        likelihood_contour_reached = log_L >= term_cond.log_L_contour
        done, termination_reason = _set_done_bit(likelihood_contour_reached, 5,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.efficiency_threshold is not None:
        efficiency_too_low = efficiency <= term_cond.efficiency_threshold
        done, termination_reason = _set_done_bit(efficiency_too_low, 6,
                                                 done=done, termination_reason=termination_reason)

    done, termination_reason = _set_done_bit(plateau, 7,
                                             done=done, termination_reason=termination_reason)

    return done, termination_reason


def _single_uniform_sample(key: PRNGKey, model: BaseAbstractModel) -> Sample:
    """
    Gets a single sample strictly within -inf bound (the entire prior), accounting for forbidden regions.

    Args:
        key: PRNGKey
        model: the model to use.

    Returns:
        a sample
    """

    log_L_constraint = jnp.asarray(-jnp.inf, float_type)

    class CarryState(NamedTuple):
        key: PRNGKey
        U: UType
        log_L: MeasureType
        num_likelihood_evals: IntArray

    def cond(carry_state: CarryState):
        return carry_state.log_L <= log_L_constraint

    def body(carry_state: CarryState) -> CarryState:
        key, sample_key = random.split(carry_state.key, 2)
        U = model.sample_U(key=sample_key)
        log_L = model.forward(U=U)
        num_likelihood_evals = carry_state.num_likelihood_evals + jnp.ones_like(carry_state.num_likelihood_evals)
        return CarryState(key=key, U=U, log_L=log_L, num_likelihood_evals=num_likelihood_evals)

    key, sample_key = random.split(key, 2)
    init_U = model.sample_U(key=sample_key)
    init_log_L = model.forward(init_U)
    init_carry_state = CarryState(
        key=key,
        U=init_U,
        log_L=init_log_L,
        num_likelihood_evals=jnp.asarray(1, int_type)
    )

    carry_state = lax.while_loop(cond_fun=cond, body_fun=body, init_val=init_carry_state)

    sample = Sample(
        U_sample=carry_state.U,
        log_L_constraint=log_L_constraint,
        log_L=carry_state.log_L,
        num_likelihood_evaluations=carry_state.num_likelihood_evals
    )
    return sample


def draw_uniform_samples(key: PRNGKey, num_live_points: int, model: BaseAbstractModel) -> Sample:
    """
    Get initial live points from uniformly sampling the entire prior.

    Args:
        key: PRNGKey
        num_live_points: the number of live points
        model: the model

    Returns:
        uniformly drawn samples within -inf bound
    """

    def body(carry_unused: Any, key: PRNGKey) -> Tuple[Any, Sample]:
        return carry_unused, _single_uniform_sample(key=key, model=model)

    _, samples = lax.scan(body, (), random.split(key, num_live_points))
    return samples


def create_init_state(key: PRNGKey, num_live_points: int, max_samples: int,
                      model: BaseAbstractModel) -> StaticStandardNestedSamplerState:
    """
    Return an initial sample collection, that will be incremented by the sampler.

    Args:
        key: PRNGKey
        num_live_points: the number of live points
        max_samples: the maximum number of samples
        model: the model to use.

    Returns:
        sample collection
    """

    def _repeat(a):
        return jnp.repeat(a[None], repeats=max_samples, axis=0, total_repeat_length=max_samples)

    sample_collection = StaticStandardSampleCollection(
        sender_node_idx=jnp.zeros(max_samples, dtype=int_type),
        log_L=jnp.full((max_samples,), jnp.inf, dtype=float_type),
        U_samples=_repeat(model.U_placeholder),
        num_likelihood_evaluations=jnp.full((max_samples,), 0, dtype=int_type),
        phantom=jnp.full((max_samples,), False, dtype=jnp.bool_)
    )

    key, sample_key = random.split(key, 2)
    init_samples = draw_uniform_samples(key=sample_key, num_live_points=num_live_points, model=model)
    # Merge the initial samples into the sample collection
    sample_collection = sample_collection._replace(
        log_L=sample_collection.log_L.at[:num_live_points].set(init_samples.log_L),
        U_samples=sample_collection.U_samples.at[:num_live_points].set(init_samples.U_sample),
        num_likelihood_evaluations=sample_collection.num_likelihood_evaluations.at[:num_live_points].set(
            init_samples.num_likelihood_evaluations)
    )

    return StaticStandardNestedSamplerState(
        key=key,
        next_sample_idx=jnp.asarray(num_live_points, int_type),
        sample_collection=sample_collection,
        front_idx=jnp.arange(num_live_points, int_type)
    )


def unbatch_state(state: StaticStandardNestedSamplerState) -> StaticStandardNestedSamplerState:
    """
    Remove the batch dimension from the state.

    Args:
        state: the state with batch dimension

    Returns:
        the state without batch dimension
    """
    if len(state.next_sample_idx.shape) == 1:
        return state

    key = state.key[0]  # Take first key
    next_sample_idx = jnp.sum(state.next_sample_idx)  # Next insert will be sum
    # Shifts are the cumulative sum of the number of samples per batch dimension
    shifts = jnp.concatenate([jnp.asarray([0], int_type), jnp.cumsum(state.next_sample_idx[:-1])])
    sender_node_idx = jnp.where(
        state.sample_collection.sender_node_idx.astype(jnp.bool_),
        state.sample_collection.sender_node_idx + shifts[:, None],
        state.sample_collection.sender_node_idx
    )
    # Front indices are shifted like senders
    front_idx = remove_chunk_dim(
        state.front_idx + shifts[:, None]
    )
    return StaticStandardNestedSamplerState(
        key=key,
        next_sample_idx=next_sample_idx,
        sample_collection=remove_chunk_dim(
            state.sample_collection._replace(sender_node_idx=sender_node_idx)
        ),
        front_idx=front_idx
    )


def test_unbatch_state():
    # Example of two batched states, first dimension is has 1 sample, second has 2 samples. Max samples is 2 for both.
    # num_live_points=1 for both.
    batched_state = StaticStandardNestedSamplerState(
        key=random.split(random.PRNGKey(42), 2),
        next_sample_idx=jnp.asarray([1, 2]),
        sample_collection=StaticStandardSampleCollection(
            sender_node_idx=jnp.asarray([[0, 0], [0, 1]]),
            log_L=jnp.asarray([[1., jnp.inf], [2., 3.]]),
            U_samples=jnp.asarray([[[1., 1.], [0., 0.]], [[2., 2.], [3., 3.]]]),
            num_likelihood_evaluations=jnp.asarray([[1, 0], [3, 4]]),
            phantom=jnp.asarray([[True, False], [True, True]])
        ),
        front_idx=jnp.asarray([[0], [1]])
    )

    unbatched_state = unbatch_state(batched_state)

    # The second element of first batch is not measure so it should be at end of unbatched
    expected_state = StaticStandardNestedSamplerState(
        key=batched_state.key[0],
        next_sample_idx=jnp.asarray(3, int_type),
        sample_collection=StaticStandardSampleCollection(
            sender_node_idx=jnp.asarray([0, 0, 2, 0]),
            log_L=jnp.asarray([1., 2., 3., jnp.inf]),
            U_samples=jnp.asarray([[1., 1.], [2., 2.], [3., 3.], [0., 0.]]),
            num_likelihood_evaluations=jnp.asarray([1, 3, 4, 0]),
            phantom=jnp.asarray([True, True, True, False])
        ),
        front_idx=jnp.asarray([0, 2])
    )
    # compare pytrees
    for a, b in zip(tree_leaves(unbatched_state), tree_leaves(expected_state)):
        assert jnp.all(a == b)


class StandardStaticNestedSampler(BaseAbstractNestedSampler):
    """
    A static nested sampler that uses a fixed number of live points. This uses a uniform sampler to generate the
    initial set of samples down to an efficiency threshold, then uses a provided sampler to generate the rest of the
    samples until the termination condition is met.
    """

    def __init__(self, init_efficiency_threshold: float, sampler: BaseAbstractSampler, num_live_points: int,
                 model: BaseAbstractModel, max_samples: int, num_parallel_workers: int = 1):
        self.init_efficiency_threshold = init_efficiency_threshold
        self.sampler = sampler
        self.num_live_points = int(num_live_points)
        self.num_parallel_workers = int(num_parallel_workers)
        remainder = max_samples % self.num_live_points
        extra = (max_samples - remainder) % self.num_live_points
        if extra > 0:
            logger.warning(
                f"Increasing max_samples ({max_samples}) by {extra} to closest multiple of "
                f"num_live_points {self.num_live_points}."
            )
        max_samples = int(max_samples + extra)
        if self.num_parallel_workers > 1:
            logger.info(f"Using {self.num_parallel_workers} parallel workers, each running identical samplers.")
        super().__init__(model=model, max_samples=max_samples)

    def __repr__(self):
        return f"StandardStaticNestedSampler(init_efficiency_threshold={self.init_efficiency_threshold}, " \
               f"sampler={self.sampler}, num_live_points={self.num_live_points}, model={self.model}, " \
               f"max_samples={self.max_samples}, num_parallel_workers={self.num_parallel_workers})"

    def _to_results(self, termination_reason: IntArray, state: StaticStandardNestedSamplerState,
                    trim: bool) -> NestedSamplerResults:

        num_samples = jnp.minimum(state.next_sample_idx, state.sample_collection.log_L.size)

        sample_collection = state.sample_collection

        if trim:
            if isinstance(num_samples, core.Tracer):
                raise RuntimeError("Tracer detected, but expected imperative context.")
            sample_collection = tree_map(lambda x: x[:num_samples], sample_collection)

            sample_tree = SampleTreeGraph(
                sender_node_idx=sample_collection.sender_node_idx,
                log_L=sample_collection.log_L
            )

            live_point_counts = count_crossed_edges(sample_tree=sample_tree)
            log_L = sample_tree.log_L[live_point_counts.samples_indices]
            num_live_points = live_point_counts.num_live_points

            final_evidence_stats, per_sample_evidence_stats = compute_evidence_stats(
                log_L=log_L,
                num_live_points=num_live_points
            )
        else:
            sample_tree = SampleTreeGraph(
                sender_node_idx=sample_collection.sender_node_idx,
                log_L=sample_collection.log_L
            )

            live_point_counts = count_crossed_edges(sample_tree=sample_tree, num_samples=num_samples)
            log_L = sample_tree.log_L[live_point_counts.samples_indices]
            num_live_points = live_point_counts.num_live_points

            final_evidence_stats, per_sample_evidence_stats = compute_evidence_stats(
                log_L=log_L,
                num_live_points=num_live_points,
                num_samples=num_samples
            )

        log_Z_mean, log_Z_var = linear_to_log_stats(
            log_f_mean=final_evidence_stats.log_Z_mean,
            log_f2_mean=final_evidence_stats.log_Z2_mean
        )
        log_Z_uncert = jnp.sqrt(log_Z_var)

        # Kish's ESS = [sum dZ]^2 / [sum dZ^2]
        ESS = effective_sample_size(final_evidence_stats.log_Z_mean, final_evidence_stats.log_dZ2_mean)

        samples = vmap(self.model.transform)(sample_collection.reservoir.point_U)

        log_L_samples = sample_collection.reservoir.log_L
        dp_mean = LogSpace(per_sample_evidence_stats.log_dZ_mean)
        dp_mean = normalise_log_space(dp_mean)
        H_mean = LogSpace(jnp.where(jnp.isneginf(dp_mean.log_abs_val),
                                    -jnp.inf,
                                    dp_mean.log_abs_val + log_L_samples)).sum().value - log_Z_mean
        X_mean = LogSpace(per_sample_evidence_stats.log_X_mean)
        num_likelihood_evaluations_per_sample = sample_collection.reservoir.num_likelihood_evaluations
        total_num_likelihood_evaluations = jnp.sum(num_likelihood_evaluations_per_sample)
        num_live_points_per_sample = num_live_points
        efficiency = LogSpace(jnp.log(num_samples) - jnp.log(total_num_likelihood_evaluations))

        log_posterior_density = sample_collection.reservoir.log_L + vmap(self.model.log_prob_prior)(
            sample_collection.reservoir.point_U)

        total_num_slices = jnp.sum(sample_collection.reservoir.num_slices)

        return NestedSamplerResults(
            log_Z_mean=log_Z_mean,  # estimate of log(E[Z])
            log_Z_uncert=log_Z_uncert,  # estimate of log(StdDev[Z])
            ESS=ESS,  # estimate of Kish's effective sample size
            H_mean=H_mean,  # estimate of E[int log(L) L dp/Z]
            total_num_samples=num_samples,  # int, the total number of samples collected.
            log_L_samples=log_L_samples,  # log(L) of each sample
            log_dp_mean=dp_mean.log_abs_val,
            log_posterior_density=log_posterior_density,
            # log(E[dZ]) of each sample, where dZ is how much it contributes to the total evidence.
            # log(StdDev[dZ]) of each sample, where dZ is how much it contributes to the total evidence.
            log_X_mean=X_mean.log_abs_val,  # log(E[U]) of each sample
            num_likelihood_evaluations_per_sample=num_likelihood_evaluations_per_sample,
            # how many likelihood evaluations were made per sample.
            num_live_points_per_sample=num_live_points_per_sample,
            # how many live points were taken for the samples.
            total_num_likelihood_evaluations=total_num_likelihood_evaluations,
            # how many likelihood evaluations were made in total,
            # sum of num_likelihood_evaluations_per_sample.
            total_num_slices=total_num_slices,
            log_efficiency=efficiency.log_abs_val,
            # total_num_samples / total_num_likelihood_evaluations
            termination_reason=termination_reason,  # termination condition as bit mask
            num_slices=sample_collection.reservoir.num_slices,
            samples=samples,
            U_samples=sample_collection.reservoir.point_U
        )

    def _run(self, key: PRNGKey, term_cond: TerminationCondition) -> Tuple[IntArray, StaticStandardNestedSamplerState]:
        # Create sampler threads.

        def thread(key: PRNGKey):
            state = create_init_state(
                key=key,
                num_live_points=self.num_live_points,
                max_samples=self.max_samples,
                model=self.model
            )

            # Uniform sampling down to a given mean efficiency
            uniform_sampler = UniformSampler(model=self.model)
            termination_reason, state = _single_thread_ns(
                init_state=state,
                termination_cond=TerminationCondition(
                    efficiency_threshold=jnp.asarray(self.init_efficiency_threshold),
                    max_samples=jnp.asarray(self.max_samples)
                ),
                sampler=uniform_sampler,
                num_samples_per_sync=self.num_live_points
            )

            # Continue sampling with provided sampler until user-defined termination condition is met
            termination_reason, state = _single_thread_ns(
                init_state=state,
                termination_cond=term_cond,
                sampler=self.sampler,
                num_samples_per_sync=self.num_live_points
            )

            return termination_reason, state

        parallel_ns = pmap(thread, axis_name='i')

        keys = random.split(key, self.num_parallel_workers)
        batched_termination_reason, batched_state = parallel_ns(keys)
        state = unbatch_state(state=batched_state)

        return batched_termination_reason, state
