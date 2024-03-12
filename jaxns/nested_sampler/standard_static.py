import warnings
from typing import Tuple, NamedTuple, Any, Union

import jax
from jax import random, pmap, tree_map, numpy as jnp, lax, core, vmap
from jax._src.lax import parallel

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.cumulative_ops import cumulative_op_static
from jaxns.internals.log_semiring import LogSpace, normalise_log_space
from jaxns.internals.logging import logger
from jaxns.internals.shrinkage_statistics import compute_evidence_stats, init_evidence_calc, \
    update_evicence_calculation, EvidenceUpdateVariables, _update_evidence_calc_op
from jaxns.internals.stats import linear_to_log_stats, effective_sample_size
from jaxns.internals.tree_structure import SampleTreeGraph, count_crossed_edges, unbatch_state
from jaxns.internals.types import TerminationCondition
from jaxns.internals.types import TerminationCondition
from jaxns.internals.types import TerminationCondition, IntArray, PRNGKey, BoolArray, int_type, UType, MeasureType, \
    float_type, \
    TerminationConditionDisjunction, \
    TerminationConditionConjunction, Sample, StaticStandardSampleCollection, \
    StaticStandardNestedSamplerState, NestedSamplerResults, EvidenceCalculation, TerminationRegister
from jaxns.nested_sampler.bases import BaseAbstractNestedSampler
from jaxns.samplers.abc import SamplerState
from jaxns.samplers.bases import BaseAbstractSampler
from jaxns.samplers.uniform_samplers import UniformSampler

__all__ = [
    'TerminationCondition',
    'StandardStaticNestedSampler'
]


def _inter_sync_shrinkage_process(
        init_state: StaticStandardNestedSamplerState,
        init_termination_register: TerminationRegister,
        sampler: BaseAbstractSampler,
        num_samples: int) -> Tuple[StaticStandardNestedSamplerState, TerminationRegister]:
    """
    Run nested sampling until `num_samples` samples are collected.

    Args:
        init_state: the state of the nested sampler at the start
        init_termination_register: the termination register at the start
        sampler: sampler to use
        num_samples: number of samples to take, i.e. work to do, must be >= front_size

    Returns:
        sampler state with samples added
    """

    front_size = init_state.front_idx.size
    if num_samples < front_size:
        raise RuntimeError(f"num_samples ({num_samples}) must be >= front_size ({front_size})")

    max_num_samples = init_state.sample_collection.log_L.size

    class CarryType(NamedTuple):
        front_sample_collection: StaticStandardSampleCollection
        sampler_state: SamplerState
        key: PRNGKey
        front_idx: IntArray
        next_sample_idx: IntArray
        evidence_calc: EvidenceCalculation

    class ResultType(NamedTuple):
        replace_idx: IntArray
        sample_collection: StaticStandardSampleCollection

    def body(carry: CarryType, unused_X: IntArray) -> Tuple[CarryType, ResultType]:
        front_loc = jnp.argmin(carry.front_sample_collection.log_L)
        dead_idx = carry.front_idx[front_loc]

        # Node index is based on root of 0, so sample-nodes are 1-indexed
        dead_node_idx = dead_idx + jnp.asarray(1, int_type)

        log_L_contour = carry.front_sample_collection.log_L[front_loc]

        # Update evidence calculation
        next_evidence_calculation = update_evicence_calculation(
            evidence_calculation=carry.evidence_calc,
            update=EvidenceUpdateVariables(
                num_live_points=jnp.asarray(carry.front_idx.size, float_type),
                log_L_next=log_L_contour
            )
        )

        key, sample_key = random.split(carry.key, 2)

        sample, phantom_samples = sampler.get_sample(
            key=sample_key,
            log_L_constraint=log_L_contour,
            sampler_state=carry.sampler_state
        )
        # Replace sample in front_sample_collection
        front_sample_collection = carry.front_sample_collection._replace(
            sender_node_idx=carry.front_sample_collection.sender_node_idx.at[front_loc].set(dead_node_idx),
            log_L=carry.front_sample_collection.log_L.at[front_loc].set(sample.log_L),
            U_samples=carry.front_sample_collection.U_samples.at[front_loc].set(sample.U_sample),
            num_likelihood_evaluations=carry.front_sample_collection.num_likelihood_evaluations.at[front_loc].set(
                sample.num_likelihood_evaluations),
            # Phantom samples are not on the front, so don't need to be updated from default of False
        )
        front_idx = carry.front_idx.at[front_loc].set(carry.next_sample_idx)

        # Set (non-phantom) sample as the next sample

        new_replace_idx = [
            carry.next_sample_idx[None]
        ]
        new_sender_node_idx = [
            dead_node_idx[None]
        ]
        new_log_L = [
            sample.log_L[None]
        ]
        new_U_samples = [
            sample.U_sample[None]
        ]
        new_num_likelihood_evaluations = [
            sample.num_likelihood_evaluations[None]
        ]
        new_phantom = [
            jnp.zeros((1,), jnp.bool_)
        ]

        next_sample_idx = jnp.minimum(carry.next_sample_idx + 1, max_num_samples)

        # Set phantom samples, whose sender nodes are all the dead point. These do not get set on the front.

        num_phantom = phantom_samples.log_L.size
        new_replace_idx.append(
            (next_sample_idx + jnp.arange(num_phantom)).astype(next_sample_idx.dtype)
        )
        new_sender_node_idx.append(
            jnp.full((num_phantom,), dead_node_idx)
        )
        new_log_L.append(
            phantom_samples.log_L
        )
        new_U_samples.append(
            phantom_samples.U_sample
        )
        new_num_likelihood_evaluations.append(
            phantom_samples.num_likelihood_evaluations
        )
        new_phantom.append(
            jnp.ones((num_phantom,), dtype=jnp.bool_)
        )

        next_sample_idx = jnp.minimum(next_sample_idx + num_phantom, max_num_samples)

        new_sample_collection = StaticStandardSampleCollection(
            sender_node_idx=jnp.concatenate(new_sender_node_idx, axis=0),
            log_L=jnp.concatenate(new_log_L, axis=0),
            U_samples=jnp.concatenate(new_U_samples, axis=0),
            num_likelihood_evaluations=jnp.concatenate(new_num_likelihood_evaluations, axis=0),
            phantom=jnp.concatenate(new_phantom, axis=0)
        )
        new_replace_idx = jnp.concatenate(new_replace_idx, axis=0)

        # Fast update of sampler state given a new sample collection that satisfies the front
        sampler_state = sampler.post_process(sample_collection=front_sample_collection,
                                             sampler_state=carry.sampler_state)

        new_carry = CarryType(
            front_sample_collection=front_sample_collection,
            sampler_state=sampler_state,
            key=key,
            front_idx=front_idx,
            next_sample_idx=next_sample_idx,
            evidence_calc=next_evidence_calculation
        )

        new_return = ResultType(
            replace_idx=new_replace_idx,
            sample_collection=new_sample_collection
        )

        return new_carry, new_return

    # Sampler state is created before all this work. Quickly updated during shrinkage.
    init_sampler_state = sampler.pre_process(state=init_state)
    init_front_sample_collection = tree_map(lambda x: x[init_state.front_idx], init_state.sample_collection)
    key, carry_key = random.split(init_state.key)
    init_carry = CarryType(
        sampler_state=init_sampler_state,
        key=carry_key,
        front_idx=init_state.front_idx,
        front_sample_collection=init_front_sample_collection,
        next_sample_idx=init_state.next_sample_idx,
        evidence_calc=init_termination_register.evidence_calc
    )
    out_carry, out_return = lax.scan(body, init_carry, jnp.arange(num_samples), unroll=1)

    # Replace the samples in the sample collection with out_return counterparts.
    sample_collection = tree_map(
        lambda x, y: x.at[out_return.replace_idx].set(y),
        init_state.sample_collection,
        out_return.sample_collection
    )
    # Note, discard front_sample_collection since it's already in out_return, and we've replaced the whole front.

    # Take front_idx and next_sample_idx from carry, which have been kept up-to-date at every iteration.
    state = StaticStandardNestedSamplerState(
        key=key,
        next_sample_idx=out_carry.next_sample_idx,
        sample_collection=sample_collection,
        front_idx=out_carry.front_idx
    )
    # Update termination register
    _n = init_state.front_idx.size
    _num_samples = _n
    evidence_calc_with_remaining, _ = cumulative_op_static(
        op=_update_evidence_calc_op,
        init=out_carry.evidence_calc,
        xs=EvidenceUpdateVariables(
            num_live_points=jnp.arange(_n, 0., -1., float_type),
            log_L_next=jnp.sort(out_carry.front_sample_collection.log_L)
        ),
    )
    num_likelihood_evaluations = init_termination_register.num_likelihood_evaluations + jnp.sum(
        out_return.sample_collection.num_likelihood_evaluations)
    efficiency = out_return.sample_collection.log_L.size / num_likelihood_evaluations
    plateau = jnp.all(jnp.equal(out_carry.front_sample_collection.log_L, out_carry.front_sample_collection.log_L[0]))
    termination_register = TerminationRegister(
        num_samples_used=out_carry.next_sample_idx,
        evidence_calc=out_carry.evidence_calc,
        evidence_calc_with_remaining=evidence_calc_with_remaining,
        num_likelihood_evaluations=num_likelihood_evaluations,
        log_L_contour=out_carry.evidence_calc.log_L,
        efficiency=efficiency,
        plateau=plateau
    )
    return state, termination_register


def _single_thread_ns(init_state: StaticStandardNestedSamplerState,
                      init_termination_register: TerminationRegister,
                      termination_cond: TerminationCondition,
                      sampler: BaseAbstractSampler,
                      num_samples_per_sync: int,
                      verbose: bool = False) -> Tuple[
    StaticStandardNestedSamplerState, TerminationRegister, IntArray]:
    """
    Runs a single thread of static nested sampling until a stopping condition is reached. Runs `num_samples_per_sync`
    between updating samples to limit memory ops.

    Args:
        init_state: the state of the nested sampler at the start
        termination_cond: the termination condition
        sampler: the sampler to use
        num_samples_per_sync: number of samples to take per all-gather
        verbose: whether to log debug messages.

    Returns:
        final sampler state
    """

    # Update the termination condition to stop before going over the maximum number of samples.
    space_needed_per_sync = num_samples_per_sync * (sampler.num_phantom() + 1)
    termination_cond = termination_cond._replace(
        max_samples=jnp.minimum(
            termination_cond.max_samples,
            init_state.sample_collection.log_L.size - space_needed_per_sync
        )
    )

    class CarryType(NamedTuple):
        state: StaticStandardNestedSamplerState
        termination_register: TerminationRegister

    def cond(carry: CarryType) -> BoolArray:
        done, termination_reason = determine_termination(
            term_cond=termination_cond,
            termination_register=carry.termination_register
        )
        return jnp.bitwise_not(done)

    def body(carry: CarryType) -> CarryType:
        # Devices are independent, i.e. expect no communication between them in sampler.
        state, termination_register = _inter_sync_shrinkage_process(
            init_state=carry.state,
            sampler=sampler,
            num_samples=num_samples_per_sync,
            init_termination_register=carry.termination_register
        )
        if verbose:
            log_Z_mean, log_Z_var = linear_to_log_stats(
                log_f_mean=termination_register.evidence_calc_with_remaining.log_Z_mean,
                log_f2_mean=termination_register.evidence_calc_with_remaining.log_Z2_mean)
            log_Z_uncert = jnp.sqrt(log_Z_var)
            jax.debug.print(
                "-------\n"
                "Num samples: {num_samples}\n"
                "Num likelihood evals: {num_likelihood_evals}\n"
                "Efficiency: {efficiency}\n"
                "log(L) contour: {log_L_contour}\n"
                "log(Z) est.: {log_Z_mean} +- {log_Z_uncert}",
                num_samples=termination_register.num_samples_used,
                num_likelihood_evals=termination_register.num_likelihood_evaluations,
                efficiency=termination_register.efficiency,
                log_L_contour=termination_register.log_L_contour,
                log_Z_mean=log_Z_mean,
                log_Z_uncert=log_Z_uncert
            )

        return CarryType(state=state, termination_register=termination_register)

    init_carry_state = CarryType(
        state=init_state,
        termination_register=init_termination_register
    )

    carry_state: CarryType = lax.while_loop(
        cond_fun=cond,
        body_fun=body,
        init_val=init_carry_state
    )

    _, termination_reason = determine_termination(
        term_cond=termination_cond,
        termination_register=carry_state.termination_register
    )

    return carry_state.state, carry_state.termination_register, termination_reason


def create_init_termination_register() -> TerminationRegister:
    """
    Initialise the termination register.

    Returns:
        The initial termination register.
    """
    return TerminationRegister(
        num_samples_used=jnp.asarray(0, int_type),
        evidence_calc=init_evidence_calc(),
        evidence_calc_with_remaining=init_evidence_calc(),
        num_likelihood_evaluations=jnp.asarray(0, int_type),
        log_L_contour=jnp.asarray(-jnp.inf, float_type),
        efficiency=jnp.asarray(0., float_type),
        plateau=jnp.asarray(False, bool)
    )


def determine_termination(
        term_cond: Union[TerminationConditionDisjunction, TerminationConditionConjunction, TerminationCondition],
        termination_register: TerminationRegister) -> Tuple[BoolArray, IntArray]:
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
        termination_register: register of termination variables to check against termination condition

    Returns:
        boolean done signal, and termination reason
    """

    termination_reason = jnp.asarray(0, int_type)
    done = jnp.asarray(False, jnp.bool_)

    def _set_done_bit(bit_done, bit_reason, done, termination_reason):
        if bit_done.size > 1:
            raise RuntimeError("bit_done must be a scalar.")
        done = jnp.bitwise_or(bit_done, done)
        termination_reason += jnp.where(bit_done,
                                        jnp.asarray(2 ** bit_reason, int_type),
                                        jnp.asarray(0, int_type))
        return done, termination_reason

    if isinstance(term_cond, TerminationConditionConjunction):
        for c in term_cond.conds:
            _done, _reason = determine_termination(term_cond=c, termination_register=termination_register)
            done = jnp.bitwise_and(_done, done)
            termination_reason = jnp.bitwise_and(_reason, termination_reason)
        return done, termination_reason

    if isinstance(term_cond, TerminationConditionDisjunction):
        for c in term_cond.conds:
            _done, _reason = determine_termination(term_cond=c, termination_register=termination_register)
            done = jnp.bitwise_or(_done, done)
            termination_reason = jnp.bitwise_or(_reason, termination_reason)
        return done, termination_reason

    if term_cond.live_evidence_frac is not None:
        warnings.warn("live_evidence_frac is deprecated, use dlogZ instead.")

    if term_cond.max_samples is not None:
        # used all points
        reached_max_samples = termination_register.num_samples_used >= term_cond.max_samples
        done, termination_reason = _set_done_bit(reached_max_samples, 0,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.evidence_uncert is not None:
        _, log_Z_var = linear_to_log_stats(
            log_f_mean=termination_register.evidence_calc_with_remaining.log_Z_mean,
            log_f2_mean=termination_register.evidence_calc_with_remaining.log_Z2_mean)
        evidence_uncert_low_enough = log_Z_var <= jnp.square(term_cond.evidence_uncert)
        done, termination_reason = _set_done_bit(evidence_uncert_low_enough, 1,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.dlogZ is not None:
        # (Z_remaining + Z_current) / Z_remaining < exp(dlogZ)
        log_Z_mean_1, log_Z_var_1 = linear_to_log_stats(
            log_f_mean=termination_register.evidence_calc_with_remaining.log_Z_mean,
            log_f2_mean=termination_register.evidence_calc_with_remaining.log_Z2_mean)

        log_Z_mean_0, log_Z_var_0 = linear_to_log_stats(
            log_f_mean=termination_register.evidence_calc.log_Z_mean,
            log_f2_mean=termination_register.evidence_calc.log_Z2_mean)

        small_remaining_evidence = jnp.less(
            log_Z_mean_1 - log_Z_mean_0, term_cond.dlogZ
        )
        done, termination_reason = _set_done_bit(small_remaining_evidence, 2,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.ess is not None:
        # Kish's ESS = [sum weights]^2 / [sum weights^2]
        ess = effective_sample_size(termination_register.evidence_calc_with_remaining.log_Z_mean,
                                    termination_register.evidence_calc_with_remaining.log_dZ2_mean)
        ess_reached = ess >= term_cond.ess
        done, termination_reason = _set_done_bit(ess_reached, 3,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.max_num_likelihood_evaluations is not None:
        num_likelihood_evaluations = jnp.sum(termination_register.num_likelihood_evaluations)
        too_max_likelihood_evaluations = num_likelihood_evaluations >= term_cond.max_num_likelihood_evaluations
        done, termination_reason = _set_done_bit(too_max_likelihood_evaluations, 4,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.log_L_contour is not None:
        likelihood_contour_reached = termination_register.log_L_contour >= term_cond.log_L_contour
        done, termination_reason = _set_done_bit(likelihood_contour_reached, 5,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.efficiency_threshold is not None:
        efficiency_too_low = termination_register.efficiency < term_cond.efficiency_threshold
        done, termination_reason = _set_done_bit(efficiency_too_low, 6,
                                                 done=done, termination_reason=termination_reason)

    done, termination_reason = _set_done_bit(termination_register.plateau, 7,
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


def draw_uniform_samples(key: PRNGKey, num_live_points: int, model: BaseAbstractModel, method: str = 'vmap') -> Sample:
    """
    Get initial live points from uniformly sampling the entire prior.

    Args:
        key: PRNGKey
        num_live_points: the number of live points
        model: the model
        method: which way to draw the init points. vmap is vectorised, and for performant but uses more memory.

    Returns:
        uniformly drawn samples within -inf bound
    """

    keys = random.split(key, num_live_points)
    if method == 'vmap':
        return jax.vmap(lambda _key: _single_uniform_sample(key=_key, model=model))(keys)
    elif method == 'scan':

        def body(carry_unused: Any, key: PRNGKey) -> Tuple[Any, Sample]:
            return carry_unused, _single_uniform_sample(key=key, model=model)

        _, samples = lax.scan(body, (), keys)

        return samples
    else:
        raise ValueError(f'Invalid method {method}')


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
        front_idx=jnp.arange(num_live_points, dtype=int_type)
    )


class StandardStaticNestedSampler(BaseAbstractNestedSampler):
    """
    A static nested sampler that uses a fixed number of live points. This uses a uniform sampler to generate the
    initial set of samples down to an efficiency threshold, then uses a provided sampler to generate the rest of the
    samples until the termination condition is met.
    """

    def __init__(self, init_efficiency_threshold: float, sampler: BaseAbstractSampler, num_live_points: int,
                 model: BaseAbstractModel, max_samples: int, num_parallel_workers: int = 1, verbose: bool = False):
        """
        Initialise the static nested sampler.

        Args:
            init_efficiency_threshold: the efficiency threshold to use for the initial uniform sampling. If 0 then
                turns it off.
            sampler: the sampler to use after the initial uniform sampling.
            num_live_points: the number of live points to use.
            model: the model to use.
            max_samples: the maximum number of samples to take.
            num_parallel_workers: number of parallel workers to use. Defaults to 1. Experimental feature.
            verbose: whether to log as we go.
        """
        self.init_efficiency_threshold = init_efficiency_threshold
        self.sampler = sampler
        self.num_live_points = int(num_live_points)
        self.num_parallel_workers = int(num_parallel_workers)
        self.verbose = bool(verbose)
        remainder = max_samples % self.num_live_points
        extra = (max_samples - remainder) % self.num_live_points
        if extra > 0:
            warnings.warn(
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
            num_live_points = live_point_counts.num_live_points
            log_L = sample_tree.log_L[live_point_counts.samples_indices]
            U_samples = sample_collection.U_samples[live_point_counts.samples_indices]
            num_likelihood_evaluations = sample_collection.num_likelihood_evaluations[live_point_counts.samples_indices]

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
            num_live_points = live_point_counts.num_live_points
            log_L = sample_tree.log_L[live_point_counts.samples_indices]
            U_samples = sample_collection.U_samples[live_point_counts.samples_indices]
            num_likelihood_evaluations = sample_collection.num_likelihood_evaluations[live_point_counts.samples_indices]

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

        # Correction by sqrt(k+1)
        total_phantom_samples = jnp.sum(sample_collection.phantom.astype(int_type))
        phantom_fraction = total_phantom_samples / num_samples  # k / (k+1)
        k = phantom_fraction / (1. - phantom_fraction)
        log_Z_uncert = log_Z_uncert * jnp.sqrt(1. + k)

        # Kish's ESS = [sum dZ]^2 / [sum dZ^2]
        ESS = effective_sample_size(final_evidence_stats.log_Z_mean, final_evidence_stats.log_dZ2_mean)
        ESS = ESS / (1. + k)

        samples = vmap(self.model.transform)(U_samples)
        parametrised_samples = vmap(self.model.transform_parametrised)(U_samples)

        log_L_samples = log_L
        dp_mean = LogSpace(per_sample_evidence_stats.log_dZ_mean)
        dp_mean = normalise_log_space(dp_mean)
        H_mean_instable = -((dp_mean * LogSpace(jnp.log(jnp.abs(log_L_samples)),
                                                jnp.sign(log_L_samples))).sum().value - log_Z_mean)
        # H \approx E[-log(compression)] = E[-log(X)] (More stable than E[log(L) - log(Z)]
        H_mean_stable = -((dp_mean * LogSpace(jnp.log(-per_sample_evidence_stats.log_X_mean))).sum().value)
        H_mean = jnp.where(jnp.isfinite(H_mean_instable), H_mean_instable, H_mean_stable)
        X_mean = LogSpace(per_sample_evidence_stats.log_X_mean)
        num_likelihood_evaluations_per_sample = num_likelihood_evaluations
        total_num_likelihood_evaluations = jnp.sum(num_likelihood_evaluations_per_sample)
        num_live_points_per_sample = num_live_points
        efficiency = LogSpace(jnp.log(num_samples) - jnp.log(total_num_likelihood_evaluations))

        log_posterior_density = log_L + vmap(self.model.log_prob_prior)(
            U_samples)

        return NestedSamplerResults(
            log_Z_mean=log_Z_mean,  # estimate of log(E[Z])
            log_Z_uncert=log_Z_uncert,  # estimate of log(StdDev[Z])
            ESS=ESS,  # estimate of Kish's effective sample size
            H_mean=H_mean,  # estimate of E[int log(L) L dp/Z]
            total_num_samples=num_samples,  # int, the total number of samples collected.
            total_phantom_samples=total_phantom_samples,  # int, the total number of phantom samples collected.
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
            log_efficiency=efficiency.log_abs_val,
            # total_num_samples / total_num_likelihood_evaluations
            termination_reason=termination_reason,  # termination condition as bit mask
            samples=samples,
            parametrised_samples=parametrised_samples,
            U_samples=U_samples
        )

    def _run(self, key: PRNGKey, term_cond: TerminationCondition) -> Tuple[IntArray, StaticStandardNestedSamplerState]:
        # Create sampler threads.

        def replica(key: PRNGKey) -> Tuple[StaticStandardNestedSamplerState, IntArray]:
            state = create_init_state(
                key=key,
                num_live_points=self.num_live_points,
                max_samples=self.max_samples,
                model=self.model
            )
            termination_register = create_init_termination_register()

            if self.init_efficiency_threshold > 0.:
                # Uniform sampling down to a given mean efficiency
                uniform_sampler = UniformSampler(model=self.model)
                termination_cond = TerminationCondition(
                    efficiency_threshold=jnp.asarray(self.init_efficiency_threshold),
                    dlogZ=jnp.asarray(0., float_type),
                    max_samples=jnp.asarray(self.max_samples)
                )
                state, termination_register, termination_reason = _single_thread_ns(
                    init_state=state,
                    init_termination_register=termination_register,
                    termination_cond=termination_cond,
                    sampler=uniform_sampler,
                    num_samples_per_sync=self.num_live_points,
                    verbose=self.verbose
                )

            # Continue sampling with provided sampler until user-defined termination condition is met.
            state, termination_register, termination_reason = _single_thread_ns(
                init_state=state,
                init_termination_register=termination_register,
                termination_cond=term_cond,
                sampler=self.sampler,
                num_samples_per_sync=self.num_live_points,
                verbose=self.verbose
            )
            if self.num_parallel_workers > 1:
                # We need to do a final sampling run to make all the chains consistent,
                #  to a likelihood contour (i.e. standardise on L(X)). Would mean that some workers are idle.
                target_log_L_contour = jnp.max(
                    parallel.all_gather(termination_register.log_L_contour, 'i')
                )
                termination_cond = TerminationCondition(
                    dlogZ=jnp.asarray(0., float_type),
                    log_L_contour=target_log_L_contour,
                    max_samples=jnp.asarray(self.max_samples)
                )
                state, termination_register, termination_reason = _single_thread_ns(
                    init_state=state,
                    init_termination_register=termination_register,
                    termination_cond=termination_cond,
                    sampler=self.sampler,
                    num_samples_per_sync=self.num_live_points,
                    verbose=self.verbose
                )

            return state, termination_reason

        if self.num_parallel_workers > 1:
            parallel_ns = pmap(replica, axis_name='i')

            keys = random.split(key, self.num_parallel_workers)
            batched_state, termination_reason = parallel_ns(keys)
            state = unbatch_state(batched_state=batched_state)
        else:
            state, termination_reason = replica(key)

        return termination_reason, state
