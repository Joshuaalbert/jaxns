import dataclasses
import warnings
from functools import partial
from typing import List, Optional, Tuple, NamedTuple, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map
from jaxlib import xla_client

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.cumulative_ops import cumulative_op_static
from jaxns.internals.log_semiring import LogSpace, normalise_log_space
from jaxns.internals.maps import create_mesh, tree_device_put, replace_index
from jaxns.internals.mixed_precision import mp_policy
from jaxns.internals.shrinkage_statistics import EvidenceUpdateVariables, _update_evidence_calc_op, \
    compute_evidence_stats
from jaxns.internals.stats import linear_to_log_stats, effective_sample_size_kish
from jaxns.internals.tree_structure import SampleTreeGraph, count_crossed_edges
from jaxns.internals.types import PRNGKey, IntArray, BoolArray
from jaxns.nested_samplers.abc import AbstractNestedSampler
from jaxns.nested_samplers.common.initialisation import create_init_state, create_init_termination_register
from jaxns.nested_samplers.common.termination import determine_termination
from jaxns.nested_samplers.common.types import TerminationCondition, NestedSamplerState, TerminationRegister, \
    SampleCollection, LivePointCollection, NestedSamplerResults
from jaxns.samplers.abc import AbstractSampler
from jaxns.samplers.abc import EphemeralState
from jaxns.samplers.uniform_samplers import UniformSampler

__all__ = [
    'ShardedStaticNestedSampler'
]


def _add_samples_to_state(sample_collection: LivePointCollection,
                          state: NestedSamplerState,
                          is_phantom: bool) -> NestedSamplerState:
    """
    Adds samples to state.

    Args:
        sample_collection: batched [N] samples
        state: state
        is_phantom: whether samples are phantom

    Returns:
        updated state
    """
    replace_idx = state.next_sample_idx
    num_samples = np.shape(sample_collection.log_L)[0]
    sample_collection = SampleCollection(
        sender_node_idx=replace_index(
            state.sample_collection.sender_node_idx,
            sample_collection.sender_node_idx,
            replace_idx
        ),
        log_L=replace_index(state.sample_collection.log_L, sample_collection.log_L, replace_idx),
        U_samples=replace_index(state.sample_collection.U_samples, sample_collection.U_sample, replace_idx),
        num_likelihood_evaluations=replace_index(
            state.sample_collection.num_likelihood_evaluations,
            sample_collection.num_likelihood_evaluations,
            replace_idx
        ),
        phantom=replace_index(
            state.sample_collection.phantom,
            jnp.full((num_samples,), is_phantom, jnp.bool_),
            state.next_sample_idx
        )
    )
    num_added = jnp.asarray(num_samples, mp_policy.index_dtype)
    next_sample_idx = state.next_sample_idx + num_added
    # Wrap at the number of samples (this is a trick for global optimisation that doesn't care about the entire progress)
    next_sample_idx = next_sample_idx % np.shape(state.sample_collection.log_L)[0]
    state = NestedSamplerState(
        key=state.key,
        next_sample_idx=next_sample_idx,
        sample_collection=sample_collection,
        num_samples=state.num_samples + num_added
    )
    return state


def _collect_shell(
        mesh: Mesh,
        live_point_collection: LivePointCollection,
        state: NestedSamplerState,
        termination_register: TerminationRegister,
        sampler: AbstractSampler,
        sampler_state: Any
) -> Tuple[LivePointCollection, NestedSamplerState, TerminationRegister]:
    """
    Run nested sampling until `num_samples` samples are collected.

    Args:
        mesh: the device mesh to use
        live_point_collection: the live point collection
        state: the state of the nested sampler at the start
        termination_register: the termination register at the start
        sampler: sampler to use
        sampler_state: the sampler state to use for sampling

    Returns:
        live_point_collection: the live point collection
        state: the state of the nested sampler at the end
        termination_register: the termination register at the end
    """

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(PartitionSpec('shard', ), PartitionSpec(), PartitionSpec()),
        out_specs=(PartitionSpec('shard', ), PartitionSpec('shard', )),
        check_rep=False
    )
    def get_samples(sample_keys, log_L_contour, sampler_state: Any):
        def body(carry, sample_key):
            sample, phantom_samples = sampler.get_sample(
                key=sample_key,
                log_L_constraint=log_L_contour,
                sampler_state=sampler_state
            )
            return carry, (sample, phantom_samples)

        _, (sample, phantom_samples) = jax.lax.scan(body, None, sample_keys)
        # phantom samples are [k, ...] and samples are [...] so
        phantom_samples = jax.tree.map(
            lambda x: jax.lax.reshape(x, (int(np.prod(np.shape(x)[:2])),) + np.shape(x)[2:]),
            phantom_samples
        )
        return sample, phantom_samples

    # Find and discard shell
    front_size = np.shape(live_point_collection.log_L)[0]
    # TODO: could make this a parameter
    shell_size = front_size // 2
    # TODO: always leave live points sorted so that we don't need to do again.
    discard_locs = jnp.argsort(live_point_collection.log_L)[:shell_size]
    # TODO: compute insert index KS-statistic
    discarded_sample_collection: LivePointCollection = jax.tree.map(lambda x: x[discard_locs], live_point_collection)
    state = _add_samples_to_state(
        sample_collection=discarded_sample_collection,
        state=state,
        is_phantom=False
    )

    # Replace the discarded samples
    key, sample_key = jax.random.split(state.key, 2)
    state = state._replace(key=key)
    supremum_index = discard_locs[-1]
    log_L_contour = live_point_collection.log_L[supremum_index]
    sharded_sample_keys = tree_device_put(jax.random.split(sample_key, shell_size), mesh, ('shard',))
    new_samples, phantom_samples = get_samples(sharded_sample_keys, log_L_contour, sampler_state)
    # Sender is the maximum log_L sample from discarded, i.e. that last added discarded
    new_sample_collection = LivePointCollection(
        sender_node_idx=jnp.full((shell_size,), state.next_sample_idx, mp_policy.index_dtype),
        U_sample=new_samples.U_sample,
        log_L=new_samples.log_L,
        log_L_constraint=new_samples.log_L_constraint,
        num_likelihood_evaluations=new_samples.num_likelihood_evaluations
    )
    live_point_collection: LivePointCollection = jax.tree.map(
        lambda x, update: x.at[discard_locs, ...].set(update),
        live_point_collection,
        new_sample_collection
    )

    # Add phantom samples (this is an option of user, controlled by `k`)

    num_phantom = np.shape(phantom_samples.log_L)[0]
    phantom_collection = LivePointCollection(
        sender_node_idx=jnp.full((num_phantom,), state.next_sample_idx, mp_policy.index_dtype),
        U_sample=phantom_samples.U_sample,
        log_L=phantom_samples.log_L,
        log_L_constraint=phantom_samples.log_L_constraint,
        num_likelihood_evaluations=phantom_samples.num_likelihood_evaluations
    )
    state = _add_samples_to_state(
        sample_collection=phantom_collection,
        state=state,
        is_phantom=True
    )

    # Update termination register
    evidence_calc, _ = cumulative_op_static(
        op=_update_evidence_calc_op,
        init=termination_register.evidence_calc,
        xs=EvidenceUpdateVariables(
            num_live_points=jnp.full((shell_size,), front_size, mp_policy.measure_dtype),
            log_L_next=discarded_sample_collection.log_L
        ),
    )
    sorted_log_L_live = jnp.sort(live_point_collection.log_L)
    evidence_calc_with_remaining, _ = cumulative_op_static(
        op=_update_evidence_calc_op,
        init=evidence_calc,
        xs=EvidenceUpdateVariables(
            num_live_points=jnp.arange(front_size, 0., -1., mp_policy.measure_dtype),
            log_L_next=sorted_log_L_live
        ),
    )
    # Note we consider phantom samples requiring 0 num_likelihood_evaluations
    num_likelihood_evaluations = termination_register.num_likelihood_evaluations + jnp.sum(
        new_samples.num_likelihood_evaluations)
    # We determine efficiency
    efficiency = jnp.asarray(front_size / jnp.sum(live_point_collection.num_likelihood_evaluations),
                             mp_policy.measure_dtype)
    plateau = jnp.all(jnp.equal(live_point_collection.log_L, live_point_collection.log_L[0]))
    relative_spread = 2. * jnp.abs(sorted_log_L_live[-1] - sorted_log_L_live[0]) / jnp.abs(
        sorted_log_L_live[0] + sorted_log_L_live[-1])
    absolute_spread = jnp.abs(sorted_log_L_live[-1] - sorted_log_L_live[0])
    termination_register = TerminationRegister(
        num_samples_used=state.num_samples,
        evidence_calc=evidence_calc,
        evidence_calc_with_remaining=evidence_calc_with_remaining,
        num_likelihood_evaluations=num_likelihood_evaluations,
        log_L_contour=log_L_contour,
        efficiency=efficiency,
        plateau=plateau,
        relative_spread=relative_spread,
        absolute_spread=absolute_spread
    )
    return live_point_collection, state, termination_register


def _main_ns_thread(
        mesh: Mesh,
        live_point_collection: LivePointCollection,
        state: NestedSamplerState,
        termination_register: TerminationRegister,
        termination_cond: TerminationCondition,
        sampler: AbstractSampler,
        num_discards_per_iteration: int = 1,
        verbose: bool = False
) -> Tuple[LivePointCollection, NestedSamplerState, TerminationRegister, IntArray]:
    """
    Runs a single thread of static nested sampling until a stopping condition is reached. Discards 1/2 of the
    live points at once, replacing them from the supremum contour, creating a sample tree.

    Args:
        mesh: the device mesh to use
        state: the state of the nested sampler at the start
        termination_register: the termination register at the start
        termination_cond: the termination condition
        sampler: the sampler to use
        num_discards_per_iteration: number of discard shells per iteration, between processing sampler state.
        verbose: whether to log debug messages.

    Returns:
        live_point_collection: the final set of live points
        state: the final state
        termination_register: the termination register
        termination_condition: the reason for termination
    """
    if num_discards_per_iteration <= 0:
        raise ValueError("num_discards_per_iteration must be > 0 got {num_discards_per_iteration}.")

    # Update the termination condition to stop before going over the maximum number of samples.
    # TODO: if we parametrise discard size then this needs to be updated.
    samples_per_discard = np.shape(live_point_collection.log_L)[0] // 2
    phantom_per_discard = samples_per_discard * sampler.num_phantom()
    space_needed_per_iteration = num_discards_per_iteration * (samples_per_discard + phantom_per_discard)

    if termination_cond.max_samples is not None:
        termination_cond = termination_cond._replace(
            max_samples=jnp.minimum(
                termination_cond.max_samples,
                np.shape(state.sample_collection.log_L)[0] - space_needed_per_iteration
            )
        )

    class CarryType(NamedTuple):
        live_point_collection: LivePointCollection
        state: NestedSamplerState
        termination_register: TerminationRegister

    def cond(carry: CarryType) -> BoolArray:
        done, termination_reason = determine_termination(
            term_cond=termination_cond,
            termination_register=carry.termination_register
        )
        return jnp.bitwise_not(done)

    def body(carry: CarryType) -> CarryType:
        # Discard half the live points and replace them with new samples

        key, ephemeral_key = jax.random.split(carry.state.key, 2)
        ephemeral_state = EphemeralState(
            key=ephemeral_key,
            live_points_collection=carry.live_point_collection,
            termination_register=carry.termination_register
        )
        sampler_state = sampler.pre_process(ephemeral_state)

        live_point_collection, state, termination_register = ..., ..., ...  # so IDE doesnt complain
        for _ in range(num_discards_per_iteration):
            live_point_collection, state, termination_register = _collect_shell(
                mesh=mesh,
                live_point_collection=carry.live_point_collection,
                state=carry.state,
                sampler=sampler,
                termination_register=carry.termination_register,
                sampler_state=sampler_state
            )
            key, ephemeral_key = jax.random.split(carry.state.key, 2)
            ephemeral_state = EphemeralState(
                key=ephemeral_key,
                live_points_collection=carry.live_point_collection,
                termination_register=carry.termination_register
            )
            sampler_state = sampler.post_process(ephemeral_state=ephemeral_state, sampler_state=sampler_state)

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

        return CarryType(
            state=state,
            termination_register=termination_register,
            live_point_collection=live_point_collection
        )

    carry = CarryType(
        state=state,
        termination_register=termination_register,
        live_point_collection=live_point_collection
    )

    carry = jax.lax.while_loop(
        cond_fun=cond,
        body_fun=body,
        init_val=carry
    )

    _, termination_reason = determine_termination(
        term_cond=termination_cond,
        termination_register=carry.termination_register
    )

    return carry.live_point_collection, carry.state, carry.termination_register, termination_reason


@dataclasses.dataclass(eq=False)
class ShardedStaticNestedSampler(AbstractNestedSampler):
    """
    A static nested sampler that uses a fixed number of live points. This uses a uniform sampler to generate the
    initial set of samples down to an efficiency threshold, then uses a provided sampler to generate the rest of the
    samples until the termination condition is met.

    Args:
        init_efficiency_threshold: the efficiency threshold to use for the initial uniform sampling. If 0 then
            turns it off.
        sampler: the sampler to use after the initial uniform sampling.
        num_live_points: the number of live points to use.
        model: the model to use.
        max_samples: the maximum number of samples to take.
        devices: the devices to use, default is 1.
        verbose: whether to log as we go.
    """
    model: BaseAbstractModel
    max_samples: int
    init_efficiency_threshold: float
    sampler: AbstractSampler
    num_live_points: int
    devices: Optional[List[xla_client.Device]] = None
    verbose: bool = False

    def __post_init__(self):
        if self.devices is None:
            self.devices = jax.devices()
        if len(self.devices) > 1:
            print(f"Running over {len(self.devices)} devices.")
        # Make sure num_live_points // 2 is a multiple of the number of devices
        if self.num_live_points % 2 != 0:
            raise ValueError("num_live_points must be even.")
        remainder = (self.num_live_points // 2) % len(self.devices)
        extra = 2 * ((self.num_live_points // 2 - remainder) % len(self.devices))
        if extra > 0:
            warnings.warn(
                f"Increasing num_live_points ({self.num_live_points}) by {extra} to closest multiple of "
                f"num_devices {len(self.devices)}."
            )
            self.num_live_points += extra

        block_size = self.num_live_points * (1 + self.sampler.num_phantom())
        remainder = self.max_samples % block_size
        extra = (self.max_samples - remainder) % block_size
        if extra > 0:
            warnings.warn(
                f"Increasing max_samples ({self.max_samples}) by {extra} to closest multiple of "
                f"{block_size}."
            )
        self.max_samples = int(self.max_samples + extra)

    def _to_results(self, termination_reason: IntArray, state: NestedSamplerState,
                    trim: bool) -> NestedSamplerResults:

        num_samples = jnp.minimum(
            state.num_samples,
            jnp.asarray(state.sample_collection.log_L.size, mp_policy.count_dtype)
        )

        sample_collection = state.sample_collection

        if trim:
            trim_size = jnp.minimum(
                state.num_samples,
                jnp.asarray(state.sample_collection.log_L.size, mp_policy.count_dtype)
            )

            if isinstance(num_samples, core.Tracer):
                raise RuntimeError("Tracer detected, but expected imperative context.")
            sample_collection = jax.tree.map(lambda x: x[:trim_size], sample_collection)

            sample_tree = SampleTreeGraph(
                sender_node_idx=sample_collection.sender_node_idx,
                log_L=sample_collection.log_L
            )

            live_point_counts = count_crossed_edges(sample_tree=sample_tree)
            num_live_points = live_point_counts.num_live_points
            log_L = sample_tree.log_L[live_point_counts.samples_indices]
            U_samples = sample_collection.U_samples[live_point_counts.samples_indices]
            num_likelihood_evaluations = sample_collection.num_likelihood_evaluations[
                live_point_counts.samples_indices]
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
            num_likelihood_evaluations = sample_collection.num_likelihood_evaluations[
                live_point_counts.samples_indices]

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
        total_phantom_samples = jnp.sum(mp_policy.cast_to_count(sample_collection.phantom))
        phantom_fraction = total_phantom_samples / num_samples  # k / (k+1)
        k = phantom_fraction / (1. - phantom_fraction)
        log_Z_uncert = log_Z_uncert * jnp.sqrt(1. + k)

        # Kish's ESS = [sum dZ]^2 / [sum dZ^2]
        ESS = effective_sample_size_kish(final_evidence_stats.log_Z_mean, final_evidence_stats.log_dZ2_mean)
        ESS = ESS / (1. + k)

        samples = jax.vmap(self.model.transform)(U_samples)
        parametrised_samples = jax.vmap(self.model.transform_parametrised)(U_samples)

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

        log_posterior_density = log_L + jax.vmap(self.model.log_prob_prior)(
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

    def _run(self, key: PRNGKey, term_cond: TerminationCondition) -> Tuple[
        IntArray, TerminationRegister, NestedSamplerState]:
        # Create sampler threads.
        mesh = create_mesh((len(self.devices),), ('shard',), devices=self.devices)

        live_point_collection, state = create_init_state(
            key=key,
            num_live_points=self.num_live_points,
            max_samples=self.max_samples,
            model=self.model,
            mesh=mesh
        )

        termination_register = create_init_termination_register()

        if self.init_efficiency_threshold > 0.:
            # Uniform sampling down to a given mean efficiency
            uniform_sampler = UniformSampler(model=self.model)
            termination_cond = TerminationCondition(
                efficiency_threshold=jnp.asarray(self.init_efficiency_threshold, mp_policy.measure_dtype),
                dlogZ=jnp.asarray(0., mp_policy.measure_dtype),
                max_samples=jnp.asarray(self.max_samples, mp_policy.count_dtype)
            )
            live_point_collection, state, termination_register, termination_reason = _main_ns_thread(
                mesh=mesh,
                live_point_collection=live_point_collection,
                state=state,
                termination_register=termination_register,
                termination_cond=termination_cond,
                sampler=uniform_sampler,
                num_discards_per_iteration=1,
                verbose=self.verbose
            )

        # Continue sampling with provided sampler until user-defined termination condition is met.
        live_point_collection, state, termination_register, termination_reason = _main_ns_thread(
            mesh=mesh,
            live_point_collection=live_point_collection,
            state=state,
            termination_register=termination_register,
            termination_cond=term_cond,
            sampler=self.sampler,
            num_discards_per_iteration=1,
            verbose=self.verbose
        )

        state = _add_samples_to_state(
            sample_collection=live_point_collection,
            state=state,
            is_phantom=False
        )

        return termination_reason, termination_register, state