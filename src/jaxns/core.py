import dataclasses
from functools import partial
from typing import Callable, NamedTuple, Any

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
from jaxctx import CtxParams

from jaxns import allocation as _allocation_module
from jaxns.allocation import AllocationTarget
from jaxns.allocation import ParentWork
from jaxns.allocation import accept_parent_work
from jaxns.allocation import build_allocation_plan
from jaxns.allocation import validate_allocation_target
from jaxns.constrained_sampler import (
    AbstractSampler,
    GALILEAN_TRAJECTORIES,
    UniDimSliceSampler,
    _as_mode_name,
)
from jaxns.diagnostics import AllocationDiagnostics
from jaxns.diagnostics import ComputeSectorSummary
from jaxns.diagnostics import DepthDiagnostics
from jaxns.diagnostics import DiagnosticConditionSummary
from jaxns.diagnostics import ExecutionDiagnostics
from jaxns.diagnostics import GoalDiagnostics
from jaxns.diagnostics import ParentSelectionDiagnostics
from jaxns.diagnostics import SamplerDiagnostics
from jaxns.diagnostics import WorkerRuntimeDiagnostics
from jaxns.em_gmm import DirectionAdaptationContext
from jaxns.em_gmm import DirectionKernelAdaptationCoordinator
from jaxns.em_gmm import DirectionKernelDispatchRequest
from jaxns.em_gmm import DirectionKernelFitRequest
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree
from jaxns.random_utils import resample_indicies
from jaxns.race_tree import build_block_state
from jaxns.samples import Samples, SeedPoint, PhantomSamples
from jaxns.state import State
from jaxns.termination_condition import TerminationCondition
from jaxns.types import IntArray, BoolArray, FloatArray, PRNGKey


@dataclasses.dataclass(slots=True, frozen=True)
class CoreWorkBatch(PureDataclassPytree):
    valid_mask: BoolArray
    capacity: IntArray
    num_work_items: IntArray
    parent_work_id: IntArray
    work_id: IntArray
    requested_parent_idx: IntArray
    effective_parent_idx: IntArray
    target_block_idx: IntArray
    parent_block_idx: IntArray
    fallback_to_root: BoolArray
    log_L_constraint: FloatArray
    seed_idx: IntArray
    direction_snapshot_id: IntArray
    phantom_slot: IntArray


CoreWorkBatch.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class CoreResultBatch(PureDataclassPytree):
    valid_mask: BoolArray
    capacity: IntArray
    num_results: IntArray
    status: IntArray
    parent_work_id: IntArray
    work_id: IntArray
    accepted_parent_idx: IntArray
    U_samples: object
    log_L: FloatArray
    num_likelihood_evaluations: IntArray
    phantom_valid_mask: BoolArray
    phantom_log_L: FloatArray
    phantom_slot: IntArray


CoreResultBatch.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class CoreDepthEpochHistory(PureDataclassPytree):
    valid_step: BoolArray
    num_steps: IntArray
    work_batch: CoreWorkBatch
    parent_work: ParentWork
    result_num_results: IntArray
    result_log_L_sum: FloatArray


CoreDepthEpochHistory.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class CoreDepthEpochResult(PureDataclassPytree):
    key: PRNGKey
    state: State
    allocation_iteration: IntArray
    history: CoreDepthEpochHistory


CoreDepthEpochResult.register_pytree()


CORE_BOUNDARY_SCHEMA_NAMES = ("CoreWorkBatch", "CoreResultBatch")

# Retain the legacy helper export for external callers and monkeypatch-based
# tests, while the v3 pure-core path below uses the JAX work planner instead.
select_parent_work = _allocation_module.select_parent_work


def _parse_v3_options(options: dict[str, Any]) -> tuple[int, bool]:
    options = dict(options)
    delta_K = options.pop("delta_K", 1)
    if int(delta_K) <= 0:
        raise ValueError("delta_K must be positive.")

    posterior_utility = options.pop("posterior_utility", "exact")
    if posterior_utility not in ("exact", "conservative"):
        raise ValueError(
            "posterior_utility must be 'exact' or 'conservative'."
        )
    posterior_conservative = bool(options.pop("posterior_conservative", False))
    posterior_conservative = (
            posterior_conservative
            or posterior_utility == "conservative"
    )
    if options:
        unsupported = ", ".join(sorted(options))
        raise ValueError(f"Unsupported v3 run options: {unsupported}.")
    return int(delta_K), posterior_conservative


def _should_advance_after_no_work(
        state: State,
        plan,
        depth_cond: TerminationCondition,
        shell_size: int,
) -> bool:
    """Decide whether a satisfied k=0 target should advance immediately."""
    valid = np.asarray(plan.valid, dtype=bool)
    finite_log_L = np.asarray(plan.log_L_blocks, dtype=float)
    active = valid & np.isfinite(finite_log_L)
    has_distinct_contours = (
            np.unique(finite_log_L[active]).size > 1
            if np.any(active)
            else False
    )
    if has_distinct_contours:
        return True
    if depth_cond.max_samples is None:
        return False
    return int(depth_cond.max_samples) <= int(state.num_samples) + shell_size


def _first_masked_index(mask, default):
    indices = jnp.arange(mask.shape[0], dtype=mp_policy.index_dtype)
    sentinel = jnp.asarray(mask.shape[0], dtype=mp_policy.index_dtype)
    selected = jnp.min(jnp.where(mask, indices, sentinel))
    return jnp.where(selected < sentinel, selected, default)


def _sample_categorical_index(key, weights, valid_mask):
    clean_weights = jnp.where(
        valid_mask & jnp.isfinite(weights) & (weights > 0.0),
        weights,
        0.0,
    )
    has_weight = jnp.sum(clean_weights) > 0.0
    logits = jnp.where(clean_weights > 0.0, jnp.log(clean_weights), -jnp.inf)
    safe_logits = jnp.where(has_weight, logits, jnp.zeros_like(logits))
    categorical_idx = jax.random.categorical(key, safe_logits).astype(
        mp_policy.index_dtype
    )
    first_idx = _first_masked_index(
        valid_mask,
        jnp.asarray(0, dtype=mp_policy.index_dtype),
    )
    return jnp.where(has_weight, categorical_idx, first_idx)


def _plan_core_work_batch_impl(
        key,
        state: State,
        plan,
        num_parents,
        *,
        capacity: int,
) -> CoreWorkBatch:
    """Plan fixed-shape core work using static-size JAX control flow."""
    block_state = build_block_state(
        state.samples,
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
    )
    slot_ids = jnp.arange(capacity, dtype=mp_policy.index_dtype)
    invalid_idx = jnp.asarray(-1, dtype=mp_policy.index_dtype)
    invalid_log_L = jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype)
    zero_count = jnp.asarray(0, dtype=mp_policy.count_dtype)

    def scan_body(carry, slot_id):
        current_K, scan_key, work_count = carry
        target_key, parent_key, sample_key, next_key = jax.random.split(
            scan_key,
            4,
        )
        deficits = plan.target_K - current_K
        target_mask = plan.valid & (deficits > 0)

        strict_parent_exists = jax.vmap(
            lambda target_X: jnp.any(plan.valid & (target_X < plan.volume_path.X))
        )(plan.volume_path.X)
        strict_target_mask = target_mask & strict_parent_exists
        root_only = state.num_samples == state.root_out_degree
        target_mask = jnp.where(
            root_only & jnp.any(strict_target_mask),
            strict_target_mask,
            target_mask,
        )
        has_target = jnp.any(target_mask)
        slot_requested = slot_id < jnp.asarray(
            num_parents,
            dtype=mp_policy.index_dtype,
        )
        valid_slot = slot_requested & has_target

        target_idx = _sample_categorical_index(
            target_key,
            deficits.astype(mp_policy.measure_dtype),
            target_mask,
        )
        target_X = plan.volume_path.X[target_idx]
        parent_mask = plan.valid & (target_X < plan.volume_path.X)
        has_parent = jnp.any(parent_mask)
        parent_weights = jnp.where(
            parent_mask,
            target_X / plan.volume_path.X,
            0.0,
        )
        parent_block_idx = _sample_categorical_index(
            parent_key,
            parent_weights,
            parent_mask,
        )

        block_start = block_state.block_start[parent_block_idx]
        block_stop = block_state.block_stop[parent_block_idx]
        block_size = jnp.maximum(block_stop - block_start, zero_count)
        sample_offset = jax.random.randint(
            sample_key,
            (),
            minval=jnp.asarray(0, dtype=mp_policy.index_dtype),
            maxval=jnp.maximum(
                block_size,
                jnp.asarray(1, dtype=mp_policy.index_dtype),
            ),
        )
        sample_position = block_start + sample_offset
        parent_idx = block_state.block_sample_indices[sample_position]
        has_sample = has_parent & (block_size > 0) & (parent_idx >= 0)
        fallback_to_root = valid_slot & jnp.logical_not(has_sample)
        requested_parent_idx = jnp.where(
            valid_slot & has_sample,
            parent_idx,
            invalid_idx,
        )
        effective_parent_idx = requested_parent_idx
        log_L_constraint = jnp.where(
            valid_slot & has_sample,
            state.samples.log_likelihoods[parent_idx],
            invalid_log_L,
        )
        selected_parent_block_idx = jnp.where(
            valid_slot & has_sample,
            parent_block_idx,
            invalid_idx,
        )
        selected_target_idx = jnp.where(valid_slot, target_idx, invalid_idx)
        next_K = current_K.at[target_idx].add(
            jnp.where(
                valid_slot,
                jnp.asarray(1, dtype=current_K.dtype),
                jnp.asarray(0, dtype=current_K.dtype),
            )
        )

        work_item = (
            valid_slot,
            slot_id,
            requested_parent_idx,
            effective_parent_idx,
            selected_target_idx,
            selected_parent_block_idx,
            fallback_to_root,
            log_L_constraint,
            invalid_idx,
            jnp.asarray(0, dtype=mp_policy.index_dtype),
            jnp.asarray(-1, dtype=mp_policy.index_dtype),
        )
        next_count = work_count + valid_slot.astype(work_count.dtype)
        return (next_K, next_key, next_count), work_item

    (_, _, num_work_items), items = jax.lax.scan(
        scan_body,
        (
            plan.current_K,
            key,
            jnp.asarray(0, dtype=mp_policy.count_dtype),
        ),
        slot_ids,
    )
    (
        valid_mask,
        work_id,
        requested_parent_idx,
        effective_parent_idx,
        target_block_idx,
        parent_block_idx,
        fallback_to_root,
        log_L_constraint,
        seed_idx,
        direction_snapshot_id,
        phantom_slot,
    ) = items
    return CoreWorkBatch(
        valid_mask=valid_mask,
        capacity=jnp.asarray(capacity, dtype=mp_policy.count_dtype),
        num_work_items=num_work_items,
        parent_work_id=work_id,
        work_id=work_id,
        requested_parent_idx=requested_parent_idx,
        effective_parent_idx=effective_parent_idx,
        target_block_idx=target_block_idx,
        parent_block_idx=parent_block_idx,
        fallback_to_root=fallback_to_root,
        log_L_constraint=log_L_constraint,
        seed_idx=seed_idx,
        direction_snapshot_id=direction_snapshot_id,
        phantom_slot=phantom_slot,
    )


@partial(jax.jit, static_argnames=["capacity"])
def _plan_core_work_batch_jax(
        key,
        state: State,
        plan,
        num_parents,
        *,
        capacity: int,
) -> CoreWorkBatch:
    return _plan_core_work_batch_impl(
        key,
        state,
        plan,
        num_parents,
        capacity=capacity,
    )


def _parent_work_from_core_work_batch(work_batch: CoreWorkBatch) -> ParentWork:
    num_work_items = int(np.asarray(work_batch.num_work_items))
    return ParentWork(
        parent_idxs=work_batch.requested_parent_idx[:num_work_items],
        parent_log_L_constraints=work_batch.log_L_constraint[:num_work_items],
        target_block_idxs=work_batch.target_block_idx[:num_work_items],
        parent_block_idxs=work_batch.parent_block_idx[:num_work_items],
        fallback_to_root=work_batch.fallback_to_root[:num_work_items],
    )


def _pad_tree_leading_dim(tree, capacity: int):
    if tree is None:
        return None

    def pad_leaf(leaf):
        leaf = jnp.asarray(leaf)
        if leaf.shape[0] == capacity:
            return leaf
        pad_shape = (capacity - leaf.shape[0],) + leaf.shape[1:]
        padding = jnp.zeros(pad_shape, dtype=leaf.dtype)
        return jnp.concatenate([leaf, padding], axis=0)

    return jax.tree.map(pad_leaf, tree)


def _core_result_batch_from_samples(
        work_batch: CoreWorkBatch,
        parent_work: ParentWork,
        new_samples: Samples,
) -> CoreResultBatch:
    num_results = int(parent_work.parent_idxs.shape[0])
    capacity = int(np.asarray(work_batch.capacity))
    valid_mask = jnp.arange(capacity, dtype=mp_policy.index_dtype) < num_results
    if new_samples.phantom_samples.log_L.ndim == 1:
        phantom_valid = jnp.zeros((capacity,), dtype=mp_policy.bool_dtype)
        phantom_log_L = jnp.full(
            (capacity,),
            -jnp.inf,
            dtype=mp_policy.measure_dtype,
        )
    else:
        num_phantom = int(new_samples.phantom_samples.log_L.shape[1])
        phantom_valid = jnp.zeros(
            (capacity, num_phantom),
            dtype=mp_policy.bool_dtype,
        )
        phantom_log_L = jnp.full(
            (capacity, num_phantom),
            -jnp.inf,
            dtype=mp_policy.measure_dtype,
        )
        phantom_valid = phantom_valid.at[:num_results].set(
            new_samples.phantom_samples.valid_mask
        )
        phantom_log_L = phantom_log_L.at[:num_results].set(
            new_samples.phantom_samples.log_L
        )
    log_L = jnp.full((capacity,), -jnp.inf, dtype=mp_policy.measure_dtype)
    log_L = log_L.at[:num_results].set(new_samples.log_likelihoods)
    num_likelihood_evaluations = jnp.zeros(
        (capacity,),
        dtype=mp_policy.count_dtype,
    )
    num_likelihood_evaluations = num_likelihood_evaluations.at[:num_results].set(
        new_samples.num_likelihood_evaluations
    )
    accepted_parent_idx = jnp.full(
        (capacity,),
        -1,
        dtype=mp_policy.index_dtype,
    )
    accepted_parent_idx = accepted_parent_idx.at[:num_results].set(
        parent_work.parent_idxs
    )
    status = jnp.where(
        valid_mask,
        jnp.asarray(1, dtype=mp_policy.index_dtype),
        jnp.asarray(0, dtype=mp_policy.index_dtype),
    )
    return CoreResultBatch(
        valid_mask=valid_mask,
        capacity=work_batch.capacity,
        num_results=jnp.asarray(num_results, dtype=mp_policy.count_dtype),
        status=status,
        parent_work_id=work_batch.parent_work_id,
        work_id=work_batch.work_id,
        accepted_parent_idx=accepted_parent_idx,
        U_samples=_pad_tree_leading_dim(new_samples.U_samples, capacity),
        log_L=log_L,
        num_likelihood_evaluations=num_likelihood_evaluations,
        phantom_valid_mask=phantom_valid,
        phantom_log_L=phantom_log_L,
        phantom_slot=work_batch.phantom_slot,
    )


def _root_core_work_batch(root_count: int) -> CoreWorkBatch:
    slots = jnp.arange(root_count, dtype=mp_policy.index_dtype)
    return CoreWorkBatch(
        valid_mask=jnp.ones((root_count,), dtype=mp_policy.bool_dtype),
        capacity=jnp.asarray(root_count, dtype=mp_policy.count_dtype),
        num_work_items=jnp.asarray(root_count, dtype=mp_policy.count_dtype),
        parent_work_id=slots,
        work_id=slots,
        requested_parent_idx=jnp.full(
            (root_count,),
            -1,
            dtype=mp_policy.index_dtype,
        ),
        effective_parent_idx=jnp.full(
            (root_count,),
            -1,
            dtype=mp_policy.index_dtype,
        ),
        target_block_idx=jnp.full(
            (root_count,),
            -1,
            dtype=mp_policy.index_dtype,
        ),
        parent_block_idx=jnp.full(
            (root_count,),
            -1,
            dtype=mp_policy.index_dtype,
        ),
        fallback_to_root=jnp.ones((root_count,), dtype=mp_policy.bool_dtype),
        log_L_constraint=jnp.full(
            (root_count,),
            -jnp.inf,
            dtype=mp_policy.measure_dtype,
        ),
        seed_idx=slots,
        direction_snapshot_id=jnp.zeros(
            (root_count,),
            dtype=mp_policy.index_dtype,
        ),
        phantom_slot=jnp.full(
            (root_count,),
            -1,
            dtype=mp_policy.index_dtype,
        ),
    )


def _notify_core_work_batch(owner: object, work_batch: CoreWorkBatch) -> None:
    hook = getattr(owner, "_set_active_core_work_batch", None)
    if callable(hook):
        hook(work_batch)


def _clear_core_work_batch(owner: object) -> None:
    hook = getattr(owner, "_clear_active_core_work_batch", None)
    if callable(hook):
        hook()


def _notify_core_boundary(
        owner: object,
        *,
        work_batch: CoreWorkBatch,
        result_batch: CoreResultBatch,
) -> None:
    hook = getattr(owner, "_record_core_boundary", None)
    if callable(hook):
        hook(work_batch=work_batch, result_batch=result_batch)


def _core_parent_work_id_for_owner(owner: object, work_idx: int) -> int:
    hook = getattr(owner, "_core_parent_work_id_for_work_index", None)
    if callable(hook):
        return int(hook(work_idx))
    return int(work_idx)


def _jax_direction_adaptation_context(context):
    if context is None:
        return None
    if isinstance(context, dict):
        return context
    return {
        "component_means": jnp.asarray(context.component_means),
        "component_radii": jnp.asarray(context.component_radii),
        "component_rotations": jnp.asarray(context.component_rotations),
        "component_probabilities": jnp.asarray(context.component_probabilities),
        "component_integrated_volumes": jnp.asarray(
            context.component_integrated_volumes
        ),
        "kernel_version": jnp.asarray(
            context.kernel_version,
            dtype=mp_policy.index_dtype,
        ),
    }


def _sample_one_core_work_item(
        sampler: AbstractSampler,
        args,
        params,
        adaptation_context,
        state: State,
        work_batch: CoreWorkBatch,
        item,
):
    work_idx, sample_key = item
    seed_key, sampler_key = jax.random.split(sample_key)
    constraint = work_batch.log_L_constraint[work_idx]
    active_idx = (
            jnp.arange(state.samples.log_likelihoods.shape[0])
            < state.num_samples
    )
    constraint_is_nan = jnp.isnan(constraint)
    initial_seed_mask = (
            active_idx
            & jnp.logical_not(constraint_is_nan)
            & (state.samples.log_likelihoods > constraint)
    )
    no_seed = jnp.logical_not(jnp.any(initial_seed_mask))
    effective_constraint = jnp.where(
        no_seed,
        jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype),
        constraint,
    )
    seed_mask = active_idx & (state.samples.log_likelihoods > effective_constraint)
    seed_idx = _sample_categorical_index(
        seed_key,
        jnp.ones_like(state.samples.log_likelihoods, dtype=mp_policy.measure_dtype),
        seed_mask,
    )
    seed_point = SeedPoint(
        U0=jax.tree.map(lambda u: u[seed_idx], state.samples.U_samples),
        log_L0=state.samples.log_likelihoods[seed_idx],
    )
    if adaptation_context is None:
        sample = sampler.get_sample(
            sampler_key,
            effective_constraint,
            seed_point,
            args=args,
            params=params,
        )
    else:
        sample = sampler.get_sample(
            sampler_key,
            effective_constraint,
            seed_point,
            args=args,
            params=params,
            adaptation_context=adaptation_context,
        )
    effective_parent_idx = jnp.where(
        no_seed,
        jnp.asarray(-1, dtype=mp_policy.index_dtype),
        work_batch.requested_parent_idx[work_idx],
    )
    effective_parent_block_idx = jnp.where(
        no_seed,
        jnp.asarray(-1, dtype=mp_policy.index_dtype),
        work_batch.parent_block_idx[work_idx],
    )
    fallback_to_root = work_batch.fallback_to_root[work_idx] | no_seed
    return (
        sample,
        effective_parent_idx,
        effective_constraint,
        work_batch.target_block_idx[work_idx],
        effective_parent_block_idx,
        fallback_to_root,
        seed_idx,
    )


def _sample_core_work_batch_impl(
        key,
        state: State,
        work_batch: CoreWorkBatch,
        sampler: AbstractSampler,
        args,
        params,
        adaptation_context,
) -> tuple[CoreWorkBatch, ParentWork, Samples, CoreResultBatch]:
    capacity = work_batch.valid_mask.shape[0]
    slot_ids = jnp.arange(capacity, dtype=mp_policy.index_dtype)
    sample_keys = jax.random.split(key, capacity)
    (
        samples_tuple,
        parent_idxs,
        parent_log_L_constraints,
        target_block_idxs,
        parent_block_idxs,
        fallback_to_root,
        seed_idx,
    ) = jax.lax.map(
        lambda item: _sample_one_core_work_item(
            sampler,
            args,
            params,
            adaptation_context,
            state,
            work_batch,
            item,
        ),
        (slot_ids, sample_keys),
    )
    U_samples, log_likelihoods, num_likelihood_evaluations, phantom_samples = (
        samples_tuple
    )
    valid_mask = work_batch.valid_mask
    if state.samples.phantom_samples.U_samples is None:
        phantom_U_samples = None
    else:
        phantom_U_samples = jax.tree.map(
            lambda u: jnp.zeros(
                (capacity, phantom_samples.log_L.shape[1]) + u.shape[1:],
                dtype=u.dtype,
            ),
            state.samples.U_samples,
        )
    parent_work = ParentWork(
        parent_idxs=jnp.where(
            valid_mask,
            parent_idxs,
            jnp.asarray(-1, dtype=mp_policy.index_dtype),
        ),
        parent_log_L_constraints=jnp.where(
            valid_mask,
            parent_log_L_constraints,
            jnp.asarray(jnp.inf, dtype=mp_policy.measure_dtype),
        ),
        target_block_idxs=jnp.where(
            valid_mask,
            target_block_idxs,
            jnp.asarray(-1, dtype=mp_policy.index_dtype),
        ),
        parent_block_idxs=jnp.where(
            valid_mask,
            parent_block_idxs,
            jnp.asarray(-1, dtype=mp_policy.index_dtype),
        ),
        fallback_to_root=valid_mask & fallback_to_root,
    )
    phantom_valid_mask = valid_mask[:, None] & phantom_samples.valid_mask
    new_samples = Samples(
        log_L_constraints=parent_work.parent_log_L_constraints,
        log_likelihoods=jnp.where(
            valid_mask,
            log_likelihoods,
            jnp.asarray(jnp.inf, dtype=mp_policy.measure_dtype),
        ),
        U_samples=U_samples,
        out_degree=jnp.zeros((capacity,), dtype=mp_policy.count_dtype),
        num_likelihood_evaluations=jnp.where(
            valid_mask,
            num_likelihood_evaluations.astype(mp_policy.count_dtype),
            jnp.asarray(0, dtype=mp_policy.count_dtype),
        ),
        phantom_samples=PhantomSamples(
            U_samples=phantom_U_samples,
            valid_mask=phantom_valid_mask,
            log_L=jnp.where(
                phantom_valid_mask,
                phantom_samples.log_L,
                jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype),
            ),
        ),
    )
    result_batch = CoreResultBatch(
        valid_mask=valid_mask,
        capacity=work_batch.capacity,
        num_results=work_batch.num_work_items,
        status=jnp.where(
            valid_mask,
            jnp.asarray(1, dtype=mp_policy.index_dtype),
            jnp.asarray(0, dtype=mp_policy.index_dtype),
        ),
        parent_work_id=work_batch.parent_work_id,
        work_id=work_batch.work_id,
        accepted_parent_idx=parent_work.parent_idxs,
        U_samples=U_samples,
        log_L=new_samples.log_likelihoods,
        num_likelihood_evaluations=new_samples.num_likelihood_evaluations,
        phantom_valid_mask=new_samples.phantom_samples.valid_mask,
        phantom_log_L=new_samples.phantom_samples.log_L,
        phantom_slot=work_batch.phantom_slot,
    )
    result_batch = dataclasses.replace(
        result_batch,
        phantom_slot=jnp.where(
            valid_mask,
            result_batch.phantom_slot,
            jnp.asarray(-1, dtype=mp_policy.index_dtype),
        ),
    )
    work_batch = dataclasses.replace(
        work_batch,
        effective_parent_idx=parent_work.parent_idxs,
        parent_block_idx=parent_work.parent_block_idxs,
        fallback_to_root=parent_work.fallback_to_root,
        seed_idx=jnp.where(
            valid_mask,
            seed_idx,
            jnp.asarray(-1, dtype=mp_policy.index_dtype),
        ),
    )
    return work_batch, parent_work, new_samples, result_batch


def _accept_core_result_batch_impl(
        state: State,
        parent_work: ParentWork,
        new_samples: Samples,
        valid_mask,
) -> State:
    capacity = parent_work.parent_idxs.shape[0]
    valid_mask = valid_mask.astype(mp_policy.bool_dtype)
    parent_idxs = jnp.where(
        parent_work.fallback_to_root | jnp.logical_not(valid_mask),
        jnp.asarray(0, dtype=mp_policy.index_dtype),
        parent_work.parent_idxs,
    )
    delta_parent_out_degree = jnp.where(
        valid_mask & jnp.logical_not(parent_work.fallback_to_root),
        jnp.asarray(1, dtype=state.samples.out_degree.dtype),
        jnp.asarray(0, dtype=state.samples.out_degree.dtype),
    )
    root_delta = jnp.sum(
        (valid_mask & parent_work.fallback_to_root).astype(
            state.root_out_degree.dtype
        ),
        dtype=state.root_out_degree.dtype,
    )
    append_samples = dataclasses.replace(
        new_samples,
        log_L_constraints=jnp.where(
            valid_mask,
            parent_work.parent_log_L_constraints,
            jnp.asarray(jnp.inf, dtype=mp_policy.measure_dtype),
        ),
        log_likelihoods=jnp.where(
            valid_mask,
            new_samples.log_likelihoods,
            jnp.asarray(jnp.inf, dtype=mp_policy.measure_dtype),
        ),
        out_degree=jnp.zeros((capacity,), dtype=state.samples.out_degree.dtype),
        num_likelihood_evaluations=jnp.where(
            valid_mask,
            new_samples.num_likelihood_evaluations.astype(
                state.samples.num_likelihood_evaluations.dtype
            ),
            jnp.asarray(0, dtype=state.samples.num_likelihood_evaluations.dtype),
        ),
        phantom_samples=PhantomSamples(
            U_samples=new_samples.phantom_samples.U_samples,
            valid_mask=valid_mask[:, None] & new_samples.phantom_samples.valid_mask,
            log_L=jnp.where(
                valid_mask[:, None] & new_samples.phantom_samples.valid_mask,
                new_samples.phantom_samples.log_L,
                jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype),
            ),
        ),
    )
    candidate_log_likelihoods = jnp.where(
        valid_mask,
        new_samples.log_likelihoods,
        jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype),
    )
    candidate_idx = jnp.argmax(candidate_log_likelihoods)
    candidate_log_L_supremum = candidate_log_likelihoods[candidate_idx]
    candidate_U_supremum = jax.tree.map(
        lambda u: u[candidate_idx],
        new_samples.U_samples,
    )
    improves_supremum = candidate_log_L_supremum > state.log_L_supremum
    samples = state.samples.append_samples(
        insert_idx=jnp.asarray(state.num_samples, dtype=jnp.int64),
        parent_idxs=parent_idxs,
        samples=append_samples,
        delta_parent_out_degree=delta_parent_out_degree,
    ).sort()
    num_results = jnp.sum(
        valid_mask.astype(state.num_samples.dtype),
        dtype=state.num_samples.dtype,
    )
    return State(
        root_out_degree=state.root_out_degree + root_delta,
        samples=samples,
        num_samples=state.num_samples + num_results,
        log_L_supremum=jnp.where(
            improves_supremum,
            candidate_log_L_supremum,
            state.log_L_supremum,
        ),
        U_supremum=jax.tree.map(
            lambda u_new, u_old: jnp.where(improves_supremum, u_new, u_old),
            candidate_U_supremum,
            state.U_supremum,
        ),
        model=state.model,
        args=state.args,
        params=state.params,
        termination_reason=state.termination_reason,
    )


def _pure_core_transition_impl(
        key,
        state: State,
        plan,
        num_parents,
        sampler: AbstractSampler,
        args,
        params,
        adaptation_context,
        *,
        capacity: int,
):
    plan_key, sample_key = jax.random.split(key)
    work_batch = _plan_core_work_batch_impl(
        plan_key,
        state,
        plan,
        num_parents,
        capacity=capacity,
    )
    work_batch, parent_work, new_samples, result_batch = _sample_core_work_batch_impl(
        sample_key,
        state,
        work_batch,
        sampler,
        args,
        params,
        adaptation_context,
    )
    next_state = _accept_core_result_batch_impl(
        state,
        parent_work,
        new_samples,
        work_batch.valid_mask,
    )
    return next_state, work_batch, result_batch, parent_work, new_samples


@partial(jax.jit, static_argnames=["sampler", "capacity"])
def _pure_core_transition_jax(
        key,
        state: State,
        plan,
        num_parents,
        sampler: AbstractSampler,
        args,
        params,
        adaptation_context,
        *,
        capacity: int,
):
    return _pure_core_transition_impl(
        key,
        state,
        plan,
        num_parents,
        sampler,
        args,
        params,
        adaptation_context,
        capacity=capacity,
    )


def _call_pure_core_transition_jax(
        *,
        key,
        state: State,
        plan,
        num_parents,
        sampler: AbstractSampler,
        args,
        params,
        adaptation_context,
        capacity: int,
):
    return _pure_core_transition_jax(
        key,
        state,
        plan,
        num_parents,
        sampler,
        args,
        params,
        adaptation_context,
        capacity=capacity,
    )


def _empty_core_work_batch_history(
        *,
        max_epoch_steps: int,
        capacity: int,
) -> CoreWorkBatch:
    step_slots = (max_epoch_steps, capacity)
    step_ids = jnp.arange(max_epoch_steps, dtype=mp_policy.index_dtype)
    work_ids = jnp.broadcast_to(
        jnp.arange(capacity, dtype=mp_policy.index_dtype),
        step_slots,
    )
    return CoreWorkBatch(
        valid_mask=jnp.zeros(step_slots, dtype=mp_policy.bool_dtype),
        capacity=jnp.full(
            (max_epoch_steps,),
            capacity,
            dtype=mp_policy.count_dtype,
        ),
        num_work_items=jnp.zeros(
            (max_epoch_steps,),
            dtype=mp_policy.count_dtype,
        ),
        parent_work_id=work_ids,
        work_id=work_ids,
        requested_parent_idx=jnp.full(
            step_slots,
            -1,
            dtype=mp_policy.index_dtype,
        ),
        effective_parent_idx=jnp.full(
            step_slots,
            -1,
            dtype=mp_policy.index_dtype,
        ),
        target_block_idx=jnp.full(
            step_slots,
            -1,
            dtype=mp_policy.index_dtype,
        ),
        parent_block_idx=jnp.full(
            step_slots,
            -1,
            dtype=mp_policy.index_dtype,
        ),
        fallback_to_root=jnp.zeros(step_slots, dtype=mp_policy.bool_dtype),
        log_L_constraint=jnp.full(
            step_slots,
            -jnp.inf,
            dtype=mp_policy.measure_dtype,
        ),
        seed_idx=jnp.full(step_slots, -1, dtype=mp_policy.index_dtype),
        direction_snapshot_id=jnp.broadcast_to(
            step_ids[:, None],
            step_slots,
        ),
        phantom_slot=jnp.full(step_slots, -1, dtype=mp_policy.index_dtype),
    )


def _empty_parent_work_history(
        *,
        max_epoch_steps: int,
        capacity: int,
) -> ParentWork:
    step_slots = (max_epoch_steps, capacity)
    return ParentWork(
        parent_idxs=jnp.full(step_slots, -1, dtype=mp_policy.index_dtype),
        parent_log_L_constraints=jnp.full(
            step_slots,
            jnp.inf,
            dtype=mp_policy.measure_dtype,
        ),
        target_block_idxs=jnp.full(
            step_slots,
            -1,
            dtype=mp_policy.index_dtype,
        ),
        parent_block_idxs=jnp.full(
            step_slots,
            -1,
            dtype=mp_policy.index_dtype,
        ),
        fallback_to_root=jnp.zeros(step_slots, dtype=mp_policy.bool_dtype),
    )


def _set_core_work_batch_history(
        history: CoreWorkBatch,
        step_idx,
        work_batch: CoreWorkBatch,
) -> CoreWorkBatch:
    return jax.tree.map(
        lambda history_leaf, value_leaf: history_leaf.at[step_idx].set(
            value_leaf
        ),
        history,
        work_batch,
    )


def _set_parent_work_history(
        history: ParentWork,
        step_idx,
        parent_work: ParentWork,
) -> ParentWork:
    return jax.tree.map(
        lambda history_leaf, value_leaf: history_leaf.at[step_idx].set(
            value_leaf
        ),
        history,
        parent_work,
    )


def _slice_core_work_batch_history(
        history: CoreWorkBatch,
        step_idx: int,
) -> CoreWorkBatch:
    return CoreWorkBatch(
        valid_mask=history.valid_mask[step_idx],
        capacity=history.capacity[step_idx],
        num_work_items=history.num_work_items[step_idx],
        parent_work_id=history.parent_work_id[step_idx],
        work_id=history.work_id[step_idx],
        requested_parent_idx=history.requested_parent_idx[step_idx],
        effective_parent_idx=history.effective_parent_idx[step_idx],
        target_block_idx=history.target_block_idx[step_idx],
        parent_block_idx=history.parent_block_idx[step_idx],
        fallback_to_root=history.fallback_to_root[step_idx],
        log_L_constraint=history.log_L_constraint[step_idx],
        seed_idx=history.seed_idx[step_idx],
        direction_snapshot_id=history.direction_snapshot_id[step_idx],
        phantom_slot=history.phantom_slot[step_idx],
    )


def _slice_parent_work_history(
        history: ParentWork,
        step_idx: int,
        num_work_items: int,
) -> ParentWork:
    return ParentWork(
        parent_idxs=history.parent_idxs[step_idx, :num_work_items],
        parent_log_L_constraints=(
            history.parent_log_L_constraints[step_idx, :num_work_items]
        ),
        target_block_idxs=history.target_block_idxs[step_idx, :num_work_items],
        parent_block_idxs=history.parent_block_idxs[step_idx, :num_work_items],
        fallback_to_root=history.fallback_to_root[step_idx, :num_work_items],
    )


def _core_num_likelihood_evaluations(state: State):
    active = (
            jnp.arange(state.samples.num_likelihood_evaluations.shape[0])
            < state.num_samples
    )
    return jnp.sum(
        jnp.where(
            active,
            state.samples.num_likelihood_evaluations,
            jnp.asarray(0, dtype=state.samples.num_likelihood_evaluations.dtype),
        ),
        dtype=state.samples.num_likelihood_evaluations.dtype,
    )


def _core_dlogz_depth_done_impl(state: State, threshold):
    plan = build_allocation_plan(
        state=state,
        allocation_target="uniform",
        iteration=jnp.asarray(0, dtype=mp_policy.index_dtype),
        delta_K=jnp.asarray(1, dtype=mp_policy.count_dtype),
    )
    valid = plan.valid.astype(mp_policy.bool_dtype)
    active = valid & jnp.isfinite(plan.log_L_blocks)
    has_active = jnp.any(active)
    max_log_L = jnp.max(jnp.where(active, plan.log_L_blocks, -jnp.inf))
    max_log_L = jnp.where(jnp.isfinite(max_log_L), max_log_L, 0.0)
    likelihood = jnp.where(active, jnp.exp(plan.log_L_blocks - max_log_L), 0.0)
    shell_evidence = likelihood * plan.volume_path.shell_mass
    evidence = jnp.sum(jnp.where(valid, shell_evidence, 0.0))
    block_idx = jnp.arange(plan.valid.shape[0], dtype=mp_policy.index_dtype)
    last_valid_idx = jnp.max(
        jnp.where(
            active,
            block_idx,
            jnp.asarray(0, dtype=mp_policy.index_dtype),
        )
    )
    remaining = likelihood[last_valid_idx] * plan.volume_path.X[last_valid_idx]
    total = evidence + remaining
    ratio = jnp.where(
        (total > 0.0) & jnp.isfinite(total),
        remaining / total,
        jnp.asarray(jnp.inf, dtype=mp_policy.measure_dtype),
    )
    return has_active & (ratio < threshold)


def _core_depth_condition_done_impl(
        state: State,
        *,
        depth_max_samples,
        max_num_likelihood_evaluations,
        dlogz_threshold,
        has_max_num_likelihood_evaluations: bool,
        has_dlogz: bool,
):
    done = state.num_samples >= depth_max_samples
    if has_max_num_likelihood_evaluations:
        done = done | (
                _core_num_likelihood_evaluations(state)
                >= max_num_likelihood_evaluations
        )
    if has_dlogz:
        done = done | _core_dlogz_depth_done_impl(state, dlogz_threshold)
    return done


class _CoreDepthEpochCarry(NamedTuple):
    key: PRNGKey
    state: State
    allocation_iteration: IntArray
    step: IntArray
    stop: BoolArray
    valid_step: BoolArray
    work_batch_history: CoreWorkBatch
    parent_work_history: ParentWork
    result_num_results: IntArray
    result_log_L_sum: FloatArray


def _pure_core_depth_epoch_impl(
        key,
        state: State,
        allocation_iteration,
        depth_max_samples,
        max_num_likelihood_evaluations,
        dlogz_threshold,
        root_out_degree,
        delta_K,
        sampler: AbstractSampler,
        args,
        params,
        adaptation_context,
        *,
        allocation_target: AllocationTarget,
        posterior_conservative: bool,
        capacity: int,
        max_epoch_steps: int,
        has_max_num_likelihood_evaluations: bool,
        has_dlogz: bool,
) -> CoreDepthEpochResult:
    work_batch_history = _empty_core_work_batch_history(
        max_epoch_steps=max_epoch_steps,
        capacity=capacity,
    )
    parent_work_history = _empty_parent_work_history(
        max_epoch_steps=max_epoch_steps,
        capacity=capacity,
    )
    init = _CoreDepthEpochCarry(
        key=key,
        state=state,
        allocation_iteration=allocation_iteration,
        step=jnp.asarray(0, dtype=mp_policy.index_dtype),
        stop=jnp.asarray(False, dtype=mp_policy.bool_dtype),
        valid_step=jnp.zeros((max_epoch_steps,), dtype=mp_policy.bool_dtype),
        work_batch_history=work_batch_history,
        parent_work_history=parent_work_history,
        result_num_results=jnp.zeros(
            (max_epoch_steps,),
            dtype=mp_policy.count_dtype,
        ),
        result_log_L_sum=jnp.zeros(
            (max_epoch_steps,),
            dtype=mp_policy.measure_dtype,
        ),
    )

    def cond_fn(carry: _CoreDepthEpochCarry):
        remaining = depth_max_samples - carry.state.num_samples
        return (
                (carry.step < max_epoch_steps)
                & (remaining >= jnp.asarray(capacity, dtype=remaining.dtype))
                & jnp.logical_not(carry.stop)
                & jnp.logical_not(
                    _core_depth_condition_done_impl(
                        carry.state,
                        depth_max_samples=depth_max_samples,
                        max_num_likelihood_evaluations=(
                            max_num_likelihood_evaluations
                        ),
                        dlogz_threshold=dlogz_threshold,
                        has_max_num_likelihood_evaluations=(
                            has_max_num_likelihood_evaluations
                        ),
                        has_dlogz=has_dlogz,
                    )
                )
        )

    def body_fn(carry: _CoreDepthEpochCarry):
        plan = build_allocation_plan(
            state=carry.state,
            allocation_target=allocation_target,
            iteration=carry.allocation_iteration,
            delta_K=delta_K,
            root_out_degree=root_out_degree,
            posterior_conservative=posterior_conservative,
        )
        next_key, transition_key = jax.random.split(carry.key)
        next_state, work_batch, result_batch, parent_work, _ = (
            _pure_core_transition_impl(
                transition_key,
                carry.state,
                plan,
                jnp.asarray(capacity, dtype=mp_policy.count_dtype),
                sampler,
                args,
                params,
                adaptation_context,
                capacity=capacity,
            )
        )
        step_idx = carry.step
        valid_step = carry.valid_step.at[step_idx].set(True)
        work_batch_history = _set_core_work_batch_history(
            carry.work_batch_history,
            step_idx,
            work_batch,
        )
        parent_work_history = _set_parent_work_history(
            carry.parent_work_history,
            step_idx,
            parent_work,
        )
        result_num_results = carry.result_num_results.at[step_idx].set(
            result_batch.num_results
        )
        result_log_L_sum = carry.result_log_L_sum.at[step_idx].set(
            jnp.sum(
                jnp.where(
                    result_batch.valid_mask,
                    result_batch.log_L,
                    jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
                )
            )
        )
        made_work = work_batch.num_work_items > 0
        made_progress = next_state.num_samples > carry.state.num_samples
        next_allocation_iteration = carry.allocation_iteration + jnp.where(
            made_work,
            jnp.asarray(0, dtype=carry.allocation_iteration.dtype),
            jnp.asarray(1, dtype=carry.allocation_iteration.dtype),
        )
        return _CoreDepthEpochCarry(
            key=next_key,
            state=next_state,
            allocation_iteration=next_allocation_iteration,
            step=carry.step + jnp.asarray(1, dtype=mp_policy.index_dtype),
            stop=jnp.logical_not(made_work & made_progress),
            valid_step=valid_step,
            work_batch_history=work_batch_history,
            parent_work_history=parent_work_history,
            result_num_results=result_num_results,
            result_log_L_sum=result_log_L_sum,
        )

    output = jax.lax.while_loop(cond_fn, body_fn, init)
    history = CoreDepthEpochHistory(
        valid_step=output.valid_step,
        num_steps=output.step,
        work_batch=output.work_batch_history,
        parent_work=output.parent_work_history,
        result_num_results=output.result_num_results,
        result_log_L_sum=output.result_log_L_sum,
    )
    return CoreDepthEpochResult(
        key=output.key,
        state=output.state,
        allocation_iteration=output.allocation_iteration,
        history=history,
    )


@partial(
    jax.jit,
    static_argnames=[
        "sampler",
        "allocation_target",
        "posterior_conservative",
        "capacity",
        "max_epoch_steps",
        "has_max_num_likelihood_evaluations",
        "has_dlogz",
    ],
)
def _pure_core_depth_epoch_jax(
        key,
        state: State,
        allocation_iteration,
        depth_max_samples,
        max_num_likelihood_evaluations,
        dlogz_threshold,
        root_out_degree,
        delta_K,
        sampler: AbstractSampler,
        args,
        params,
        adaptation_context,
        *,
        allocation_target: AllocationTarget,
        posterior_conservative: bool,
        capacity: int,
        max_epoch_steps: int,
        has_max_num_likelihood_evaluations: bool,
        has_dlogz: bool,
) -> CoreDepthEpochResult:
    return _pure_core_depth_epoch_impl(
        key,
        state,
        allocation_iteration,
        depth_max_samples,
        max_num_likelihood_evaluations,
        dlogz_threshold,
        root_out_degree,
        delta_K,
        sampler,
        args,
        params,
        adaptation_context,
        allocation_target=allocation_target,
        posterior_conservative=posterior_conservative,
        capacity=capacity,
        max_epoch_steps=max_epoch_steps,
        has_max_num_likelihood_evaluations=has_max_num_likelihood_evaluations,
        has_dlogz=has_dlogz,
    )


def trace_inner_depth_transition_core(
        *,
        nested_sampler,
        state: State,
        depth_cond: TerminationCondition,
        allocation_target: AllocationTarget,
        key,
        delta_K: int,
        posterior_conservative: bool = False,
        max_goal_iterations: int = 2,
):
    """Return a JAXPR for the pure-core inner depth transition."""
    allocation_target = validate_allocation_target(allocation_target)
    capacity = int(nested_sampler.shell_size)
    root_out_degree = state.root_out_degree
    depth_max_samples = (
        state.samples.log_likelihoods.shape[0]
        if depth_cond.max_samples is None
        else int(depth_cond.max_samples)
    )

    def trace_fn(trace_key, trace_state):
        output = _pure_core_depth_epoch_impl(
            trace_key,
            trace_state,
            jnp.asarray(0, dtype=mp_policy.index_dtype),
            jnp.asarray(depth_max_samples, dtype=mp_policy.count_dtype),
            jnp.asarray(0, dtype=mp_policy.count_dtype),
            jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
            root_out_degree,
            jnp.asarray(delta_K, dtype=mp_policy.count_dtype),
            nested_sampler.sampler,
            nested_sampler.args,
            nested_sampler.params,
            None,
            allocation_target=allocation_target,
            posterior_conservative=posterior_conservative,
            capacity=capacity,
            max_epoch_steps=max_goal_iterations,
            has_max_num_likelihood_evaluations=False,
            has_dlogz=False,
        )
        step_idx = jnp.maximum(
            output.history.num_steps - jnp.asarray(1, dtype=mp_policy.index_dtype),
            jnp.asarray(0, dtype=mp_policy.index_dtype),
        )
        result_boundary = (
            output.history.result_num_results[step_idx],
            output.history.result_log_L_sum[step_idx],
        )
        return output.state, result_boundary

    return {
        "boundary_result": jax.make_jaxpr(trace_fn)(key, state),
    }


@dataclasses.dataclass(slots=True)
class _ExecutionDiagnosticsBuilder:
    allocation_target: str
    target_num_live_points: int
    shell_size: int
    sampler: AbstractSampler | None
    depth_condition_summaries: list[DiagnosticConditionSummary] = (
        dataclasses.field(default_factory=list)
    )
    goal_condition_summaries: list[DiagnosticConditionSummary] = (
        dataclasses.field(default_factory=list)
    )
    allocation_target_summaries: list[object] = dataclasses.field(
        default_factory=list
    )
    requested_parent_indices: list[int] = dataclasses.field(
        default_factory=list
    )
    effective_parent_indices: list[int] = dataclasses.field(
        default_factory=list
    )
    accepted_parent_indices: list[int] = dataclasses.field(
        default_factory=list
    )
    sentinel_fallback_indices: list[int] = dataclasses.field(
        default_factory=list
    )
    direction_adaptation_diagnostics: list[object] = dataclasses.field(
        default_factory=list
    )


def _allocation_summary(
        *,
        allocation_target: str,
        iteration: int,
        plan,
) -> dict[str, object]:
    summary = {
        "allocation_target": allocation_target,
        "mode": allocation_target,
        "iteration": int(iteration),
        "target_K": np.asarray(plan.target_K),
        "current_K": np.asarray(plan.current_K),
        "live_point_targets": np.asarray(plan.target_K),
        "unit_peak_utility": np.asarray(plan.unit_peak_utility),
    }
    if allocation_target == "evidence_improving":
        summary["evidence_improving_utility"] = np.asarray(
            plan.unit_peak_utility
        )
    if allocation_target == "posterior_improving":
        summary["posterior_improving_utility"] = np.asarray(
            plan.unit_peak_utility
        )
    return summary


def _record_depth_condition_summaries(
        builder: _ExecutionDiagnosticsBuilder,
        *,
        iteration: int,
        state: State,
        depth_cond: TerminationCondition,
        register,
        dlogz_summary: tuple[float, bool] | None = None,
) -> None:
    if register is None:
        num_samples = int(state.num_samples)
        num_likelihood_evaluations = int(
            np.sum(
                np.asarray(state.samples.num_likelihood_evaluations)[
                    :num_samples
                ]
            )
        )
    else:
        num_samples = int(np.asarray(register.num_samples_used))
        num_likelihood_evaluations = int(
            np.asarray(register.num_likelihood_evaluations)
        )
    if depth_cond.max_samples is not None:
        value = num_samples
        builder.depth_condition_summaries.append(
            DiagnosticConditionSummary(
                condition_name="max_samples",
                iteration=iteration,
                num_samples=num_samples,
                value=value,
                satisfied=value >= int(depth_cond.max_samples),
            )
        )
    if depth_cond.max_num_likelihood_evaluations is not None:
        value = num_likelihood_evaluations
        builder.depth_condition_summaries.append(
            DiagnosticConditionSummary(
                condition_name="max_num_likelihood_evaluations",
                iteration=iteration,
                num_samples=num_samples,
                value=value,
                satisfied=(
                    value >= int(depth_cond.max_num_likelihood_evaluations)
                ),
            )
        )
    if depth_cond.dlogZ is not None:
        if dlogz_summary is None:
            dlogz_summary = _compute_dlogz_depth_summary(
                state,
                threshold=float(depth_cond.dlogZ),
            )
        value, satisfied = dlogz_summary
        builder.depth_condition_summaries.append(
            DiagnosticConditionSummary(
                condition_name="dlogZ",
                iteration=iteration,
                num_samples=int(state.num_samples),
                value=float(value),
                satisfied=bool(satisfied),
            )
        )


def _compute_dlogz_depth_summary(
        state: State,
        *,
        threshold: float,
) -> tuple[float, bool]:
    plan = build_allocation_plan(
        state=state,
        allocation_target="uniform",
        iteration=0,
        delta_K=1,
    )
    valid = np.asarray(plan.valid, dtype=bool)
    finite = np.isfinite(np.asarray(plan.log_L_blocks, dtype=float))
    active = valid & finite
    if not np.any(active):
        return float("inf"), False
    log_L = np.asarray(plan.log_L_blocks, dtype=float)
    max_log_L = float(np.max(log_L[active]))
    likelihood = np.zeros_like(log_L, dtype=float)
    likelihood[active] = np.exp(log_L[active] - max_log_L)
    shell_evidence = likelihood * np.asarray(
        plan.volume_path.shell_mass,
        dtype=float,
    )
    evidence = float(np.sum(shell_evidence[valid]))
    last_valid_idx = int(np.where(active)[0][-1])
    remaining = float(
        likelihood[last_valid_idx]
        * np.asarray(plan.volume_path.X, dtype=float)[last_valid_idx]
    )
    total = evidence + remaining
    if total <= 0.0 or not np.isfinite(total):
        return float("inf"), False
    ratio = remaining / total
    return float(ratio), bool(ratio < threshold)


def _record_parent_selection(
        builder: _ExecutionDiagnosticsBuilder,
        *,
        requested_parent_work: ParentWork,
        accepted_parent_work: ParentWork,
) -> None:
    requested = np.asarray(requested_parent_work.parent_idxs, dtype=np.int64)
    effective = np.asarray(accepted_parent_work.parent_idxs, dtype=np.int64)
    fallback = np.asarray(
        accepted_parent_work.fallback_to_root,
        dtype=bool,
    )
    start = len(builder.requested_parent_indices)
    builder.requested_parent_indices.extend(int(x) for x in requested)
    builder.effective_parent_indices.extend(int(x) for x in effective)
    builder.accepted_parent_indices.extend(int(x) for x in effective)
    builder.sentinel_fallback_indices.extend(
        start + int(i) for i in np.where(fallback)[0]
    )


def _sampler_mode(sampler: AbstractSampler | None) -> str:
    if isinstance(sampler, UniDimSliceSampler):
        return "slice"
    if sampler is None:
        return "unknown"
    return type(sampler).__name__


def _direction_kernel_mode(sampler: AbstractSampler | None) -> str:
    mode = _as_mode_name(getattr(sampler, "direction_kernel", None))
    return "isotropic" if mode is None else mode


def _direction_kernel_requests_adaptation(
        sampler: AbstractSampler | None,
) -> bool:
    mode = _direction_kernel_mode(sampler)
    return mode in {
        "ellipsoidal",
        "ellipsoidal_gaussian",
        "gmm",
        "non_isotropic",
        "non-isotropic",
    }


def _identity_direction_context(
        *,
        d_dim: int,
        kernel_version: int = 0,
        allocation_target: str | None = None,
) -> DirectionAdaptationContext:
    return DirectionAdaptationContext(
        component_means=np.zeros((1, d_dim), dtype=float),
        component_radii=np.ones((1, d_dim), dtype=float),
        component_rotations=np.eye(d_dim, dtype=float)[None, :, :],
        component_probabilities=np.ones((1,), dtype=float),
        component_integrated_volumes=np.ones((1,), dtype=float),
        kernel_version=int(kernel_version),
        allocation_target=allocation_target,
    )


def _direction_context_has_components(context: object | None) -> bool:
    if context is None:
        return False
    for name in (
            "component_means",
            "component_radii",
            "component_rotations",
            "component_probabilities",
    ):
        if isinstance(context, dict):
            value = context.get(name)
        else:
            value = getattr(context, name, None)
        if value is None:
            return False
    return True


def _ensure_direction_context(
        *,
        context: object | None,
        d_dim: int,
        kernel_version: int = 0,
        allocation_target: str | None = None,
) -> DirectionAdaptationContext | object:
    if _direction_context_has_components(context):
        return context
    return _identity_direction_context(
        d_dim=d_dim,
        kernel_version=kernel_version,
        allocation_target=allocation_target,
    )


def _trajectory_mode(sampler: AbstractSampler | None) -> str:
    mode = _as_mode_name(getattr(sampler, "trajectory", None))
    if mode in ("straight_line", "straight_line_perfect"):
        if bool(getattr(sampler, "no_step_out", False)):
            return "straight_line_perfect_bracketing"
    return "straight_line_perfect_bracketing" if mode is None else mode


def _phantom_burn_in(sampler: AbstractSampler | None) -> int:
    burn_in = getattr(sampler, "phantom_burn_in", 0)
    return 0 if burn_in is None else int(burn_in)


def _normalise_runtime_status(status: object) -> str:
    status = str(status).lower()
    if status in {"accept", "accepted", "complete", "completed"}:
        return "accepted"
    if status in {"retry", "retried"}:
        return "retried"
    if status in {"revoke", "revoked", "cancel", "cancelled"}:
        return "revoked"
    return status


def _dispatch_field(record: object, *names: str, default=None):
    for name in names:
        if isinstance(record, dict) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


def _worker_runtime_diagnostics(ns: object) -> WorkerRuntimeDiagnostics:
    records = tuple(getattr(ns, "coordinator_dispatch_records", ()))
    if not records:
        return WorkerRuntimeDiagnostics(
            worker_count=0,
            compute_sector_summaries=(),
            runner_ids=(),
            task_ids=(),
            requested_parent_indices=jnp.asarray([], dtype=mp_policy.index_dtype),
            effective_parent_indices=jnp.asarray([], dtype=mp_policy.index_dtype),
            accepted_parent_indices=jnp.asarray([], dtype=mp_policy.index_dtype),
            in_flight_parent_targets=(),
            accepted_task_count=0,
            retried_task_count=0,
            revoked_task_count=0,
            model_compilation_times=(),
            async_identity_preserved=True,
            dispatch_records=(),
        )
    runner_id = str(_dispatch_field(records[0], "runner_id", default=""))
    records = tuple(
        record
        for record in records
        if str(_dispatch_field(record, "runner_id", default="")) == runner_id
    )
    statuses = tuple(
        _normalise_runtime_status(_dispatch_field(record, "status"))
        for record in records
    )
    dispatch_identities = tuple(
        (
            str(_dispatch_field(record, "runner_id", default="")),
            str(_dispatch_field(record, "task_id", default="")),
            str(_dispatch_field(record, "attempt_id", default="")),
            str(_dispatch_field(record, "transport_id", default="")),
        )
        for record in records
    )
    async_identity_preserved = (
            all(
                str(_dispatch_field(record, "identity_owner", default=""))
                == "coordinator"
                for record in records
            )
            and all(
                all(identity_part for identity_part in dispatch_identity)
                for dispatch_identity in dispatch_identities
            )
    )
    lb_state = getattr(ns, "runtime_lb_state", None)
    sectors = ()
    worker_count = 0
    if lb_state is not None:
        sectors = tuple(
            ComputeSectorSummary(
                sector_id=sector.sector_id,
                device_type=sector.device_type,
                device_id=sector.device_id,
                worker_count=int(sector.num_workers),
            )
            for sector in lb_state.compute_sectors.values()
        )
        worker_count = sum(sector.worker_count for sector in sectors)
    return WorkerRuntimeDiagnostics(
        worker_count=worker_count,
        compute_sector_summaries=sectors,
        runner_ids=tuple(
            str(_dispatch_field(record, "runner_id", default=""))
            for record in records
        ),
        task_ids=tuple(
            str(_dispatch_field(record, "task_id", default=""))
            for record in records
        ),
        requested_parent_indices=jnp.asarray(
            [
                int(_dispatch_field(
                    record,
                    "requested_parent_index",
                    "requested_parent_idx",
                    default=-1,
                ))
                for record in records
            ],
            dtype=mp_policy.index_dtype,
        ),
        effective_parent_indices=jnp.asarray(
            [
                int(_dispatch_field(
                    record,
                    "effective_parent_index",
                    "effective_parent_idx",
                    default=-1,
                ))
                for record in records
            ],
            dtype=mp_policy.index_dtype,
        ),
        accepted_parent_indices=jnp.asarray(
            [
                int(_dispatch_field(
                    record,
                    "accepted_parent_index",
                    "accepted_parent_idx",
                    default=-1,
                ))
                for record in records
            ],
            dtype=mp_policy.index_dtype,
        ),
        in_flight_parent_targets=tuple(
            int(_dispatch_field(
                record,
                "in_flight_parent_target",
                default=-1,
            ))
            for record in records
        ),
        accepted_task_count=sum(status == "accepted" for status in statuses),
        retried_task_count=sum(status == "retried" for status in statuses),
        revoked_task_count=sum(status == "revoked" for status in statuses),
        model_compilation_times=(),
        async_identity_preserved=async_identity_preserved,
        dispatch_records=records,
    )


def _build_execution_diagnostics(
        ns: object,
        builder: _ExecutionDiagnosticsBuilder,
        state: State,
) -> ExecutionDiagnostics:
    num_samples = int(state.num_samples)
    sample_slice = slice(0, num_samples)
    phantom_valid = np.asarray(
        state.samples.phantom_samples.valid_mask[sample_slice],
        dtype=bool,
    )
    if phantom_valid.shape[-1] == 0:
        retained_counts = jnp.zeros((num_samples,), dtype=mp_policy.count_dtype)
    else:
        retained_counts = jnp.sum(
            jnp.asarray(phantom_valid, dtype=mp_policy.count_dtype),
            axis=-1,
        )
    phantom_shape = state.samples.phantom_samples.log_L[sample_slice].shape
    if len(phantom_shape) > 1 and phantom_shape[1] == 0:
        phantom_likelihood_evaluations = jnp.zeros(
            phantom_shape,
            dtype=mp_policy.count_dtype,
        )
    else:
        # -1 marks unavailable per-phantom likelihood counts.
        phantom_likelihood_evaluations = jnp.full(
            phantom_shape,
            -1,
            dtype=mp_policy.count_dtype,
        )
    return ExecutionDiagnostics(
        allocation=AllocationDiagnostics(
            mode=builder.allocation_target,
            target_num_live_points=builder.target_num_live_points,
            shell_size=builder.shell_size,
            target_summaries=tuple(builder.allocation_target_summaries),
        ),
        parent_selection=ParentSelectionDiagnostics(
            requested_parent_indices=jnp.asarray(
                builder.requested_parent_indices,
                dtype=mp_policy.index_dtype,
            ),
            effective_parent_indices=jnp.asarray(
                builder.effective_parent_indices,
                dtype=mp_policy.index_dtype,
            ),
            accepted_parent_indices=jnp.asarray(
                builder.accepted_parent_indices,
                dtype=mp_policy.index_dtype,
            ),
            sentinel_fallback_count=len(builder.sentinel_fallback_indices),
            sentinel_fallback_indices=jnp.asarray(
                builder.sentinel_fallback_indices,
                dtype=mp_policy.index_dtype,
            ),
        ),
        depth=DepthDiagnostics(
            condition_summaries=tuple(builder.depth_condition_summaries),
        ),
        goal=GoalDiagnostics(
            condition_summaries=tuple(builder.goal_condition_summaries),
        ),
        sampler=SamplerDiagnostics(
            mode=_sampler_mode(builder.sampler),
            direction_kernel_mode=_direction_kernel_mode(builder.sampler),
            trajectory_mode=_trajectory_mode(builder.sampler),
            phantom_burn_in=_phantom_burn_in(builder.sampler),
            retained_phantom_capacity=(
                0 if builder.sampler is None else int(builder.sampler.num_phantom())
            ),
            retained_phantom_counts_per_sample=retained_counts,
            likelihood_evaluations_per_classic_sample=(
                state.samples.num_likelihood_evaluations[sample_slice]
            ),
            likelihood_evaluations_per_retained_phantom_cluster=(
                phantom_likelihood_evaluations
            ),
            direction_adaptation_diagnostics=tuple(
                builder.direction_adaptation_diagnostics
            ),
        ),
        worker_runtime=_worker_runtime_diagnostics(ns),
    )


@partial(jax.jit, inline=True, static_argnames=['num_live_points', 'num_phantom', 'max_samples', 'store_phantom_samples', 'batch_size'])
def _sample_init_state(key, num_live_points: int, max_samples: int, model: Model, num_phantom: int = 0, args=(),
                       params: CtxParams | None = None, store_phantom_samples: bool = False, batch_size: int | None = None) -> State:
    def single_sample(key):
        key, subkey = jax.random.split(key)
        U_sample = model.sample_U(subkey)
        log_L = model.log_likelihood(U_sample, args=args, params=params, allow_nan=False).astype(mp_policy.measure_dtype)
        num_likelihood_evaluations = jnp.array(1, dtype=mp_policy.count_dtype)
        carry = (key, U_sample, log_L, num_likelihood_evaluations)

        def cond_fn(carry):
            _, _, log_L, _ = carry
            return log_L <= -jnp.inf

        def body_fn(carry):
            key, _, _, num_likelihood_evaluations = carry
            key, subkey = jax.random.split(key)
            U_sample = model.sample_U(subkey)
            log_L = model.log_likelihood(U_sample, args=args, params=params, allow_nan=False).astype(mp_policy.measure_dtype)
            num_likelihood_evaluations += jnp.ones_like(num_likelihood_evaluations)
            return (key, U_sample, log_L, num_likelihood_evaluations)

        key, U_sample, log_L, num_likelihood_evaluations = jax.lax.while_loop(cond_fn, body_fn, carry)
        return U_sample, log_L, num_likelihood_evaluations

    U_samples, log_likelihoods, num_likelihood_evaluations = jax.lax.map(
        single_sample,
        jax.random.split(key, num_live_points),
        batch_size=batch_size
    )
    out_degree = jnp.zeros((num_live_points,), dtype=mp_policy.count_dtype)

    # extend each to max_samples
    def _concat(x, fill_value):
        return jax.tree.map(
            lambda x, y: jnp.concatenate([
                x,
                jnp.repeat(y[None, ...], repeats=(max_samples - num_live_points), axis=0)
            ], axis=0),
            x,
            fill_value
        )

    phantom_samples = PhantomSamples(
        valid_mask=jnp.full((num_live_points, num_phantom), False, dtype=mp_policy.bool_dtype),
        U_samples=jax.tree.map(lambda u: jnp.zeros((num_live_points, num_phantom) + u[0].shape, u.dtype), U_samples),
        log_L=jnp.full((num_live_points, num_phantom), -jnp.inf, dtype=mp_policy.measure_dtype)
    )
    phantom_samples.U_samples = None
    samples = Samples(
        log_L_constraints=jnp.full((num_live_points,), -jnp.inf, dtype=mp_policy.measure_dtype),
        log_likelihoods=log_likelihoods,
        U_samples=U_samples,
        out_degree=out_degree,
        num_likelihood_evaluations=num_likelihood_evaluations,
        phantom_samples=phantom_samples
    )

    sample_atom = Samples(
        log_likelihoods=jnp.asarray(jnp.inf, mp_policy.measure_dtype),
        out_degree=jnp.asarray(0, mp_policy.count_dtype),
        num_likelihood_evaluations=jnp.asarray(0, mp_policy.count_dtype),
        U_samples=jax.tree.map(lambda x: jnp.zeros_like(x[0]), samples.U_samples),
        phantom_samples=PhantomSamples(
            U_samples=jax.tree.map(lambda x: jnp.zeros_like(x[0]), samples.phantom_samples.U_samples),
            log_L=jnp.zeros_like(samples.phantom_samples.log_L[0]),
            valid_mask=jnp.zeros_like(samples.phantom_samples.valid_mask[0])
        ),
        log_L_constraints=jnp.asarray(jnp.inf, mp_policy.measure_dtype)
    )
    sample_atom.phantom_samples.U_samples = None

    samples = _concat(samples, sample_atom).sort()

    log_L_supremum_idx = jnp.argmax(log_likelihoods)
    log_L_supremum = log_likelihoods[log_L_supremum_idx]
    U_supremum = jax.tree.map(lambda u: u[log_L_supremum_idx], U_samples)
    # Sort samples into increasing log-likelihood order
    state = State(
        root_out_degree=jnp.array(num_live_points, dtype=mp_policy.count_dtype),
        samples=samples,
        num_samples=jnp.array(num_live_points, dtype=mp_policy.count_dtype),
        log_L_supremum=log_L_supremum,
        U_supremum=U_supremum,
        model=model,
        args=args,
        params=params,
        termination_reason=jnp.array(0, dtype=mp_policy.index_dtype)
    )
    return state


@partial(jax.jit, inline=True, static_argnames=['target_num_live_points', 'shell_size', 'batch_size'])
def _run_ns(key, state: State, target_num_live_points: int, shell_size: int, args=(),
            sampler: AbstractSampler | None = None,
            params=None,
            termination_condition: TerminationCondition | None = None,
            batch_size: int | None = None) -> State:
    """
    Perform a single nested sampling run.

    Args:
        key: PRNG key
        target_num_live_points: the number of live points to use off root
        shell_size: the number of samples to discard and replenish per iteration
        args: arguments to pass to the model
        sampler: the sampler to use to produce i.i.d. samples within likelihood constraints
        params: parameters to pass to the model
        termination_condition: the termination condition to use
        batch_size: how many likelihood evaluations to batch together

    Returns:
        A final state object containing all samples and relevant information for resuming, or evidence calculation.
    """

    # Algorithm
    # repeat until termination condition:
    # choose constraints: compute the recurrence K[i] = K[i-1] - 1 + d[i], and choose indexes where K[i] < num_live_points. I.e. we make each sample have at least a certain number of live points.
    # choose seeds: the any points with likelihoods > the contour, else reparent off root
    # sample (in parallel)

    class OuterCarry(NamedTuple):
        key: jax.Array
        state: State

    def outer_cond_fn(carry: OuterCarry) -> BoolArray:
        register = carry.state.compute_termination_register(target_num_live_points=target_num_live_points)
        done, _ = register.is_done(termination_condition)
        return jnp.logical_not(done)

    def outer_body_fn(outer_carry: OuterCarry) -> OuterCarry:
        # Select likelihood constraints to achieve minimum K[i]>target (randomly without replacement for simplicity)
        K_per_sample = outer_carry.state.samples.compute_num_live_points_per_sample(
            root_out_degree=outer_carry.state.root_out_degree,
            num_samples=outer_carry.state.num_samples
        )
        K_next_sample = K_per_sample - 1 + outer_carry.state.samples.out_degree
        select_weights = jnp.where(
            jnp.logical_and(
                jnp.arange(K_next_sample.shape[0]) < outer_carry.state.num_samples,
                K_next_sample < target_num_live_points
            ), 0, -jnp.inf)
        select_contours_key, key = jax.random.split(outer_carry.key, 2)
        parent_idxs = resample_indicies(select_contours_key, log_weights=select_weights, S=shell_size, replace=False)  # [S]
        proposed_log_L_constraints = outer_carry.state.samples.log_likelihoods[parent_idxs]  # [S]

        # TODO: give sampling a multi-ellipsoidal clustering, then use to guide sampling along preferential axes.

        def get_sample(key, log_L_constraint, parent_idx: IntArray):
            seed_key, sample_key = jax.random.split(key)
            # Get seed from samples
            i_start = jax.lax.while_loop(
                lambda i: (i < outer_carry.state.num_samples) & (outer_carry.state.samples.log_likelihoods[i] <= log_L_constraint),
                lambda i: i + 1,
                parent_idx + 1
            )
            no_seeds = i_start == outer_carry.state.num_samples
            log_L_constraint = jnp.where(no_seeds, jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype), log_L_constraint)
            delta_root_out_degree = jnp.where(no_seeds, 1, 0).astype(mp_policy.count_dtype)
            delta_parent_out_degree = jnp.where(no_seeds, 0, 1).astype(mp_policy.count_dtype)
            i_start = jnp.where(no_seeds, 0, i_start).astype(mp_policy.index_dtype)
            seed_select_idx = jax.random.randint(seed_key, (), i_start, outer_carry.state.num_samples)
            seed_point = SeedPoint(
                U0=jax.tree.map(lambda u: u[seed_select_idx], outer_carry.state.samples.U_samples),
                log_L0=outer_carry.state.samples.log_likelihoods[seed_select_idx]
            )
            return sampler.get_sample(
                sample_key, log_L_constraint, seed_point, args=args, params=params,
            ), (delta_root_out_degree, delta_parent_out_degree, log_L_constraint)

        key, subkey = jax.random.split(outer_carry.key)
        keys = jax.random.split(subkey, shell_size)
        (U_samples, log_likelihoods, num_likelihood_evaluations, phantom_samples), (
            delta_root_out_degree,
            delta_parent_out_degree,
            log_L_constraints
        ) = jax.lax.map(
            lambda x: get_sample(x[0], x[1], x[2]),
            (keys, proposed_log_L_constraints, parent_idxs),
            batch_size=batch_size
        )

        new_samples = Samples(
            log_L_constraints=log_L_constraints,
            log_likelihoods=log_likelihoods,
            U_samples=U_samples,
            out_degree=jnp.zeros((shell_size,), dtype=mp_policy.count_dtype),
            num_likelihood_evaluations=num_likelihood_evaluations,
            phantom_samples=phantom_samples
        )
        new_samples.phantom_samples.U_samples = None

        candidate_supremum_candidate_iid = jnp.argmax(new_samples.log_likelihoods)
        candidate_log_L_supremum = new_samples.log_likelihoods[candidate_supremum_candidate_iid]
        candidate_U_supremum = jax.tree.map(lambda u: u[candidate_supremum_candidate_iid], new_samples.U_samples)

        log_L_supremum = jnp.where(candidate_log_L_supremum > outer_carry.state.log_L_supremum, candidate_log_L_supremum, outer_carry.state.log_L_supremum)
        U_supremum = jax.tree.map(lambda u_new, u_old: jnp.where(candidate_log_L_supremum > outer_carry.state.log_L_supremum, u_new, u_old),
                                  candidate_U_supremum, outer_carry.state.U_supremum)
        state = State(
            root_out_degree=outer_carry.state.root_out_degree + jnp.sum(delta_root_out_degree),
            samples=outer_carry.state.samples.append_samples(
                insert_idx=outer_carry.state.num_samples,
                parent_idxs=parent_idxs,
                samples=new_samples,
                delta_parent_out_degree=delta_parent_out_degree,
            ).sort(),
            num_samples=outer_carry.state.num_samples + len(new_samples),
            log_L_supremum=log_L_supremum,
            U_supremum=U_supremum,
            model=outer_carry.state.model,
            args=outer_carry.state.args,
            params=outer_carry.state.params,
            termination_reason=outer_carry.state.termination_reason
        )

        return OuterCarry(key=key, state=state)

    init_outer_carry = OuterCarry(
        key=key,
        state=state
    )

    carry = jax.lax.while_loop(outer_cond_fn, outer_body_fn, init_outer_carry)
    return carry.state


def _sampler_uses_galilean(sampler: AbstractSampler | None) -> bool:
    return _as_mode_name(getattr(sampler, "trajectory", None)) in GALILEAN_TRAJECTORIES


def _phantom_coordinates_like_state(
        state: State,
        batch_size: int,
        num_phantom: int,
):
    if state.samples.phantom_samples.U_samples is None:
        return None
    sample_U = state.samples.U_samples
    return jax.tree.map(
        lambda u: jnp.zeros(
            (batch_size, num_phantom) + u.shape[1:],
            dtype=u.dtype,
        ),
        sample_U,
    )


def _state_direction_fitting_rows_and_weights(
        state: State,
) -> tuple[object, np.ndarray]:
    result = state.to_result().trim()
    log_weights = np.asarray(result.v3_log_posterior_weights, dtype=float)
    finite = np.isfinite(log_weights)
    weights = np.zeros_like(log_weights, dtype=float)
    if np.any(finite):
        max_log_weight = float(np.max(log_weights[finite]))
        raw_weights = np.exp(log_weights[finite] - max_log_weight)
        total = float(np.sum(raw_weights))
        if total > 0.0 and np.isfinite(total):
            weights[finite] = raw_weights / total
    num_samples = int(state.num_samples)
    rows = jax.tree.map(lambda u: u[:num_samples], state.samples.U_samples)
    return rows, weights[:num_samples]


def _run_ns_python(
        key,
        state: State,
        target_num_live_points: int,
        shell_size: int,
        args=(),
        sampler: AbstractSampler | None = None,
        params=None,
        termination_condition: TerminationCondition | None = None,
) -> State:
    """Python legacy loop for sampler modes that are not JAX-traceable."""
    while True:
        register = state.compute_termination_register(
            target_num_live_points=target_num_live_points,
        )
        done, termination_reason = register.is_done(termination_condition)
        if bool(done):
            state.termination_reason = termination_reason
            return state

        K_per_sample = state.samples.compute_num_live_points_per_sample(
            root_out_degree=state.root_out_degree,
            num_samples=state.num_samples,
        )
        K_next_sample = K_per_sample - 1 + state.samples.out_degree
        select_weights = jnp.where(
            jnp.logical_and(
                jnp.arange(K_next_sample.shape[0]) < state.num_samples,
                K_next_sample < target_num_live_points,
            ),
            0.0,
            -jnp.inf,
        )
        select_contours_key, key = jax.random.split(key)
        parent_idxs = resample_indicies(
            select_contours_key,
            log_weights=select_weights,
            S=shell_size,
            replace=False,
        )
        proposed_log_L_constraints = state.samples.log_likelihoods[parent_idxs]
        shell_keys = jax.random.split(key, shell_size + 1)
        key = shell_keys[0]
        proposal_keys = shell_keys[1:]
        num_samples = int(state.num_samples)
        sorted_log_l = np.asarray(state.samples.log_likelihoods[:num_samples])

        sample_results = []
        delta_root_out_degree = []
        delta_parent_out_degree = []
        log_L_constraints = []
        for task_key, log_L_constraint, parent_idx in zip(
                proposal_keys,
                proposed_log_L_constraints,
                parent_idxs,
        ):
            seed_key, sample_key = jax.random.split(task_key)
            i_start = int(
                np.searchsorted(
                    sorted_log_l,
                    float(log_L_constraint),
                    side="right",
                )
            )
            no_seeds = i_start == num_samples
            effective_log_L_constraint = (
                jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype)
                if no_seeds
                else jnp.asarray(log_L_constraint, dtype=mp_policy.measure_dtype)
            )
            seed_low = 0 if no_seeds else i_start
            seed_select_idx = int(
                jax.random.randint(seed_key, (), seed_low, num_samples)
            )
            seed_point = SeedPoint(
                U0=jax.tree.map(
                    lambda u: u[seed_select_idx],
                    state.samples.U_samples,
                ),
                log_L0=state.samples.log_likelihoods[seed_select_idx],
            )
            sample_results.append(
                sampler.get_sample(
                    sample_key,
                    effective_log_L_constraint,
                    seed_point,
                    args=args,
                    params=params,
                )
            )
            delta_root_out_degree.append(
                jnp.asarray(1 if no_seeds else 0, dtype=mp_policy.count_dtype)
            )
            delta_parent_out_degree.append(
                jnp.asarray(0 if no_seeds else 1, dtype=mp_policy.count_dtype)
            )
            log_L_constraints.append(effective_log_L_constraint)

        U_samples = jax.tree.map(
            lambda *values: jnp.stack(values, axis=0),
            *[item[0] for item in sample_results],
        )
        phantom_outputs = [item[3] for item in sample_results]
        if phantom_outputs[0].U_samples is None:
            phantom_U_samples = _phantom_coordinates_like_state(
                state,
                batch_size=shell_size,
                num_phantom=int(phantom_outputs[0].log_L.shape[0]),
            )
        else:
            phantom_U_samples = jax.tree.map(
                lambda *values: jnp.stack(values, axis=0),
                *[phantom.U_samples for phantom in phantom_outputs],
            )
        new_samples = Samples(
            log_L_constraints=jnp.stack(log_L_constraints, axis=0),
            log_likelihoods=jnp.stack(
                [item[1] for item in sample_results],
                axis=0,
            ),
            U_samples=U_samples,
            out_degree=jnp.zeros((shell_size,), dtype=mp_policy.count_dtype),
            num_likelihood_evaluations=jnp.stack(
                [item[2] for item in sample_results],
                axis=0,
            ).astype(mp_policy.count_dtype),
            phantom_samples=PhantomSamples(
                U_samples=phantom_U_samples,
                log_L=jnp.stack(
                    [phantom.log_L for phantom in phantom_outputs],
                    axis=0,
                ),
                valid_mask=jnp.stack(
                    [phantom.valid_mask for phantom in phantom_outputs],
                    axis=0,
                ),
            ),
        )
        new_samples.phantom_samples.U_samples = None

        candidate_idx = int(jnp.argmax(new_samples.log_likelihoods))
        candidate_log_L_supremum = new_samples.log_likelihoods[candidate_idx]
        candidate_U_supremum = jax.tree.map(
            lambda u: u[candidate_idx],
            new_samples.U_samples,
        )
        log_L_supremum = jnp.where(
            candidate_log_L_supremum > state.log_L_supremum,
            candidate_log_L_supremum,
            state.log_L_supremum,
        )
        U_supremum = jax.tree.map(
            lambda u_new, u_old: jnp.where(
                candidate_log_L_supremum > state.log_L_supremum,
                u_new,
                u_old,
            ),
            candidate_U_supremum,
            state.U_supremum,
        )
        state = State(
            root_out_degree=state.root_out_degree + jnp.sum(
                jnp.stack(delta_root_out_degree, axis=0),
            ),
            samples=state.samples.append_samples(
                insert_idx=state.num_samples,
                parent_idxs=parent_idxs,
                samples=new_samples,
                delta_parent_out_degree=jnp.stack(
                    delta_parent_out_degree,
                    axis=0,
                ),
            ).sort(),
            num_samples=state.num_samples + shell_size,
            log_L_supremum=log_L_supremum,
            U_supremum=U_supremum,
            model=state.model,
            args=state.args,
            params=state.params,
            termination_reason=state.termination_reason,
        )


@dataclasses.dataclass(slots=True)
class NestedSampler(PureDataclassPytree):
    model: Model
    target_num_live_points: int | None = None
    max_samples: int | None = None
    shell_size: int | None = None
    args: tuple = ()
    params: CtxParams | None = None
    sampler: AbstractSampler | None = None
    termination_condition: TerminationCondition | None = None
    store_phantom_samples: bool = False
    collect_phantom_samples: bool = False
    batch_size: int | None = None

    def __post_init__(self):
        U_ndims = 0
        if self.target_num_live_points is None or self.max_samples is None or self.shell_size is None or self.sampler is None:
            U_ndims = int(self.model.U_ndims(self.args, self.params))
        if self.target_num_live_points is None:
            self.target_num_live_points = 20 * U_ndims
        if self.max_samples is None:
            self.max_samples = 10000 * U_ndims
        if self.shell_size is None:
            self.shell_size = max(1, self.target_num_live_points // 2)
        max_samples = jnp.asarray(self.max_samples, dtype=mp_policy.count_dtype)
        if self.termination_condition is None:
            self.termination_condition = TerminationCondition(dlogZ=1e-2, max_samples=max_samples)
        elif self.termination_condition.max_samples is None:
            self.termination_condition.max_samples = max_samples
        else:
            self.termination_condition.max_samples = jnp.minimum(self.termination_condition.max_samples, max_samples)
        if self.sampler is None:
            self.sampler = UniDimSliceSampler(model=self.model, num_slices=max(1, 100 * U_ndims), phantom_burn_in=max(1, 20 * U_ndims), no_step_out=True,
                                              collect_phantom_samples=self.collect_phantom_samples)

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(this, ['target_num_live_points', 'max_samples', 'shell_size', 'store_phantom_samples', 'batch_size'])

    @classmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        return cls.build_unflatten(aux_data, children)

    def run(self, key: PRNGKey | None = None) -> State:
        """
        Creates an initial state with the legacy fixed-live-point compatibility
        path, and performs sampling until the termination condition is met.

        Use ``run_until_goal`` for the v3 allocation-target execution path.

        Args:
            key: PRNGKey to use for sampling

        Returns:
            the final state after running nested sampling, which can be used for evidence calculation or resuming.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        return _run(self, key)

    def resume(self, state: State, key: PRNGKey | None = None) -> State:
        """
        Resumes using the legacy fixed-live-point compatibility path until the
        termination condition is met.

        Use ``resume_until_goal`` for the v3 allocation-target execution path.

        Args:
            state: the state to resume from, which should be a valid state returned by a previous call to run or resume. The state should not have met the termination condition yet.
            key: the PRNGKey to use for sampling

        Returns:
            the final state after running nested sampling, which can be used for evidence calculation or resuming.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        if _sampler_uses_galilean(self.sampler):
            return _run_ns_python(
                key=key,
                state=state,
                target_num_live_points=int(self.target_num_live_points),
                shell_size=int(self.shell_size),
                args=self.args,
                sampler=self.sampler,
                params=self.params,
                termination_condition=self.termination_condition,
            )
        return _run_ns(
            key=key,
            state=state,
            target_num_live_points=int(self.target_num_live_points),
            shell_size=int(self.shell_size),
            args=self.args,
            sampler=self.sampler,
            params=self.params,
            termination_condition=self.termination_condition,
            batch_size=self.batch_size
        )

    def _sample_v3_root_state(self, key: PRNGKey) -> State:
        """Draw only the v3 root children from the sentinel contour."""
        root_count = int(self.target_num_live_points)
        adaptation_context = None
        if _direction_kernel_requests_adaptation(self.sampler):
            adaptation_context = _identity_direction_context(
                d_dim=int(self.model.U_ndims(self.args, self.params)),
                allocation_target=None,
            )
        if _sampler_uses_galilean(self.sampler):
            adaptation_context = {
                "force_jax_galilean": True,
                "direction_adaptation_context": adaptation_context,
            }
        outputs = []
        for sample_idx, sample_key in enumerate(jax.random.split(key, root_count)):
            seed_key, sampler_key = jax.random.split(sample_key, 2)
            seed_U = self.model.sample_U(
                seed_key,
                args=self.args,
                params=self.params,
            )
            seed_log_L = self.model.log_likelihood(
                seed_U,
                args=self.args,
                params=self.params,
                allow_nan=False,
            ).astype(mp_policy.measure_dtype)
            if _sampler_uses_galilean(self.sampler):
                num_phantom = self.sampler.num_phantom()
                outputs.append(
                    (
                        seed_U,
                        seed_log_L,
                        jnp.ones((), dtype=mp_policy.count_dtype),
                        PhantomSamples(
                            U_samples=None,
                            log_L=jnp.zeros(
                                (num_phantom,),
                                dtype=mp_policy.measure_dtype,
                            ),
                            valid_mask=jnp.zeros(
                                (num_phantom,),
                                dtype=mp_policy.bool_dtype,
                            ),
                        ),
                    )
                )
            else:
                outputs.append(
                    self._sample_constrained(
                        sampler_key,
                        jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype),
                        SeedPoint(U0=seed_U, log_L0=seed_log_L),
                        requested_parent_idx=-1,
                        effective_parent_idx=-1,
                        accepted_parent_idx=-1,
                        parent_work_id=sample_idx,
                        adaptation_context=adaptation_context,
                    )
                )

        U_samples = jax.tree.map(
            lambda *values: jnp.stack(values, axis=0),
            *[output[0] for output in outputs],
        )
        log_likelihoods = jnp.stack([output[1] for output in outputs], axis=0)
        num_likelihood_evaluations = jnp.stack(
            [output[2] for output in outputs],
            axis=0,
        ).astype(mp_policy.count_dtype)
        phantom_outputs = [output[3] for output in outputs]
        first_phantom = phantom_outputs[0]
        if first_phantom.U_samples is None:
            phantom_U_samples = None
        else:
            phantom_U_samples = jax.tree.map(
                lambda *values: jnp.stack(values, axis=0),
                *[phantom.U_samples for phantom in phantom_outputs],
            )
        root_samples = Samples(
            log_L_constraints=jnp.full(
                (root_count,),
                -jnp.inf,
                dtype=mp_policy.measure_dtype,
            ),
            log_likelihoods=log_likelihoods,
            U_samples=U_samples,
            out_degree=jnp.zeros((root_count,), dtype=mp_policy.count_dtype),
            num_likelihood_evaluations=num_likelihood_evaluations,
            phantom_samples=PhantomSamples(
                U_samples=phantom_U_samples,
                valid_mask=jnp.stack(
                    [phantom.valid_mask for phantom in phantom_outputs],
                    axis=0,
                ),
                log_L=jnp.stack(
                    [phantom.log_L for phantom in phantom_outputs],
                    axis=0,
                ),
            ),
        )
        root_work_batch = _root_core_work_batch(root_count)
        root_parent_work = ParentWork(
            parent_idxs=jnp.full(
                (root_count,),
                -1,
                dtype=mp_policy.index_dtype,
            ),
            parent_log_L_constraints=root_samples.log_L_constraints,
            target_block_idxs=jnp.full(
                (root_count,),
                -1,
                dtype=mp_policy.index_dtype,
            ),
            parent_block_idxs=jnp.full(
                (root_count,),
                -1,
                dtype=mp_policy.index_dtype,
            ),
            fallback_to_root=jnp.ones(
                (root_count,),
                dtype=mp_policy.bool_dtype,
            ),
        )
        _notify_core_boundary(
            self,
            work_batch=root_work_batch,
            result_batch=_core_result_batch_from_samples(
                work_batch=root_work_batch,
                parent_work=root_parent_work,
                new_samples=root_samples,
            ),
        )
        samples = root_samples.resize(int(self.max_samples)).sort()
        if not self.store_phantom_samples:
            samples.phantom_samples.U_samples = None

        log_L_supremum_idx = jnp.argmax(log_likelihoods)
        return State(
            root_out_degree=jnp.asarray(
                root_count,
                dtype=mp_policy.count_dtype,
            ),
            samples=samples,
            num_samples=jnp.asarray(root_count, dtype=mp_policy.count_dtype),
            log_L_supremum=log_likelihoods[log_L_supremum_idx],
            U_supremum=jax.tree.map(
                lambda u: u[log_L_supremum_idx],
                U_samples,
            ),
            model=self.model,
            args=self.args,
            params=self.params,
            termination_reason=jnp.asarray(0, dtype=mp_policy.index_dtype),
        )

    def _sample_constrained(
            self,
            key: PRNGKey,
            log_L_constraint,
            seed_point: SeedPoint,
            *,
            requested_parent_idx: int,
            effective_parent_idx: int,
            accepted_parent_idx: int,
            parent_work_id: int | None = None,
            adaptation_context=None,
    ):
        del requested_parent_idx, effective_parent_idx
        del accepted_parent_idx, parent_work_id
        if adaptation_context is None:
            return self.sampler.get_sample(
                key,
                log_L_constraint,
                seed_point,
                args=self.args,
                params=self.params,
            )
        return self.sampler.get_sample(
            key,
            log_L_constraint,
            seed_point,
            args=self.args,
            params=self.params,
            adaptation_context=adaptation_context,
        )

    def _sample_parent_work(
            self,
            key: PRNGKey,
            state: State,
            parent_work: ParentWork,
            adaptation_context=None,
    ) -> tuple[ParentWork, Samples]:
        outputs = []
        parent_idxs = []
        parent_log_L_constraints = []
        target_block_idxs = []
        parent_block_idxs = []
        fallback_to_root = []
        num_samples = int(state.num_samples)
        active_log_likelihoods = np.asarray(
            state.samples.log_likelihoods[:num_samples],
        )
        sortable_log_likelihoods = np.where(
            np.isnan(active_log_likelihoods),
            -np.inf,
            active_log_likelihoods,
        )
        sorted_active_offsets = np.argsort(
            sortable_log_likelihoods,
            kind="stable",
        )
        sorted_active_log_likelihoods = (
            sortable_log_likelihoods[sorted_active_offsets]
        )

        for work_idx, sample_key in enumerate(
                jax.random.split(key, int(parent_work.parent_idxs.shape[0]))
        ):
            seed_key, sampler_key = jax.random.split(sample_key, 2)
            constraint = parent_work.parent_log_L_constraints[work_idx]
            constraint_value = float(constraint)
            if np.isnan(constraint_value):
                first_seed_offset = num_samples
            else:
                first_seed_offset = int(
                    np.searchsorted(
                        sorted_active_log_likelihoods,
                        constraint_value,
                        side="right",
                    )
                )
            no_seed = first_seed_offset == num_samples
            if no_seed:
                constraint = jnp.asarray(
                    -jnp.inf,
                    dtype=mp_policy.measure_dtype,
                )
                first_seed_offset = int(
                    np.searchsorted(
                        sorted_active_log_likelihoods,
                        float(constraint),
                        side="right",
                    )
                )

            candidate_count = num_samples - first_seed_offset
            seed_choice = jax.random.randint(
                seed_key,
                (),
                minval=0,
                maxval=candidate_count,
            )
            seed_offset = first_seed_offset + int(seed_choice)
            seed_idx = int(sorted_active_offsets[seed_offset])
            seed_point = SeedPoint(
                U0=jax.tree.map(
                    lambda u: u[seed_idx],
                    state.samples.U_samples,
                ),
                log_L0=state.samples.log_likelihoods[seed_idx],
            )
            requested_parent_idx = int(parent_work.parent_idxs[work_idx])
            effective_parent_idx = -1 if no_seed else requested_parent_idx
            parent_work_id = _core_parent_work_id_for_owner(self, work_idx)
            outputs.append(
                self._sample_constrained(
                    sampler_key,
                    constraint,
                    seed_point,
                    requested_parent_idx=requested_parent_idx,
                    effective_parent_idx=effective_parent_idx,
                    accepted_parent_idx=effective_parent_idx,
                    parent_work_id=parent_work_id,
                    adaptation_context=adaptation_context,
                )
            )
            parent_idxs.append(
                effective_parent_idx
            )
            parent_log_L_constraints.append(float(constraint))
            target_block_idxs.append(
                int(parent_work.target_block_idxs[work_idx])
            )
            parent_block_idxs.append(
                (
                    -1
                    if no_seed
                    else int(parent_work.parent_block_idxs[work_idx])
                )
            )
            fallback_to_root.append(
                bool(no_seed or bool(parent_work.fallback_to_root[work_idx]))
            )

        U_samples = jax.tree.map(
            lambda *values: jnp.stack(values, axis=0),
            *[output[0] for output in outputs],
        )
        phantom_outputs = [output[3] for output in outputs]
        first_phantom = phantom_outputs[0]
        if first_phantom.U_samples is None:
            phantom_U_samples = _phantom_coordinates_like_state(
                state,
                batch_size=len(outputs),
                num_phantom=int(first_phantom.log_L.shape[0]),
            )
        else:
            phantom_U_samples = jax.tree.map(
                lambda *values: jnp.stack(values, axis=0),
                *[phantom.U_samples for phantom in phantom_outputs],
            )
        adjusted_parent_work = ParentWork(
            parent_idxs=jnp.asarray(parent_idxs, dtype=mp_policy.index_dtype),
            parent_log_L_constraints=jnp.asarray(
                parent_log_L_constraints,
                dtype=mp_policy.measure_dtype,
            ),
            target_block_idxs=jnp.asarray(
                target_block_idxs,
                dtype=mp_policy.index_dtype,
            ),
            parent_block_idxs=jnp.asarray(
                parent_block_idxs,
                dtype=mp_policy.index_dtype,
            ),
            fallback_to_root=jnp.asarray(
                fallback_to_root,
                dtype=mp_policy.bool_dtype,
            ),
        )
        new_samples = Samples(
            log_L_constraints=adjusted_parent_work.parent_log_L_constraints,
            log_likelihoods=jnp.stack(
                [output[1] for output in outputs],
                axis=0,
            ),
            U_samples=U_samples,
            out_degree=jnp.zeros((len(outputs),), dtype=mp_policy.count_dtype),
            num_likelihood_evaluations=jnp.stack(
                [output[2] for output in outputs],
                axis=0,
            ).astype(mp_policy.count_dtype),
            phantom_samples=PhantomSamples(
                U_samples=phantom_U_samples,
                valid_mask=jnp.stack(
                    [phantom.valid_mask for phantom in phantom_outputs],
                    axis=0,
                ),
                log_L=jnp.stack(
                    [phantom.log_L for phantom in phantom_outputs],
                    axis=0,
                ),
            ),
        )
        if (
                state.samples.phantom_samples.U_samples is None
                and not self.store_phantom_samples
        ):
            new_samples.phantom_samples.U_samples = None
        return adjusted_parent_work, new_samples

    def _depth_condition_done(
            self,
            state: State,
            depth_cond: TerminationCondition,
            *,
            iteration: int | None = None,
            diagnostics_builder: _ExecutionDiagnosticsBuilder | None = None,
    ) -> bool:
        unsupported_fields = [
            field.name
            for field in dataclasses.fields(depth_cond)
            if (
                    field.name
                    not in {
                        "max_samples",
                        "max_num_likelihood_evaluations",
                        "dlogZ",
                    }
                    and getattr(depth_cond, field.name) is not None
            )
        ]
        if unsupported_fields:
            unsupported = ", ".join(unsupported_fields)
            raise ValueError(
                f"Unsupported v3 depth condition field(s): {unsupported}."
            )
        register = None
        if (
                diagnostics_builder is not None
                and depth_cond.max_num_likelihood_evaluations is not None
        ):
            register = state.compute_termination_register(
                target_num_live_points=int(self.target_num_live_points),
            )
            register.is_done(depth_cond)
        dlogz_summary = None
        if depth_cond.dlogZ is not None:
            dlogz_summary = _compute_dlogz_depth_summary(
                state,
                threshold=float(depth_cond.dlogZ),
            )
        if diagnostics_builder is not None and iteration is not None:
            _record_depth_condition_summaries(
                diagnostics_builder,
                iteration=iteration,
                state=state,
                depth_cond=depth_cond,
                register=register,
                dlogz_summary=dlogz_summary,
            )

        if depth_cond.max_samples is not None:
            if int(state.num_samples) >= int(depth_cond.max_samples):
                return True
        if depth_cond.max_num_likelihood_evaluations is not None:
            num_samples = int(state.num_samples)
            num_likelihood_evaluations = int(
                np.sum(
                    np.asarray(state.samples.num_likelihood_evaluations)[
                        :num_samples
                    ]
                )
            )
            if (
                    num_likelihood_evaluations
                    >= int(depth_cond.max_num_likelihood_evaluations)
            ):
                return True
        if depth_cond.dlogZ is not None:
            if dlogz_summary is None:
                dlogz_summary = _compute_dlogz_depth_summary(
                    state,
                    threshold=float(depth_cond.dlogZ),
                )
            return bool(dlogz_summary[1])
        return False

    def run_until_goal(
            self,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition,
            allocation_target: AllocationTarget = "uniform",
            key: PRNGKey | None = None,
            max_goal_iterations: int = 100,
            **options,
    ) -> State:
        allocation_target = validate_allocation_target(allocation_target)
        delta_K, posterior_conservative = _parse_v3_options(options)
        if key is None:
            key = jax.random.PRNGKey(42)
        key, init_key = jax.random.split(key)
        state = self._sample_v3_root_state(init_key)
        return self._resume_until_goal(
            state=state,
            goal_cond=goal_cond,
            depth_cond=depth_cond,
            allocation_target=allocation_target,
            delta_K=delta_K,
            posterior_conservative=posterior_conservative,
            key=key,
            max_goal_iterations=max_goal_iterations,
        )

    def resume_until_goal(
            self,
            state: State,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition,
            allocation_target: AllocationTarget = "uniform",
            key: PRNGKey | None = None,
            max_goal_iterations: int = 100,
            **options,
    ) -> State:
        allocation_target = validate_allocation_target(allocation_target)
        delta_K, posterior_conservative = _parse_v3_options(options)
        if key is None:
            key = jax.random.PRNGKey(42)
        return self._resume_until_goal(
            state=state,
            goal_cond=goal_cond,
            depth_cond=depth_cond,
            allocation_target=allocation_target,
            delta_K=delta_K,
            posterior_conservative=posterior_conservative,
            key=key,
            max_goal_iterations=max_goal_iterations,
        )

    def _resume_until_goal(
            self,
            state: State,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition,
            allocation_target: AllocationTarget,
            delta_K: int,
            posterior_conservative: bool,
            key: PRNGKey,
            max_goal_iterations: int,
    ) -> State:
        current = dataclasses.replace(state, execution_diagnostics=None)
        initial_root_out_degree = int(state.root_out_degree)
        direction_adaptation_enabled = _direction_kernel_requests_adaptation(
            self.sampler
        )
        direction_d_dim = int(self.model.U_ndims(self.args, self.params))
        direction_coordinator = DirectionKernelAdaptationCoordinator.initial()
        direction_distinct_shell_count = 0
        direction_last_successful_shell_count = 0
        direction_last_log_likelihood = None
        diagnostics_builder = _ExecutionDiagnosticsBuilder(
            allocation_target=allocation_target,
            target_num_live_points=int(self.target_num_live_points),
            shell_size=int(self.shell_size),
            sampler=self.sampler,
        )
        prior_diagnostics = getattr(state, "execution_diagnostics", None)
        if prior_diagnostics is not None:
            prior_parent_selection = getattr(
                prior_diagnostics,
                "parent_selection",
                None,
            )
            if prior_parent_selection is not None:
                diagnostics_builder.requested_parent_indices.extend(
                    int(x)
                    for x in np.asarray(
                        prior_parent_selection.requested_parent_indices
                    )
                )
                diagnostics_builder.effective_parent_indices.extend(
                    int(x)
                    for x in np.asarray(
                        prior_parent_selection.effective_parent_indices
                    )
                )
                diagnostics_builder.accepted_parent_indices.extend(
                    int(x)
                    for x in np.asarray(
                        prior_parent_selection.accepted_parent_indices
                    )
                )
                diagnostics_builder.sentinel_fallback_indices.extend(
                    int(x)
                    for x in np.asarray(
                        prior_parent_selection.sentinel_fallback_indices
                    )
                )

        def finish(final_state: State) -> State:
            diagnostics = _build_execution_diagnostics(
                self,
                diagnostics_builder,
                final_state,
            )
            return dataclasses.replace(
                final_state,
                execution_diagnostics=diagnostics,
            )

        goal_iteration = 0
        allocation_iteration = 0
        allocation_iteration_limit = (
                int(self.max_samples)
                + int(state.root_out_degree)
                + int(max_goal_iterations)
        )
        while goal_iteration < int(max_goal_iterations):
            goal_reached = bool(goal_cond(current))
            diagnostics_builder.goal_condition_summaries.append(
                DiagnosticConditionSummary(
                    condition_name="goal_cond",
                    iteration=len(
                        diagnostics_builder.goal_condition_summaries
                    ),
                    num_samples=int(current.num_samples),
                    value=int(current.num_samples),
                    satisfied=goal_reached,
                )
            )
            goal_iteration += 1
            if goal_reached:
                return finish(current)
            if self._depth_condition_done(
                    current,
                    depth_cond,
            ):
                return finish(current)
            if int(current.num_samples) >= int(self.max_samples):
                return finish(current)
            while not self._depth_condition_done(
                    current,
                    depth_cond,
                    iteration=len(
                        {
                            summary.iteration
                            for summary in (
                                diagnostics_builder
                                .depth_condition_summaries
                            )
                        }
                    ),
                    diagnostics_builder=diagnostics_builder,
            ):
                remaining = int(self.max_samples) - int(current.num_samples)
                if remaining <= 0:
                    return finish(current)
                shell_capacity = min(int(self.shell_size), remaining)
                if type(self) is NestedSampler:
                    depth_max_samples = (
                        int(self.max_samples)
                        if depth_cond.max_samples is None
                        else min(int(depth_cond.max_samples), int(self.max_samples))
                    )
                    depth_remaining = depth_max_samples - int(current.num_samples)
                    if depth_remaining <= 0:
                        continue
                    epoch_capacity = min(shell_capacity, depth_remaining)
                    max_epoch_steps = min(
                        4,
                        max(
                            1,
                            (depth_remaining + epoch_capacity - 1)
                            // epoch_capacity,
                        ),
                    )
                    if _sampler_uses_galilean(self.sampler):
                        max_epoch_steps = 1
                    plan = build_allocation_plan(
                        state=current,
                        allocation_target=allocation_target,
                        iteration=allocation_iteration,
                        delta_K=delta_K,
                        root_out_degree=initial_root_out_degree,
                        posterior_conservative=posterior_conservative,
                    )
                    diagnostics_builder.allocation_target_summaries.append(
                        _allocation_summary(
                            allocation_target=allocation_target,
                            iteration=allocation_iteration,
                            plan=plan,
                        )
                    )
                    direction_adaptation_context = None
                    if direction_adaptation_enabled:
                        shells_since_success = (
                                direction_distinct_shell_count
                                - direction_last_successful_shell_count
                        )
                        fit_eligible = (
                                direction_coordinator.successful_update_count == 0
                                or shells_since_success
                                >= direction_coordinator.update_every_shells
                        )
                        if fit_eligible:
                            key, fit_key = jax.random.split(key)
                            fit_rows, fit_weights = (
                                _state_direction_fitting_rows_and_weights(current)
                            )
                            fit_result = (
                                direction_coordinator.request_direction_kernel_fit(
                                    DirectionKernelFitRequest(
                                        shell_epoch=(
                                            direction_distinct_shell_count
                                        ),
                                        allocation_target=allocation_target,
                                        samples_U=fit_rows,
                                        posterior_weights=fit_weights,
                                        key=fit_key,
                                    )
                                )
                            )
                            diagnostics_builder.direction_adaptation_diagnostics.append(
                                fit_result.diagnostics
                            )
                            direction_coordinator = fit_result.coordinator
                            if not fit_result.diagnostics.fallback_active:
                                direction_last_successful_shell_count = (
                                    direction_distinct_shell_count
                                )

                        snapshot = direction_coordinator.prepare_dispatch_snapshot(
                            DirectionKernelDispatchRequest(
                                chain_id=(
                                    f"{allocation_target}-"
                                    f"{direction_distinct_shell_count}"
                                ),
                                shell_epoch=direction_distinct_shell_count,
                                allocation_target=allocation_target,
                            )
                        )
                        direction_adaptation_context = _ensure_direction_context(
                            context=snapshot.direction_adaptation_context(),
                            d_dim=direction_d_dim,
                            kernel_version=snapshot.kernel_version,
                            allocation_target=allocation_target,
                        )
                    before = int(current.num_samples)
                    epoch_result = _pure_core_depth_epoch_jax(
                        key,
                        current,
                        jnp.asarray(
                            allocation_iteration,
                            dtype=mp_policy.index_dtype,
                        ),
                        jnp.asarray(
                            depth_max_samples,
                            dtype=mp_policy.count_dtype,
                        ),
                        jnp.asarray(
                            (
                                0
                                if depth_cond.max_num_likelihood_evaluations
                                is None
                                else int(
                                    depth_cond
                                    .max_num_likelihood_evaluations
                                )
                            ),
                            dtype=mp_policy.count_dtype,
                        ),
                        jnp.asarray(
                            0.0 if depth_cond.dlogZ is None else float(
                                depth_cond.dlogZ
                            ),
                            dtype=mp_policy.measure_dtype,
                        ),
                        jnp.asarray(
                            initial_root_out_degree,
                            dtype=mp_policy.count_dtype,
                        ),
                        jnp.asarray(delta_K, dtype=mp_policy.count_dtype),
                        self.sampler,
                        self.args,
                        self.params,
                        _jax_direction_adaptation_context(
                            direction_adaptation_context
                        ),
                        allocation_target=allocation_target,
                        posterior_conservative=posterior_conservative,
                        capacity=epoch_capacity,
                        max_epoch_steps=max_epoch_steps,
                        has_max_num_likelihood_evaluations=(
                            depth_cond.max_num_likelihood_evaluations
                            is not None
                        ),
                        has_dlogz=depth_cond.dlogZ is not None,
                    )
                    key = epoch_result.key
                    current = epoch_result.state
                    allocation_iteration = int(
                        np.asarray(epoch_result.allocation_iteration)
                    )
                    num_epoch_steps = int(
                        np.asarray(epoch_result.history.num_steps)
                    )
                    for epoch_step in range(num_epoch_steps):
                        if not bool(
                                np.asarray(
                                    epoch_result.history.valid_step[epoch_step]
                                )
                        ):
                            continue
                        work_batch = _slice_core_work_batch_history(
                            epoch_result.history.work_batch,
                            epoch_step,
                        )
                        num_work_items = int(
                            np.asarray(work_batch.num_work_items)
                        )
                        if num_work_items == 0:
                            continue
                        requested_parent_work = (
                            _parent_work_from_core_work_batch(work_batch)
                        )
                        accepted_parent_work = _slice_parent_work_history(
                            epoch_result.history.parent_work,
                            epoch_step,
                            num_work_items,
                        )
                        _record_parent_selection(
                            diagnostics_builder,
                            requested_parent_work=requested_parent_work,
                            accepted_parent_work=accepted_parent_work,
                        )
                        if direction_adaptation_enabled:
                            shell_values = np.asarray(
                                (
                                    accepted_parent_work
                                    .parent_log_L_constraints
                                ),
                                dtype=float,
                            )
                            for shell_value in np.unique(shell_values):
                                if not np.isfinite(shell_value):
                                    continue
                                if (
                                        direction_last_log_likelihood is None
                                        or float(shell_value)
                                        != float(direction_last_log_likelihood)
                                ):
                                    direction_distinct_shell_count += 1
                                    direction_last_log_likelihood = float(
                                        shell_value
                                    )
                    if allocation_iteration > allocation_iteration_limit:
                        return finish(current)
                    if int(current.num_samples) <= before and num_epoch_steps == 0:
                        return finish(current)
                    continue
                plan = build_allocation_plan(
                    state=current,
                    allocation_target=allocation_target,
                    iteration=allocation_iteration,
                    delta_K=delta_K,
                    root_out_degree=initial_root_out_degree,
                    posterior_conservative=posterior_conservative,
                )
                diagnostics_builder.allocation_target_summaries.append(
                    _allocation_summary(
                        allocation_target=allocation_target,
                        iteration=allocation_iteration,
                        plan=plan,
                    )
                )
                direction_adaptation_context = None
                if direction_adaptation_enabled:
                    shells_since_success = (
                            direction_distinct_shell_count
                            - direction_last_successful_shell_count
                    )
                    fit_eligible = (
                            direction_coordinator.successful_update_count == 0
                            or shells_since_success
                            >= direction_coordinator.update_every_shells
                    )
                    if fit_eligible:
                        key, fit_key = jax.random.split(key)
                        fit_rows, fit_weights = (
                            _state_direction_fitting_rows_and_weights(current)
                        )
                        fit_result = (
                            direction_coordinator.request_direction_kernel_fit(
                                DirectionKernelFitRequest(
                                    shell_epoch=direction_distinct_shell_count,
                                    allocation_target=allocation_target,
                                    samples_U=fit_rows,
                                    posterior_weights=fit_weights,
                                    key=fit_key,
                                )
                            )
                        )
                        diagnostics_builder.direction_adaptation_diagnostics.append(
                            fit_result.diagnostics
                        )
                        direction_coordinator = fit_result.coordinator
                        if not fit_result.diagnostics.fallback_active:
                            direction_last_successful_shell_count = (
                                direction_distinct_shell_count
                            )

                    snapshot = direction_coordinator.prepare_dispatch_snapshot(
                        DirectionKernelDispatchRequest(
                            chain_id=(
                                f"{allocation_target}-"
                                f"{direction_distinct_shell_count}"
                            ),
                            shell_epoch=direction_distinct_shell_count,
                            allocation_target=allocation_target,
                        )
                    )
                    direction_adaptation_context = _ensure_direction_context(
                        context=snapshot.direction_adaptation_context(),
                        d_dim=direction_d_dim,
                        kernel_version=snapshot.kernel_version,
                        allocation_target=allocation_target,
                    )
                key, transition_key = jax.random.split(key)
                if type(self) is NestedSampler:
                    (
                        next_current,
                        work_batch,
                        result_batch,
                        parent_work,
                        new_samples,
                    ) = _call_pure_core_transition_jax(
                        key=transition_key,
                        state=current,
                        plan=plan,
                        num_parents=jnp.asarray(
                            shell_capacity,
                            dtype=mp_policy.count_dtype,
                        ),
                        sampler=self.sampler,
                        args=self.args,
                        params=self.params,
                        adaptation_context=_jax_direction_adaptation_context(
                            direction_adaptation_context
                        ),
                        capacity=shell_capacity,
                    )
                    requested_parent_work = _parent_work_from_core_work_batch(
                        work_batch
                    )
                    accepted_parent_work = ParentWork(
                        parent_idxs=parent_work.parent_idxs[
                            :int(np.asarray(work_batch.num_work_items))
                        ],
                        parent_log_L_constraints=(
                            parent_work.parent_log_L_constraints[
                                :int(np.asarray(work_batch.num_work_items))
                            ]
                        ),
                        target_block_idxs=parent_work.target_block_idxs[
                            :int(np.asarray(work_batch.num_work_items))
                        ],
                        parent_block_idxs=parent_work.parent_block_idxs[
                            :int(np.asarray(work_batch.num_work_items))
                        ],
                        fallback_to_root=parent_work.fallback_to_root[
                            :int(np.asarray(work_batch.num_work_items))
                        ],
                    )
                else:
                    key, plan_key, sample_key = jax.random.split(key, 3)
                    work_batch = _plan_core_work_batch_jax(
                        plan_key,
                        current,
                        plan,
                        jnp.asarray(
                            shell_capacity,
                            dtype=mp_policy.count_dtype,
                        ),
                        capacity=shell_capacity,
                    )
                    requested_parent_work = _parent_work_from_core_work_batch(
                        work_batch
                    )
                    if int(np.asarray(work_batch.num_work_items)) == 0:
                        allocation_iteration += 1
                        if allocation_iteration > allocation_iteration_limit:
                            return finish(current)
                        continue
                    _notify_core_work_batch(self, work_batch)
                    try:
                        accepted_parent_work, new_samples = self._sample_parent_work(
                            key=sample_key,
                            state=current,
                            parent_work=requested_parent_work,
                            adaptation_context=direction_adaptation_context,
                        )
                    finally:
                        _clear_core_work_batch(self)
                    result_batch = _core_result_batch_from_samples(
                        work_batch=work_batch,
                        parent_work=accepted_parent_work,
                        new_samples=new_samples,
                    )
                    next_current = accept_parent_work(
                        state=current,
                        parent_work=accepted_parent_work,
                        new_samples=new_samples,
                    )
                if int(np.asarray(work_batch.num_work_items)) == 0:
                    allocation_iteration += 1
                    if allocation_iteration > allocation_iteration_limit:
                        return finish(current)
                    continue
                _notify_core_boundary(
                    self,
                    work_batch=work_batch,
                    result_batch=result_batch,
                )
                _record_parent_selection(
                    diagnostics_builder,
                    requested_parent_work=requested_parent_work,
                    accepted_parent_work=accepted_parent_work,
                )
                before = int(current.num_samples)
                current = next_current
                if direction_adaptation_enabled:
                    shell_values = np.asarray(
                        accepted_parent_work.parent_log_L_constraints,
                        dtype=float,
                    )
                    for shell_value in np.unique(shell_values):
                        if not np.isfinite(shell_value):
                            continue
                        if (
                                direction_last_log_likelihood is None
                                or float(shell_value)
                                != float(direction_last_log_likelihood)
                        ):
                            direction_distinct_shell_count += 1
                            direction_last_log_likelihood = float(shell_value)
                if int(current.num_samples) <= before:
                    return finish(current)
        return finish(current)

    def _with_depth_condition(
            self,
            depth_cond: TerminationCondition,
    ) -> "NestedSampler":
        return dataclasses.replace(
            self,
            termination_condition=depth_cond,
            store_phantom_samples=False,
        )


NestedSampler.register_pytree()


def _run(self: NestedSampler, key) -> State:
    key, init_key = jax.random.split(key)
    state = _sample_init_state(
        key=init_key,
        num_live_points=int(self.target_num_live_points),
        max_samples=int(self.max_samples),
        model=self.model,
        num_phantom=(
            int(self.sampler.num_phantom()) if self.sampler is not None else 0
        ),
        args=self.args,
        params=self.params,
        store_phantom_samples=self.store_phantom_samples,
        batch_size=self.batch_size
    )
    # `_sample_init_state` is jitted and may reconstruct dataclass-pytree model
    # metadata with tracer leaves. Restore the Python metadata before the state
    # enters `_run_ns`, whose while-loop carry compares static metadata.
    state = dataclasses.replace(
        state,
        model=self.model,
        args=self.args,
        params=self.params,
    )
    if _sampler_uses_galilean(self.sampler):
        state = _run_ns_python(
            key=key,
            state=state,
            target_num_live_points=int(self.target_num_live_points),
            shell_size=int(self.shell_size),
            args=self.args,
            sampler=self.sampler,
            params=self.params,
            termination_condition=self.termination_condition,
        )
        return dataclasses.replace(
            state,
            model=self.model,
            args=self.args,
            params=self.params,
        )
    state = _run_ns(
        key=key,
        state=state,
        target_num_live_points=int(self.target_num_live_points),
        shell_size=int(self.shell_size),
        args=self.args,
        sampler=self.sampler,
        params=self.params,
        termination_condition=self.termination_condition,
        batch_size=self.batch_size
    )
    return dataclasses.replace(
        state,
        model=self.model,
        args=self.args,
        params=self.params,
    )
