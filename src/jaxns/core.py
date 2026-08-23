"""Depth-first nested sampling core described by the v3 paper."""

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jaxctx import CtxParams

from jaxns.allocation import AllocationPlan, build_allocation_plan
from jaxns.constrained_sampler import (
    AbstractSampler,
    ConstrainedSampleBatch,
    UniDimSliceSampler,
)
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree
from jaxns.race_tree import (
    BlockState,
    build_block_state,
    initialise_likelihood_order,
)
from jaxns.samples import PhantomSamples, Samples, SeedPoint
from jaxns.state import State, termination_register_from_volume_path
from jaxns.termination_condition import (
    TerminationCondition,
    TerminationRegister,
)
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey


@dataclasses.dataclass(slots=True, frozen=True)
class CoreWorkBatch(PureDataclassPytree):
    """Fixed-shape description of one vmapped replacement batch."""

    valid: BoolArray
    requested_parent_idx: IntArray
    effective_parent_idx: IntArray
    target_block_idx: IntArray
    requested_parent_block_idx: IntArray
    effective_parent_block_idx: IntArray
    fallback_to_shallower: BoolArray
    reused_seed: BoolArray
    requested_log_L_constraint: FloatArray
    log_L_constraint: FloatArray
    seed_idx: IntArray

    @property
    def num_valid(self) -> IntArray:
        """Number of accepted prefix lanes in this static schedule."""
        return jnp.sum(self.valid, dtype=mp_policy.count_dtype)


CoreWorkBatch.register_pytree()


def _first_masked_index(mask: BoolArray) -> IntArray:
    indices = jnp.arange(mask.shape[0], dtype=mp_policy.index_dtype)
    return jnp.min(
        jnp.where(mask, indices, jnp.asarray(mask.shape[0], indices.dtype))
    )


def _categorical_masked(key: PRNGKey, weights, mask: BoolArray) -> IntArray:
    """Sample a masked index without materialising a candidate payload."""
    weights = jnp.where(mask & jnp.isfinite(weights) & (weights > 0), weights, 0.0)
    has_weight = jnp.sum(weights) > 0
    logits = jnp.where(weights > 0, jnp.log(weights), -jnp.inf)
    sampled = jax.random.categorical(
        key,
        jnp.where(has_weight, logits, jnp.zeros_like(logits)),
    ).astype(mp_policy.index_dtype)
    return jnp.where(has_weight, sampled, _first_masked_index(mask))


def _depth_relevant_blocks(
        plan: AllocationPlan,
        depth_cond: TerminationCondition,
) -> BoolArray:
    """Mask blocks before the expectation-based evidence/posterior cutoff."""
    valid = plan.valid
    log_shell_mass = jnp.where(
        valid & (plan.volume_path.shell_mass > 0),
        jnp.log(plan.volume_path.shell_mass),
        -jnp.inf,
    )
    log_dZ = plan.log_L_blocks + log_shell_mass
    log_Z_prefix = jax.lax.associative_scan(jnp.logaddexp, log_dZ)
    log_remaining = (
        plan.log_L_blocks
        + jnp.where(
            plan.volume_path.X > 0,
            jnp.log(plan.volume_path.X),
            -jnp.inf,
        )
    )
    remaining_fraction = jnp.exp(
        log_remaining - jnp.logaddexp(log_Z_prefix, log_remaining)
    )
    relevant = valid
    if depth_cond.dlogZ is not None:
        relevant = relevant & (remaining_fraction >= depth_cond.dlogZ)

    if depth_cond.cummax_XL_frac is not None:
        log_XL = jnp.where(valid, log_remaining, -jnp.inf)
        log_XL_peak = jax.lax.associative_scan(jnp.maximum, log_XL)
        relevant = relevant & (
            log_XL
            >= log_XL_peak + jnp.log(depth_cond.cummax_XL_frac)
        )
    return relevant


def _stationary_seed_mask(
        state: State,
        log_L_constraint: FloatArray,
        from_root: BoolArray,
) -> BoolArray:
    """Return exact stationary seeds for a requested parent contour.

    A classic sample is stationary at lambda exactly when its own generation
    interval contains lambda: parent_constraint <= lambda < likelihood. Root
    fallback is deliberately narrower and may only use sentinel children.
    """
    valid = (
        jnp.arange(state.samples.log_likelihoods.shape[0])
        < state.num_samples
    )
    interval_contains = (
        (state.samples.log_L_constraints <= log_L_constraint)
        & (state.samples.log_likelihoods > log_L_constraint)
    )
    if state.samples.parent_idx is None:
        root_child = jnp.isneginf(state.samples.log_L_constraints)
    else:
        root_child = state.samples.parent_idx == -1
    return valid & jnp.where(from_root, root_child, interval_contains)


def _sample_parent_from_block(
        key: PRNGKey,
        block_state: BlockState,
        block_idx: IntArray,
) -> IntArray:
    safe_block_idx = jnp.maximum(block_idx, 0)
    start = block_state.block_start[safe_block_idx]
    size = block_state.block_size[safe_block_idx]
    offset = jax.random.randint(
        key,
        (),
        minval=jnp.asarray(0, mp_policy.index_dtype),
        maxval=jnp.maximum(size, 1),
    )
    parent_idx = block_state.block_sample_indices[start + offset]
    return jnp.where(block_idx >= 0, parent_idx, -1).astype(
        mp_policy.index_dtype
    )


def _closest_seedable_parent_block(
        state: State,
        block_state: BlockState,
        requested_block_idx: IntArray,
) -> IntArray:
    """Resolve a requested contour to itself or its closest seedable ancestor."""
    requested_constraint = jnp.where(
        requested_block_idx >= 0,
        block_state.log_L_blocks[jnp.maximum(requested_block_idx, 0)],
        jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
    )
    requested_has_seed = jnp.any(
        _stationary_seed_mask(
            state,
            requested_constraint,
            requested_block_idx < 0,
        )
    )

    # The fallback is rarely taken. Scan ancestors one at a time so the common
    # scheduler path never constructs a blocks-by-samples eligibility matrix.
    def fallback_cond(fallback_carry):
        candidate_block, found = fallback_carry
        return (candidate_block >= 0) & jnp.logical_not(found)

    def fallback_body(fallback_carry):
        candidate_block, _ = fallback_carry
        candidate_constraint = block_state.log_L_blocks[candidate_block]
        found = jnp.any(
            _stationary_seed_mask(
                state,
                candidate_constraint,
                jnp.asarray(False, mp_policy.bool_dtype),
            )
        )
        return jnp.where(found, candidate_block, candidate_block - 1), found

    fallback_block_idx, fallback_found = jax.lax.while_loop(
        fallback_cond,
        fallback_body,
        (
            requested_block_idx - 1,
            jnp.asarray(False, mp_policy.bool_dtype),
        ),
    )
    return jnp.where(
        requested_has_seed,
        requested_block_idx,
        jnp.where(fallback_found, fallback_block_idx, -1),
    )


def _plan_work_batch(
        key: PRNGKey,
        state: State,
        block_state: BlockState,
        plan: AllocationPlan,
        relevant: BoolArray,
        shell_size: int,
        max_valid_lanes: IntArray | None = None,
) -> CoreWorkBatch:
    """Schedule a batch while accounting for its expected in-flight depth."""
    slots = jnp.arange(shell_size, dtype=mp_policy.index_dtype)
    block_indices = jnp.arange(plan.valid.shape[0], dtype=mp_policy.index_dtype)
    plan_key, seed_draw_key = jax.random.split(key)

    if max_valid_lanes is None:
        max_valid_lanes = jnp.asarray(shell_size, mp_policy.index_dtype)

    def plan_one(carry, slot):
        expected_K, schedule_key = carry
        (
            target_key,
            requested_block_key,
            requested_parent_key,
            fallback_parent_key,
            next_key,
        ) = jax.random.split(schedule_key, 5)

        deficits = jnp.maximum(
            plan.target_K.astype(mp_policy.measure_dtype) - expected_K,
            0.0,
        )
        target_mask = relevant & (deficits > 0)
        lane_valid = jnp.any(target_mask) & (slot < max_valid_lanes)
        last_relevant = jnp.max(
            jnp.where(relevant, block_indices, jnp.asarray(0, block_indices.dtype))
        )
        fallback_target_mask = relevant & (block_indices == last_relevant)
        target_idx = _categorical_masked(
            target_key,
            jnp.where(target_mask, deficits, 1.0),
            jnp.where(jnp.any(target_mask), target_mask, fallback_target_mask),
        )

        # Candidate zero is the root sentinel (X=1); candidate h+1 is block h.
        parent_X = jnp.concatenate(
            [jnp.ones((1,), plan.volume_path.X.dtype), plan.volume_path.X]
        )
        parent_mask = jnp.concatenate(
            [
                jnp.ones((1,), mp_policy.bool_dtype),
                plan.valid & (block_indices < target_idx),
            ]
        )
        parent_weights = plan.volume_path.X[target_idx] / jnp.maximum(
            parent_X,
            jnp.finfo(parent_X.dtype).tiny,
        )
        requested_choice = _categorical_masked(
            requested_block_key,
            parent_weights,
            parent_mask,
        )
        requested_block_idx = requested_choice - 1
        requested_parent_idx = _sample_parent_from_block(
            requested_parent_key,
            block_state,
            requested_block_idx,
        )
        effective_block_idx = _closest_seedable_parent_block(
            state,
            block_state,
            requested_block_idx,
        )
        effective_constraint = jnp.where(
            effective_block_idx >= 0,
            plan.log_L_blocks[jnp.maximum(effective_block_idx, 0)],
            jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        )
        requested_constraint = jnp.where(
            requested_block_idx >= 0,
            plan.log_L_blocks[jnp.maximum(requested_block_idx, 0)],
            jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        )
        effective_parent_idx = jnp.where(
            effective_block_idx == requested_block_idx,
            requested_parent_idx,
            _sample_parent_from_block(
                fallback_parent_key,
                block_state,
                effective_block_idx,
            ),
        )

        # Before likelihoods are known, a thread from X_parent reaches X_g
        # with probability X_g / X_parent. Account for all S scheduled threads
        # so later slots do not repeatedly target the same nominal deficit.
        effective_parent_X = jnp.where(
            effective_block_idx >= 0,
            plan.volume_path.X[jnp.maximum(effective_block_idx, 0)],
            1.0,
        )
        crosses_block = plan.valid & (block_indices > effective_block_idx)
        expected_contribution = jnp.where(
            crosses_block,
            jnp.minimum(plan.volume_path.X / effective_parent_X, 1.0),
            0.0,
        )
        expected_K = expected_K + jnp.where(
            lane_valid,
            expected_contribution,
            0.0,
        )
        planned = (
            lane_valid,
            requested_parent_idx,
            effective_parent_idx,
            target_idx,
            requested_block_idx,
            effective_block_idx,
            effective_block_idx != requested_block_idx,
            requested_constraint,
            effective_constraint,
        )
        return (expected_K, next_key), planned

    (_, _), planned = jax.lax.scan(
        plan_one,
        (
            plan.current_K.astype(mp_policy.measure_dtype),
            plan_key,
        ),
        slots,
    )
    (
        lane_valid,
        requested_parent_idx,
        effective_parent_idx,
        target_idx,
        requested_block_idx,
        effective_block_idx,
        fallback_to_shallower,
        requested_constraint,
        effective_constraint,
    ) = planned

    # Enforce distinct seeds only among lanes with the same stationary target
    # distribution. Excluding a seed selected at a different contour biases
    # nested eligibility sets: a deep lane can consume a high-likelihood seed
    # and leave a shallow lane with a non-stationary low-ranked population.
    # Sorting only the S lane identities groups equal contours without ever
    # sorting or moving the N scientific payloads.
    seed_lane_order = jnp.argsort(
        jnp.where(lane_valid, effective_block_idx, plan.valid.shape[0]),
        stable=True,
    )
    seed_keys = jax.random.split(seed_draw_key, shell_size)

    def choose_seed(carry, lane_idx):
        current_block, used_at_block, used_in_batch, seed_indices, reused = carry
        lane_block = effective_block_idx[lane_idx]
        same_block = lane_block == current_block
        used_at_block = jnp.where(
            same_block,
            used_at_block,
            jnp.zeros_like(used_at_block),
        )
        seed_mask = _stationary_seed_mask(
            state,
            effective_constraint[lane_idx],
            lane_block < 0,
        )
        unused_seed_mask = seed_mask & jnp.logical_not(used_at_block)
        preferred_seed_mask = jnp.where(
            jnp.any(unused_seed_mask),
            unused_seed_mask,
            seed_mask,
        )
        seed_idx = _categorical_masked(
            seed_keys[lane_idx],
            jnp.ones(seed_mask.shape, mp_policy.measure_dtype),
            preferred_seed_mask,
        )
        is_valid = lane_valid[lane_idx]
        reused_seed = is_valid & used_in_batch[seed_idx]
        used_at_block = used_at_block.at[seed_idx].set(
            used_at_block[seed_idx] | is_valid
        )
        used_in_batch = used_in_batch.at[seed_idx].set(
            used_in_batch[seed_idx] | is_valid
        )
        seed_indices = seed_indices.at[lane_idx].set(seed_idx)
        reused = reused.at[lane_idx].set(reused_seed)
        return lane_block, used_at_block, used_in_batch, seed_indices, reused

    _, _, _, seed_idx, reused_seed = jax.lax.fori_loop(
        0,
        shell_size,
        lambda order_idx, carry: choose_seed(
            carry,
            seed_lane_order[order_idx],
        ),
        (
            jnp.asarray(-2, dtype=mp_policy.index_dtype),
            jnp.zeros(
                state.samples.log_likelihoods.shape,
                dtype=mp_policy.bool_dtype,
            ),
            jnp.zeros(
                state.samples.log_likelihoods.shape,
                dtype=mp_policy.bool_dtype,
            ),
            jnp.zeros((shell_size,), dtype=mp_policy.index_dtype),
            jnp.zeros((shell_size,), dtype=mp_policy.bool_dtype),
        ),
    )
    return CoreWorkBatch(
        valid=lane_valid,
        requested_parent_idx=requested_parent_idx,
        effective_parent_idx=effective_parent_idx,
        target_block_idx=target_idx,
        requested_parent_block_idx=requested_block_idx,
        effective_parent_block_idx=effective_block_idx,
        fallback_to_shallower=fallback_to_shallower,
        reused_seed=reused_seed,
        requested_log_L_constraint=requested_constraint,
        log_L_constraint=effective_constraint,
        seed_idx=seed_idx,
    )


def _sample_work_batch(
        key: PRNGKey,
        state: State,
        sampler: AbstractSampler,
        work: CoreWorkBatch,
) -> ConstrainedSampleBatch:
    """Run every replacement lane concurrently with one full ``vmap``."""
    keys = jax.random.split(key, work.seed_idx.shape[0])

    def sample_one(sample_key, constraint, seed_idx):
        seed = SeedPoint(
            U0=jax.tree.map(lambda u: u[seed_idx], state.samples.U_samples),
            log_L0=state.samples.log_likelihoods[seed_idx],
        )
        return sampler.get_sample(
            sample_key,
            constraint,
            seed,
            args=state.args,
            params=state.params,
        )

    sampled = jax.vmap(sample_one)(
        keys,
        work.log_L_constraint,
        work.seed_idx,
    )
    U_samples, log_likelihoods, num_evals, phantom_samples = sampled
    return ConstrainedSampleBatch(
        U_samples=U_samples,
        log_likelihoods=log_likelihoods,
        num_likelihood_evaluations=num_evals,
        phantom_samples=phantom_samples,
    )


def _accept_work_batch(
        state: State,
        work: CoreWorkBatch,
        batch: ConstrainedSampleBatch,
        *,
        store_phantom_samples: bool,
) -> State:
    shell_size = batch.log_likelihoods.shape[0]
    # V3 shrinkage needs phantom likelihoods and cluster identity, not phantom
    # coordinates. Discarding coordinates keeps the persistent state compact;
    # the legacy flag is retained only for constructor compatibility.
    del store_phantom_samples
    stored_phantoms = dataclasses.replace(
        batch.phantom_samples,
        U_samples=None,
    )
    new_samples = Samples(
        log_L_constraints=work.log_L_constraint,
        log_likelihoods=batch.log_likelihoods,
        U_samples=batch.U_samples,
        out_degree=jnp.zeros((shell_size,), mp_policy.count_dtype),
        num_likelihood_evaluations=(
            batch.num_likelihood_evaluations.astype(
                mp_policy.count_dtype
            )
        ),
        phantom_samples=stored_phantoms,
        parent_idx=work.effective_parent_idx,
        requested_parent_idx=work.requested_parent_idx,
        requested_log_L_constraint=work.requested_log_L_constraint,
        seed_idx=work.seed_idx,
    )
    appended = state.samples.set_slice(state.num_samples, new_samples)
    parent_slots = jnp.maximum(work.effective_parent_idx, 0)
    parent_delta = (work.valid & (work.effective_parent_idx >= 0)).astype(
        appended.out_degree.dtype
    )
    out_degree = appended.out_degree.at[parent_slots].add(parent_delta)
    appended = dataclasses.replace(appended, out_degree=out_degree)

    classic_idx = jnp.argmax(
        jnp.where(work.valid, batch.log_likelihoods, -jnp.inf)
    )
    candidate_log_L = batch.log_likelihoods[classic_idx]
    candidate_U = jax.tree.map(
        lambda u: u[classic_idx],
        batch.U_samples,
    )
    improves = candidate_log_L > state.log_L_supremum
    return dataclasses.replace(
        state,
        root_out_degree=(
            state.root_out_degree
            + jnp.sum(work.valid & (work.effective_parent_idx < 0)).astype(
                state.root_out_degree.dtype
            )
        ),
        samples=appended,
        num_samples=(
            state.num_samples
            + work.num_valid.astype(state.num_samples.dtype)
        ),
        log_L_supremum=jnp.where(
            improves,
            candidate_log_L,
            state.log_L_supremum,
        ),
        U_supremum=jax.tree.map(
            lambda u_new, u_old: jnp.where(improves, u_new, u_old),
            candidate_U,
            state.U_supremum,
        ),
        depth_loop_iter=state.depth_loop_iter + jnp.asarray(
            1,
            state.depth_loop_iter.dtype,
        ),
        num_reparented=(
            state.num_reparented
            + jnp.sum(work.valid & work.fallback_to_shallower).astype(
                state.num_reparented.dtype
            )
        ),
        num_reused_seeds=(
            state.num_reused_seeds
            + jnp.sum(work.valid & work.reused_seed).astype(
                state.num_reused_seeds.dtype
            )
        ),
        likelihood_order=(
            None
            if state.likelihood_order is None
            else state.likelihood_order.insert(
                appended.log_likelihoods,
                state.num_samples,
                work.valid,
            )
        ),
    )


class _DepthCarry(NamedTuple):
    key: PRNGKey
    state: State
    block_state: BlockState
    plan: AllocationPlan
    relevant: BoolArray
    register: TerminationRegister


@partial(
    jax.jit,
    static_argnames=(
        "shell_size",
        "allocation_target",
        "delta_K",
        "store_phantom_samples",
        "max_samples",
    ),
)
def _run_depth(
        key: PRNGKey,
        state: State,
        sampler: AbstractSampler,
        depth_cond: TerminationCondition,
        *,
        shell_size: int,
        allocation_target: str,
        delta_K: int,
        store_phantom_samples: bool,
        max_samples: int,
) -> State:
    """Run one allocation depth epoch entirely in compiled JAX."""

    def build_depth_view(current_state):
        block_state = build_block_state(
            current_state.samples,
            root_out_degree=current_state.root_out_degree,
            num_samples=current_state.num_samples,
            likelihood_order=current_state.likelihood_order,
        )
        plan = build_allocation_plan(
            state=current_state,
            allocation_target=allocation_target,
            iteration=current_state.goal_loop_iter,
            delta_K=jnp.asarray(delta_K, mp_policy.count_dtype),
            block_state=block_state,
        )
        relevant = _depth_relevant_blocks(plan, depth_cond)
        register = termination_register_from_volume_path(
            current_state,
            block_state,
            plan.volume_path.X,
            plan.volume_path.shell_mass,
        )
        return block_state, plan, relevant, register

    def cond(carry: _DepthCarry):
        has_gap = jnp.any(carry.plan.under_allocated(carry.relevant))
        has_buffer = (
            carry.state.num_samples + shell_size
            <= carry.state.samples.log_likelihoods.shape[0]
        )
        sample_limit = jnp.asarray(max_samples, mp_policy.count_dtype)
        if depth_cond.max_samples is not None:
            sample_limit = jnp.minimum(
                sample_limit,
                depth_cond.max_samples.astype(mp_policy.count_dtype),
            )
        below_global_limit = carry.state.num_samples < sample_limit
        depth_done, _ = carry.register.is_done(depth_cond)
        return (
            has_gap
            & has_buffer
            & below_global_limit
            & jnp.logical_not(depth_done)
        )

    def body(carry: _DepthCarry):
        plan_key, sample_key, next_key = jax.random.split(carry.key, 3)
        sample_limit = jnp.asarray(max_samples, mp_policy.count_dtype)
        if depth_cond.max_samples is not None:
            sample_limit = jnp.minimum(
                sample_limit,
                depth_cond.max_samples.astype(mp_policy.count_dtype),
            )
        work = _plan_work_batch(
            plan_key,
            carry.state,
            carry.block_state,
            carry.plan,
            carry.relevant,
            shell_size,
            max_valid_lanes=(
                sample_limit.astype(mp_policy.index_dtype)
                - carry.state.num_samples.astype(mp_policy.index_dtype)
            ),
        )
        sampled_batch = _sample_work_batch(
            sample_key,
            carry.state,
            sampler,
            work,
        )
        next_state = _accept_work_batch(
            carry.state,
            work,
            sampled_batch,
            store_phantom_samples=store_phantom_samples,
        )
        next_block_state, next_plan, next_relevant, next_register = (
            build_depth_view(next_state)
        )
        return _DepthCarry(
            key=next_key,
            state=next_state,
            block_state=next_block_state,
            plan=next_plan,
            relevant=next_relevant,
            register=next_register,
        )

    block_state, plan, relevant, register = build_depth_view(state)
    initial_carry = _DepthCarry(
        key=key,
        state=state,
        block_state=block_state,
        plan=plan,
        relevant=relevant,
        register=register,
    )
    return jax.lax.while_loop(cond, body, initial_carry).state


@partial(
    jax.jit,
    static_argnames=(
        "root_degree",
        "sample_capacity",
        "num_phantom",
        "store_phantom_samples",
    ),
)
def _sample_init_state(
        key: PRNGKey,
        model: Model,
        args,
        params,
        *,
        root_degree: int,
        sample_capacity: int,
        num_phantom: int,
        store_phantom_samples: bool,
) -> State:
    """Draw the root sentinel children with a single vectorised prior call."""

    def sample_root(root_key):
        def draw(draw_key):
            U = model.sample_U(draw_key, args=args, params=params)
            log_L = model.log_likelihood(
                U,
                args=args,
                params=params,
                allow_nan=False,
            ).astype(mp_policy.measure_dtype)
            return draw_key, U, log_L, jnp.asarray(1, mp_policy.count_dtype)

        draw_key, U, log_L, num_evals = draw(root_key)

        def invalid(carry):
            _, _, likelihood, _ = carry
            return likelihood <= -jnp.inf

        def redraw(carry):
            old_key, _, _, old_evals = carry
            next_key, proposal_key = jax.random.split(old_key)
            _, next_U, next_log_L, _ = draw(proposal_key)
            return next_key, next_U, next_log_L, old_evals + 1

        _, U, log_L, num_evals = jax.lax.while_loop(
            invalid,
            redraw,
            (draw_key, U, log_L, num_evals),
        )
        return U, log_L, num_evals

    U_samples, log_likelihoods, num_evals = jax.vmap(sample_root)(
        jax.random.split(key, root_degree)
    )
    del store_phantom_samples
    phantom_U = None
    root_samples = Samples(
        log_L_constraints=jnp.full(
            (root_degree,),
            -jnp.inf,
            mp_policy.measure_dtype,
        ),
        log_likelihoods=log_likelihoods,
        U_samples=U_samples,
        out_degree=jnp.zeros((root_degree,), mp_policy.count_dtype),
        num_likelihood_evaluations=num_evals,
        phantom_samples=PhantomSamples(
            U_samples=phantom_U,
            valid_mask=jnp.zeros(
                (root_degree, num_phantom),
                mp_policy.bool_dtype,
            ),
            log_L=jnp.full(
                (root_degree, num_phantom),
                -jnp.inf,
                mp_policy.measure_dtype,
            ),
        ),
        parent_idx=jnp.full(
            (root_degree,),
            -1,
            mp_policy.index_dtype,
        ),
        requested_parent_idx=jnp.full(
            (root_degree,),
            -1,
            mp_policy.index_dtype,
        ),
        requested_log_L_constraint=jnp.full(
            (root_degree,),
            -jnp.inf,
            mp_policy.measure_dtype,
        ),
        seed_idx=jnp.full(
            (root_degree,),
            -1,
            mp_policy.index_dtype,
        ),
    ).resize(sample_capacity)
    supremum_idx = jnp.argmax(log_likelihoods)
    return State(
        root_out_degree=jnp.asarray(root_degree, mp_policy.count_dtype),
        samples=root_samples,
        num_samples=jnp.asarray(root_degree, mp_policy.count_dtype),
        log_L_supremum=log_likelihoods[supremum_idx],
        U_supremum=jax.tree.map(lambda u: u[supremum_idx], U_samples),
        termination_reason=jnp.asarray(0, mp_policy.count_dtype),
        model=model,
        args=args,
        params=params,
        likelihood_order=initialise_likelihood_order(
            root_samples.log_likelihoods,
            jnp.asarray(root_degree, mp_policy.count_dtype),
        ),
    )


@dataclasses.dataclass(slots=True, frozen=True)
class NestedSampler(PureDataclassPytree):
    """Object-oriented configuration and Python goal-loop driver for v3."""

    model: Model
    target_num_live_points: int | None = None
    root_allocation_degree: int | None = None
    max_samples: int | None = None
    shell_size: int | None = None
    batch_size: int | None = None
    args: tuple = ()
    params: CtxParams | None = None
    sampler: AbstractSampler | None = None
    termination_condition: TerminationCondition | None = None
    store_phantom_samples: bool = False
    collect_phantom_samples: bool = False
    allocation_target: Literal[
        "uniform",
        "evidence_improving",
        "posterior_improving",
    ] = "uniform"
    delta_K: int | None = None
    initial_capacity: int | None = None

    def __post_init__(self):
        U_ndims = int(self.model.U_ndims(self.args, self.params))
        root_degree = self.root_allocation_degree
        if root_degree is None:
            root_degree = self.target_num_live_points
        if root_degree is None:
            # Match v2's robust default number of independent Markov chains.
            # Merely recording phantoms must not change the sampled race tree.
            root_degree = max(1, 30 * U_ndims)
        if (
            self.root_allocation_degree is not None
            and self.target_num_live_points is not None
            and self.root_allocation_degree != self.target_num_live_points
        ):
            raise ValueError(
                "root_allocation_degree and target_num_live_points disagree."
            )
        shell_size = self.shell_size
        if shell_size is None:
            # A wider vmap increases exposure to the slowest data-dependent
            # rejection loop in the batch. Ten chains per dimension retains
            # useful CPU batching without the long-tail cost observed at the
            # former half-root width on multimodal problems.
            shell_size = min(root_degree, max(1, 10 * U_ndims))
        max_samples = self.max_samples
        if max_samples is None:
            max_samples = max(root_degree + shell_size, 100 * root_degree)
        if max_samples < root_degree:
            raise ValueError("max_samples must hold all root samples.")
        delta_K = self.delta_K
        if delta_K is None:
            # One outer allocation step should normally create one full
            # replacement batch. A unit increment leaves S-1 vmapped sampler
            # lanes idle and adds a Python/device round trip per sample.
            delta_K = shell_size
        if shell_size <= 0 or delta_K <= 0:
            raise ValueError("shell_size and delta_K must be positive.")

        sampler = self.sampler
        if sampler is None:
            num_slices = max(1, 5 * U_ndims)
            retained_phantoms = U_ndims if self.collect_phantom_samples else 0
            phantom_burn_in = num_slices - 1 - retained_phantoms
            sampler = UniDimSliceSampler(
                model=self.model,
                num_slices=num_slices,
                no_step_out=True,
                gradient_guided=False,
                collect_phantom_samples=self.collect_phantom_samples,
                phantom_burn_in=max(0, phantom_burn_in),
            )
        if isinstance(sampler, UniDimSliceSampler):
            if not sampler.no_step_out:
                raise ValueError("v3 currently requires perfect/no-step-out bracketing.")
            if sampler.gradient_guided:
                raise ValueError("Gradient-guided sampling is deferred from v3.")

        termination_condition = self.termination_condition
        if termination_condition is None:
            termination_condition = TerminationCondition(
                # Match the released v2 scientific stopping goal exactly so
                # accuracy/performance comparisons cannot benefit from an
                # earlier termination threshold.
                dlogZ=jnp.log1p(
                    jnp.asarray(1e-3, mp_policy.measure_dtype)
                ),
                max_samples=jnp.asarray(max_samples, mp_policy.count_dtype),
            )
        elif termination_condition.max_samples is None:
            termination_condition = dataclasses.replace(
                termination_condition,
                max_samples=jnp.asarray(max_samples, mp_policy.count_dtype),
            )
        initial_capacity = self.initial_capacity
        if initial_capacity is None:
            initial_capacity = min(
                max_samples + shell_size - 1,
                max(root_degree + shell_size, root_degree + 64 * shell_size),
            )
        initial_capacity = max(root_degree, int(initial_capacity))

        object.__setattr__(self, "target_num_live_points", root_degree)
        object.__setattr__(self, "root_allocation_degree", root_degree)
        object.__setattr__(self, "shell_size", int(shell_size))
        object.__setattr__(self, "max_samples", int(max_samples))
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "termination_condition", termination_condition)
        object.__setattr__(self, "initial_capacity", initial_capacity)
        object.__setattr__(self, "delta_K", int(delta_K))

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(
            this,
            [
                "target_num_live_points",
                "root_allocation_degree",
                "max_samples",
                "shell_size",
                "batch_size",
                "store_phantom_samples",
                "collect_phantom_samples",
                "allocation_target",
                "delta_K",
                "initial_capacity",
            ],
        )

    @classmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        return cls.build_unflatten(aux_data, children)

    def initialise(self, key: PRNGKey | None = None) -> State:
        """Create a resumable immutable root state."""
        if key is None:
            key = jax.random.PRNGKey(42)
        init_key, run_key = jax.random.split(key)
        state = _sample_init_state(
            init_key,
            self.model,
            self.args,
            self.params,
            root_degree=int(self.root_allocation_degree),
            sample_capacity=int(self.initial_capacity),
            num_phantom=int(self.sampler.num_phantom()),
            store_phantom_samples=self.store_phantom_samples,
        )
        return dataclasses.replace(state, random_key=run_key)

    def run(self, key: PRNGKey | None = None) -> State:
        """Run until the default expectation-based goal is satisfied."""
        if key is None:
            key = jax.random.PRNGKey(42)

        def default_goal(state: State) -> bool:
            if int(state.goal_loop_iter) == 0:
                return False
            done, _ = state.compute_termination_register().is_done(
                self.termination_condition
            )
            return bool(done)

        return self.run_until_goal(default_goal, key=key)

    def run_until_goal(
            self,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition | None = None,
            key: PRNGKey | None = None,
    ) -> State:
        """Run a Python goal loop around compiled JAX depth epochs."""
        state = self.initialise(key)
        return self._resume_until_goal(
            state,
            goal_cond,
            depth_cond=depth_cond,
            key=None,
        )

    def resume_until_goal(
            self,
            state: State,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition | None = None,
            key: PRNGKey | None = None,
    ) -> State:
        """Resume an immutable state under a user-provided Python goal."""
        return self._resume_until_goal(
            state,
            goal_cond,
            depth_cond=depth_cond,
            key=key,
        )

    def _resume_until_goal(
            self,
            state: State,
            goal_cond: Callable[[State], bool],
            *,
            depth_cond: TerminationCondition | None,
            key: PRNGKey | None,
    ) -> State:
        if depth_cond is None:
            depth_cond = self.termination_condition
        if key is None:
            key = state.random_key
        if key is None:
            key = jax.random.PRNGKey(42)
        consecutive_no_progress = 0
        while not bool(goal_cond(state)):
            num_samples = int(state.num_samples)
            capacity = state.samples.log_likelihoods.shape[0]
            if num_samples + self.shell_size > capacity:
                storage_capacity = self.max_samples + self.shell_size - 1
                if capacity >= storage_capacity:
                    break
                new_capacity = min(
                    storage_capacity,
                    max(capacity * 2, num_samples + 64 * self.shell_size),
                )
                state = state.resize(new_capacity)
            depth_key, key = jax.random.split(key)
            next_state = _run_depth(
                depth_key,
                state,
                self.sampler,
                depth_cond,
                shell_size=int(self.shell_size),
                allocation_target=self.allocation_target,
                delta_K=int(self.delta_K),
                store_phantom_samples=self.store_phantom_samples,
                max_samples=int(self.max_samples),
            )
            state = dataclasses.replace(
                next_state,
                goal_loop_iter=next_state.goal_loop_iter + jnp.asarray(
                    1,
                    next_state.goal_loop_iter.dtype,
                ),
                random_key=key,
            )
            if int(state.num_samples) == num_samples:
                consecutive_no_progress += 1
            else:
                consecutive_no_progress = 0
            # The paper schedule deliberately spends its first epoch exposing
            # the root allocation before it can request work. A second empty
            # epoch means the supplied depth condition cannot advance this
            # state, so returning control is safer than spinning in Python.
            if consecutive_no_progress >= 2:
                break
            if int(state.num_samples) >= self.max_samples:
                break
        done, reason = state.compute_termination_register().is_done(
            self.termination_condition
        )
        return dataclasses.replace(
            state,
            termination_reason=jnp.where(
                done,
                reason,
                state.termination_reason,
            ),
        )

    def run_single_iteration(
            self,
            state: State | None = None,
            depth_cond: TerminationCondition | None = None,
            key: PRNGKey | None = None,
    ) -> State:
        """Run exactly one compiled depth epoch."""
        if state is None:
            state = self.initialise(key)
            key = state.random_key
        elif key is None:
            key = state.random_key
        if key is None:
            key = jax.random.PRNGKey(42)
        if depth_cond is None:
            depth_cond = self.termination_condition
        depth_key, next_key = jax.random.split(key)
        next_state = _run_depth(
            depth_key,
            state,
            self.sampler,
            depth_cond,
            shell_size=int(self.shell_size),
            allocation_target=self.allocation_target,
            delta_K=int(self.delta_K),
            store_phantom_samples=self.store_phantom_samples,
            max_samples=int(self.max_samples),
        )
        return dataclasses.replace(next_state, random_key=next_key)


NestedSampler.register_pytree()
