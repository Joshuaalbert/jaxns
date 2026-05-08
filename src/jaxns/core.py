import dataclasses
from functools import partial
from typing import Callable, NamedTuple, Any

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
from jaxctx import CtxParams

from jaxns.allocation import AllocationTarget
from jaxns.allocation import ParentWork
from jaxns.allocation import accept_parent_work
from jaxns.allocation import build_allocation_plan
from jaxns.allocation import select_parent_work
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
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree
from jaxns.random_utils import resample_indicies
from jaxns.samples import Samples, SeedPoint, PhantomSamples
from jaxns.state import State
from jaxns.termination_condition import TerminationCondition
from jaxns.types import IntArray, BoolArray, PRNGKey


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
            jnp.sum(state.samples.num_likelihood_evaluations[:num_samples])
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
    if not store_phantom_samples:
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
    if not store_phantom_samples:
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
        if new_samples.phantom_samples.U_samples is None:
            new_samples.phantom_samples.U_samples = (
                _phantom_coordinates_like_state(
                    outer_carry.state,
                    batch_size=shell_size,
                    num_phantom=int(phantom_samples.log_L.shape[1]),
                )
            )

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
        if state.samples.phantom_samples.U_samples is None:
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
        outputs = []
        for sample_key in jax.random.split(key, root_count):
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
            outputs.append(
                self._sample_constrained(
                    sampler_key,
                    jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype),
                    SeedPoint(U0=seed_U, log_L0=seed_log_L),
                    requested_parent_idx=-1,
                    effective_parent_idx=-1,
                    accepted_parent_idx=-1,
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
        samples = Samples(
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
        ).resize(int(self.max_samples)).sort()
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
    ):
        del requested_parent_idx, effective_parent_idx, accepted_parent_idx
        return self.sampler.get_sample(
            key,
            log_L_constraint,
            seed_point,
            args=self.args,
            params=self.params,
        )

    def _sample_parent_work(
            self,
            key: PRNGKey,
            state: State,
            parent_work: ParentWork,
    ) -> tuple[ParentWork, Samples]:
        outputs = []
        parent_idxs = []
        parent_log_L_constraints = []
        target_block_idxs = []
        parent_block_idxs = []
        fallback_to_root = []
        log_likelihoods = jnp.asarray(state.samples.log_likelihoods)
        num_samples = int(state.num_samples)
        sample_indices = jnp.arange(num_samples)

        for work_idx, sample_key in enumerate(
                jax.random.split(key, int(parent_work.parent_idxs.shape[0]))
        ):
            seed_key, sampler_key = jax.random.split(sample_key, 2)
            constraint = parent_work.parent_log_L_constraints[work_idx]
            candidate_mask = log_likelihoods[:num_samples] > constraint
            candidate_indices = sample_indices[candidate_mask]
            no_seed = int(candidate_indices.shape[0]) == 0
            if no_seed:
                constraint = jnp.asarray(
                    -jnp.inf,
                    dtype=mp_policy.measure_dtype,
                )
                candidate_mask = log_likelihoods[:num_samples] > constraint
                candidate_indices = sample_indices[candidate_mask]

            seed_choice = jax.random.randint(
                seed_key,
                (),
                minval=0,
                maxval=candidate_indices.shape[0],
            )
            seed_idx = candidate_indices[seed_choice]
            seed_point = SeedPoint(
                U0=jax.tree.map(
                    lambda u: u[seed_idx],
                    state.samples.U_samples,
                ),
                log_L0=state.samples.log_likelihoods[seed_idx],
            )
            requested_parent_idx = int(parent_work.parent_idxs[work_idx])
            effective_parent_idx = -1 if no_seed else requested_parent_idx
            outputs.append(
                self._sample_constrained(
                    sampler_key,
                    constraint,
                    seed_point,
                    requested_parent_idx=requested_parent_idx,
                    effective_parent_idx=effective_parent_idx,
                    accepted_parent_idx=effective_parent_idx,
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
        if state.samples.phantom_samples.U_samples is None:
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
                jnp.sum(state.samples.num_likelihood_evaluations[:num_samples])
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
        current = state
        initial_root_out_degree = int(state.root_out_degree)
        diagnostics_builder = _ExecutionDiagnosticsBuilder(
            allocation_target=allocation_target,
            target_num_live_points=int(self.target_num_live_points),
            shell_size=int(self.shell_size),
            sampler=self.sampler,
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

        skip_goal_check = False
        for iteration in range(max_goal_iterations):
            if not skip_goal_check:
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
                if goal_reached:
                    return finish(current)
            skip_goal_check = False
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
                plan = build_allocation_plan(
                    state=current,
                    allocation_target=allocation_target,
                    iteration=iteration,
                    delta_K=delta_K,
                    root_out_degree=initial_root_out_degree,
                    posterior_conservative=posterior_conservative,
                )
                diagnostics_builder.allocation_target_summaries.append(
                    _allocation_summary(
                        allocation_target=allocation_target,
                        iteration=iteration,
                        plan=plan,
                    )
                )
                key, select_key, sample_key = jax.random.split(key, 3)
                parent_work = select_parent_work(
                    key=select_key,
                    state=current,
                    plan=plan,
                    num_parents=min(int(self.shell_size), remaining),
                )
                if int(parent_work.parent_idxs.shape[0]) == 0:
                    if iteration == 0 and _should_advance_after_no_work(
                            state=current,
                            plan=plan,
                            depth_cond=depth_cond,
                            shell_size=int(self.shell_size),
                    ):
                        skip_goal_check = True
                    break
                requested_parent_work = parent_work
                parent_work, new_samples = self._sample_parent_work(
                    key=sample_key,
                    state=current,
                    parent_work=parent_work,
                )
                _record_parent_selection(
                    diagnostics_builder,
                    requested_parent_work=requested_parent_work,
                    accepted_parent_work=parent_work,
                )
                before = int(current.num_samples)
                current = accept_parent_work(
                    state=current,
                    parent_work=parent_work,
                    new_samples=new_samples,
                )
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
