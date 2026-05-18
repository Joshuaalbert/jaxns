from __future__ import annotations

import dataclasses
import inspect
import re
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxns.allocation as allocation
import jaxns.core as core_module
from jaxns.constrained_sampler import AbstractSampler
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.termination_condition import TerminationCondition
from tests.test_v3_run_pattern import DeterministicContourSampler
from tests.test_v3_run_pattern import _goal_after_state_counts
from tests.test_v3_run_pattern import _make_deterministic_nested_sampler
from tests.test_v3_run_pattern import _make_indexed_seed_state
from tests.test_v3_run_pattern import _make_samples
from tests.test_v3_run_pattern import _make_state


SCHEMA_NAME_EXPORT_KEYWORDS = (
    "CORE",
    "BOUNDARY",
)

WORK_SCHEMA_FIELD_GROUPS = {
    "validity mask": ("valid_mask", "work_valid_mask"),
    "static capacity": ("capacity", "max_work_items", "work_capacity"),
    "active count": ("num_work_items", "work_count", "num_valid"),
    "parent work identity": ("parent_work_id", "work_id"),
    "requested parent": ("requested_parent_idx", "requested_parent_index"),
    "effective parent": ("effective_parent_idx", "effective_parent_index"),
    "target block": ("target_block_idx", "target_block_index"),
    "parent block": ("parent_block_idx", "parent_block_index"),
    "sentinel fallback": ("fallback_to_root", "sentinel_fallback"),
    "likelihood constraint": ("log_L_constraint", "log_L_constraints"),
    "seed identity": ("seed_idx", "seed_index", "seed_identity"),
    "direction snapshot": ("direction_snapshot_id", "direction_kernel_id"),
    "phantom slot": ("phantom_slot", "phantom_slot_idx"),
}

RESULT_SCHEMA_FIELD_GROUPS = {
    "validity mask": ("valid_mask", "result_valid_mask"),
    "static capacity": ("capacity", "max_result_items", "result_capacity"),
    "active count": ("num_results", "result_count", "num_valid"),
    "status mask": ("status", "status_code", "status_mask"),
    "parent work identity": ("parent_work_id", "work_id"),
    "accepted parent": ("accepted_parent_idx", "accepted_parent_index"),
    "unit coordinates": ("U_samples", "u_samples"),
    "likelihood": ("log_L", "log_likelihoods"),
    "likelihood eval count": (
        "num_likelihood_evaluations",
        "likelihood_evaluation_count",
    ),
    "phantom validity": ("phantom_valid_mask", "phantom_mask"),
    "phantom likelihood": ("phantom_log_L", "log_L_phantom"),
}

ACCEPTED_SCHEMA_FIELD_NAMES = (
    "work_schema",
    "result_schema",
    "work",
    "result",
)


def _goal_after_num_samples(min_num_samples: int):
    def goal_cond(state):
        return int(state.num_samples) >= min_num_samples

    return goal_cond


@dataclasses.dataclass(frozen=True, slots=True)
class _CallbackToyModel(PureDataclassPytree):
    """Toy model that records executed likelihood callbacks."""

    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 1

    def sample_U(self, key, args=(), params=None):
        del args, params
        return jax.random.uniform(key, minval=0.0, maxval=1.0)

    def transform_to_X(self, U, args=(), params=None):
        del args, params
        return U

    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan: bool = True,
    ):
        del args, params, allow_nan
        jax.debug.callback(
            lambda _: _LIKELIHOOD_CALLBACK_HITS.append(1),
            jnp.asarray(U),
            ordered=True,
        )
        return -jnp.square(jnp.asarray(U) - 0.25)

    def log_prior(self, U, args=(), params=None):
        del args, params
        return jnp.where((U >= 0.0) & (U <= 1.0), 0.0, -jnp.inf)


_CallbackToyModel.register_pytree()

_LIKELIHOOD_CALLBACK_HITS: list[int] = []
_SAMPLER_FORWARDING_HITS: list[int] = []


@dataclasses.dataclass(frozen=True, slots=True)
class _ArgsParamsForwardingSampler(AbstractSampler):
    """Sampler that asserts non-root core work receives user context."""

    expected_args: tuple
    expected_params: dict[str, float]

    def num_phantom(self) -> int:
        return 0

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point,
            args=(),
            params=None,
    ):
        del key
        assert args == self.expected_args
        assert params == self.expected_params
        jax.debug.callback(
            lambda _: _SAMPLER_FORWARDING_HITS.append(1),
            log_L_constraint,
            ordered=True,
        )
        log_L = jnp.where(
            jnp.isneginf(log_L_constraint),
            seed_point.log_L0,
            log_L_constraint,
        ) + jnp.asarray(params["offset"])
        return (
            seed_point.U0,
            log_L,
            jnp.asarray(1, dtype=mp_policy.count_dtype),
            core_module.PhantomSamples(
                U_samples=None,
                valid_mask=jnp.zeros((0,), dtype=mp_policy.bool_dtype),
                log_L=jnp.zeros((0,), dtype=mp_policy.measure_dtype),
            ),
        )


def _make_core_work_batch(
        *,
        valid_mask: tuple[bool, ...],
        log_l_constraints: tuple[float, ...],
) -> core_module.CoreWorkBatch:
    capacity = len(valid_mask)
    slots = jnp.arange(capacity, dtype=mp_policy.index_dtype)
    return core_module.CoreWorkBatch(
        valid_mask=jnp.asarray(valid_mask, dtype=mp_policy.bool_dtype),
        capacity=jnp.asarray(capacity, dtype=mp_policy.count_dtype),
        num_work_items=jnp.asarray(sum(valid_mask), dtype=mp_policy.count_dtype),
        parent_work_id=slots,
        work_id=slots,
        requested_parent_idx=jnp.full(
            (capacity,),
            0,
            dtype=mp_policy.index_dtype,
        ),
        effective_parent_idx=jnp.full(
            (capacity,),
            -1,
            dtype=mp_policy.index_dtype,
        ),
        target_block_idx=jnp.zeros((capacity,), dtype=mp_policy.index_dtype),
        parent_block_idx=jnp.zeros((capacity,), dtype=mp_policy.index_dtype),
        fallback_to_root=jnp.zeros((capacity,), dtype=mp_policy.bool_dtype),
        log_L_constraint=jnp.asarray(
            log_l_constraints,
            dtype=mp_policy.measure_dtype,
        ),
        seed_idx=jnp.full((capacity,), -1, dtype=mp_policy.index_dtype),
        direction_snapshot_id=jnp.zeros(
            (capacity,),
            dtype=mp_policy.index_dtype,
        ),
        phantom_slot=jnp.full((capacity,), -1, dtype=mp_policy.index_dtype),
    )


def _discover_boundary_schema_types():
    for export_name, schema_names in vars(core_module).items():
        if not (
                export_name.endswith("SCHEMA_NAMES")
                and all(
                    keyword in export_name
                    for keyword in SCHEMA_NAME_EXPORT_KEYWORDS
                )
        ):
            continue
        if isinstance(schema_names, str):
            continue
        schema_names = tuple(schema_names)
        if not schema_names or not all(
                isinstance(name, str) and name
                for name in schema_names
        ):
            continue
        schema_types = tuple(
            getattr(core_module, schema_name, None)
            for schema_name in schema_names
        )
        if all(schema_type is not None for schema_type in schema_types):
            return export_name, tuple(zip(schema_names, schema_types))

    for hook_name, hook in vars(core_module).items():
        if not (
                callable(hook)
                and "core" in hook_name.lower()
                and "boundary" in hook_name.lower()
                and "schema" in hook_name.lower()
        ):
            continue
        schemas = hook()
        schema_pairs = _schema_pairs_from_hook_result(schemas)
        if schema_pairs:
            return hook_name, schema_pairs

    pytest.fail(
        "Ticket 0020 requires an internal fixed-shape pure-core/runtime "
        "boundary to be documented through either an exported "
        "*CORE*BOUNDARY*SCHEMA_NAMES constants tuple or a callable boundary "
        "schema hook in jaxns.core."
    )


def _schema_pairs_from_hook_result(schemas):
    if isinstance(schemas, dict):
        return tuple(
            (str(name), schema_type)
            for name, schema_type in schemas.items()
        )
    if isinstance(schemas, tuple) and len(schemas) == 2:
        return (
            ("work_schema", schemas[0]),
            ("result_schema", schemas[1]),
        )
    return ()


def _field_names(boundary_type: object) -> frozenset[str]:
    if dataclasses.is_dataclass(boundary_type):
        return frozenset(
            field.name for field in dataclasses.fields(boundary_type)
        )
    annotations = getattr(boundary_type, "__annotations__", None)
    if annotations is not None:
        return frozenset(annotations)
    named_tuple_fields = getattr(boundary_type, "_fields", None)
    if named_tuple_fields is not None:
        return frozenset(named_tuple_fields)
    return frozenset()


def _require_field_group(
        fields: frozenset[str],
        group_name: str,
        aliases: tuple[str, ...],
        *,
        schema_name: str,
) -> None:
    assert any(alias in fields for alias in aliases), (
        f"{schema_name} must expose {group_name} metadata through one of "
        f"{aliases}. Fields were: {sorted(fields)}."
    )


def _schema_pair_by_role(schema_pairs):
    by_role = {}
    for schema_name, schema_type in schema_pairs:
        normalized = schema_name.lower()
        for role in ACCEPTED_SCHEMA_FIELD_NAMES:
            if role in normalized:
                by_role[role] = (schema_name, schema_type)
    if "work_schema" not in by_role and "work" in by_role:
        by_role["work_schema"] = by_role["work"]
    if "result_schema" not in by_role and "result" in by_role:
        by_role["result_schema"] = by_role["result"]
    return by_role


def test_pure_core_boundary_exposes_documented_static_shape_schemas():
    export_name, schema_pairs = _discover_boundary_schema_types()
    schemas_by_role = _schema_pair_by_role(schema_pairs)

    assert "work_schema" in schemas_by_role, (
        f"{export_name} must identify the fixed-shape core work schema."
    )
    assert "result_schema" in schemas_by_role, (
        f"{export_name} must identify the fixed-shape core result schema."
    )

    work_name, work_type = schemas_by_role["work_schema"]
    result_name, result_type = schemas_by_role["result_schema"]
    work_fields = _field_names(work_type)
    result_fields = _field_names(result_type)

    for group_name, aliases in WORK_SCHEMA_FIELD_GROUPS.items():
        _require_field_group(
            work_fields,
            group_name,
            aliases,
            schema_name=work_name,
        )
    for group_name, aliases in RESULT_SCHEMA_FIELD_GROUPS.items():
        _require_field_group(
            result_fields,
            group_name,
            aliases,
            schema_name=result_name,
        )


def test_pure_core_trace_contract_rejects_no_arg_dummy_hook():
    def dummy_no_arg_jaxpr_hook():
        return (
            "lambda ; a:f32[4]. let "
            "b:f32[4] = while[body_jaxpr={} cond_jaxpr={}] a "
            "in (b,) ; carry_shape result_shape"
        )

    assert not _trace_candidate_accepts_real_run_arguments(
        "pure_core_jaxpr_smoke_dummy",
        dummy_no_arg_jaxpr_hook,
    )


def test_pure_core_trace_contract_rejects_string_only_dummy_trace():
    def dummy_realistic_jaxpr_hook(
            *,
            state,
            depth_cond,
            allocation_target,
            key,
            nested_sampler,
    ):
        del state, depth_cond, allocation_target, key, nested_sampler
        return (
            "lambda ; state:f32[8]. let "
            "carry:f32[8] = while[body_jaxpr={} cond_jaxpr={}] state "
            "in (carry,) ; result output static_shape"
        )

    assert _trace_candidate_accepts_real_run_arguments(
        "pure_core_realistic_string_dummy",
        dummy_realistic_jaxpr_hook,
    )
    assert not _trace_result_has_jax_control_flow(
        dummy_realistic_jaxpr_hook(
            state=object(),
            depth_cond=object(),
            allocation_target="uniform",
            key=object(),
            nested_sampler=object(),
        )
    )


def test_run_resume_inner_depth_transition_exposes_real_jax_trace():
    trace_name, traced = _trace_real_inner_depth_transition()
    trace_text = _trace_result_text(traced)
    control_flow_text = _valid_jaxpr_control_flow_text(trace_text)

    assert control_flow_text is not None, (
        f"{trace_name} must return a JAXPR/lowered object containing JAX "
        "control flow such as while, scan, or cond for the inner "
        "run_until_goal/resume_until_goal depth/transition path."
    )
    assert _trace_text_has_static_shape_evidence(trace_text), (
        f"{trace_name} must expose static-shape carry/result evidence, such "
        "as input/output avals, shape signatures, or lowered tensor shapes."
    )
    assert _trace_text_has_carry_result_evidence(trace_text), (
        f"{trace_name} must identify the JAX carry and result/boundary shapes "
        "for the inner depth/transition path. The public Python goal_cond "
        "outer loop is allowed, but the expensive depth transition must be "
        "trace-visible."
    )


def _trace_real_inner_depth_transition() -> tuple[str, object]:
    nested_sampler = _make_deterministic_nested_sampler(
        max_samples=8,
        shell_size=2,
        num_phantom=2,
    )
    state = _make_state(
        root_out_degree=2,
        log_likelihoods=(0.0, 1.0, 2.0),
        out_degree=(1, 0, 0),
        log_L_constraints=(-np.inf, -np.inf, 0.0),
        max_samples=8,
        num_phantom=2,
    )
    depth_cond = TerminationCondition(max_samples=5)
    arguments = _InnerTraceArguments(
        nested_sampler=nested_sampler,
        state=state,
        depth_cond=depth_cond,
        allocation_target="uniform",
        key=jax.random.PRNGKey(91),
        delta_K=2,
    )

    failures = []
    for name, candidate in _iter_inner_depth_trace_candidates(nested_sampler):
        if not _trace_candidate_accepts_real_run_arguments(name, candidate):
            continue
        try:
            traced = _call_trace_candidate(candidate, arguments)
        except TypeError as error:
            failures.append(f"{name}: {error}")
            continue
        if _trace_result_has_jax_control_flow(traced):
            return name, traced
        failures.append(f"{name}: returned no JAX control-flow trace")

    failure_text = "; ".join(failures) if failures else "no candidates found"
    pytest.fail(
        "Ticket 0020 requires a concrete trace/lower/JAXPR hook for the real "
        "inner run_until_goal/resume_until_goal depth transition. The hook "
        "must be either a public NestedSampler method or an internal "
        "jaxns.core callable with realistic arguments such as NestedSampler, "
        "State, depth_cond, allocation_target, key, and delta_K. The public "
        f"Python goal_cond loop remains allowed. Candidate failures: "
        f"{failure_text}."
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _InnerTraceArguments:
    nested_sampler: object
    state: object
    depth_cond: object
    allocation_target: str
    key: object
    delta_K: int


def _iter_inner_depth_trace_candidates(nested_sampler):
    nested_sampler_candidate_names = (
        "trace_run_until_goal_core",
        "trace_resume_until_goal_core",
        "trace_inner_depth_transition",
        "trace_pure_core_transition",
        "lower_run_until_goal_core",
        "lower_resume_until_goal_core",
        "lower_inner_depth_transition",
        "lower_pure_core_transition",
        "make_run_until_goal_jaxpr",
        "make_resume_until_goal_jaxpr",
        "make_inner_depth_transition_jaxpr",
    )
    for name in nested_sampler_candidate_names:
        candidate = getattr(nested_sampler, name, None)
        if callable(candidate):
            yield f"NestedSampler.{name}", candidate

    for name, candidate in vars(core_module).items():
        if not callable(candidate):
            continue
        lower_name = name.lower()
        if not (
                any(token in lower_name for token in ("trace", "lower", "jaxpr"))
                and any(token in lower_name for token in ("run", "resume", "depth", "transition"))
                and any(token in lower_name for token in ("core", "jax"))
        ):
            continue
        yield f"jaxns.core.{name}", candidate


def _trace_candidate_accepts_real_run_arguments(
        name: str,
        candidate: Callable,
) -> bool:
    del name
    try:
        signature = inspect.signature(candidate)
    except (TypeError, ValueError):
        return False
    parameters = tuple(signature.parameters.values())
    required = tuple(
        parameter
        for parameter in parameters
        if (
                parameter.default is inspect.Parameter.empty
                and parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
        )
    )
    if not required:
        return False

    names = {parameter.name for parameter in parameters}
    has_runner = bool(names & {"nested_sampler", "runner", "ns", "sampler"})
    has_state = "state" in names or "initial_state" in names
    has_depth = bool(names & {"depth_cond", "termination_condition"})
    has_allocation = "allocation_target" in names
    has_key = "key" in names
    return has_state and has_depth and has_allocation and has_key and (
        has_runner or inspect.ismethod(candidate)
    )


def _call_trace_candidate(candidate: Callable, arguments: _InnerTraceArguments):
    signature = inspect.signature(candidate)
    kwargs = {}
    positional = []
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        value = _trace_argument_for_parameter(parameter.name, arguments)
        if value is _MISSING_TRACE_ARGUMENT:
            if parameter.default is inspect.Parameter.empty:
                raise TypeError(
                    f"missing realistic trace argument for {parameter.name!r}"
                )
            continue
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            kwargs[parameter.name] = value
    return candidate(*positional, **kwargs)


_MISSING_TRACE_ARGUMENT = object()


def _trace_argument_for_parameter(
        name: str,
        arguments: _InnerTraceArguments,
):
    if name in {"nested_sampler", "runner", "ns", "sampler"}:
        return arguments.nested_sampler
    if name in {"state", "initial_state"}:
        return arguments.state
    if name in {"depth_cond", "termination_condition"}:
        return arguments.depth_cond
    if name == "allocation_target":
        return arguments.allocation_target
    if name == "key":
        return arguments.key
    if name in {"delta_K", "delta_k"}:
        return arguments.delta_K
    if name in {"posterior_conservative", "collect_diagnostics"}:
        return False
    if name in {"max_goal_iterations", "max_iterations"}:
        return 2
    return _MISSING_TRACE_ARGUMENT


def _trace_result_has_jax_control_flow(result: object) -> bool:
    if isinstance(result, (str, bytes)):
        return False
    return _valid_jaxpr_control_flow_text(_trace_result_text(result)) is not None


def _trace_result_text(result: object) -> str:
    return "\n".join(_trace_result_text_parts(result))


def _trace_result_text_parts(result: object):
    if result is None:
        return ()
    if isinstance(result, (str, bytes)):
        return (result.decode() if isinstance(result, bytes) else result,)
    if isinstance(result, dict):
        parts = []
        for key, value in result.items():
            parts.append(str(key))
            parts.extend(_trace_result_text_parts(value))
        return tuple(parts)
    if isinstance(result, (tuple, list)):
        parts = []
        for value in result:
            parts.extend(_trace_result_text_parts(value))
        return tuple(parts)

    jaxpr = getattr(result, "jaxpr", None)
    if jaxpr is not None:
        return (str(jaxpr),)
    as_text = getattr(result, "as_text", None)
    if callable(as_text):
        try:
            return (as_text(),)
        except TypeError:
            pass

    type_name = type(result).__name__.lower()
    module_name = type(result).__module__.lower()
    if "jax" in module_name or any(
            token in type_name
            for token in ("jaxpr", "lower", "compiled", "trace")
    ):
        return (str(result),)
    return (str(type(result)),)


def _valid_jaxpr_control_flow_text(text: str) -> str | None:
    lower_text = text.lower()
    has_control_flow = any(
        token in lower_text
        for token in ("while", "scan", "cond")
    )
    has_jaxpr_shape = (
        "lambda" in lower_text
        or "let" in lower_text
        or "jaxpr" in lower_text
    )
    if has_control_flow and has_jaxpr_shape:
        return text
    return None


def _trace_text_has_static_shape_evidence(text: str) -> bool:
    lower_text = text.lower()
    if any(
            token in lower_text
            for token in (
                "shapedarray",
                "input_avals",
                "out_avals",
                "aval",
                "static_shape",
                "shape_signature",
                "tensor<",
            )
    ):
        return True
    return bool(
        re.search(
            r"\b(?:f|s|u|i|bool)[0-9]*\[[0-9,\s]+\]",
            lower_text,
        )
    )


def _trace_text_has_carry_result_evidence(text: str) -> bool:
    lower_text = text.lower()
    has_carry = any(
        token in lower_text
        for token in ("carry", "state", "num_samples", "samples")
    )
    has_result = any(
        token in lower_text
        for token in ("result", "output", "out_avals", "boundary")
    )
    return has_carry and has_result


def test_v3_root_initialization_keeps_static_capacity_and_sentinel_metadata():
    nested_sampler = _make_deterministic_nested_sampler(
        max_samples=7,
        shell_size=2,
        num_phantom=2,
    )

    state = nested_sampler.run_until_goal(
        goal_cond=_goal_after_num_samples(0),
        depth_cond=TerminationCondition(max_samples=4),
        allocation_target="uniform",
        key=jax.random.PRNGKey(1),
    )

    num_samples = int(state.num_samples)
    assert num_samples == int(state.root_out_degree) == 2
    assert state.samples.log_likelihoods.shape == (7,)
    assert state.samples.out_degree.shape == (7,)
    assert state.samples.phantom_samples.log_L.shape == (7, 2)
    np.testing.assert_array_equal(
        np.asarray(state.samples.log_L_constraints[:num_samples]),
        np.asarray([-np.inf, -np.inf]),
    )
    np.testing.assert_array_equal(
        np.asarray(state.samples.out_degree[:num_samples]),
        np.asarray([0, 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(state.samples.phantom_samples.valid_mask[:num_samples]),
        np.ones((num_samples, 2), dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(state.samples.num_likelihood_evaluations[:num_samples]),
        np.full((num_samples,), 3, dtype=int),
    )
    assert state.samples.U_samples.dtype == mp_policy.measure_dtype
    assert state.samples.phantom_samples.log_L.dtype == mp_policy.measure_dtype
    assert np.all(
        np.isfinite(
            np.asarray(state.samples.phantom_samples.log_L[:num_samples])
        )
    )


def test_direct_pure_core_large_max_samples_uses_bounded_state_capacity():
    nested_sampler = _make_deterministic_nested_sampler(
        max_samples=10_000_000,
        shell_size=2,
    )

    state = nested_sampler.run_until_goal(
        goal_cond=lambda _: False,
        depth_cond=TerminationCondition(max_samples=10),
        allocation_target="uniform",
        key=jax.random.PRNGKey(3),
        max_goal_iterations=1,
    )

    assert int(state.num_samples) == 10
    sample_capacity = int(state.samples.log_likelihoods.shape[0])
    assert sample_capacity == 32
    assert sample_capacity & (sample_capacity - 1) == 0
    assert sample_capacity < int(nested_sampler.max_samples)
    assert sample_capacity >= int(state.num_samples)


def test_collected_sample_prefix_result_matches_trimmed_result():
    nested_sampler = _make_deterministic_nested_sampler(
        max_samples=64,
        shell_size=2,
    )
    state = nested_sampler.run_until_goal(
        goal_cond=lambda _: False,
        depth_cond=TerminationCondition(max_samples=10),
        allocation_target="uniform",
        key=jax.random.PRNGKey(5),
        max_goal_iterations=1,
    )

    prefix_result = (
        core_module._state_with_collected_sample_prefix(state).to_result()
    )
    trimmed_result = state.to_result().trim()

    assert int(prefix_result.total_num_samples) == 10
    assert prefix_result.log_L.shape == (10,)
    np.testing.assert_allclose(
        np.asarray(prefix_result.log_Z_mean),
        np.asarray(trimmed_result.log_Z_mean),
    )
    np.testing.assert_allclose(
        np.asarray(prefix_result.log_Z_uncert),
        np.asarray(trimmed_result.log_Z_uncert),
    )
    np.testing.assert_allclose(
        np.asarray(prefix_result.v3_log_posterior_weights),
        np.asarray(trimmed_result.v3_log_posterior_weights),
    )


def test_direct_pure_core_resume_grows_bounded_capacity_by_bucket():
    nested_sampler = _make_deterministic_nested_sampler(
        max_samples=10_000_000,
        shell_size=2,
    )
    state = nested_sampler.run_until_goal(
        goal_cond=lambda _: False,
        depth_cond=TerminationCondition(max_samples=10),
        allocation_target="uniform",
        key=jax.random.PRNGKey(6),
        max_goal_iterations=1,
    )

    resumed = nested_sampler.resume_until_goal(
        state=state,
        goal_cond=lambda _: False,
        depth_cond=TerminationCondition(max_samples=34),
        allocation_target="uniform",
        key=jax.random.PRNGKey(7),
        max_goal_iterations=1,
    )

    assert int(state.samples.log_likelihoods.shape[0]) == 32
    assert int(resumed.num_samples) == 34
    assert int(resumed.samples.log_likelihoods.shape[0]) == 64
    assert int(resumed.samples.log_likelihoods.shape[0]) < int(
        nested_sampler.max_samples
    )


def test_pure_core_work_batch_uses_right_side_plateau_for_strict_seed():
    state = _make_indexed_seed_state(
        root_out_degree=1,
        log_likelihoods=(0.0, 1.0, 1.0, 1.0, 2.0),
        out_degree=(1, 0, 0, 0, 0),
        max_samples=8,
    )
    work_batch = _make_core_work_batch(
        valid_mask=(True,),
        log_l_constraints=(1.0,),
    )

    adjusted_work, _, new_samples, result_batch = (
        core_module._sample_core_work_batch_impl(
            jax.random.PRNGKey(7),
            state,
            work_batch,
            DeterministicContourSampler(),
            args=(),
            params=None,
            adaptation_context=None,
        )
    )

    assert int(np.asarray(adjusted_work.seed_idx[0])) == 4
    assert int(np.asarray(result_batch.num_results)) == 1
    assert int(np.asarray(new_samples.num_likelihood_evaluations[0])) == 1


def test_pure_core_non_root_work_forwards_args_and_params_to_sampler():
    _SAMPLER_FORWARDING_HITS.clear()
    expected_args = ("context", 7)
    expected_params = {"offset": 0.5}
    state = _make_state(
        root_out_degree=1,
        log_likelihoods=(0.0, 1.0),
        out_degree=(1, 0),
        max_samples=4,
    )
    work_batch = _make_core_work_batch(
        valid_mask=(True,),
        log_l_constraints=(0.0,),
    )
    sampler = _ArgsParamsForwardingSampler(
        expected_args=expected_args,
        expected_params=expected_params,
    )

    _, _, new_samples, result_batch = core_module._sample_core_work_batch_impl(
        jax.random.PRNGKey(11),
        state,
        work_batch,
        sampler,
        args=expected_args,
        params=expected_params,
        adaptation_context=None,
    )
    jax.block_until_ready(result_batch.log_L)

    assert _SAMPLER_FORWARDING_HITS == [1]
    np.testing.assert_allclose(
        np.asarray(new_samples.log_likelihoods),
        np.asarray([0.5]),
    )
    assert int(np.asarray(new_samples.num_likelihood_evaluations[0])) == 1


def test_pure_core_invalid_unidim_slice_slots_skip_likelihood_callbacks():
    _LIKELIHOOD_CALLBACK_HITS.clear()
    state = _make_state(
        root_out_degree=1,
        log_likelihoods=(0.0, 1.0),
        out_degree=(1, 0),
        max_samples=4,
    )
    work_batch = _make_core_work_batch(
        valid_mask=(False, False),
        log_l_constraints=(0.0, 0.0),
    )
    sampler = UniDimSliceSampler(
        model=_CallbackToyModel(),
        num_slices=2,
        no_step_out=True,
        collect_phantom_samples=False,
    )

    adjusted_work, _, new_samples, result_batch = (
        core_module._sample_core_work_batch_impl(
            jax.random.PRNGKey(13),
            state,
            work_batch,
            sampler,
            args=(),
            params=None,
            adaptation_context=None,
        )
    )
    jax.block_until_ready(result_batch.log_L)

    assert _LIKELIHOOD_CALLBACK_HITS == []
    np.testing.assert_array_equal(
        np.asarray(adjusted_work.seed_idx),
        np.asarray([-1, -1]),
    )
    np.testing.assert_array_equal(
        np.asarray(new_samples.num_likelihood_evaluations),
        np.asarray([0, 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(result_batch.valid_mask),
        np.asarray([False, False]),
    )


def test_depth_runs_inside_goal_boundary_before_goal_recheck():
    initial_state = _make_state(
        root_out_degree=2,
        log_likelihoods=(0.0, 1.0, 2.0),
        out_degree=(1, 0, 0),
        log_L_constraints=(-np.inf, -np.inf, 0.0),
        max_samples=8,
    )
    observations: list[int] = []

    state = _make_deterministic_nested_sampler(max_samples=8).resume_until_goal(
        state=initial_state,
        goal_cond=_goal_after_state_counts(
            observations,
            required_observations=2,
        ),
        depth_cond=TerminationCondition(max_samples=5),
        allocation_target="uniform",
        delta_K=4,
        key=jax.random.PRNGKey(53),
    )

    assert observations == [3, 5]
    assert int(state.num_samples) == 5


def test_max_goal_iterations_caps_callbacks_not_depth_progress():
    observations: list[int] = []

    state = _make_deterministic_nested_sampler(max_samples=8).run_until_goal(
        goal_cond=_goal_after_state_counts(
            observations,
            required_observations=99,
        ),
        depth_cond=TerminationCondition(max_samples=8),
        allocation_target="uniform",
        key=jax.random.PRNGKey(57),
        max_goal_iterations=2,
    )

    assert len(observations) == 2
    assert observations[0] == 2
    assert 2 < observations[1] < 8
    assert int(state.num_samples) == 8


def test_allocation_parent_acceptance_preserves_in_flight_identity():
    state = _make_state(
        root_out_degree=1,
        log_likelihoods=(0.0, 1.0),
        out_degree=(1, 0),
        log_L_constraints=(-np.inf, 0.0),
        max_samples=3,
    )
    parent_work = allocation.ParentWork(
        parent_idxs=jnp.asarray([0], dtype=jnp.int32),
        parent_log_L_constraints=jnp.asarray([0.0]),
        target_block_idxs=jnp.asarray([1], dtype=jnp.int32),
        parent_block_idxs=jnp.asarray([0], dtype=jnp.int32),
        fallback_to_root=jnp.asarray([False]),
    )
    new_samples = _make_samples(
        log_likelihoods=(1.5,),
        out_degree=(0,),
        log_L_constraints=(99.0,),
        max_samples=1,
    )

    accepted = allocation.accept_parent_work(
        state=state,
        parent_work=parent_work,
        new_samples=new_samples,
    )

    assert int(accepted.root_out_degree) == 1
    assert int(accepted.num_samples) == 3
    np.testing.assert_array_equal(
        np.asarray(accepted.samples.out_degree[:3]),
        np.asarray([2, 0, 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(accepted.samples.log_L_constraints[:3]),
        np.asarray([-np.inf, 0.0, 0.0]),
    )


def test_acceptance_fallback_mutates_root_not_requested_parent():
    state = _make_state(
        root_out_degree=2,
        log_likelihoods=(0.0, 0.0),
        out_degree=(0, 0),
        log_L_constraints=(-np.inf, -np.inf),
        max_samples=3,
    )
    parent_work = allocation.ParentWork(
        parent_idxs=jnp.asarray([-1], dtype=jnp.int32),
        parent_log_L_constraints=jnp.asarray([-np.inf]),
        target_block_idxs=jnp.asarray([0], dtype=jnp.int32),
        parent_block_idxs=jnp.asarray([-1], dtype=jnp.int32),
        fallback_to_root=jnp.asarray([True]),
    )
    new_samples = _make_samples(
        log_likelihoods=(0.25,),
        out_degree=(0,),
        log_L_constraints=(0.0,),
        max_samples=1,
    )

    accepted = allocation.accept_parent_work(
        state=state,
        parent_work=parent_work,
        new_samples=new_samples,
    )

    assert int(accepted.root_out_degree) == 3
    assert int(accepted.num_samples) == 3
    np.testing.assert_array_equal(
        np.asarray(accepted.samples.out_degree[:3]),
        np.asarray([0, 0, 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(accepted.samples.log_L_constraints[:3]),
        np.asarray([-np.inf, -np.inf, -np.inf]),
    )


def test_run_until_goal_pure_core_does_not_call_python_parent_selection(
        monkeypatch,
) -> None:
    nested_sampler = _make_deterministic_nested_sampler(
        max_samples=5,
        shell_size=1,
    )

    def fail_python_parent_selection(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "pure-core run_until_goal must not call the Python "
            "select_parent_work orchestration helper"
        )

    monkeypatch.setattr(
        core_module,
        "select_parent_work",
        fail_python_parent_selection,
    )

    state = nested_sampler.run_until_goal(
        goal_cond=_goal_after_num_samples(3),
        depth_cond=TerminationCondition(max_samples=3),
        allocation_target="uniform",
        key=jax.random.PRNGKey(0),
        max_goal_iterations=3,
    )

    assert int(state.num_samples) >= 3
