from __future__ import annotations

import dataclasses
import importlib
import inspect
import threading
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxns.allocation import ParentWork
from jaxns.constrained_sampler import AbstractSampler
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.pytree import PureDataclassPytree
from jaxns.samples import PhantomSamples
from jaxns.samples import Samples
from jaxns.samples import SeedPoint
from jaxns.state import State
from jaxns.termination_condition import TerminationCondition


REQUIRED_REQUEST_FIELDS = frozenset((
    "protocol_version",
    "runner_id",
    "task_id",
    "attempt_id",
    "transport_id",
    "compile_identity_digest",
    "eval_id",
    "U_bytes",
    "U_shape_tree",
    "requested_dtype_policy",
    "deadline_ms",
))

REQUIRED_RESPONSE_FIELDS = frozenset((
    "protocol_version",
    "runner_id",
    "task_id",
    "attempt_id",
    "transport_id",
    "compile_identity_digest",
    "eval_id",
    "status",
    "log_L",
    "error_type",
    "error_message",
    "worker_id",
    "cache_event",
    "elapsed_seconds",
))

REQUEST_FORBIDDEN_FIELD_TOKENS = (
    "model",
    "args",
    "params",
    "sampler",
    "phantom",
    "direction",
    "parent",
    "seed",
    "buffer",
    "trajectory",
    "state",
)

RESPONSE_FORBIDDEN_FIELD_TOKENS = (
    "U_",
    "u_",
    "model",
    "args",
    "params",
    "sampler",
    "phantom",
    "direction",
    "parent",
    "seed",
    "buffer",
    "trajectory",
    "sample",
    "state",
    "compiled",
)

# Ticket 0018 names jaxns.runtime and jaxns.diagnostics as the agreed public
# inspection surfaces for runtime payloads, scheduling, and diagnostics.
PUBLIC_CONTRACT_MODULES = ("jaxns.runtime", "jaxns.diagnostics")


class NestedUCase(NamedTuple):
    U: object
    forbidden_context_tokens: tuple[bytes, ...]


class EvalIdentityFixture(NamedTuple):
    runner_id: str
    task_id: str
    attempt_id: str
    transport_id: str
    compile_identity_digest: str
    eval_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class DispatchToyModel(PureDataclassPytree):
    centre: float = 0.25

    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 2

    def sample_U(self, key, args=(), params=None):
        del args, params
        return jax.random.uniform(key, shape=(2,), minval=0.0, maxval=1.0)

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
        delta = jnp.asarray(U) - jnp.asarray(self.centre)
        return -jnp.sum(jnp.square(delta))

    def log_prior(self, U, args=(), params=None):
        del args, params
        U = jnp.asarray(U)
        inside = jnp.logical_and(jnp.all(U >= 0.0), jnp.all(U <= 1.0))
        return jnp.where(inside, 0.0, -jnp.inf)


DispatchToyModel.register_pytree()


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerExceptionModel(DispatchToyModel):

    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan: bool = True,
    ):
        del U, args, params, allow_nan
        raise RuntimeError("deterministic worker exception fixture")


WorkerExceptionModel.register_pytree()


class BlockingLikelihoodGate:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()


_BLOCKING_LIKELIHOOD_GATES: dict[str, BlockingLikelihoodGate] = {}


@dataclasses.dataclass(frozen=True, slots=True)
class BlockingDispatchModel(DispatchToyModel):
    gate_id: str = ""

    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan: bool = True,
    ):
        del args, params, allow_nan
        gate_id = self.gate_id
        centre = float(self.centre)
        dtype = jnp.asarray(U).dtype

        def blocked_log_likelihood(u_value):
            gate = _BLOCKING_LIKELIHOOD_GATES[gate_id]
            gate.started.set()
            if not gate.release.wait(timeout=5.0):
                raise TimeoutError(
                    "Timed out waiting to release blocking likelihood."
                )
            delta = np.asarray(u_value) - centre
            return np.asarray(-np.sum(np.square(delta)), dtype=dtype)

        return jax.pure_callback(
            blocked_log_likelihood,
            jax.ShapeDtypeStruct((), dtype),
            U,
        )


BlockingDispatchModel.register_pytree()


@dataclasses.dataclass(slots=True)
class MutableDispatchModel(PureDataclassPytree):
    centre: float = 0.25

    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 2

    def sample_U(self, key, args=(), params=None):
        del args, params
        return jax.random.uniform(key, shape=(2,), minval=0.0, maxval=1.0)

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
        del allow_nan
        arg_shift = jnp.asarray(args[0]["shift"]) if args else 0.0
        param_shift = (
            jnp.asarray(params["shift"])
            if params is not None
            else 0.0
        )
        centre = jnp.asarray(self.centre) + arg_shift + param_shift
        delta = jnp.asarray(U) - centre
        return -jnp.sum(jnp.square(delta))

    def log_prior(self, U, args=(), params=None):
        del args, params
        U = jnp.asarray(U)
        inside = jnp.logical_and(jnp.all(U >= 0.0), jnp.all(U <= 1.0))
        return jnp.where(inside, 0.0, -jnp.inf)


MutableDispatchModel.register_pytree()


class NonSerializableLocalSampler(AbstractSampler):
    """Sampler fixture whose local state must not cross the worker boundary."""

    def __init__(self) -> None:
        self.seed_choice = "seed-choice-must-stay-local"
        self.direction_snapshot = {"basis": np.eye(2)}
        self.phantom_buffer = np.asarray([0.1, 0.2])

    def __getstate__(self):
        raise AssertionError(
            "likelihood dispatch must not pickle constrained-sampler state"
        )

    def num_phantom(self) -> int:
        return 2

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point: SeedPoint,
            args=(),
            params=None,
            adaptation_context=None,
    ):
        del key, log_L_constraint, args, params, adaptation_context
        log_L = seed_point.log_L0 + jnp.asarray(0.1)
        phantom_samples = PhantomSamples(
            U_samples=jnp.stack([seed_point.U0, seed_point.U0]),
            valid_mask=jnp.asarray([True, True]),
            log_L=jnp.asarray([log_L - 0.01, log_L - 0.02]),
        )
        return (
            seed_point.U0,
            log_L,
            jnp.asarray(1, dtype=jnp.int32),
            phantom_samples,
        )


class ParentConcurrencyGate:
    """Reusable barrier for proving parent samplers can run concurrently."""

    def __init__(
            self,
            *,
            required_arrivals: int,
            timeout_seconds: float,
            skip_initial_arrivals: int = 0,
    ):
        self.required_arrivals = int(required_arrivals)
        self.timeout_seconds = float(timeout_seconds)
        self.skip_initial_arrivals = int(skip_initial_arrivals)
        self.arrival_count = 0
        self.release_count = 0
        self.timeout_count = 0
        self._condition = threading.Condition()

    def arrive_and_wait(self) -> None:
        with self._condition:
            self.arrival_count += 1
            effective_arrivals = self.arrival_count - self.skip_initial_arrivals
            if effective_arrivals <= 0:
                return
            if effective_arrivals >= self.required_arrivals:
                self.release_count += 1
                self._condition.notify_all()
                return
            released = self._condition.wait_for(
                lambda: self.release_count > 0,
                timeout=self.timeout_seconds,
            )
            if not released:
                self.timeout_count += 1


@dataclasses.dataclass(slots=True, frozen=True)
class ParentConcurrencyProbeSampler(UniDimSliceSampler):
    """Sampler fixture that exposes whether two parent tasks overlap."""

    gate: ParentConcurrencyGate | None = None

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point: SeedPoint,
            args=(),
            params=None,
            adaptation_context=None,
    ):
        del key, log_L_constraint, args, params
        assert isinstance(adaptation_context, dict)
        proposal_log_likelihood_fn = adaptation_context.get(
            "proposal_log_likelihood_fn"
        )
        assert callable(proposal_log_likelihood_fn)
        if self.gate is not None:
            self.gate.arrive_and_wait()
        log_L = proposal_log_likelihood_fn(seed_point.U0)
        log_L = jnp.asarray(log_L, dtype=seed_point.log_L0.dtype)
        phantom_samples = PhantomSamples(
            U_samples=None,
            valid_mask=jnp.zeros((0,), dtype=bool),
            log_L=jnp.zeros((0,), dtype=seed_point.log_L0.dtype),
        )
        return (
            seed_point.U0,
            log_L,
            jnp.asarray(1, dtype=jnp.int32),
            phantom_samples,
        )


def _runtime_module():
    return importlib.import_module("jaxns.runtime")


def _public_symbol(
        name: str,
        *,
        modules: tuple[str, ...] = PUBLIC_CONTRACT_MODULES,
):
    for module_name in modules:
        module = importlib.import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    searched = ", ".join(modules)
    pytest.fail(
        f"Ticket 0018 requires public {name} reachable from {searched}."
    )


def _field_names(cls: type) -> tuple[str, ...]:
    assert dataclasses.is_dataclass(cls), (
        f"{cls.__name__} must be a dataclass so its payload schema is "
        "publicly inspectable."
    )
    return tuple(field.name for field in dataclasses.fields(cls))


def _assert_no_forbidden_fields(
        field_names: tuple[str, ...],
        forbidden_tokens: tuple[str, ...],
) -> None:
    offending = tuple(
        field_name
        for field_name in field_names
        for token in forbidden_tokens
        if token in field_name
    )
    assert offending == (), (
        "Likelihood-eval per-request/response payloads must not expose "
        f"runner-local or model payload fields, got {offending!r}."
    )


def _nested_u_case() -> NestedUCase:
    return NestedUCase(
        U={
            "left": jnp.asarray([0.125, 0.25], dtype=jnp.float32),
            "right": (
                np.asarray([[0.5], [0.75]], dtype=np.float32),
                {"scalar": jnp.asarray(0.875, dtype=jnp.float32)},
            ),
        },
        forbidden_context_tokens=(
            b"MODEL_BYTES_SENTINEL",
            b"ARGS_BYTES_SENTINEL",
            b"PARAMS_BYTES_SENTINEL",
            b"SAMPLER_STATE_SENTINEL",
            b"PHANTOM_BUFFER_SENTINEL",
            b"DIRECTION_SNAPSHOT_SENTINEL",
            b"PARENT_ARRAY_SENTINEL",
        ),
    )


def _identity_fixture(
        *,
        task_id: str = "task-000001",
        attempt_id: str = "attempt-000001",
        transport_id: str = "transport-000001",
        eval_id: str = "eval-000001",
) -> EvalIdentityFixture:
    return EvalIdentityFixture(
        runner_id="runner-000001",
        task_id=task_id,
        attempt_id=attempt_id,
        transport_id=transport_id,
        compile_identity_digest="compile-digest-000001",
        eval_id=eval_id,
    )


def _shape_tree_from_public_api(runtime, U: object):
    ShapeDtypeTree = _public_symbol("ShapeDtypeTree")
    for method_name in ("from_pytree", "from_value", "from_tree"):
        method = getattr(ShapeDtypeTree, method_name, None)
        if callable(method):
            return method(U)
    for function_name in (
            "shape_dtype_tree_from_pytree",
            "make_shape_dtype_tree",
    ):
        function = getattr(runtime, function_name, None)
        if callable(function):
            return function(U)
    pytest.fail(
        "Ticket 0018 requires a public ShapeDtypeTree constructor or helper "
        "for deriving static pytree leaf shapes/dtypes from U."
    )


def _request_dataclass_from_u(
        runtime,
        U: object,
        *,
        identity: EvalIdentityFixture | None = None,
        requested_dtype_policy: str = "float32",
        deadline_ms: int | None = None,
):
    Request = _public_symbol("LikelihoodEvalRequest")
    if identity is None:
        identity = _identity_fixture()
    return Request(
        protocol_version=1,
        runner_id=identity.runner_id,
        task_id=identity.task_id,
        attempt_id=identity.attempt_id,
        transport_id=identity.transport_id,
        compile_identity_digest=identity.compile_identity_digest,
        eval_id=identity.eval_id,
        U_bytes=runtime.pickle_payload(U),
        U_shape_tree=_shape_tree_from_public_api(runtime, U),
        requested_dtype_policy=requested_dtype_policy,
        deadline_ms=deadline_ms,
    )


def _response_dataclass(
        *,
        identity: EvalIdentityFixture | None = None,
        status: str = "ok",
        log_L: float | None = -1.25,
        cache_event: str = "hit",
        error_type: str | None = None,
        error_message: str | None = None,
        worker_id: str = "worker-000001",
):
    Response = _public_symbol("LikelihoodEvalResponse")
    if identity is None:
        identity = _identity_fixture()
    return Response(
        protocol_version=1,
        runner_id=identity.runner_id,
        task_id=identity.task_id,
        attempt_id=identity.attempt_id,
        transport_id=identity.transport_id,
        compile_identity_digest=identity.compile_identity_digest,
        eval_id=identity.eval_id,
        status=status,
        log_L=log_L,
        error_type=error_type if status != "ok" else None,
        error_message=error_message if status != "ok" else None,
        worker_id=worker_id,
        cache_event=cache_event,
        elapsed_seconds=0.0,
    )


def _assert_trees_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(
            np.asarray(left_leaf),
            np.asarray(right_leaf),
        )


def _require_method(obj: object, name: str):
    method = getattr(obj, name, None)
    assert callable(method), (
        f"{type(obj).__name__} must expose public {name}(...) for the "
        "Ticket 0018 likelihood-eval dispatch contract."
    )
    return method


def _require_any_method(obj: object, names: tuple[str, ...]):
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return method
    joined = ", ".join(f"{name}(...)" for name in names)
    assert False, (
        f"{type(obj).__name__} must expose one of {joined} for the "
        "Ticket 0018 likelihood-eval dispatch contract."
    )


def _read_attr(obj: object, *names: str):
    containers = [obj]
    for candidate in tuple(containers):
        for container_name in ("identity", "request", "response", "record"):
            if isinstance(candidate, dict):
                nested = candidate.get(container_name)
            else:
                nested = getattr(candidate, container_name, None)
            if nested is not None:
                containers.append(nested)
    for container in containers:
        for name in names:
            if isinstance(container, dict) and name in container:
                return container[name]
            if hasattr(container, name):
                return getattr(container, name)
    pytest.fail(
        "Expected diagnostics/record field not exposed: "
        f"{', '.join(names)}."
    )


def _read_int(obj: object, *names: str) -> int:
    return int(_read_attr(obj, *names))


def _read_float(obj: object, *names: str) -> float:
    return float(_read_attr(obj, *names))


def _read_status(obj: object) -> str:
    return str(_read_attr(obj, "status", "routing_status")).lower()


def _coerce_sequence(value: object) -> tuple[object, ...]:
    if callable(value):
        value = value()
    if value is None:
        return ()
    if isinstance(value, dict):
        for field_name in ("records", "events", "items", "entries"):
            if field_name in value:
                return tuple(value[field_name])
    for field_name in ("records", "events", "items", "entries"):
        if hasattr(value, field_name):
            return tuple(getattr(value, field_name))
    return tuple(value)


def _public_runtime_snapshot(runner: object, lb: object | None = None) -> object:
    for source in tuple(item for item in (runner, lb) if item is not None):
        for method_name in (
                "likelihood_dispatch_snapshot",
                "runtime_dispatch_snapshot",
                "get_likelihood_dispatch_snapshot",
                "get_runtime_dispatch_snapshot",
        ):
            method = getattr(source, method_name, None)
            if callable(method):
                return method()
    pytest.fail(
        "Ticket 0018 requires a public runner/LB dispatch snapshot exposing "
        "accepted sample, out-degree, phantom, and allocation mutation counts."
    )


def _mutation_counts(snapshot: object) -> tuple[int, int, int, int]:
    return (
        _read_int(snapshot, "accepted_sample_count"),
        _read_int(snapshot, "out_degree_update_count"),
        _read_int(snapshot, "phantom_cluster_update_count"),
        _read_int(snapshot, "allocation_update_count"),
    )


def _assert_no_statistical_mutation(
        before: object,
        after: object,
) -> None:
    assert _mutation_counts(after) == _mutation_counts(before)


def _public_likelihood_diagnostics(
        runner: object,
        lb: object | None = None,
) -> object:
    for source in tuple(item for item in (runner, lb) if item is not None):
        for method_name in (
                "likelihood_dispatch_diagnostics",
                "get_likelihood_dispatch_diagnostics",
                "get_diagnostics",
                "diagnostics",
        ):
            method = getattr(source, method_name, None)
            if callable(method):
                diagnostics = method()
                if diagnostics is not None:
                    return diagnostics
    pytest.fail(
        "Ticket 0018 requires public runner/LB likelihood dispatch "
        "diagnostics for worker count, device classes, cache, and capacity."
    )


def _dispatch_record_sequence(diagnostics: object) -> tuple[object, ...]:
    for field_name in (
            "likelihood_eval_records",
            "dispatch_records",
            "eval_records",
    ):
        if isinstance(diagnostics, dict) and field_name in diagnostics:
            return _coerce_sequence(diagnostics[field_name])
        if hasattr(diagnostics, field_name):
            return _coerce_sequence(getattr(diagnostics, field_name))
    return ()


def _response_from_dispatch_record(record: object) -> object | None:
    if isinstance(record, dict):
        return record.get("response")
    return getattr(record, "response", None)


def _completed_ok_eval_records(records: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(
        record
        for record in records
        if (
                _response_from_dispatch_record(record) is not None
                and _read_status(_response_from_dispatch_record(record)) == "ok"
        )
    )


def _completed_eval_count_by_worker(diagnostics: object) -> dict[str, int]:
    return {
        str(worker_id): int(completed_count)
        for worker_id, completed_count in dict(
            _read_attr(diagnostics, "completed_eval_count_by_worker")
        ).items()
    }


def _completed_ok_eval_count_from_diagnostics(diagnostics: object) -> int:
    return sum(_completed_eval_count_by_worker(diagnostics).values())


def _completed_worker_ids_from_diagnostics(diagnostics: object) -> set[str]:
    return {
        worker_id
        for worker_id, completed_count in (
            _completed_eval_count_by_worker(diagnostics).items()
        )
        if completed_count > 0
    }


def _eval_identity_tuple(value: object) -> tuple[str, str, str, str]:
    return (
        str(_read_attr(value, "task_id")),
        str(_read_attr(value, "attempt_id")),
        str(_read_attr(value, "transport_id")),
        str(_read_attr(value, "eval_id")),
    )


def _assert_model_log_likelihood(
        response: object,
        model: DispatchToyModel,
        U: object,
) -> None:
    assert _read_status(response) == "ok"
    actual = float(_read_attr(response, "log_L"))
    expected = float(model.log_likelihood(U, allow_nan=False))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def _normalise_error_category(value: object) -> str:
    return str(value).replace("_", "").replace("-", "").lower()


def _assert_structured_failed_response(
        response: object,
        *,
        error_category: str,
        cache_event: str,
) -> None:
    assert _read_status(response) == "failed"
    assert _read_attr(response, "log_L") is None
    assert (
        _normalise_error_category(_read_attr(response, "error_type"))
        == _normalise_error_category(error_category)
    )
    assert str(_read_attr(response, "error_message"))
    assert str(_read_attr(response, "cache_event")) == cache_event


def _evaluate_likelihood(worker: object, request: object) -> object:
    return _require_any_method(
        worker,
        ("evaluate_likelihood", "handle_likelihood_eval"),
    )(request)


def _make_compile_identity(runtime, model: object, U: object):
    RuntimeCompileIdentity = _public_symbol("RuntimeCompileIdentity")
    from_problem = getattr(RuntimeCompileIdentity, "from_problem", None)
    if callable(from_problem):
        return from_problem(
            model=model,
            args=(),
            params=None,
            dtype_policy="float32",
            device_class="cpu",
            U_shape_tree=_shape_tree_from_public_api(runtime, U),
        )
    builder = getattr(runtime, "build_runtime_compile_identity", None)
    if callable(builder):
        return builder(
            model=model,
            args=(),
            params=None,
            dtype_policy="float32",
            device_class="cpu",
            U_shape_tree=_shape_tree_from_public_api(runtime, U),
        )
    pytest.fail(
        "Ticket 0018 requires public RuntimeCompileIdentity.from_problem(...) "
        "or build_runtime_compile_identity(...) with U shape/dtype metadata."
    )


def _compile_identity_digest(identity: object) -> str:
    return str(
        _read_attr(
            identity,
            "identity_digest",
            "compile_identity_digest",
            "digest",
        )
    )


def _register_identity(target: object, identity: object, model: object) -> None:
    register = _require_any_method(
        target,
        (
            "register_compile_identity",
            "register_runtime_compile_identity",
        ),
    )
    register(identity=identity, model=model, args=(), params=None)


def _diagnostics_from(target: object):
    diagnostics = _require_any_method(
        target,
        (
            "likelihood_dispatch_diagnostics",
            "get_diagnostics",
            "diagnostics",
        ),
    )
    return diagnostics()


def test_likelihood_eval_request_response_public_payload_schema_is_small():
    Request = _public_symbol("LikelihoodEvalRequest")
    Response = _public_symbol("LikelihoodEvalResponse")

    request_fields = _field_names(Request)
    response_fields = _field_names(Response)

    assert REQUIRED_REQUEST_FIELDS <= set(request_fields)
    assert REQUIRED_RESPONSE_FIELDS <= set(response_fields)
    _assert_no_forbidden_fields(
        request_fields,
        REQUEST_FORBIDDEN_FIELD_TOKENS,
    )
    _assert_no_forbidden_fields(
        response_fields,
        RESPONSE_FORBIDDEN_FIELD_TOKENS,
    )


def test_likelihood_eval_request_round_trips_nested_u_without_context_payloads():
    runtime = _runtime_module()
    u_case = _nested_u_case()
    request = _request_dataclass_from_u(runtime, u_case.U)
    response = _response_dataclass(log_L=-3.5)

    round_tripped_request = runtime.unpickle_payload(
        runtime.pickle_payload(request)
    )
    round_tripped_response = runtime.unpickle_payload(
        runtime.pickle_payload(response)
    )
    encoded_request = runtime.pickle_payload(round_tripped_request)
    decoded_U = runtime.unpickle_payload(round_tripped_request.U_bytes)

    _assert_trees_equal(decoded_U, u_case.U)
    assert round_tripped_request.compile_identity_digest == (
        "compile-digest-000001"
    )
    assert round_tripped_request.eval_id == "eval-000001"
    assert round_tripped_response.status == "ok"
    assert type(round_tripped_response.log_L) is float
    assert round_tripped_response.log_L == -3.5
    for forbidden_token in u_case.forbidden_context_tokens:
        assert forbidden_token not in encoded_request


def test_runtime_compile_identity_declares_static_u_shape_dtype_only():
    RuntimeCompileIdentity = _public_symbol("RuntimeCompileIdentity")
    fields = set(_field_names(RuntimeCompileIdentity))

    assert "identity_digest" in fields
    assert any(
        field_name in fields
        for field_name in ("model_digest", "serialized_model_digest")
    )
    assert any(
        field_name in fields
        for field_name in ("args_digest", "serialized_args_digest")
    )
    assert any(
        field_name in fields
        for field_name in ("params_digest", "serialized_params_digest")
    )
    assert "dtype_policy" in fields
    assert "device_class" in fields
    assert "U_shape_tree" in fields

    forbidden = tuple(
        field_name
        for field_name in sorted(fields)
        if any(
            token in field_name
            for token in (
                "sampler",
                "phantom",
                "parent",
                "direction",
                "seed",
                "buffer",
            )
        )
    )
    assert forbidden == (), (
        "RuntimeCompileIdentity for likelihood dispatch must not depend on "
        f"sampler-local state, got {forbidden!r}."
    )


def test_worker_evaluates_model_once_then_reuses_matching_compile_cache():
    runtime = _runtime_module()
    Worker = _public_symbol("LikelihoodEvalWorker")
    model = DispatchToyModel()
    first_U = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    second_U = jnp.asarray([0.5, 0.75], dtype=jnp.float32)
    mismatched_U = jnp.asarray([[0.125, 0.25]], dtype=jnp.float32)
    identity = _make_compile_identity(runtime, model, first_U)
    worker = Worker(worker_id="worker-000001", device_class="cpu")

    _register_identity(worker, identity, model)
    before = _diagnostics_from(worker)
    assert _read_int(before, "compile_count", "jit_compile_count") == 0

    digest = _compile_identity_digest(identity)
    first_response = _evaluate_likelihood(
        worker,
        _request_dataclass_from_u(
            runtime,
            first_U,
            identity=_identity_fixture(eval_id="eval-000001")._replace(
                compile_identity_digest=digest,
            ),
        ),
    )
    second_response = _evaluate_likelihood(
        worker,
        _request_dataclass_from_u(
            runtime,
            second_U,
            identity=_identity_fixture(eval_id="eval-000002")._replace(
                compile_identity_digest=digest,
            ),
        ),
    )
    rejected_response = _evaluate_likelihood(
        worker,
        _request_dataclass_from_u(
            runtime,
            mismatched_U,
            identity=_identity_fixture(eval_id="eval-000003")._replace(
                compile_identity_digest=digest,
            ),
        ),
    )
    after = _diagnostics_from(worker)

    _assert_model_log_likelihood(first_response, model, first_U)
    assert first_response.cache_event == "compile"
    assert type(_read_attr(first_response, "log_L")) is float
    _assert_model_log_likelihood(second_response, model, second_U)
    assert second_response.cache_event == "hit"
    _assert_structured_failed_response(
        rejected_response,
        error_category="shape_mismatch",
        cache_event="rejected",
    )
    assert _read_int(after, "compile_count", "jit_compile_count") == 1
    assert _read_int(after, "cache_hit_count") == 1
    assert _read_int(after, "rejected_shape_cache_count") == 1


def test_worker_rejects_shape_dtype_and_pytree_mismatches_without_recompile():
    runtime = _runtime_module()
    Worker = _public_symbol("LikelihoodEvalWorker")
    model = DispatchToyModel()
    valid_U = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    identity = _make_compile_identity(runtime, model, valid_U)
    worker = Worker(worker_id="worker-000001", device_class="cpu")

    _register_identity(worker, identity, model)
    digest = _compile_identity_digest(identity)
    valid_response = _evaluate_likelihood(
        worker,
        _request_dataclass_from_u(
            runtime,
            valid_U,
            identity=_identity_fixture(eval_id="eval-000001")._replace(
                compile_identity_digest=digest,
            ),
        ),
    )
    assert _read_status(valid_response) == "ok"
    assert str(_read_attr(valid_response, "cache_event")) == "compile"
    compile_count_after_valid = _read_int(
        _diagnostics_from(worker),
        "compile_count",
        "jit_compile_count",
    )

    mismatch_cases = (
        (
            jnp.asarray([[0.125, 0.25]], dtype=jnp.float32),
            "ShapeMismatch",
            "eval-shape-mismatch",
        ),
        (
            jnp.asarray([1, 2], dtype=jnp.int32),
            "DtypeMismatch",
            "eval-dtype-mismatch",
        ),
        (
            {"proposal": jnp.asarray([0.125, 0.25], dtype=jnp.float32)},
            "PytreeMismatch",
            "eval-pytree-mismatch",
        ),
    )
    for U, expected_error_type, eval_id in mismatch_cases:
        response = _evaluate_likelihood(
            worker,
            _request_dataclass_from_u(
                runtime,
                U,
                identity=_identity_fixture(eval_id=eval_id)._replace(
                    compile_identity_digest=digest,
                ),
            ),
        )
        _assert_structured_failed_response(
            response,
            error_category=expected_error_type,
            cache_event="rejected",
        )
        assert _read_int(
            _diagnostics_from(worker),
            "compile_count",
            "jit_compile_count",
        ) == compile_count_after_valid

    diagnostics = _diagnostics_from(worker)
    assert _read_int(diagnostics, "compile_count", "jit_compile_count") == 1
    assert _read_int(diagnostics, "rejected_shape_cache_count") == 3


def test_worker_returns_structured_failures_for_bad_payload_identity_and_timeout():
    runtime = _runtime_module()
    Worker = _public_symbol("LikelihoodEvalWorker")
    model = DispatchToyModel()
    valid_U = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    identity = _make_compile_identity(runtime, model, valid_U)
    digest = _compile_identity_digest(identity)
    worker = Worker(worker_id="worker-000001", device_class="cpu")
    _register_identity(worker, identity, model)

    base_request = _request_dataclass_from_u(
        runtime,
        valid_U,
        identity=_identity_fixture()._replace(
            compile_identity_digest=digest,
        ),
    )
    compile_count_before_failures = _read_int(
        _diagnostics_from(worker),
        "compile_count",
        "jit_compile_count",
    )
    failure_cases = (
        (
            dataclasses.replace(base_request, U_bytes=b"not-a-pickle"),
            "MalformedUPayload",
        ),
        (
            dataclasses.replace(
                base_request,
                compile_identity_digest="unknown-compile-identity",
            ),
            "UnknownCompileIdentity",
        ),
        (
            dataclasses.replace(base_request, deadline_ms=0),
            "LikelihoodEvalTimeout",
        ),
    )
    for request, expected_error_type in failure_cases:
        response = _evaluate_likelihood(worker, request)
        _assert_structured_failed_response(
            response,
            error_category=expected_error_type,
            cache_event="rejected",
        )
        assert _read_int(
            _diagnostics_from(worker),
            "compile_count",
            "jit_compile_count",
        ) == compile_count_before_failures

    exception_model = WorkerExceptionModel()
    exception_identity = _make_compile_identity(
        runtime,
        exception_model,
        valid_U,
    )
    exception_worker = Worker(worker_id="worker-000002", device_class="cpu")
    _register_identity(exception_worker, exception_identity, exception_model)
    compile_count_before_exception = _read_int(
        _diagnostics_from(exception_worker),
        "compile_count",
        "jit_compile_count",
    )
    exception_response = _evaluate_likelihood(
        exception_worker,
        _request_dataclass_from_u(
            runtime,
            valid_U,
            identity=_identity_fixture(eval_id="eval-worker-exception")._replace(
                compile_identity_digest=_compile_identity_digest(
                    exception_identity
                ),
            ),
        ),
    )
    _assert_structured_failed_response(
        exception_response,
        error_category="worker_exception",
        cache_event="rejected",
    )
    assert _read_int(
        _diagnostics_from(exception_worker),
        "compile_count",
        "jit_compile_count",
    ) == compile_count_before_exception


def test_worker_validates_request_metadata_and_dtype_policy_against_identity():
    runtime = _runtime_module()
    Worker = _public_symbol("LikelihoodEvalWorker")
    model = DispatchToyModel()
    valid_U = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    identity = _make_compile_identity(runtime, model, valid_U)
    digest = _compile_identity_digest(identity)
    worker = Worker(worker_id="worker-000001", device_class="cpu")
    _register_identity(worker, identity, model)

    base_request = _request_dataclass_from_u(
        runtime,
        valid_U,
        identity=_identity_fixture()._replace(
            compile_identity_digest=digest,
        ),
    )
    tampered_metadata_response = _evaluate_likelihood(
        worker,
        dataclasses.replace(
            base_request,
            U_shape_tree=_shape_tree_from_public_api(
                runtime,
                jnp.asarray([[0.125, 0.25]], dtype=jnp.float32),
            ),
        ),
    )
    dtype_policy_response = _evaluate_likelihood(
        worker,
        dataclasses.replace(
            base_request,
            requested_dtype_policy="float64",
            eval_id="eval-dtype-policy",
        ),
    )
    diagnostics = _diagnostics_from(worker)

    _assert_structured_failed_response(
        tampered_metadata_response,
        error_category="RequestMetadataShapeMismatch",
        cache_event="rejected",
    )
    _assert_structured_failed_response(
        dtype_policy_response,
        error_category="DtypePolicyMismatch",
        cache_event="rejected",
    )
    assert _read_int(diagnostics, "compile_count", "jit_compile_count") == 0
    assert _read_int(diagnostics, "rejected_shape_cache_count") == 2


def test_worker_returns_structured_failure_for_malformed_u_shape_tree_metadata():
    runtime = _runtime_module()
    Worker = _public_symbol("LikelihoodEvalWorker")
    model = DispatchToyModel()
    valid_U = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    identity = _make_compile_identity(runtime, model, valid_U)
    digest = _compile_identity_digest(identity)
    worker = Worker(worker_id="worker-000001", device_class="cpu")
    _register_identity(worker, identity, model)

    request = _request_dataclass_from_u(
        runtime,
        valid_U,
        identity=_identity_fixture()._replace(
            compile_identity_digest=digest,
        ),
    )
    response = _evaluate_likelihood(
        worker,
        dataclasses.replace(
            request,
            U_shape_tree={"malformed": "metadata"},
        ),
    )
    diagnostics = _diagnostics_from(worker)

    _assert_structured_failed_response(
        response,
        error_category="RequestMetadataMalformedUShapeTreeMetadata",
        cache_event="rejected",
    )
    assert _read_int(diagnostics, "compile_count", "jit_compile_count") == 0
    assert _read_int(diagnostics, "rejected_shape_cache_count") == 1


def test_worker_rejects_registered_identity_for_wrong_device_class():
    runtime = _runtime_module()
    Worker = _public_symbol("LikelihoodEvalWorker")
    RuntimeCompileIdentity = _public_symbol("RuntimeCompileIdentity")
    model = DispatchToyModel()
    valid_U = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    identity = RuntimeCompileIdentity.from_problem(
        model=model,
        args=(),
        params=None,
        dtype_policy="float32",
        device_class="gpu",
        U_shape_tree=_shape_tree_from_public_api(runtime, valid_U),
    )
    digest = _compile_identity_digest(identity)
    worker = Worker(worker_id="worker-000001", device_class="cpu")
    _register_identity(worker, identity, model)

    response = _evaluate_likelihood(
        worker,
        _request_dataclass_from_u(
            runtime,
            valid_U,
            identity=_identity_fixture()._replace(
                compile_identity_digest=digest,
            ),
        ),
    )
    diagnostics = _diagnostics_from(worker)

    _assert_structured_failed_response(
        response,
        error_category="DeviceClassMismatch",
        cache_event="rejected",
    )
    assert _read_int(diagnostics, "compile_count", "jit_compile_count") == 0
    assert _read_int(diagnostics, "cache_hit_count") == 0


def test_single_worker_capacity_queues_second_eval_then_starts_it_in_order():
    runtime = _runtime_module()
    Scheduler = _public_symbol("LikelihoodEvalScheduler")
    model = DispatchToyModel()
    U = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    identity = _make_compile_identity(runtime, model, U)
    scheduler = Scheduler(requested_worker_specs=("cpu:*:1",))

    _register_identity(scheduler, identity, model)
    start_eval = _require_method(scheduler, "start_likelihood_eval")
    complete_eval = _require_method(scheduler, "complete_likelihood_eval")
    digest = _compile_identity_digest(identity)
    first_identity = _identity_fixture(eval_id="eval-000001")._replace(
        compile_identity_digest=digest,
    )
    second_identity = _identity_fixture(
        transport_id="transport-000002",
        eval_id="eval-000002",
    )._replace(compile_identity_digest=digest)
    first_record = start_eval(
        request=_request_dataclass_from_u(
            runtime,
            U,
            identity=first_identity,
        )
    )
    second_record = start_eval(
        request=_request_dataclass_from_u(
            runtime,
            U + jnp.asarray([0.25, 0.25], dtype=jnp.float32),
            identity=second_identity,
        )
    )
    worker_id = str(_read_attr(first_record, "worker_id"))
    completion = complete_eval(
        worker_id=worker_id,
        response=_response_dataclass(
            identity=first_identity,
            log_L=float(model.log_likelihood(U, allow_nan=False)),
            cache_event="compile",
            worker_id=worker_id,
        ),
    )
    started_after_completion = _read_attr(
        completion,
        "started_record",
        "next_started_record",
    )
    diagnostics = _diagnostics_from(scheduler)

    assert _read_status(first_record) in {"started", "active", "running"}
    assert _read_status(second_record) in {"queued", "pending"}
    assert _eval_identity_tuple(started_after_completion) == (
        second_identity.task_id,
        second_identity.attempt_id,
        second_identity.transport_id,
        second_identity.eval_id,
    )
    assert _read_status(started_after_completion) in {
        "started",
        "active",
        "running",
    }
    assert str(_read_attr(started_after_completion, "worker_id")) == worker_id
    max_by_worker = dict(_read_attr(diagnostics, "max_active_evals_per_worker"))
    assert max(max_by_worker.values()) == 1
    assert _read_int(diagnostics, "max_active_evals_pool") == 1
    assert _read_int(diagnostics, "queued_eval_count") == 1


def test_multi_worker_capacity_never_exceeds_one_active_eval_per_worker():
    runtime = _runtime_module()
    Scheduler = _public_symbol("LikelihoodEvalScheduler")
    model = DispatchToyModel()
    U = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    identity = _make_compile_identity(runtime, model, U)
    scheduler = Scheduler(requested_worker_specs=("cpu:*:2",))

    _register_identity(scheduler, identity, model)
    start_eval = _require_method(scheduler, "start_likelihood_eval")
    digest = _compile_identity_digest(identity)
    records = []
    for idx in range(3):
        records.append(
            start_eval(
                request=_request_dataclass_from_u(
                    runtime,
                    U + jnp.asarray([0.05 * idx, 0.0], dtype=jnp.float32),
                    identity=_identity_fixture(
                        transport_id=f"transport-00000{idx + 1}",
                        eval_id=f"eval-00000{idx + 1}",
                    )._replace(compile_identity_digest=digest),
                )
            )
        )
    diagnostics = _diagnostics_from(scheduler)
    active_records = tuple(
        record
        for record in records
        if _read_status(record) in {"started", "active", "running", "rerouted"}
    )

    assert len(active_records) == 2
    assert len({str(_read_attr(record, "worker_id")) for record in active_records}) == 2
    assert _read_status(records[2]) in {"queued", "pending", "rerouted"}
    max_by_worker = dict(_read_attr(diagnostics, "max_active_evals_per_worker"))
    assert max(max_by_worker.values()) == 1
    assert _read_int(diagnostics, "observed_worker_count", "worker_count") == 2
    assert _read_int(diagnostics, "max_active_evals_pool") <= 2


def test_scheduler_rejects_stale_response_with_old_compile_identity_digest():
    runtime = _runtime_module()
    Scheduler = _public_symbol("LikelihoodEvalScheduler")
    model = DispatchToyModel()
    old_model = DispatchToyModel(centre=0.75)
    U = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    current_identity = _make_compile_identity(runtime, model, U)
    old_identity = _make_compile_identity(runtime, old_model, U)
    current_digest = _compile_identity_digest(current_identity)
    old_digest = _compile_identity_digest(old_identity)
    shared_identity = _identity_fixture(eval_id="eval-stale-identity")
    current_request = _request_dataclass_from_u(
        runtime,
        U,
        identity=shared_identity._replace(
            compile_identity_digest=current_digest,
        ),
    )
    scheduler = Scheduler(requested_worker_specs=("cpu:*:1",))

    _register_identity(scheduler, old_identity, old_model)
    _register_identity(scheduler, current_identity, model)
    current_record = scheduler.start_likelihood_eval(
        request=current_request,
    )
    worker_id = str(_read_attr(current_record, "worker_id"))
    stale_completion = scheduler.complete_likelihood_eval(
        worker_id=worker_id,
        response=_response_dataclass(
            identity=shared_identity._replace(
                compile_identity_digest=old_digest,
            ),
            log_L=-100.0,
            cache_event="compile",
            worker_id=worker_id,
        ),
    )
    still_started = scheduler.wait_for_started_record(
        current_request,
        timeout_seconds=0.0,
    )
    current_completion = scheduler.complete_likelihood_eval(
        worker_id=worker_id,
        response=_response_dataclass(
            identity=shared_identity._replace(
                compile_identity_digest=current_digest,
            ),
            log_L=float(model.log_likelihood(U, allow_nan=False)),
            cache_event="compile",
            worker_id=worker_id,
        ),
    )
    diagnostics = _diagnostics_from(scheduler)

    assert _read_attr(stale_completion, "completed_record") is None
    assert _read_attr(stale_completion, "started_record") is None
    assert still_started is current_record
    assert _read_attr(current_completion, "completed_record") is current_record
    assert _completed_eval_count_by_worker(diagnostics) == {worker_id: 1}
    assert _read_int(diagnostics, "dispatch_eval_count") == 1
    assert _read_int(diagnostics, "compile_count", "jit_compile_count") == 1
    assert _read_int(diagnostics, "cache_hit_count") == 0


def test_queued_likelihood_eval_timeout_cancels_queue_and_routes_failure():
    runtime = _runtime_module()
    Scheduler = _public_symbol("LikelihoodEvalScheduler")
    Router = _public_symbol("LikelihoodEvalResponseRouter")
    model = DispatchToyModel()
    U = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    identity = _make_compile_identity(runtime, model, U)
    digest = _compile_identity_digest(identity)
    scheduler = Scheduler(requested_worker_specs=("cpu:*:1",))
    router = Router()

    _register_identity(scheduler, identity, model)
    start_eval = _require_method(scheduler, "start_likelihood_eval")
    complete_eval = _require_method(scheduler, "complete_likelihood_eval")
    cancel_eval = _require_method(scheduler, "cancel_likelihood_eval")
    route_eval_response = _require_any_method(
        router,
        ("route_eval_response", "route_response"),
    )
    first_identity = _identity_fixture(eval_id="eval-000001")._replace(
        compile_identity_digest=digest,
    )
    second_identity = _identity_fixture(
        transport_id="transport-000002",
        eval_id="eval-000002",
    )._replace(compile_identity_digest=digest)
    first_request = _request_dataclass_from_u(
        runtime,
        U,
        identity=first_identity,
    )
    second_request = _request_dataclass_from_u(
        runtime,
        U + jnp.asarray([0.25, 0.25], dtype=jnp.float32),
        identity=second_identity,
    )
    router.register_pending(first_request)
    router.register_pending(second_request)

    first_record = start_eval(request=first_request)
    queued_record = start_eval(request=second_request)
    failed_response = cancel_eval(request=second_request)
    routed_failure = route_eval_response(failed_response)
    completion = complete_eval(
        worker_id=str(_read_attr(first_record, "worker_id")),
        response=_response_dataclass(
            identity=first_identity,
            log_L=float(model.log_likelihood(U, allow_nan=False)),
            cache_event="compile",
            worker_id=str(_read_attr(first_record, "worker_id")),
        ),
    )
    diagnostics = _diagnostics_from(scheduler)
    snapshot = router.snapshot()

    assert _read_status(first_record) in {"started", "active", "running"}
    assert _read_status(queued_record) in {"queued", "pending"}
    _assert_structured_failed_response(
        failed_response,
        error_category="LikelihoodEvalTimeout",
        cache_event="rejected",
    )
    assert _read_status(routed_failure) in {"delivered", "completed", "ok"}
    assert _read_attr(
        completion,
        "started_record",
        "next_started_record",
    ) is None
    assert _read_int(snapshot, "pending_eval_count") == 1
    assert _read_int(snapshot, "delivered_eval_count") == 1
    assert _read_int(diagnostics, "failed_eval_count") == 1


def test_queued_dispatch_timeout_cancellation_is_atomic_with_completion_race(
        monkeypatch,
):
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    model = DispatchToyModel()
    U = jnp.asarray([0.125, 0.25], dtype=jnp.float32)

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=3,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=3),
            batch_size=None,
        )
        scheduler = runner._ensure_likelihood_scheduler()
        assert callable(
            _require_method(
                scheduler,
                "wait_for_started_record_or_cancel_likelihood_eval",
            )
        )

        blocking_request = runner.make_likelihood_eval_request(
            U=U,
            task_id="task-blocking",
            attempt_id="attempt-blocking",
            transport_id="transport-blocking",
            eval_id="eval-blocking",
        )
        blocking_record = scheduler.start_likelihood_eval(
            request=blocking_request,
        )
        worker_id = str(_read_attr(blocking_record, "worker_id"))
        worker = scheduler.worker_for_id(worker_id)

        original_cancel = scheduler.cancel_likelihood_eval
        legacy_cancel_entered = threading.Event()
        raced_completion_finished = threading.Event()
        raced_completion_errors = []

        def racing_cancel(
                *,
                request,
                error_type="LikelihoodEvalTimeout",
                error_message=(
                    "Timed out waiting for a queued likelihood evaluation "
                    "to start."
                ),
        ):
            legacy_cancel_entered.set()

            def complete_active_request() -> None:
                try:
                    blocking_response = worker.evaluate_likelihood(
                        blocking_request
                    )
                    scheduler.complete_likelihood_eval(
                        worker_id=worker_id,
                        response=blocking_response,
                    )
                except Exception as error:
                    raced_completion_errors.append(error)
                finally:
                    raced_completion_finished.set()

            completion_thread = threading.Thread(
                target=complete_active_request,
                name="likelihood-timeout-race-completion",
            )
            completion_thread.start()
            assert raced_completion_finished.wait(timeout=2.0)
            completion_thread.join(timeout=1.0)
            assert not completion_thread.is_alive()
            if raced_completion_errors:
                raise raced_completion_errors[0]
            return original_cancel(
                request=request,
                error_type=error_type,
                error_message=error_message,
            )

        monkeypatch.setattr(scheduler, "cancel_likelihood_eval", racing_cancel)

        timed_out_response = runner.dispatch_likelihood_eval(
            U=U + jnp.asarray([0.25, 0.25], dtype=jnp.float32),
            task_id="task-timeout",
            attempt_id="attempt-timeout",
            transport_id="transport-timeout",
            eval_id="eval-timeout",
            deadline_ms=0,
        )
        assert not legacy_cancel_entered.is_set()
        _assert_structured_failed_response(
            timed_out_response,
            error_category="LikelihoodEvalTimeout",
            cache_event="rejected",
        )

        blocking_response = worker.evaluate_likelihood(blocking_request)
        completion = scheduler.complete_likelihood_eval(
            worker_id=worker_id,
            response=blocking_response,
        )
        subsequent_response = runner.dispatch_likelihood_eval(
            U=U + jnp.asarray([0.1, 0.0], dtype=jnp.float32),
            task_id="task-after-timeout",
            attempt_id="attempt-after-timeout",
            transport_id="transport-after-timeout",
            eval_id="eval-after-timeout",
        )
        diagnostics = lb.likelihood_dispatch_diagnostics()

    assert _read_status(blocking_record) in {"started", "active", "running"}
    assert _read_attr(completion, "started_record", "next_started_record") is None
    _assert_model_log_likelihood(
        subsequent_response,
        model,
        U + jnp.asarray([0.1, 0.0], dtype=jnp.float32),
    )
    assert _read_int(diagnostics, "failed_eval_count") == 1


def test_started_likelihood_eval_cancellation_keeps_worker_occupied_until_completion():
    runtime = _runtime_module()
    Scheduler = _public_symbol("LikelihoodEvalScheduler")
    gate_id = "started-cancel-occupancy"
    gate = BlockingLikelihoodGate()
    _BLOCKING_LIKELIHOOD_GATES[gate_id] = gate
    model = BlockingDispatchModel(gate_id=gate_id)
    U = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    identity = _make_compile_identity(runtime, model, U)
    digest = _compile_identity_digest(identity)
    scheduler = Scheduler(requested_worker_specs=("cpu:*:1",))

    try:
        _register_identity(scheduler, identity, model)
        start_eval = _require_method(scheduler, "start_likelihood_eval")
        complete_eval = _require_method(scheduler, "complete_likelihood_eval")
        cancel_eval = _require_method(scheduler, "cancel_likelihood_eval")
        first_identity = _identity_fixture(eval_id="eval-000001")._replace(
            compile_identity_digest=digest,
        )
        second_identity = _identity_fixture(
            transport_id="transport-000002",
            eval_id="eval-000002",
        )._replace(compile_identity_digest=digest)
        first_request = _request_dataclass_from_u(
            runtime,
            U,
            identity=first_identity,
        )
        second_U = U + jnp.asarray([0.25, 0.25], dtype=jnp.float32)
        second_request = _request_dataclass_from_u(
            runtime,
            second_U,
            identity=second_identity,
        )

        first_record = start_eval(request=first_request)
        queued_second_record = start_eval(request=second_request)
        worker_id = str(_read_attr(first_record, "worker_id"))
        worker = scheduler.worker_for_id(worker_id)
        first_result: dict[str, object] = {}
        first_errors: list[BaseException] = []

        def complete_first_physical_eval() -> None:
            try:
                response = worker.evaluate_likelihood(first_request)
                completion = complete_eval(
                    worker_id=worker_id,
                    response=response,
                )
                first_result["response"] = response
                first_result["completion"] = completion
            except BaseException as error:
                first_errors.append(error)

        first_thread = threading.Thread(
            target=complete_first_physical_eval,
            name="started-cancel-physical-worker",
        )
        first_thread.start()
        assert gate.started.wait(timeout=5.0)

        failed_response = cancel_eval(
            request=first_request,
            error_type="LikelihoodEvalCancelled",
            error_message="Cancelled started likelihood evaluation.",
        )
        second_started_before_physical_completion = (
            scheduler.wait_for_started_record(
                second_request,
                timeout_seconds=0.0,
            )
        )
        gate.release.set()
        first_thread.join(timeout=5.0)
        assert not first_thread.is_alive()
        if first_errors:
            raise first_errors[0]

        stale_completion = first_result["completion"]
        stale_response = first_result["response"]
        second_started_record = _read_attr(
            stale_completion,
            "started_record",
            "next_started_record",
        )
        second_response = worker.evaluate_likelihood(second_request)
        second_completion = complete_eval(
            worker_id=worker_id,
            response=second_response,
        )
        diagnostics = _diagnostics_from(scheduler)
    finally:
        _BLOCKING_LIKELIHOOD_GATES.pop(gate_id, None)

    assert _read_status(first_record) in {"started", "active", "running"}
    assert _read_status(queued_second_record) in {"queued", "pending"}
    _assert_structured_failed_response(
        failed_response,
        error_category="LikelihoodEvalCancelled",
        cache_event="rejected",
    )
    assert second_started_before_physical_completion is None
    assert _read_status(stale_response) == "ok"
    assert _read_attr(stale_response, "error_type") is None
    assert second_started_record is not None
    assert _eval_identity_tuple(second_started_record) == (
        second_identity.task_id,
        second_identity.attempt_id,
        second_identity.transport_id,
        second_identity.eval_id,
    )
    assert _read_attr(
        stale_completion,
        "completed_record",
    ) is None
    assert str(_read_attr(second_started_record, "worker_id")) == worker_id
    assert _read_status(second_response) == "ok"
    assert _read_attr(second_response, "error_type") is None
    expected_second_log_likelihood = -np.sum(
        np.square(np.asarray(second_U) - model.centre)
    )
    np.testing.assert_allclose(
        float(_read_attr(second_response, "log_L")),
        expected_second_log_likelihood,
        rtol=1e-6,
        atol=1e-7,
    )
    assert _eval_identity_tuple(
        _read_attr(second_completion, "completed_record")
    ) == (
        second_identity.task_id,
        second_identity.attempt_id,
        second_identity.transport_id,
        second_identity.eval_id,
    )
    assert _read_int(diagnostics, "failed_eval_count") == 1
    assert _completed_eval_count_by_worker(diagnostics) == {worker_id: 1}
    assert _completed_ok_eval_count_from_diagnostics(diagnostics) == 1
    assert _read_int(diagnostics, "dispatch_eval_count") == 2
    assert (
        _read_int(diagnostics, "compile_count", "jit_compile_count")
        + _read_int(diagnostics, "cache_hit_count")
    ) == 1
    assert _read_int(diagnostics, "max_active_evals_pool") == 1


def test_response_router_ignores_duplicate_stale_and_late_eval_responses():
    runtime = _runtime_module()
    Router = _public_symbol("LikelihoodEvalResponseRouter")
    router = Router()
    request = _request_dataclass_from_u(
        runtime,
        jnp.asarray([0.125, 0.25], dtype=jnp.float32),
    )
    register_pending = _require_method(router, "register_pending")
    route_eval_response = _require_any_method(
        router,
        ("route_eval_response", "route_response"),
    )
    snapshot = _require_method(router, "snapshot")

    register_pending(request)
    before = snapshot()
    delivered = route_eval_response(_response_dataclass(log_L=-0.5))
    after_delivered = snapshot()
    duplicate = route_eval_response(_response_dataclass(log_L=-0.5))
    stale = route_eval_response(
        _response_dataclass(
            identity=_identity_fixture(
                attempt_id="attempt-previous",
            )._replace(
                compile_identity_digest="compile-digest-000001",
            ),
            log_L=-0.25,
        )
    )
    late_eval = route_eval_response(
        _response_dataclass(
            identity=_identity_fixture(eval_id="eval-late")._replace(
                compile_identity_digest="compile-digest-000001",
            ),
            log_L=-0.125,
        )
    )
    after_rejected = snapshot()

    assert _read_status(delivered) in {"delivered", "completed", "ok"}
    assert _read_status(duplicate) in {
        "duplicate_eval_response",
        "duplicate_task_result",
    }
    assert _read_status(stale) in {
        "stale_eval_response",
        "stale_task_result",
        "mismatched_result_identity",
    }
    assert _read_status(late_eval) in {
        "late_eval_response",
        "stale_eval_response",
        "stale_task_result",
        "mismatched_result_identity",
    }
    for field_name in (
            "accepted_sample_count",
            "out_degree_update_count",
            "phantom_cluster_update_count",
            "allocation_update_count",
    ):
        assert _read_int(after_delivered, field_name) == (
            _read_int(before, field_name)
        )
        assert _read_int(after_rejected, field_name) == (
            _read_int(before, field_name)
        )


def test_response_router_does_not_deliver_stale_compile_identity_response():
    runtime = _runtime_module()
    Router = _public_symbol("LikelihoodEvalResponseRouter")
    current_identity = _identity_fixture(
        eval_id="eval-stale-compile-identity",
    )._replace(compile_identity_digest="compile-digest-current")
    old_identity = current_identity._replace(
        compile_identity_digest="compile-digest-old",
    )
    request = _request_dataclass_from_u(
        runtime,
        jnp.asarray([0.125, 0.25], dtype=jnp.float32),
        identity=current_identity,
    )
    router = Router()
    register_pending = _require_method(router, "register_pending")
    route_eval_response = _require_any_method(
        router,
        ("route_eval_response", "route_response"),
    )

    register_pending(request)
    stale = route_eval_response(
        _response_dataclass(
            identity=old_identity,
            log_L=-100.0,
        )
    )
    after_stale = router.snapshot()
    delivered = route_eval_response(
        _response_dataclass(
            identity=current_identity,
            log_L=-0.5,
        )
    )
    after_delivered = router.snapshot()

    assert _read_status(stale) in {
        "stale_eval_response",
        "stale_task_result",
        "mismatched_result_identity",
    }
    assert _read_int(after_stale, "pending_eval_count") == 1
    assert _read_int(after_stale, "delivered_eval_count") == 0
    assert _read_status(delivered) in {"delivered", "completed", "ok"}
    assert _read_int(after_delivered, "pending_eval_count") == 0
    assert _read_int(after_delivered, "delivered_eval_count") == 1


def test_local_lb_raw_likelihood_response_does_not_mutate_runner_statistics():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    model = DispatchToyModel()
    U = jnp.asarray([0.125, 0.5], dtype=jnp.float32)

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=True,
            target_num_live_points=2,
            max_samples=3,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=3),
            batch_size=None,
        )
        dispatch_likelihood = _require_any_method(
            runner,
            (
                "dispatch_likelihood_eval",
                "evaluate_likelihood_via_dispatch",
                "request_likelihood_eval",
            ),
        )

        before = _public_runtime_snapshot(runner, lb)
        response = dispatch_likelihood(
            U=U,
            task_id="task-000001",
            attempt_id="attempt-000001",
            transport_id="transport-000001",
            eval_id="eval-000001",
        )
        after = _public_runtime_snapshot(runner, lb)

    _assert_model_log_likelihood(response, model, U)
    _assert_no_statistical_mutation(before, after)


def test_raw_likelihood_response_is_rejected_as_sampler_completion_payload():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    runtime = _runtime_module()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=DispatchToyModel(),
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=3,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=3),
            batch_size=None,
        )
        before = tuple(runner.runtime_acceptance_ledger.accepted_task_ids)
        dispatch_record = runner.prepare_runtime_dispatch(
            requested_parent_idx=0,
            effective_parent_idx=0,
            accepted_parent_idx=0,
            effective_log_L_constraint=-0.5,
            accepted_log_L_constraint=-0.5,
            seed_id="seed-000001",
            phantom_cluster_id="phantom-cluster-000001",
        )
        identity = _identity_fixture(
            task_id=str(_read_attr(dispatch_record, "task_id")),
            attempt_id=str(_read_attr(dispatch_record, "attempt_id")),
            transport_id=str(_read_attr(dispatch_record, "transport_id")),
        )._replace(
            runner_id=str(_read_attr(dispatch_record, "runner_id")),
            compile_identity_digest=(
                runner.runtime_compile_identity.identity_digest
            ),
        )
        rejected_record = runner.complete_runtime_dispatch(
            dispatch_record=dispatch_record,
            result=runtime.LikelihoodEvalResponse(
                protocol_version=1,
                runner_id=identity.runner_id,
                task_id=identity.task_id,
                attempt_id=identity.attempt_id,
                transport_id=identity.transport_id,
                compile_identity_digest=identity.compile_identity_digest,
                eval_id=identity.eval_id,
                status="ok",
                log_L=-0.25,
                error_type=None,
                error_message=None,
                worker_id="worker-000001",
                cache_event="hit",
                elapsed_seconds=0.0,
            ),
            current_parent_idx=0,
            current_effective_log_L_constraint=-0.5,
        )
        after = tuple(runner.runtime_acceptance_ledger.accepted_task_ids)

    assert _read_status(rejected_record) == "invalid_result_payload"
    assert after == before


def test_local_sampler_state_is_not_required_for_likelihood_dispatch_payload():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=DispatchToyModel(),
            collect_phantoms=True,
            sampler=NonSerializableLocalSampler(),
            target_num_live_points=2,
            max_samples=3,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=3),
            batch_size=None,
        )

        make_request = _require_any_method(
            runner,
            (
                "make_likelihood_eval_request",
                "prepare_likelihood_eval_request",
            ),
        )
        signature = inspect.signature(make_request)
        assert "U" in signature.parameters
        request = make_request(
            U=jnp.asarray([0.125, 0.25], dtype=jnp.float32),
            task_id="task-000001",
            attempt_id="attempt-000001",
            transport_id="transport-000001",
            eval_id="eval-000001",
        )

    request_fields = _field_names(type(request))
    _assert_no_forbidden_fields(
        request_fields,
        REQUEST_FORBIDDEN_FIELD_TOKENS,
    )


def test_local_lb_run_keeps_sampler_state_local_and_dispatches_likelihood_evals():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    sampler = NonSerializableLocalSampler()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=DispatchToyModel(),
            collect_phantoms=True,
            sampler=sampler,
            target_num_live_points=2,
            max_samples=3,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=3),
            batch_size=None,
        )
        state = runner.run_until_goal(
            goal_cond=lambda state: int(state.num_samples) >= 3,
            depth_cond=TerminationCondition(max_samples=3),
            allocation_target="uniform",
            key=jax.random.PRNGKey(7),
            max_goal_iterations=2,
        )

    assert int(state.num_samples) == 3
    assert sampler.seed_choice == "seed-choice-must-stay-local"
    assert sampler.direction_snapshot["basis"].shape == (2, 2)
    np.testing.assert_array_equal(sampler.phantom_buffer, [0.1, 0.2])


def test_real_unidim_sampler_proposal_likelihoods_cross_dispatch_boundary():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    model = DispatchToyModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=2,
        no_step_out=True,
        collect_phantom_samples=False,
    )

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=False,
            sampler=sampler,
            target_num_live_points=2,
            max_samples=3,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=3),
            batch_size=None,
        )
        state = runner.run_until_goal(
            goal_cond=lambda state: int(state.num_samples) >= 3,
            depth_cond=TerminationCondition(max_samples=3),
            allocation_target="uniform",
            key=jax.random.PRNGKey(11),
            max_goal_iterations=2,
        )
        diagnostics = _public_likelihood_diagnostics(runner, lb)

    num_samples = int(state.num_samples)
    expected_probe_count = int(
        np.sum(np.asarray(
            state.samples.num_likelihood_evaluations[:num_samples]
        ))
    )

    assert int(state.num_samples) == 3
    assert expected_probe_count >= 2 * num_samples
    assert _completed_ok_eval_count_from_diagnostics(
        diagnostics
    ) == expected_probe_count
    assert _read_int(diagnostics, "dispatch_eval_count") == expected_probe_count


def test_local_lb_proposal_dispatch_reuses_worker_compile_cache():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    model = DispatchToyModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=3,
        no_step_out=True,
        collect_phantom_samples=False,
    )

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=False,
            sampler=sampler,
            target_num_live_points=2,
            max_samples=4,
            shell_size=2,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        state = runner.run_until_goal(
            goal_cond=lambda state: int(state.num_samples) >= 4,
            depth_cond=TerminationCondition(max_samples=4),
            allocation_target="uniform",
            key=jax.random.PRNGKey(17),
            max_goal_iterations=2,
        )
        diagnostics = _public_likelihood_diagnostics(runner, lb)

    completed_ok_eval_count = _completed_ok_eval_count_from_diagnostics(
        diagnostics
    )
    observed_worker_count = _read_int(
        diagnostics,
        "observed_worker_count",
        "worker_count",
    )
    compile_count = _read_int(diagnostics, "compile_count", "jit_compile_count")
    cache_hit_count = _read_int(diagnostics, "cache_hit_count")

    assert int(state.num_samples) == 4
    assert completed_ok_eval_count > 0
    assert _read_int(diagnostics, "distinct_compile_identity_count") == 1
    assert compile_count <= observed_worker_count
    assert cache_hit_count == completed_ok_eval_count - compile_count
    assert cache_hit_count > 0
    assert _read_int(diagnostics, "rejected_shape_cache_count") == 0


def test_local_lb_reuses_registered_problem_snapshot_without_per_eval_deserialize(
        monkeypatch,
):
    runtime = _runtime_module()
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    model = DispatchToyModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=3,
        no_step_out=True,
        collect_phantom_samples=False,
    )
    deserialize_count = 0
    original_deserialize = runtime.SerializedModelProblem.deserialize_problem

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=False,
            sampler=sampler,
            target_num_live_points=2,
            max_samples=4,
            shell_size=2,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        warmup_response = runner.dispatch_likelihood_eval(
            U=jnp.asarray([0.25, 0.25], dtype=jnp.float32),
            task_id="task-warmup",
            attempt_id="attempt-warmup",
            transport_id="transport-warmup",
            eval_id="eval-warmup",
        )
        assert _read_status(warmup_response) == "ok"
        completed_before = _completed_ok_eval_count_from_diagnostics(
            _public_likelihood_diagnostics(runner, lb)
        )

        def spy_deserialize(serialized_problem):
            nonlocal deserialize_count
            deserialize_count += 1
            return original_deserialize(serialized_problem)

        monkeypatch.setattr(
            runtime.SerializedModelProblem,
            "deserialize_problem",
            spy_deserialize,
        )

        state = runner.run_until_goal(
            goal_cond=lambda state: int(state.num_samples) >= 4,
            depth_cond=TerminationCondition(max_samples=4),
            allocation_target="uniform",
            key=jax.random.PRNGKey(19),
            max_goal_iterations=2,
        )
        diagnostics = _public_likelihood_diagnostics(runner, lb)

    completed_after = _completed_ok_eval_count_from_diagnostics(diagnostics)
    completed_during_run = completed_after - completed_before
    observed_worker_count = _read_int(
        diagnostics,
        "observed_worker_count",
        "worker_count",
    )

    assert int(state.num_samples) == 4
    assert completed_during_run > observed_worker_count
    assert deserialize_count <= observed_worker_count


def test_local_lb_dispatch_reuses_scheduler_registration_and_static_u_tree(
        monkeypatch,
):
    runtime = _runtime_module()
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    model = DispatchToyModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=3,
        no_step_out=True,
        collect_phantom_samples=False,
    )
    register_count = 0
    shape_tree_count = 0
    request_private_shape_tree_count = 0
    original_register = (
        runtime.LocalLoadBalancerState.register_likelihood_compile_identity
    )
    original_from_pytree = runtime.ShapeDtypeTree.from_pytree
    original_private_from_pytree = runtime._shape_dtype_tree_from_pytree

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=False,
            sampler=sampler,
            target_num_live_points=2,
            max_samples=5,
            shell_size=2,
            termination_condition=TerminationCondition(max_samples=5),
            batch_size=None,
        )
        warmup_response = runner.dispatch_likelihood_eval(
            U=jnp.asarray([0.25, 0.25], dtype=jnp.float32),
            task_id="task-warmup-overhead",
            attempt_id="attempt-warmup-overhead",
            transport_id="transport-warmup-overhead",
            eval_id="eval-warmup-overhead",
        )
        assert _read_status(warmup_response) == "ok"
        completed_before = _completed_ok_eval_count_from_diagnostics(
            _public_likelihood_diagnostics(runner, lb)
        )

        def spy_register(lb_state, *, identity, serialized_problem):
            nonlocal register_count
            register_count += 1
            return original_register(
                lb_state,
                identity=identity,
                serialized_problem=serialized_problem,
            )

        def spy_from_pytree(cls, value):
            del cls
            nonlocal shape_tree_count
            shape_tree_count += 1
            return original_from_pytree(value)

        def spy_private_from_pytree(value, tree_cls=runtime.ShapeDtypeTree):
            nonlocal request_private_shape_tree_count
            for frame_info in inspect.stack(context=0):
                if frame_info.function == "make_likelihood_eval_request":
                    request_private_shape_tree_count += 1
                    break
            return original_private_from_pytree(value, tree_cls=tree_cls)

        monkeypatch.setattr(
            runtime.LocalLoadBalancerState,
            "register_likelihood_compile_identity",
            spy_register,
        )
        monkeypatch.setattr(
            runtime.ShapeDtypeTree,
            "from_pytree",
            classmethod(spy_from_pytree),
        )
        monkeypatch.setattr(
            runtime,
            "_shape_dtype_tree_from_pytree",
            spy_private_from_pytree,
        )

        state = runner.run_until_goal(
            goal_cond=lambda state: int(state.num_samples) >= 5,
            depth_cond=TerminationCondition(max_samples=5),
            allocation_target="uniform",
            key=jax.random.PRNGKey(29),
            max_goal_iterations=3,
        )
        diagnostics = _public_likelihood_diagnostics(runner, lb)

    completed_after = _completed_ok_eval_count_from_diagnostics(diagnostics)
    completed_during_run = completed_after - completed_before
    observed_worker_count = _read_int(
        diagnostics,
        "observed_worker_count",
        "worker_count",
    )

    assert int(state.num_samples) == 5
    assert completed_during_run > 2 * observed_worker_count
    assert register_count <= observed_worker_count
    assert shape_tree_count <= observed_worker_count
    assert request_private_shape_tree_count == 0


def test_local_lb_parent_shell_dispatch_uses_available_workers_concurrently():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    model = DispatchToyModel()
    gate = ParentConcurrencyGate(
        required_arrivals=2,
        timeout_seconds=2.0,
    )
    sampler = ParentConcurrencyProbeSampler(
        model=model,
        num_slices=1,
        no_step_out=True,
        collect_phantom_samples=False,
        gate=gate,
    )
    state = State(
        root_out_degree=jnp.asarray(0, dtype=jnp.int32),
        samples=Samples(
            log_L_constraints=jnp.asarray([-jnp.inf, -jnp.inf, -jnp.inf]),
            log_likelihoods=jnp.asarray([-0.25, -0.125, -0.0625]),
            U_samples=jnp.asarray(
                [[0.2, 0.4], [0.6, 0.8], [0.3, 0.5]],
                dtype=jnp.float32,
            ),
            out_degree=jnp.zeros((3,), dtype=jnp.int32),
            num_likelihood_evaluations=jnp.ones((3,), dtype=jnp.int32),
            phantom_samples=PhantomSamples(
                U_samples=None,
                valid_mask=jnp.zeros((3, 0), dtype=bool),
                log_L=jnp.zeros((3, 0)),
            ),
        ),
        num_samples=jnp.asarray(3, dtype=jnp.int32),
        log_L_supremum=jnp.asarray(-0.0625),
        U_supremum=jnp.asarray([0.3, 0.5], dtype=jnp.float32),
        termination_reason=jnp.asarray(0, dtype=jnp.int32),
        model=model,
    )
    parent_work = ParentWork(
        parent_idxs=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        parent_log_L_constraints=jnp.asarray(
            [-jnp.inf, -jnp.inf, -jnp.inf]
        ),
        target_block_idxs=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        parent_block_idxs=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        fallback_to_root=jnp.asarray([False, False, False]),
    )
    parent_key = jax.random.PRNGKey(23)
    expected_seed_offsets = []
    for sample_key in jax.random.split(parent_key, 3):
        seed_key, _ = jax.random.split(sample_key, 2)
        expected_seed_offsets.append(
            int(jax.random.randint(seed_key, (), minval=0, maxval=3))
        )
    expected_U_samples = np.asarray(state.samples.U_samples)[
        expected_seed_offsets
    ]
    expected_log_likelihoods = -np.sum(
        np.square(expected_U_samples - model.centre),
        axis=1,
    )

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:2"])
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=False,
            sampler=sampler,
            target_num_live_points=3,
            max_samples=6,
            shell_size=3,
            termination_condition=TerminationCondition(max_samples=6),
            batch_size=None,
        )
        adjusted_parent_work, samples = runner._sample_parent_work(
            key=parent_key,
            state=state,
            parent_work=parent_work,
        )
        coordinator_records = tuple(runner.coordinator_dispatch_records)
        diagnostics = _public_likelihood_diagnostics(runner, lb)

    completed_worker_ids = _completed_worker_ids_from_diagnostics(diagnostics)
    max_by_worker = dict(_read_attr(diagnostics, "max_active_evals_per_worker"))
    accepted_parent_order = [
        int(_read_attr(record, "requested_parent_idx"))
        for record in coordinator_records
        if _read_status(record) == "accepted"
    ]

    assert adjusted_parent_work.parent_idxs.tolist() == [0, 1, 2]
    assert accepted_parent_order == [0, 1, 2]
    assert samples.log_likelihoods.shape == (3,)
    assert samples.U_samples.shape == (3, 2)
    np.testing.assert_allclose(np.asarray(samples.U_samples), expected_U_samples)
    np.testing.assert_allclose(
        np.asarray(samples.log_likelihoods),
        expected_log_likelihoods,
        rtol=1e-6,
    )
    assert gate.release_count > 0
    assert gate.timeout_count == 0
    assert _read_int(diagnostics, "observed_worker_count", "worker_count") == 2
    assert _read_int(diagnostics, "max_active_evals_pool") >= 2
    assert _completed_ok_eval_count_from_diagnostics(diagnostics) == 3
    assert len(completed_worker_ids) >= 2
    assert max_by_worker
    assert all(active <= 1 for active in max_by_worker.values())


def test_local_lb_likelihood_dispatch_matches_direct_model_value():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    model = DispatchToyModel()
    U = jnp.asarray([0.125, 0.5], dtype=jnp.float32)

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=3,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=3),
            batch_size=None,
        )
        dispatch_likelihood = _require_any_method(
            runner,
            (
                "dispatch_likelihood_eval",
                "evaluate_likelihood_via_dispatch",
                "request_likelihood_eval",
            ),
        )
        response = dispatch_likelihood(
            U=U,
            task_id="task-000001",
            attempt_id="attempt-000001",
            transport_id="transport-000001",
            eval_id="eval-000001",
        )

    _assert_model_log_likelihood(response, model, U)
    response_fields = _field_names(type(response))
    _assert_no_forbidden_fields(
        response_fields,
        RESPONSE_FORBIDDEN_FIELD_TOKENS,
    )


def test_local_lb_worker_pool_capacity_diagnostics_are_reported():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:2"])
        runner = lb.get_nested_sampler(
            model=DispatchToyModel(),
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        state = runner.run_until_goal(
            goal_cond=lambda state: int(state.num_samples) >= 4,
            depth_cond=TerminationCondition(max_samples=4),
            allocation_target="uniform",
            key=jax.random.PRNGKey(13),
            max_goal_iterations=2,
        )
        diagnostics = _public_likelihood_diagnostics(runner, lb)

    assert int(state.num_samples) == 4
    observed_worker_count = _read_int(
        diagnostics,
        "observed_worker_count",
        "worker_count",
    )
    assert observed_worker_count == 2
    max_by_worker = dict(_read_attr(diagnostics, "max_active_evals_per_worker"))
    assert max_by_worker
    assert all(active <= 1 for active in max_by_worker.values())
    assert _read_int(diagnostics, "max_active_evals_pool") <= observed_worker_count
    assert _read_attr(diagnostics, "observed_worker_device_classes") == ("cpu",)


def test_two_local_lb_runners_share_likelihood_workers_without_bleed():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    U = jnp.asarray([0.125, 0.5], dtype=jnp.float32)

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        first_runner = lb.get_nested_sampler(
            model=DispatchToyModel(centre=0.2),
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=3,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=3),
            batch_size=None,
        )
        second_runner = lb.get_nested_sampler(
            model=DispatchToyModel(centre=0.6),
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=3,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=3),
            batch_size=None,
        )
        scheduler = lb.load_balancer_state.likelihood_eval_scheduler()
        assert scheduler is first_runner._ensure_likelihood_scheduler()
        assert scheduler is second_runner._ensure_likelihood_scheduler()

        first_request = first_runner.make_likelihood_eval_request(
            U=U,
            task_id="task-shared",
            attempt_id="attempt-shared",
            transport_id="transport-shared",
            eval_id="eval-shared",
        )
        second_request = second_runner.make_likelihood_eval_request(
            U=U,
            task_id="task-shared",
            attempt_id="attempt-shared",
            transport_id="transport-shared",
            eval_id="eval-shared",
        )
        first_record = scheduler.start_likelihood_eval(
            request=first_request,
        )
        second_record = scheduler.start_likelihood_eval(
            request=second_request,
        )

        assert _read_status(first_record) in {
            "started",
            "active",
            "running",
        }
        assert _read_status(second_record) in {"queued", "pending"}
        assert scheduler.wait_for_started_record(
            second_request,
            timeout_seconds=0.0,
        ) is None

        worker_id = str(_read_attr(first_record, "worker_id"))
        worker = scheduler.worker_for_id(worker_id)
        first_response = worker.evaluate_likelihood(first_request)
        first_completion = scheduler.complete_likelihood_eval(
            worker_id=worker_id,
            response=first_response,
        )
        second_started = _read_attr(
            first_completion,
            "started_record",
            "next_started_record",
        )
        second_response = worker.evaluate_likelihood(second_request)
        scheduler.complete_likelihood_eval(
            worker_id=worker_id,
            response=second_response,
        )
        diagnostics = lb.likelihood_dispatch_diagnostics()

    _assert_model_log_likelihood(
        first_response,
        DispatchToyModel(centre=0.2),
        U,
    )
    _assert_model_log_likelihood(
        second_response,
        DispatchToyModel(centre=0.6),
        U,
    )
    assert _eval_identity_tuple(second_started) == (
        second_request.task_id,
        second_request.attempt_id,
        second_request.transport_id,
        second_request.eval_id,
    )
    assert first_request.runner_id != second_request.runner_id
    assert first_request.compile_identity_digest != (
        second_request.compile_identity_digest
    )
    assert first_runner.runtime_acceptance_ledger.accepted_task_ids == ()
    assert second_runner.runtime_acceptance_ledger.accepted_task_ids == ()
    assert _read_int(diagnostics, "observed_worker_count", "worker_count") == 1
    assert _read_int(diagnostics, "compile_count", "jit_compile_count") == 2
    assert _read_int(diagnostics, "distinct_compile_identity_count") == 2
    assert _completed_ok_eval_count_from_diagnostics(diagnostics) == 2


def test_runner_scheduler_cache_refreshes_after_other_client_spec_churn():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    U = jnp.asarray([0.125, 0.5], dtype=jnp.float32)

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=DispatchToyModel(centre=0.2),
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=3,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=3),
            batch_size=None,
        )
        original_scheduler = runner._ensure_likelihood_scheduler()
        original_specs = original_scheduler.requested_worker_specs

        with LoadBalancerClient(address="local") as other_lb:
            other_lb.add_workers(["cpu:*:2"])
            churn_scheduler = (
                other_lb.load_balancer_state.likelihood_eval_scheduler()
            )

        current_scheduler = lb.load_balancer_state.likelihood_eval_scheduler()
        refreshed_scheduler = runner._ensure_likelihood_scheduler()
        response = runner.dispatch_likelihood_eval(
            U=U,
            task_id="task-spec-churn",
            attempt_id="attempt-spec-churn",
            transport_id="transport-spec-churn",
            eval_id="eval-spec-churn",
        )
        current_diagnostics = current_scheduler.likelihood_dispatch_diagnostics()
        lb_diagnostics = lb.likelihood_dispatch_diagnostics()
        original_diagnostics = (
            original_scheduler.likelihood_dispatch_diagnostics()
        )

    assert churn_scheduler is not original_scheduler
    assert current_scheduler is not original_scheduler
    assert current_scheduler.requested_worker_specs == original_specs
    assert refreshed_scheduler is current_scheduler
    _assert_model_log_likelihood(response, DispatchToyModel(centre=0.2), U)
    assert _completed_ok_eval_count_from_diagnostics(current_diagnostics) == 1
    assert _completed_ok_eval_count_from_diagnostics(lb_diagnostics) == 1
    assert _completed_ok_eval_count_from_diagnostics(original_diagnostics) == 0


def test_runner_refreshes_compile_identity_after_worker_device_class_churn():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    U = jnp.asarray([0.125, 0.5], dtype=jnp.float32)
    cpu_lb = LoadBalancerClient(address="local")
    gpu_lb = LoadBalancerClient(address="local")

    try:
        cpu_lb.add_workers(["cpu:*:1"])
        runner = cpu_lb.get_nested_sampler(
            model=DispatchToyModel(centre=0.2),
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=3,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=3),
            batch_size=None,
        )
        original_identity = runner.runtime_compile_identity
        assert str(_read_attr(original_identity, "device_class")) == "cpu"

        gpu_lb.add_workers(["gpu:*:1"])
        cpu_lb.shutdown()

        request = runner.make_likelihood_eval_request(
            U=U,
            task_id="task-device-churn",
            attempt_id="attempt-device-churn",
            transport_id="transport-device-churn",
            eval_id="eval-device-churn",
        )
        response = runner.dispatch_likelihood_eval(
            U=U,
            task_id=request.task_id,
            attempt_id=request.attempt_id,
            transport_id=request.transport_id,
            eval_id=request.eval_id,
        )
        diagnostics = gpu_lb.likelihood_dispatch_diagnostics()
    finally:
        cpu_lb.shutdown()
        gpu_lb.shutdown()

    refreshed_identity = runner.runtime_compile_identity
    assert str(_read_attr(refreshed_identity, "device_class")) == "gpu"
    assert request.compile_identity_digest == refreshed_identity.identity_digest
    assert refreshed_identity.identity_digest != original_identity.identity_digest
    assert response.compile_identity_digest == refreshed_identity.identity_digest
    assert str(_read_attr(response, "worker_id")).startswith("worker-")
    _assert_model_log_likelihood(response, DispatchToyModel(centre=0.2), U)
    assert _read_attr(diagnostics, "observed_worker_device_classes") == (
        "gpu",
    )
    assert _completed_ok_eval_count_from_diagnostics(diagnostics) == 1


def test_local_lb_compile_registration_uses_serialized_problem_snapshot():
    LoadBalancerClient = _public_symbol("LoadBalancerClient")
    original_model = MutableDispatchModel(centre=0.1)
    original_args = (
        {"shift": jnp.asarray(0.05, dtype=jnp.float32)},
    )
    original_params = {"shift": jnp.asarray(0.02, dtype=jnp.float32)}
    U = jnp.asarray([0.4, 0.6], dtype=jnp.float32)

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=original_model,
            args=original_args,
            params=original_params,
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=3,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=3),
            batch_size=None,
        )
        original_digest = runner.runtime_compile_identity.identity_digest

        original_model.centre = 0.7
        original_args[0]["shift"] = jnp.asarray(0.2, dtype=jnp.float32)
        original_params["shift"] = jnp.asarray(0.3, dtype=jnp.float32)

        response = runner.dispatch_likelihood_eval(
            U=U,
            task_id="task-drift",
            attempt_id="attempt-drift",
            transport_id="transport-drift",
            eval_id="eval-drift",
        )
        diagnostics = lb.likelihood_dispatch_diagnostics()

    snapshot_expected = MutableDispatchModel(centre=0.1).log_likelihood(
        U,
        args=({"shift": jnp.asarray(0.05, dtype=jnp.float32)},),
        params={"shift": jnp.asarray(0.02, dtype=jnp.float32)},
        allow_nan=False,
    )
    mutated_expected = original_model.log_likelihood(
        U,
        args=original_args,
        params=original_params,
        allow_nan=False,
    )

    assert response.status == "ok"
    assert response.compile_identity_digest == original_digest
    np.testing.assert_allclose(
        float(response.log_L),
        float(snapshot_expected),
        rtol=1e-6,
        atol=1e-7,
    )
    assert not np.isclose(float(response.log_L), float(mutated_expected))
    assert _read_int(diagnostics, "compile_count", "jit_compile_count") == 1


def test_likelihood_dispatch_diagnostics_report_cache_capacity_and_workers():
    required_fields = {
        "requested_worker_specs",
        "observed_worker_count",
        "observed_worker_device_classes",
        "dispatch_latency_seconds",
        "dispatch_throughput_per_second",
        "compile_count",
        "cache_hit_count",
        "rejected_shape_cache_count",
        "distinct_compile_identity_count",
        "dispatch_eval_count",
        "queued_eval_count",
        "failed_eval_count",
        "max_active_evals_per_worker",
        "max_active_evals_pool",
        "completed_eval_count_by_worker",
    }

    for module_name in ("jaxns.runtime", "jaxns.diagnostics"):
        Diagnostics = _public_symbol(
            "LikelihoodDispatchDiagnostics",
            modules=(module_name,),
        )
        fields = set(_field_names(Diagnostics))
        missing = required_fields - fields
        assert missing == set()

    diagnostics = Diagnostics(
        requested_worker_specs=("cpu:*:2",),
        observed_worker_count=2,
        observed_worker_device_classes=("cpu",),
        dispatch_latency_seconds=(0.0, 0.0),
        dispatch_throughput_per_second=2.0,
        compile_count=1,
        cache_hit_count=1,
        rejected_shape_cache_count=1,
        distinct_compile_identity_count=1,
        dispatch_eval_count=3,
        queued_eval_count=1,
        failed_eval_count=0,
        max_active_evals_per_worker={"worker-000001": 1, "worker-000002": 1},
        max_active_evals_pool=2,
        completed_eval_count_by_worker={
            "worker-000001": 2,
            "worker-000002": 1,
        },
    )

    assert _read_attr(diagnostics, "requested_worker_specs") == ("cpu:*:2",)
    assert _read_int(diagnostics, "observed_worker_count") == 2
    assert _read_attr(diagnostics, "observed_worker_device_classes") == (
        "cpu",
    )
    assert tuple(_read_attr(diagnostics, "dispatch_latency_seconds")) == (
        0.0,
        0.0,
    )
    assert _read_float(diagnostics, "dispatch_throughput_per_second") == 2.0
    assert _read_int(diagnostics, "compile_count") == 1
    assert _read_int(diagnostics, "cache_hit_count") == 1
    assert _read_int(diagnostics, "rejected_shape_cache_count") == 1
    assert _read_int(diagnostics, "distinct_compile_identity_count") == 1
    assert _read_int(diagnostics, "dispatch_eval_count") == 3
    assert _read_int(diagnostics, "queued_eval_count") == 1
    assert _read_int(diagnostics, "failed_eval_count") == 0
    assert max(
        dict(_read_attr(diagnostics, "max_active_evals_per_worker")).values()
    ) == 1
    assert _read_int(diagnostics, "max_active_evals_pool") == 2
    assert sum(
        dict(_read_attr(diagnostics, "completed_eval_count_by_worker")).values()
    ) == 3
