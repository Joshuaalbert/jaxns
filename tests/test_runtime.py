from __future__ import annotations

import dataclasses
import importlib
import inspect
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxctx import CtxParams
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.constrained_sampler import AbstractSampler
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree
from jaxns.samples import PhantomSamples
from jaxns.samples import SeedPoint
from jaxns.state import State
from jaxns.termination_condition import TerminationCondition

tfpd = tfp.distributions


@dataclasses.dataclass(frozen=True, slots=True)
class ToyModel(PureDataclassPytree):
    centre: float = 0.25

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
        return -jnp.square(jnp.asarray(U) - self.centre)

    def log_prior(self, U, args=(), params=None):
        del args, params
        inside = jnp.logical_and(U >= 0.0, U <= 1.0)
        return jnp.where(inside, 0.0, -jnp.inf)


ToyModel.register_pytree()


def runtime_prior_model_with_parameter(scale: float):
    x = Prior(tfpd.Uniform(low=0.0, high=1.0), name="x").realise()
    theta = Prior(tfpd.Normal(loc=scale, scale=0.25), name="theta").parameter()
    return -jnp.square(x - theta)


def runtime_prior_model_with_nested_args(config: dict[str, object]):
    x = Prior(tfpd.Uniform(low=0.0, high=1.0), name="x").realise()
    prior_config = config["prior"]
    weight_config = config["weights"]
    theta = Prior(
        tfpd.Normal(
            loc=jnp.asarray(prior_config["loc"]),
            scale=jnp.asarray(prior_config["scale"]),
        ),
        name="theta",
    ).parameter()
    return -jnp.square((x - theta) * jnp.asarray(weight_config["likelihood"]))


def make_toy_model() -> ToyModel:
    return ToyModel()


class ParsedWorkerSpec(NamedTuple):
    device_type: str
    device_ids: tuple[str, ...]
    num_workers_per_device: int


class RuntimeProblem(NamedTuple):
    model: object
    args: tuple[object, ...]
    params: dict[str, object]


class RuntimeDispatchRecord(NamedTuple):
    runner_id: str
    task_id: str
    attempt_id: str
    transport_id: str
    requested_parent_idx: int
    effective_parent_idx: int
    accepted_parent_idx: int
    status: str
    worker_id: str
    sector_id: str
    identity_owner: str


class RuntimeLifecycleRecord(NamedTuple):
    runner_id: str
    task_id: str
    attempt_id: str
    transport_id: str
    attempt_number: int
    delivery_number: int
    requested_parent_idx: int
    effective_parent_idx: int
    accepted_parent_idx: int
    effective_log_L_constraint: float
    accepted_log_L_constraint: float
    seed_id: str
    phantom_cluster_id: str
    status: str
    worker_id: str
    sector_id: str
    identity_owner: str


class RuntimeStateSnapshot(NamedTuple):
    root_out_degree: np.ndarray
    num_samples: np.ndarray
    k_values: np.ndarray
    log_L_constraints: np.ndarray
    log_likelihoods: np.ndarray
    out_degree: np.ndarray
    num_likelihood_evaluations: np.ndarray
    U_samples: object
    phantom_U_samples: object
    phantom_valid_mask: np.ndarray
    phantom_log_L: np.ndarray


# Pre-0018 legacy fixtures below assert the old serialized constrained-sampler
# worker-task path. Ticket 0018 supersedes that path for ordinary runtime work:
# workers should receive only likelihood-eval U payloads after identity
# registration, while sampler state remains runner-local.
PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP = pytest.mark.skip(
    reason=(
        "pre-0018 legacy serialized constrained-sampler worker-task "
        "coverage; Ticket 0018 ordinary LoadBalancerClient work must use "
        "likelihood-eval dispatch with sampler state runner-local"
    )
)
PRE_0018_LEGACY_COMPILE_IDENTITY_SKIP = pytest.mark.skip(
    reason=(
        "pre-0018 RuntimeCompileIdentity included sampler/phantom execution "
        "payload details; Ticket 0018 ordinary compile identity is model, "
        "args, params, dtype policy, device class, and U shape/tree"
    )
)


class SerializedWorkerOnlySampler(AbstractSampler):
    """Pre-0018 sampler fixture that requires a serialized worker boundary."""

    def __init__(self, *, worker_instance: bool = False):
        self.worker_instance = worker_instance

    def __getstate__(self):
        return {}

    def __setstate__(self, state) -> None:
        del state
        self.worker_instance = True

    def num_phantom(self) -> int:
        return 0

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point,
            args=(),
            params=None,
            adaptation_context=None,
    ):
        del key, log_L_constraint, args, params, adaptation_context
        if not self.worker_instance:
            raise AssertionError(
                "sampler must execute from serialized worker payload"
            )
        return (
            seed_point.U0,
            seed_point.log_L0,
            jnp.asarray(1, dtype=jnp.int32),
            PhantomSamples(
                U_samples=None,
                valid_mask=jnp.zeros((0,), dtype=bool),
                log_L=jnp.zeros(
                    (0,),
                    dtype=jnp.asarray(seed_point.log_L0).dtype,
                ),
            ),
        )


class SerializedModelEvaluatingSampler(AbstractSampler):
    """Sampler fixture that requires worker-side model injection."""

    def __init__(self, model=None):
        self.model = model

    def __getstate__(self):
        return {}

    def __setstate__(self, state) -> None:
        del state
        self.model = None

    def num_phantom(self) -> int:
        return 0

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point,
            args=(),
            params=None,
            adaptation_context=None,
    ):
        del key, log_L_constraint, adaptation_context
        if self.model is None:
            raise AssertionError(
                "worker sampler must receive serialized model payload"
            )
        log_L = self.model.log_likelihood(
            seed_point.U0,
            args=args,
            params=params,
            allow_nan=False,
        )
        return (
            seed_point.U0,
            log_L,
            jnp.asarray(1, dtype=jnp.int32),
            PhantomSamples(
                U_samples=None,
                valid_mask=jnp.zeros((0,), dtype=bool),
                log_L=jnp.zeros((0,), dtype=jnp.asarray(log_L).dtype),
            ),
        )


class RetryOnceSerializedWorkerSampler(AbstractSampler):
    """Sampler fixture that fails once for one serialized finite task."""

    _failed_fingerprints: set[tuple[float, float, float]] = set()
    _failure_count: int = 0

    def __init__(self, *, worker_instance: bool = False):
        self.worker_instance = worker_instance

    def __getstate__(self):
        return {}

    def __setstate__(self, state) -> None:
        del state
        self.worker_instance = True

    @classmethod
    def reset_failures(cls) -> None:
        cls._failed_fingerprints = set()
        cls._failure_count = 0

    @classmethod
    def observed_failure_count(cls) -> int:
        return cls._failure_count

    def num_phantom(self) -> int:
        return 0

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point,
            args=(),
            params=None,
            adaptation_context=None,
    ):
        del key, args, params, adaptation_context
        if not self.worker_instance:
            raise AssertionError(
                "sampler must execute from serialized worker payload"
            )

        constraint = float(log_L_constraint)
        fingerprint = (
            constraint,
            float(seed_point.log_L0),
            float(jnp.ravel(jnp.asarray(seed_point.U0))[0]),
        )
        if (
                np.isfinite(constraint)
                and not self._failed_fingerprints
                and fingerprint not in self._failed_fingerprints
        ):
            self._failed_fingerprints.add(fingerprint)
            self.__class__._failure_count += 1
            raise RuntimeError("deterministic worker retry fixture failure")

        return (
            seed_point.U0,
            seed_point.log_L0,
            jnp.asarray(1, dtype=jnp.int32),
            PhantomSamples(
                U_samples=None,
                valid_mask=jnp.zeros((0,), dtype=bool),
                log_L=jnp.zeros(
                    (0,),
                    dtype=jnp.asarray(seed_point.log_L0).dtype,
                ),
            ),
        )


class IdentityConfigSampler(AbstractSampler):
    """Sampler fixture whose pickled config affects cache identity."""

    def __init__(self, *, config_value: int):
        self.config_value = config_value

    def num_phantom(self) -> int:
        return 0

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point,
            args=(),
            params=None,
            adaptation_context=None,
    ):
        del key, log_L_constraint, args, params, adaptation_context
        return (
            seed_point.U0,
            seed_point.log_L0,
            jnp.asarray(1, dtype=jnp.int32),
            PhantomSamples(
                U_samples=None,
                valid_mask=jnp.zeros((0,), dtype=bool),
                log_L=jnp.zeros(
                    (0,),
                    dtype=jnp.asarray(seed_point.log_L0).dtype,
                ),
            ),
        )


class CountingSerializedWorkerSampler(AbstractSampler):
    """Sampler fixture that records worker-side deserialisation count."""

    _deserialise_count: int = 0

    def __init__(self, *, worker_instance: bool = False):
        self.worker_instance = worker_instance

    def __getstate__(self):
        return {}

    def __setstate__(self, state) -> None:
        del state
        self.worker_instance = True
        self.__class__._deserialise_count += 1

    @classmethod
    def reset_counts(cls) -> None:
        cls._deserialise_count = 0

    @classmethod
    def deserialise_count(cls) -> int:
        return cls._deserialise_count

    def num_phantom(self) -> int:
        return 0

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point,
            args=(),
            params=None,
            adaptation_context=None,
    ):
        del key, log_L_constraint, args, params, adaptation_context
        if not self.worker_instance:
            raise AssertionError(
                "sampler must execute from serialized worker payload"
            )
        return (
            seed_point.U0,
            seed_point.log_L0,
            jnp.asarray(1, dtype=jnp.int32),
            PhantomSamples(
                U_samples=None,
                valid_mask=jnp.zeros((0,), dtype=bool),
                log_L=jnp.zeros(
                    (0,),
                    dtype=jnp.asarray(seed_point.log_L0).dtype,
                ),
            ),
        )


class AdaptationContextRecordingSampler(AbstractSampler):
    """Sampler fixture that records per-task adaptation contexts."""

    _observed_kernel_versions: list[int | None] = []
    _deserialise_count: int = 0

    def __init__(self, *, worker_instance: bool = False):
        self.worker_instance = worker_instance

    def __getstate__(self):
        return {}

    def __setstate__(self, state) -> None:
        del state
        self.worker_instance = True
        self.__class__._deserialise_count += 1

    @classmethod
    def reset_counts(cls) -> None:
        cls._observed_kernel_versions = []
        cls._deserialise_count = 0

    @classmethod
    def observed_kernel_versions(cls) -> tuple[int | None, ...]:
        return tuple(cls._observed_kernel_versions)

    @classmethod
    def deserialise_count(cls) -> int:
        return cls._deserialise_count

    def num_phantom(self) -> int:
        return 0

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point,
            args=(),
            params=None,
            adaptation_context=None,
    ):
        del key, log_L_constraint, args, params
        if not self.worker_instance:
            raise AssertionError(
                "sampler must execute from serialized worker payload"
            )
        self.__class__._observed_kernel_versions.append(
            None
            if adaptation_context is None
            else int(getattr(adaptation_context, "kernel_version"))
        )
        return (
            seed_point.U0,
            seed_point.log_L0,
            jnp.asarray(1, dtype=jnp.int32),
            PhantomSamples(
                U_samples=None,
                valid_mask=jnp.zeros((0,), dtype=bool),
                log_L=jnp.zeros(
                    (0,),
                    dtype=jnp.asarray(seed_point.log_L0).dtype,
                ),
            ),
        )


class MixedPhantomCoordinateSampler(AbstractSampler):
    """Sampler that returns coordinate phantoms for root calls only."""

    def __init__(self, *, coordinate_calls: int):
        self.coordinate_calls = coordinate_calls
        self.call_count = 0

    def num_phantom(self) -> int:
        return 1

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point,
            args=(),
            params=None,
            adaptation_context=None,
    ):
        del key, log_L_constraint, args, params, adaptation_context
        self.call_count += 1
        if self.call_count <= self.coordinate_calls:
            phantom_U = jax.tree.map(
                lambda u: jnp.expand_dims(u, axis=0),
                seed_point.U0,
            )
        else:
            phantom_U = None
        return (
            seed_point.U0,
            seed_point.log_L0,
            jnp.asarray(1, dtype=jnp.int32),
            PhantomSamples(
                U_samples=phantom_U,
                valid_mask=jnp.ones((1,), dtype=bool),
                log_L=jnp.expand_dims(seed_point.log_L0, axis=0),
            ),
        )


_MISSING = object()


def _load_balancer_client():
    runtime = importlib.import_module("jaxns.runtime")
    return runtime.LoadBalancerClient


def _runtime_module():
    return importlib.import_module("jaxns.runtime")


def _runtime_problem() -> RuntimeProblem:
    return RuntimeProblem(
        model=make_toy_model(),
        args=(
            {"literal": "payload", "array": jnp.asarray([1.0, 2.0])},
            ("nested", 3),
        ),
        params={
            "theta": jnp.asarray([0.25, 0.5]),
            "nested": {"enabled": True, "scale": jnp.asarray(2.0)},
        },
    )


def _runtime_ctx_problem(
        *,
        scale: float = 0.75,
        param_value: float = 0.75,
) -> tuple[Model, tuple[float], CtxParams]:
    model = Model(prior_model=runtime_prior_model_with_parameter)
    args = (scale,)
    params = model.init_params(
        key=jax.random.PRNGKey(31),
        args=args,
    )
    params["theta"] = jnp.asarray(param_value)
    assert isinstance(params, CtxParams)
    return model, args, params


def _compile_identity_test_kwargs() -> dict[str, object]:
    return {
        "target_num_live_points": 2,
        "max_samples": 4,
        "shell_size": 1,
        "termination_condition": TerminationCondition(max_samples=4),
        "batch_size": None,
    }


def _compile_identity_runner(
        client,
        *,
        args_scale: float = 0.75,
        params_scale: float = 0.75,
        collect_phantoms: bool = False,
        sampler_config: int = 3,
        sampler_kind: str = "identity_config",
):
    model, _, params = _runtime_ctx_problem(
        scale=params_scale,
        param_value=params_scale,
    )
    if sampler_kind == "identity_config":
        sampler = IdentityConfigSampler(config_value=sampler_config)
    elif sampler_kind == "serialized_worker_only":
        sampler = SerializedWorkerOnlySampler()
    else:
        raise ValueError(f"Unknown sampler_kind {sampler_kind!r}.")
    return client.get_nested_sampler(
        model=model,
        args=(args_scale,),
        params=params,
        collect_phantoms=collect_phantoms,
        sampler=sampler,
        **_compile_identity_test_kwargs(),
    )


def _get_attr_or_key(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value[name]
    return getattr(value, name)


def _get_worker_count(value: object) -> int:
    if isinstance(value, dict):
        count = value.get(
            "num_workers_per_device",
            value.get("workers_per_device"),
        )
    else:
        count = getattr(
            value,
            "num_workers_per_device",
            getattr(value, "workers_per_device", None),
        )
    return int(count)


def _normalise_worker_spec(value: object) -> ParsedWorkerSpec:
    device_type = str(_get_attr_or_key(value, "device_type"))
    device_ids = _get_attr_or_key(value, "device_ids")
    if device_ids == "*":
        normalised_device_ids = ("*",)
    else:
        normalised_device_ids = tuple(
            str(device_id) for device_id in device_ids
        )
    return ParsedWorkerSpec(
        device_type=device_type,
        device_ids=normalised_device_ids,
        num_workers_per_device=_get_worker_count(value),
    )


def _assert_trees_equal(left: object, right: object) -> None:
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if hasattr(left_leaf, "shape") or hasattr(right_leaf, "shape"):
            np.testing.assert_array_equal(
                np.asarray(left_leaf),
                np.asarray(right_leaf),
            )
        else:
            assert left_leaf == right_leaf


def _small_runtime_sampler(model, collect_phantoms: bool = False):
    return UniDimSliceSampler(
        model=model,
        num_slices=3,
        no_step_out=True,
        collect_phantom_samples=collect_phantoms,
        phantom_burn_in=1,
    )


def _run_small_runtime_runner(runner, max_samples: int = 5) -> State:
    state = runner.run_until_goal(
        goal_cond=lambda state: int(state.num_samples) >= max_samples,
        depth_cond=TerminationCondition(max_samples=max_samples),
        allocation_target="uniform",
        key=jax.random.PRNGKey(11),
        max_goal_iterations=4,
    )
    assert isinstance(state, State)
    assert int(state.num_samples) == max_samples
    assert np.all(np.isfinite(np.asarray(
        state.samples.log_likelihoods[:max_samples]
    )))
    return state


def _direct_record_field(
        record: object,
        name: str,
        default: object = _MISSING,
) -> object:
    if isinstance(record, dict) and name in record:
        return record[name]
    if hasattr(record, name):
        return getattr(record, name)
    return default


def _record_containers(record: object) -> tuple[object, ...]:
    containers = [record]
    for field_name in (
            "metadata",
            "identity",
            "result_identity",
            "task_identity",
            "attempt_identity",
            "transport_identity",
            "parent_metadata",
            "lifecycle",
            "request",
            "worker_request",
            "dispatch_request",
    ):
        nested = _direct_record_field(record, field_name)
        if nested is not _MISSING and nested is not None:
            containers.append(nested)
    return tuple(containers)


def _record_field(
        record: object,
        *field_names: str,
        default: object = _MISSING,
) -> object:
    for container in _record_containers(record):
        for field_name in field_names:
            value = _direct_record_field(container, field_name)
            if value is not _MISSING:
                return value
    if default is not _MISSING:
        return default
    pytest.fail(
        "Runtime dispatch records must expose one of: "
        f"{', '.join(field_names)}."
    )


def _coerce_record_sequence(value: object) -> tuple[object, ...]:
    if callable(value):
        value = value()
    if value is None:
        return ()
    if isinstance(value, dict):
        for field_name in ("records", "events", "entries", "items"):
            if field_name in value:
                return tuple(value[field_name])
    for field_name in ("records", "events", "entries", "items"):
        if hasattr(value, field_name):
            return tuple(getattr(value, field_name))
    return tuple(value)


def _dispatch_record_sources(source: object) -> tuple[object, ...]:
    sources = [source]
    for candidate in tuple(sources):
        diagnostics = getattr(candidate, "diagnostics", None)
        if diagnostics is not None:
            sources.append(diagnostics)
            worker_runtime = getattr(diagnostics, "worker_runtime", None)
            if worker_runtime is not None:
                sources.append(worker_runtime)
        worker_runtime = getattr(candidate, "worker_runtime", None)
        if worker_runtime is not None:
            sources.append(worker_runtime)
    return tuple(sources)


def _public_dispatch_records(*sources: object) -> tuple[object, ...]:
    for source in sources:
        for candidate in _dispatch_record_sources(source):
            if not hasattr(candidate, "coordinator_dispatch_records"):
                continue
            records = _coerce_record_sequence(
                getattr(candidate, "coordinator_dispatch_records")
            )
            if records:
                return records
    pytest.fail(
        "Load-balanced runs must expose public coordinator_dispatch_records."
    )


def _normalise_dispatch_status(status: object) -> str:
    status_name = str(status).lower()
    if status_name in {"accept", "accepted", "complete", "completed"}:
        return "accepted"
    if status_name in {"duplicate", "duplicate_task_result"}:
        return "duplicate_task_result"
    if status_name in {"failed", "failure", "worker_failed"}:
        return "failed"
    if status_name in {"pending", "prepared"}:
        return "pending"
    if status_name in {"retry", "retried"}:
        return "retried"
    if status_name in {"revoke", "revoked", "cancel", "cancelled"}:
        return "revoked"
    if status_name in {"stale_parent", "stale_parent_target"}:
        return "stale_parent_target"
    if status_name in {"stale_result", "stale_task_result"}:
        return "stale_task_result"
    if status_name == "mismatched_result_identity":
        return "mismatched_result_identity"
    return status_name


def _required_int(value: object, field_name: str) -> int:
    if value is None:
        pytest.fail(f"{field_name} must use an explicit sentinel, not None.")
    return int(value)


def _required_float(value: object, field_name: str) -> float:
    if value is None:
        pytest.fail(f"{field_name} must be explicit, not None.")
    return float(value)


def _normalise_dispatch_record(record: object) -> RuntimeDispatchRecord:
    identity_owner = str(
        _record_field(
            record,
            "identity_owner",
            "issued_by",
            "owner",
        )
    ).lower()
    return RuntimeDispatchRecord(
        runner_id=str(
            _record_field(record, "runner_id", "coordinator_runner_id")
        ),
        task_id=str(_record_field(record, "task_id", "coordinator_task_id")),
        attempt_id=str(
            _record_field(record, "attempt_id", "coordinator_attempt_id")
        ),
        transport_id=str(
            _record_field(
                record,
                "transport_id",
                "delivery_id",
                "coordinator_transport_id",
            )
        ),
        requested_parent_idx=_required_int(
            _record_field(
                record,
                "requested_parent_idx",
                "requested_parent_index",
            ),
            "requested_parent_idx",
        ),
        effective_parent_idx=_required_int(
            _record_field(
                record,
                "effective_parent_idx",
                "effective_parent_index",
                "effective_parent",
            ),
            "effective_parent_idx",
        ),
        accepted_parent_idx=_required_int(
            _record_field(
                record,
                "accepted_parent_idx",
                "accepted_parent_index",
                "accepted_parent",
            ),
            "accepted_parent_idx",
        ),
        status=_normalise_dispatch_status(
            _record_field(record, "status", "task_status")
        ),
        worker_id=str(
            _record_field(record, "worker_id", "worker_identity")
        ),
        sector_id=str(
            _record_field(record, "sector_id", "compute_sector_id")
        ),
        identity_owner=identity_owner,
    )


def _normalise_lifecycle_record(record: object) -> RuntimeLifecycleRecord:
    identity_owner = str(
        _record_field(
            record,
            "identity_owner",
            "issued_by",
            "owner",
        )
    ).lower()
    return RuntimeLifecycleRecord(
        runner_id=str(
            _record_field(record, "runner_id", "coordinator_runner_id")
        ),
        task_id=str(_record_field(record, "task_id", "coordinator_task_id")),
        attempt_id=str(
            _record_field(record, "attempt_id", "coordinator_attempt_id")
        ),
        transport_id=str(
            _record_field(
                record,
                "transport_id",
                "delivery_id",
                "coordinator_transport_id",
            )
        ),
        attempt_number=_required_int(
            _record_field(record, "attempt_number", "retry_number"),
            "attempt_number",
        ),
        delivery_number=_required_int(
            _record_field(
                record,
                "delivery_number",
                "transport_delivery_number",
            ),
            "delivery_number",
        ),
        requested_parent_idx=_required_int(
            _record_field(
                record,
                "requested_parent_idx",
                "requested_parent_index",
            ),
            "requested_parent_idx",
        ),
        effective_parent_idx=_required_int(
            _record_field(
                record,
                "effective_parent_idx",
                "effective_parent_index",
                "effective_parent",
            ),
            "effective_parent_idx",
        ),
        accepted_parent_idx=_required_int(
            _record_field(
                record,
                "accepted_parent_idx",
                "accepted_parent_index",
                "accepted_parent",
            ),
            "accepted_parent_idx",
        ),
        effective_log_L_constraint=_required_float(
            _record_field(
                record,
                "effective_log_L_constraint",
                "effective_strict_contour",
            ),
            "effective_log_L_constraint",
        ),
        accepted_log_L_constraint=_required_float(
            _record_field(
                record,
                "accepted_log_L_constraint",
                "accepted_strict_contour",
            ),
            "accepted_log_L_constraint",
        ),
        seed_id=str(_record_field(record, "seed_id", "seed_identity")),
        phantom_cluster_id=str(
            _record_field(
                record,
                "phantom_cluster_id",
                "phantom_likelihood_cluster_id",
            )
        ),
        status=_normalise_dispatch_status(
            _record_field(record, "status", "task_status")
        ),
        worker_id=str(
            _record_field(record, "worker_id", "worker_identity")
        ),
        sector_id=str(
            _record_field(record, "sector_id", "compute_sector_id")
        ),
        identity_owner=identity_owner,
    )


def _tree_to_numpy(value: object) -> object:
    if value is None:
        return None
    return jax.tree.map(lambda leaf: np.asarray(leaf), value)


def _runtime_state_snapshot(state: State) -> RuntimeStateSnapshot:
    k_values = state.samples.compute_num_live_points_per_sample(
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
    )
    return RuntimeStateSnapshot(
        root_out_degree=np.asarray(state.root_out_degree),
        num_samples=np.asarray(state.num_samples),
        k_values=np.asarray(k_values),
        log_L_constraints=np.asarray(state.samples.log_L_constraints),
        log_likelihoods=np.asarray(state.samples.log_likelihoods),
        out_degree=np.asarray(state.samples.out_degree),
        num_likelihood_evaluations=np.asarray(
            state.samples.num_likelihood_evaluations
        ),
        U_samples=_tree_to_numpy(state.samples.U_samples),
        phantom_U_samples=_tree_to_numpy(
            state.samples.phantom_samples.U_samples
        ),
        phantom_valid_mask=np.asarray(
            state.samples.phantom_samples.valid_mask
        ),
        phantom_log_L=np.asarray(state.samples.phantom_samples.log_L),
    )


def _assert_optional_trees_equal(left: object, right: object) -> None:
    assert (left is None) == (right is None)
    if left is not None:
        _assert_trees_equal(left, right)


def _assert_runtime_state_snapshot_equal(
        left: RuntimeStateSnapshot,
        right: RuntimeStateSnapshot,
) -> None:
    np.testing.assert_array_equal(left.root_out_degree, right.root_out_degree)
    np.testing.assert_array_equal(left.num_samples, right.num_samples)
    np.testing.assert_array_equal(left.k_values, right.k_values)
    np.testing.assert_array_equal(
        left.log_L_constraints,
        right.log_L_constraints,
    )
    np.testing.assert_array_equal(
        left.log_likelihoods,
        right.log_likelihoods,
    )
    np.testing.assert_array_equal(left.out_degree, right.out_degree)
    np.testing.assert_array_equal(
        left.num_likelihood_evaluations,
        right.num_likelihood_evaluations,
    )
    _assert_optional_trees_equal(left.U_samples, right.U_samples)
    _assert_optional_trees_equal(
        left.phantom_U_samples,
        right.phantom_U_samples,
    )
    np.testing.assert_array_equal(
        left.phantom_valid_mask,
        right.phantom_valid_mask,
    )
    np.testing.assert_array_equal(left.phantom_log_L, right.phantom_log_L)


def _require_runner_method(runner, method_name: str):
    method = getattr(runner, method_name, None)
    assert callable(method), (
        "Runtime runners must expose public "
        f"{method_name}(...) for dispatch lifecycle tests."
    )
    return method


def _assert_method_has_no_state_parameter(method) -> None:
    assert "state" not in inspect.signature(method).parameters


def _prepare_runtime_dispatch(
        runner,
        *,
        requested_parent_idx: int = 0,
        effective_parent_idx: int = 0,
        accepted_parent_idx: int = 0,
        effective_log_L_constraint: float = -0.5,
        accepted_log_L_constraint: float = -0.5,
        seed_id: str = "seed-000001",
        phantom_cluster_id: str = "phantom-cluster-000001",
) -> object:
    method = _require_runner_method(runner, "prepare_runtime_dispatch")
    _assert_method_has_no_state_parameter(method)
    return method(
        requested_parent_idx=requested_parent_idx,
        effective_parent_idx=effective_parent_idx,
        accepted_parent_idx=accepted_parent_idx,
        effective_log_L_constraint=effective_log_L_constraint,
        accepted_log_L_constraint=accepted_log_L_constraint,
        seed_id=seed_id,
        phantom_cluster_id=phantom_cluster_id,
    )


def _fail_runtime_dispatch(runner, dispatch_record: object) -> object:
    method = _require_runner_method(runner, "mark_runtime_dispatch_failed")
    _assert_method_has_no_state_parameter(method)
    return method(
        dispatch_record=dispatch_record,
        error="worker failed before completion",
    )


def _retry_runtime_dispatch(
        runner,
        dispatch_record: object,
        *,
        reason: str = "retry requested",
) -> object:
    method = _require_runner_method(runner, "retry_runtime_dispatch")
    _assert_method_has_no_state_parameter(method)
    return method(dispatch_record=dispatch_record, reason=reason)


def _revoke_runtime_dispatch(runner, dispatch_record: object) -> object:
    method = _require_runner_method(runner, "revoke_runtime_dispatch")
    _assert_method_has_no_state_parameter(method)
    return method(
        dispatch_record=dispatch_record,
        reason="parent target was superseded",
    )


def _complete_runtime_dispatch(
        runner,
        dispatch_record: object,
        worker_result: object,
        *,
        current_parent_idx: int,
        current_log_L_constraint: float,
) -> object:
    method = _require_runner_method(runner, "complete_runtime_dispatch")
    _assert_method_has_no_state_parameter(method)
    return method(
        dispatch_record=dispatch_record,
        result=worker_result,
        current_parent_idx=current_parent_idx,
        current_effective_log_L_constraint=current_log_L_constraint,
    )


def _worker_result_for_lifecycle_record(record: object, runtime):
    lifecycle_record = _normalise_lifecycle_record(record)
    return runtime.WorkerResult(
        identity=runtime.WorkerResultIdentity(
            task_id=lifecycle_record.task_id,
            attempt_id=lifecycle_record.attempt_id,
            transport_id=lifecycle_record.transport_id,
            worker_id=lifecycle_record.worker_id,
            sector_id=lifecycle_record.sector_id,
        ),
        payload=_constrained_sampler_completion_payload(runtime),
    )


def _constrained_sampler_completion_payload(runtime):
    return runtime.ConstrainedSamplerCompletionPayload(
        U_sample=jnp.asarray(0.25, dtype=jnp.float32),
        log_L=jnp.asarray(-0.25, dtype=jnp.float32),
        num_likelihood_evaluations=jnp.asarray(1, dtype=jnp.int32),
        phantom_samples=PhantomSamples(
            U_samples=None,
            valid_mask=jnp.zeros((0,), dtype=bool),
            log_L=jnp.zeros((0,), dtype=jnp.float32),
        ),
    )


def _worker_result_with_runtime_stats_for_lifecycle_record(
        record: object,
        runtime,
        *,
        dispatch_latency_seconds: float = 1.25,
        payload_cache_latency_seconds: float = 0.125,
        sampler_execution_latency_seconds: float = 1.0,
):
    worker_result = _worker_result_for_lifecycle_record(record, runtime)
    return dataclasses.replace(
        worker_result,
        payload=runtime._WorkerTaskExecutionStats(
            output=worker_result.payload,
            sampler_loop_mode="python",
            dispatch_latency_seconds=dispatch_latency_seconds,
            payload_cache_latency_seconds=payload_cache_latency_seconds,
            sampler_execution_latency_seconds=sampler_execution_latency_seconds,
        ),
    )


def _mismatched_worker_result_for_lifecycle_record(
        record: object,
        runtime,
        *,
        mismatch_field: str,
):
    lifecycle_record = _normalise_lifecycle_record(record)
    task_id = lifecycle_record.task_id
    attempt_id = lifecycle_record.attempt_id
    transport_id = lifecycle_record.transport_id
    if mismatch_field == "task_id":
        task_id = f"mismatched-{task_id}"
    elif mismatch_field == "attempt_id":
        attempt_id = f"mismatched-{attempt_id}"
    elif mismatch_field == "transport_id":
        transport_id = f"mismatched-{transport_id}"
    else:
        raise ValueError(f"Unknown mismatch field {mismatch_field!r}.")
    return runtime.WorkerResult(
        identity=runtime.WorkerResultIdentity(
            task_id=task_id,
            attempt_id=attempt_id,
            transport_id=transport_id,
            worker_id=lifecycle_record.worker_id,
            sector_id=lifecycle_record.sector_id,
        ),
        payload=_constrained_sampler_completion_payload(runtime),
    )


def _lifecycle_parent_metadata(
        record: RuntimeLifecycleRecord,
) -> tuple[int, int, int, float, float, str, str]:
    return (
        record.requested_parent_idx,
        record.effective_parent_idx,
        record.accepted_parent_idx,
        record.effective_log_L_constraint,
        record.accepted_log_L_constraint,
        record.seed_id,
        record.phantom_cluster_id,
    )


def _assert_lifecycle_parent_metadata_preserved(
        reference: RuntimeLifecycleRecord,
        *records: RuntimeLifecycleRecord,
) -> None:
    expected = _lifecycle_parent_metadata(reference)
    for record in records:
        assert _lifecycle_parent_metadata(record) == expected


def _lifecycle_records_for_task(
        task_id: str,
        *sources: object,
) -> tuple[RuntimeLifecycleRecord, ...]:
    records = tuple(
        _normalise_lifecycle_record(record)
        for record in _public_dispatch_records(*sources)
        if str(_record_field(record, "task_id", "coordinator_task_id"))
        == task_id
    )
    assert records
    return records


def _accepted_dispatch_records(
        *sources: object,
) -> tuple[RuntimeDispatchRecord, ...]:
    records = tuple(
        _normalise_dispatch_record(record)
        for record in _public_dispatch_records(*sources)
    )
    accepted = tuple(
        record for record in records if record.status == "accepted"
    )
    assert accepted
    return accepted


def _accepted_raw_dispatch_records_for_runner(
        runner,
        *sources: object,
) -> tuple[object, ...]:
    runner_id = _runner_id(runner)
    raw_records = tuple(
        record
        for record in _public_dispatch_records(*sources)
        if str(_record_field(record, "runner_id", "coordinator_runner_id"))
        == runner_id
        and _normalise_dispatch_status(
            _record_field(record, "status", "task_status")
        )
        == "accepted"
    )
    assert raw_records
    return raw_records


def _runner_id(runner) -> str:
    return str(getattr(runner, "runtime_runner_identity").runner_id)


def _accepted_dispatch_records_for_runner(
        runner,
        *sources: object,
) -> tuple[RuntimeDispatchRecord, ...]:
    runner_id = _runner_id(runner)
    runner_records = tuple(
        record
        for record in _accepted_dispatch_records(*sources)
        if record.runner_id == runner_id
    )
    assert runner_records
    return runner_records


def _assert_accepted_dispatch_is_non_duplicating(
        accepted_records: tuple[RuntimeDispatchRecord, ...],
        runner,
) -> None:
    task_ids = tuple(record.task_id for record in accepted_records)
    assert len(task_ids) == len(set(task_ids))
    for record in accepted_records:
        assert record.identity_owner == "coordinator"
        assert record.runner_id
        assert record.task_id
        assert record.attempt_id
        assert record.transport_id
        assert record.task_id != record.attempt_id
        assert record.task_id != record.transport_id
        assert record.attempt_id != record.transport_id
        assert record.requested_parent_idx >= -1
        assert record.effective_parent_idx >= -1
        assert record.accepted_parent_idx >= -1
        assert record.worker_id
        assert record.sector_id

    ledger = getattr(runner, "runtime_acceptance_ledger", None)
    if ledger is not None:
        assert set(ledger.accepted_task_ids) == set(task_ids)


def _accepted_raw_diagnostic_records_for_runner(
        runner,
        state: State,
) -> tuple[object, ...]:
    diagnostics = getattr(state, "execution_diagnostics", None)
    assert diagnostics is not None
    worker_runtime = getattr(diagnostics, "worker_runtime", None)
    assert worker_runtime is not None
    runner_id = _runner_id(runner)
    records = tuple(
        record
        for record in worker_runtime.dispatch_records
        if str(_record_field(record, "runner_id", "coordinator_runner_id"))
        == runner_id
        and _normalise_dispatch_status(
            _record_field(record, "status", "task_status")
        )
        == "accepted"
    )
    assert records
    return records


def _runtime_latency_values(record: object) -> tuple[float, float, float]:
    values = tuple(
        float(_record_field(record, field_name))
        for field_name in (
            "dispatch_latency_seconds",
            "payload_cache_latency_seconds",
            "sampler_execution_latency_seconds",
        )
    )
    assert np.all(np.isfinite(np.asarray(values, dtype=float)))
    assert all(value >= 0.0 for value in values)
    return values


def _runtime_sampler_loop_mode(record: object) -> str:
    mode = str(_record_field(record, "sampler_loop_mode")).lower()
    assert mode in {"python", "fused"}
    return mode


def _serialized_problem_payload(record: object, runtime):
    payload = _direct_record_field(record, "serialized_problem")
    if payload is not _MISSING:
        if isinstance(payload, runtime.SerializedModelProblem):
            return payload
        if all(
                _direct_record_field(payload, field_name) is not _MISSING
                for field_name in ("model_bytes", "args_bytes", "params_bytes")
        ):
            return payload
    pytest.fail(
        "Worker dispatch records must expose serialized_problem with the "
        "model, args, and params payload sent to the worker."
    )


def _runtime_compile_identity(runner) -> object:
    identity = getattr(runner, "runtime_compile_identity", _MISSING)
    assert identity is not _MISSING, (
        "Runtime runners must expose public runtime_compile_identity for "
        "compile/cache identity diagnostics."
    )
    if callable(identity):
        identity = identity()
    assert identity is not None
    assert not isinstance(identity, int), (
        "runtime_compile_identity must be a deterministic value, not a "
        "Python id(...) integer."
    )
    return identity


def _record_runtime_compile_identity(record: object) -> object:
    identity = _record_field(record, "runtime_compile_identity")
    if callable(identity):
        identity = identity()
    assert identity is not None
    assert not isinstance(identity, int), (
        "Dispatch runtime_compile_identity must be deterministic, not a "
        "Python id(...) integer."
    )
    return identity


def _record_client_id(record: object) -> str:
    return str(_record_field(record, "client_id"))


def _restore_problem_from_payload(payload, runtime):
    deserialize_problem = getattr(payload, "deserialize_problem", None)
    if callable(deserialize_problem):
        return deserialize_problem()
    return runtime.ModelProblem(
        model=runtime.deserialize_model(payload.model_bytes),
        args=runtime.deserialize_args(payload.args_bytes),
        params=runtime.deserialize_params(payload.params_bytes),
        collect_phantoms=bool(
            _direct_record_field(payload, "collect_phantoms", False)
        ),
    )


def _sector_resource_summary(client) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (
            sector.device_type,
            sector.device_id,
            int(sector.num_workers),
        )
        for sector in client.compute_sectors
    )


def _request_load_balancer_shutdown(client) -> None:
    request_shutdown = getattr(client, "request_shutdown", None)
    assert callable(request_shutdown), (
        "LoadBalancerClient must expose public request_shutdown() so a "
        "connected coordinator can request shared load-balancer shutdown."
    )
    request_shutdown()


def test_load_balancer_client_is_public_at_jaxns_runtime():
    LoadBalancerClient = _load_balancer_client()

    assert inspect.isclass(LoadBalancerClient)


def test_add_workers_parses_public_worker_specs():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        parsed_specs = lb.add_workers(["cpu:*:5", "gpu:0,1:10"])

    assert [_normalise_worker_spec(spec) for spec in parsed_specs] == [
        ParsedWorkerSpec(
            device_type="cpu",
            device_ids=("*",),
            num_workers_per_device=5,
        ),
        ParsedWorkerSpec(
            device_type="gpu",
            device_ids=("0", "1"),
            num_workers_per_device=10,
        ),
    ]
    assert parsed_specs[0].num_workers_per_device == 5


def test_add_workers_registers_compute_sectors_in_local_state():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:5", "gpu:0,1:10"])
        compute_sectors = lb.compute_sectors

    assert [
        (
            sector.sector_id,
            sector.device_type,
            sector.device_id,
            sector.num_workers,
        )
        for sector in compute_sectors
    ] == [
        ("sector-000001", "cpu", "*", 5),
        ("sector-000002", "gpu", "0", 10),
        ("sector-000003", "gpu", "1", 10),
    ]


@pytest.mark.parametrize(
    "worker_spec",
    [
        "tpu:*:1",
        "cpu:*:0",
        "cpu:*:-1",
        "cpu:*:two",
        "cpu:*",
        "gpu::1",
        "gpu:0,,1:2",
        "gpu:0,0:2",
        " cpu:*:1",
        "cpu:*:1 ",
        None,
    ],
)
def test_add_workers_rejects_invalid_specs(worker_spec: object):
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        with pytest.raises(
            ValueError,
            match=r"(?i)worker spec|device|num_workers|positive",
        ):
            lb.add_workers([worker_spec])


@pytest.mark.parametrize("specs", ["cpu:*:5", None, object()])
def test_add_workers_rejects_invalid_specs_collection(specs: object):
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        with pytest.raises(ValueError, match="specs"):
            lb.add_workers(specs)


def test_get_nested_sampler_returns_v3_runner_and_preserves_problem_payload():
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()
    problem = _runtime_problem()

    with LoadBalancerClient(address="local") as lb:
        runner = lb.get_nested_sampler(
            model=problem.model,
            args=problem.args,
            params=problem.params,
            collect_phantoms=True,
            target_num_live_points=2,
            max_samples=8,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=8),
            batch_size=1,
        )

    assert callable(runner.run_until_goal)
    assert callable(runner.resume_until_goal)
    assert "goal_cond" in inspect.signature(runner.run_until_goal).parameters
    assert "state" in inspect.signature(runner.resume_until_goal).parameters
    assert (
        "allocation_target"
        in inspect.signature(runner.run_until_goal).parameters
    )

    assert getattr(runner, "model") == problem.model
    _assert_trees_equal(getattr(runner, "args"), problem.args)
    _assert_trees_equal(getattr(runner, "params"), problem.params)
    assert getattr(runner, "collect_phantom_samples") is True

    runner_identity = getattr(runner, "runtime_runner_identity")
    problem_payload = getattr(runner, "runtime_problem_payload")
    acceptance_ledger = getattr(runner, "runtime_acceptance_ledger")
    assert runner_identity == runtime.RunnerIdentity(
        runner_id="runner-000001",
        client_id="client-000001",
    )
    assert isinstance(problem_payload, runtime.SerializedModelProblem)
    assert isinstance(acceptance_ledger, runtime.AcceptanceLedger)
    restored_problem = problem_payload.deserialize_problem()
    assert restored_problem.model == problem.model
    _assert_trees_equal(restored_problem.args, problem.args)
    _assert_trees_equal(restored_problem.params, problem.params)
    assert restored_problem.collect_phantoms is True

    nested_sampler_signature = inspect.signature(NestedSampler.run_until_goal)
    runner_signature = inspect.signature(runner.run_until_goal)
    for parameter_name in ("goal_cond", "depth_cond", "allocation_target"):
        assert parameter_name in runner_signature.parameters
        assert (
            runner_signature.parameters[parameter_name].default
            == nested_sampler_signature.parameters[parameter_name].default
        )


@PRE_0018_LEGACY_COMPILE_IDENTITY_SKIP
def test_pre_0018_legacy_runtime_compile_identity_is_sampler_sensitive():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        base_runner = _compile_identity_runner(lb)
        same_runner = _compile_identity_runner(lb)
        args_changed_runner = _compile_identity_runner(
            lb,
            args_scale=0.95,
        )
        params_changed_runner = _compile_identity_runner(
            lb,
            params_scale=0.95,
        )
        collect_changed_runner = _compile_identity_runner(
            lb,
            collect_phantoms=True,
        )
        sampler_config_changed_runner = _compile_identity_runner(
            lb,
            sampler_config=5,
        )
        sampler_type_changed_runner = _compile_identity_runner(
            lb,
            sampler_kind="serialized_worker_only",
        )
        toy_model_runner = lb.get_nested_sampler(
            model=ToyModel(centre=0.25),
            sampler=IdentityConfigSampler(config_value=3),
            **_compile_identity_test_kwargs(),
        )
        toy_model_changed_runner = lb.get_nested_sampler(
            model=ToyModel(centre=0.55),
            sampler=IdentityConfigSampler(config_value=3),
            **_compile_identity_test_kwargs(),
        )

        base_identity = _runtime_compile_identity(base_runner)
        same_identity = _runtime_compile_identity(same_runner)
        args_changed_identity = _runtime_compile_identity(
            args_changed_runner
        )
        params_changed_identity = _runtime_compile_identity(
            params_changed_runner
        )
        collect_changed_identity = _runtime_compile_identity(
            collect_changed_runner
        )
        sampler_config_changed_identity = _runtime_compile_identity(
            sampler_config_changed_runner
        )
        sampler_type_changed_identity = _runtime_compile_identity(
            sampler_type_changed_runner
        )
        toy_model_identity = _runtime_compile_identity(toy_model_runner)
        toy_model_changed_identity = _runtime_compile_identity(
            toy_model_changed_runner
        )

        lb.add_workers(["cpu:*:3"])
        same_device_type_runner = _compile_identity_runner(lb)
        same_device_type_identity = _runtime_compile_identity(
            same_device_type_runner
        )

        lb.add_workers(["gpu:0:1"])
        device_type_changed_runner = _compile_identity_runner(lb)
        device_type_changed_identity = _runtime_compile_identity(
            device_type_changed_runner
        )

    with LoadBalancerClient(address="local") as one_worker_lb:
        one_worker_lb.add_workers(["cpu:*:1"])
        one_worker_runner = _compile_identity_runner(one_worker_lb)
        one_worker_identity = _runtime_compile_identity(one_worker_runner)

    with LoadBalancerClient(address="local") as three_worker_lb:
        three_worker_lb.add_workers(["cpu:*:3"])
        three_worker_runner = _compile_identity_runner(three_worker_lb)
        three_worker_identity = _runtime_compile_identity(
            three_worker_runner
        )

    assert base_runner is not same_runner
    assert base_runner.model is not same_runner.model
    assert base_runner.sampler is not same_runner.sampler
    assert base_identity == same_identity
    assert base_identity == same_device_type_identity
    assert one_worker_identity == three_worker_identity
    assert toy_model_identity != toy_model_changed_identity
    assert args_changed_identity != base_identity
    assert params_changed_identity != base_identity
    assert collect_changed_identity != base_identity
    assert sampler_config_changed_identity != base_identity
    assert sampler_type_changed_identity != base_identity
    assert device_type_changed_identity != base_identity


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_serialized_worker_only_sampler_rejects_direct_core_execution():
    model = make_toy_model()
    runner = NestedSampler(
        model=model,
        sampler=SerializedWorkerOnlySampler(),
        target_num_live_points=2,
        max_samples=4,
        shell_size=1,
        termination_condition=TerminationCondition(max_samples=4),
        batch_size=None,
    )

    with pytest.raises(AssertionError, match="serialized worker payload"):
        _run_small_runtime_runner(runner, max_samples=4)


def test_v3_direct_run_respects_store_phantom_samples_flag():
    model = make_toy_model()
    no_store_runner = NestedSampler(
        model=model,
        collect_phantom_samples=True,
        store_phantom_samples=False,
        sampler=MixedPhantomCoordinateSampler(coordinate_calls=2),
        target_num_live_points=2,
        max_samples=5,
        shell_size=1,
        termination_condition=TerminationCondition(max_samples=5),
        batch_size=None,
    )
    store_runner = NestedSampler(
        model=model,
        collect_phantom_samples=True,
        store_phantom_samples=True,
        sampler=MixedPhantomCoordinateSampler(coordinate_calls=2),
        target_num_live_points=2,
        max_samples=5,
        shell_size=1,
        termination_condition=TerminationCondition(max_samples=5),
        batch_size=None,
    )

    no_store_state = _run_small_runtime_runner(no_store_runner, max_samples=5)
    store_state = _run_small_runtime_runner(store_runner, max_samples=5)

    assert no_store_runner.store_phantom_samples is False
    assert store_runner.store_phantom_samples is True
    assert no_store_state.samples.phantom_samples.U_samples is None
    assert store_state.samples.phantom_samples.U_samples is not None
    assert store_state.samples.phantom_samples.valid_mask.shape[1] > 0
    assert store_state.samples.phantom_samples.log_L.shape[1] > 0


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_local_load_balanced_runner_executes_with_coordinator_dispatch_trace():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:2"])
        model = make_toy_model()
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=False,
            sampler=SerializedWorkerOnlySampler(),
            target_num_live_points=2,
            max_samples=5,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=5),
            batch_size=None,
        )

        state = _run_small_runtime_runner(runner, max_samples=5)
        accepted_records = _accepted_dispatch_records_for_runner(
            runner,
            lb,
            runner,
            state,
        )

    assert int(state.root_out_degree) >= int(runner.target_num_live_points)
    assert len(accepted_records) >= (
        int(state.num_samples) - int(runner.target_num_live_points)
    )
    assert {record.runner_id for record in accepted_records} == {
        _runner_id(runner)
    }
    assert {record.sector_id for record in accepted_records} == {
        "sector-000001"
    }
    _assert_accepted_dispatch_is_non_duplicating(accepted_records, runner)


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_local_runtime_reuses_worker_payload_until_last_client_teardown():
    LoadBalancerClient = _load_balancer_client()
    CountingSerializedWorkerSampler.reset_counts()
    lb_state = None

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        lb_state = lb.load_balancer_state
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=False,
            sampler=CountingSerializedWorkerSampler(),
            target_num_live_points=2,
            max_samples=5,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=5),
            batch_size=None,
        )

        state = _run_small_runtime_runner(runner, max_samples=5)

        assert int(state.num_samples) == 5
        assert lb_state.worker_runtime_cache_size() == 1
        assert CountingSerializedWorkerSampler.deserialise_count() == 1

    assert lb_state is not None
    assert lb_state.worker_runtime_cache_size() == 0


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_local_runtime_reuses_stable_sampler_payload_for_dispatches(
        monkeypatch,
):
    runtime = _runtime_module()
    LoadBalancerClient = _load_balancer_client()
    serialize_calls = []
    original_serialize_sampler = runtime.serialize_sampler

    def counting_serialize_sampler(sampler):
        serialize_calls.append(type(sampler).__qualname__)
        return original_serialize_sampler(sampler)

    monkeypatch.setattr(
        runtime,
        "serialize_sampler",
        counting_serialize_sampler,
    )

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=False,
            sampler=SerializedWorkerOnlySampler(),
            target_num_live_points=2,
            max_samples=5,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=5),
            batch_size=None,
        )

        state = _run_small_runtime_runner(runner, max_samples=5)
        accepted_records = _accepted_dispatch_records_for_runner(
            runner,
            lb,
            runner,
            state,
        )

    assert len(accepted_records) >= 3
    assert len(serialize_calls) <= 2, (
        "stable runtime compile identity/payload cache should avoid "
        "serializing the sampler once per accepted local-LB dispatch; "
        f"serialize_calls={serialize_calls!r}, "
        f"accepted_dispatches={len(accepted_records)}"
    )


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_worker_payload_cache_keeps_per_dispatch_adaptation_context_fresh():
    runtime = _runtime_module()
    DirectionAdaptationContext = importlib.import_module(
        "jaxns.em_gmm"
    ).DirectionAdaptationContext
    AdaptationContextRecordingSampler.reset_counts()
    serialized_problem = runtime.SerializedModelProblem.from_problem(
        model=make_toy_model(),
    )
    sampler = AdaptationContextRecordingSampler()
    sampler_bytes = runtime.serialize_sampler(sampler)
    compile_identity = runtime._build_runtime_compile_identity(
        serialized_problem=serialized_problem,
        sampler=sampler,
        worker_device_types=("cpu",),
    )
    lb_state = runtime.LocalLoadBalancerState(address="local-test")
    seed_point = SeedPoint(U0=jnp.asarray(0.25), log_L0=jnp.asarray(0.0))

    def make_task(kernel_version: int):
        return runtime.SerializedWorkerTask(
            serialized_problem=serialized_problem,
            sampler_bytes=sampler_bytes,
            key_bytes=runtime.pickle_payload(jax.random.PRNGKey(kernel_version)),
            log_L_constraint_bytes=runtime.pickle_payload(jnp.asarray(-np.inf)),
            seed_point_bytes=runtime.pickle_payload(seed_point),
            adaptation_context_bytes=runtime.pickle_payload(
                DirectionAdaptationContext(kernel_version=kernel_version)
            ),
            runtime_compile_identity=compile_identity,
        )

    runtime._execute_serialized_worker_task(
        make_task(3),
        runtime_lb_state=lb_state,
    )
    runtime._execute_serialized_worker_task(
        make_task(4),
        runtime_lb_state=lb_state,
    )

    assert lb_state.worker_runtime_cache_size() == 1
    assert AdaptationContextRecordingSampler.deserialise_count() == 1
    assert AdaptationContextRecordingSampler.observed_kernel_versions() == (3, 4)


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_unidim_worker_cache_keeps_fresh_context_and_explicit_loop_mode(
        monkeypatch,
):
    runtime = _runtime_module()
    DirectionAdaptationContext = importlib.import_module(
        "jaxns.em_gmm"
    ).DirectionAdaptationContext
    observed_contexts = []

    def recording_get_sample(
            self,
            key,
            log_L_constraint,
            seed_point,
            args=(),
            params=None,
            adaptation_context=None,
    ):
        del self, key, log_L_constraint, args, params
        observed_contexts.append(adaptation_context)
        return (
            seed_point.U0,
            seed_point.log_L0,
            jnp.asarray(1, dtype=jnp.int32),
            PhantomSamples(
                U_samples=None,
                valid_mask=jnp.zeros((0,), dtype=bool),
                log_L=jnp.zeros(
                    (0,),
                    dtype=jnp.asarray(seed_point.log_L0).dtype,
                ),
            ),
        )

    monkeypatch.setattr(
        UniDimSliceSampler,
        "get_sample",
        recording_get_sample,
    )

    model = make_toy_model()
    serialized_problem = runtime.SerializedModelProblem.from_problem(
        model=model,
    )
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=3,
        no_step_out=True,
        direction_kernel="ellipsoidal",
    )
    sampler_bytes = runtime.serialize_sampler(sampler)
    compile_identity = runtime._build_runtime_compile_identity(
        serialized_problem=serialized_problem,
        sampler=sampler,
        worker_device_types=("cpu",),
    )
    lb_state = runtime.LocalLoadBalancerState(address="local-test")
    seed_point = SeedPoint(U0=jnp.asarray(0.25), log_L0=jnp.asarray(0.0))

    def make_task(kernel_version: int):
        return runtime.SerializedWorkerTask(
            serialized_problem=serialized_problem,
            sampler_bytes=sampler_bytes,
            key_bytes=runtime.pickle_payload(jax.random.PRNGKey(kernel_version)),
            log_L_constraint_bytes=runtime.pickle_payload(jnp.asarray(-np.inf)),
            seed_point_bytes=runtime.pickle_payload(seed_point),
            adaptation_context_bytes=runtime.pickle_payload(
                DirectionAdaptationContext(kernel_version=kernel_version)
            ),
            runtime_compile_identity=compile_identity,
        )

    runtime._execute_serialized_worker_task(
        make_task(3),
        runtime_lb_state=lb_state,
    )
    runtime._execute_serialized_worker_task(
        make_task(4),
        runtime_lb_state=lb_state,
    )

    assert lb_state.worker_runtime_cache_size() == 1
    observed_versions = tuple(
        int(context["direction_adaptation_context"].kernel_version)
        for context in observed_contexts
    )
    observed_loop_modes = tuple(
        str(context["sampler_loop_mode"]).lower()
        for context in observed_contexts
    )
    assert observed_versions == (3, 4)
    assert observed_loop_modes
    assert set(observed_loop_modes) <= {"python", "fused"}


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_runtime_dispatch_mutates_acceptance_ledger_once_per_task_id():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:2"])
        model = make_toy_model()
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=False,
            sampler=SerializedWorkerOnlySampler(),
            target_num_live_points=2,
            max_samples=5,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=5),
            batch_size=None,
        )

        state = _run_small_runtime_runner(runner, max_samples=5)
        accepted_records = _accepted_dispatch_records_for_runner(
            runner,
            lb,
            runner,
            state,
        )

    accepted_task_ids = tuple(record.task_id for record in accepted_records)
    assert len(accepted_records) == len(set(accepted_task_ids))
    assert set(runner.runtime_acceptance_ledger.accepted_task_ids) == set(
        accepted_task_ids
    )


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_local_runtime_retries_failed_worker_task_before_acceptance():
    LoadBalancerClient = _load_balancer_client()
    RetryOnceSerializedWorkerSampler.reset_failures()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=True,
            sampler=RetryOnceSerializedWorkerSampler(),
            target_num_live_points=2,
            max_samples=5,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=5),
            batch_size=None,
        )

        state = _run_small_runtime_runner(runner, max_samples=5)
        raw_records = _public_dispatch_records(lb, runner, state)
        failed_task_ids = tuple(
            dict.fromkeys(
                str(_record_field(record, "task_id", "coordinator_task_id"))
                for record in raw_records
                if _normalise_dispatch_status(
                    _record_field(record, "status", "task_status")
                )
                == "failed"
            )
        )
        assert failed_task_ids
        assert len(failed_task_ids) == 1

        failed_task_id = failed_task_ids[0]
        task_records = _lifecycle_records_for_task(
            failed_task_id,
            lb,
            runner,
            state,
        )
        accepted_records = _accepted_dispatch_records_for_runner(
            runner,
            lb,
            runner,
            state,
        )

    assert RetryOnceSerializedWorkerSampler.observed_failure_count() == 1
    assert [record.status for record in task_records] == [
        "pending",
        "failed",
        "retried",
        "accepted",
    ]

    pending, failed, retried, accepted = task_records
    assert {
        pending.task_id,
        failed.task_id,
        retried.task_id,
        accepted.task_id,
    } == {failed_task_id}
    assert failed.attempt_id == pending.attempt_id
    assert failed.transport_id == pending.transport_id
    assert failed.attempt_number == 1
    assert retried.attempt_id != pending.attempt_id
    assert retried.transport_id != pending.transport_id
    assert retried.attempt_number == 2
    assert retried.delivery_number == 1
    assert accepted.attempt_id == retried.attempt_id
    assert accepted.transport_id == retried.transport_id
    assert accepted.attempt_number == 2
    assert accepted.delivery_number == 1
    assert pending.seed_id not in {"", "None"}
    assert pending.phantom_cluster_id not in {"", "None"}
    assert pending.accepted_parent_idx >= 0
    assert np.isfinite(pending.accepted_log_L_constraint)
    _assert_lifecycle_parent_metadata_preserved(
        pending,
        failed,
        retried,
        accepted,
    )

    accepted_task_ids = runner.runtime_acceptance_ledger.accepted_task_ids
    assert accepted_task_ids.count(failed_task_id) == 1
    assert {record.task_id for record in accepted_records} == set(
        accepted_task_ids
    )
    assert len(accepted_records) == int(state.num_samples)
    assert len(accepted_task_ids) == int(state.num_samples)

    snapshot = _runtime_state_snapshot(state)
    num_samples = int(snapshot.num_samples)
    assert num_samples == 5
    assert int(snapshot.root_out_degree) >= int(runner.target_num_live_points)
    assert np.all(snapshot.out_degree[:num_samples] >= 0)
    assert (
        int(snapshot.root_out_degree)
        + int(np.sum(snapshot.out_degree[:num_samples]))
    ) == num_samples
    assert np.all(snapshot.k_values[:num_samples] >= 0)
    assert np.all(np.isfinite(snapshot.log_likelihoods[:num_samples]))


def test_in_process_scheduler_round_robins_assignments_across_sectors():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:2", "gpu:0,1:2"])
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=False,
            sampler=IdentityConfigSampler(config_value=3),
            **_compile_identity_test_kwargs(),
        )
        raw_records = tuple(
            _prepare_runtime_dispatch(
                runner,
                seed_id=f"seed-fairness-{idx:06d}",
                phantom_cluster_id=f"phantom-fairness-{idx:06d}",
            )
            for idx in range(6)
        )
        records = tuple(
            _normalise_lifecycle_record(record) for record in raw_records
        )

    expected_sector_sequence = (
        "sector-000001",
        "sector-000002",
        "sector-000003",
        "sector-000001",
        "sector-000002",
        "sector-000003",
    )
    assert tuple(record.status for record in records) == ("pending",) * 6
    assert tuple(record.sector_id for record in records) == (
        expected_sector_sequence
    )
    assert tuple(
        (record.sector_id, record.worker_id) for record in records
    ) == (
        ("sector-000001", "worker-sector-000001-000001"),
        ("sector-000002", "worker-sector-000002-000002"),
        ("sector-000003", "worker-sector-000003-000001"),
        ("sector-000001", "worker-sector-000001-000002"),
        ("sector-000002", "worker-sector-000002-000001"),
        ("sector-000003", "worker-sector-000003-000002"),
    )
    assert {record.runner_id for record in records} == {_runner_id(runner)}
    assert {record.identity_owner for record in records} == {"coordinator"}

    worker_ids_by_sector = {
        sector_id: {
            record.worker_id
            for record in records
            if record.sector_id == sector_id
        }
        for sector_id in expected_sector_sequence[:3]
    }
    assert {
        sector_id: len(worker_ids)
        for sector_id, worker_ids in worker_ids_by_sector.items()
    } == {
        "sector-000001": 2,
        "sector-000002": 2,
        "sector-000003": 2,
    }


def test_local_load_balanced_runner_executes_default_sampler():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:2"])
        model = make_toy_model()
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )

        state = _run_small_runtime_runner(runner, max_samples=4)
        accepted_records = _accepted_dispatch_records_for_runner(
            runner,
            lb,
            runner,
            state,
        )

    assert int(state.num_samples) == 4
    assert len(accepted_records) >= 2
    _assert_accepted_dispatch_is_non_duplicating(accepted_records, runner)


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_two_local_clients_execute_shared_workers_with_isolated_dispatch_ids():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as first:
        with LoadBalancerClient(address="local") as second:
            first.add_workers(["cpu:*:2"])
            assert first.compute_sectors == second.compute_sectors

            first_model = ToyModel(centre=0.2)
            second_model = ToyModel(centre=0.6)
            first_runner = first.get_nested_sampler(
                model=first_model,
                collect_phantoms=False,
                sampler=SerializedWorkerOnlySampler(),
                target_num_live_points=2,
                max_samples=5,
                shell_size=1,
                termination_condition=TerminationCondition(max_samples=5),
                batch_size=None,
            )
            second_runner = second.get_nested_sampler(
                model=second_model,
                collect_phantoms=False,
                sampler=SerializedWorkerOnlySampler(),
                target_num_live_points=2,
                max_samples=5,
                shell_size=1,
                termination_condition=TerminationCondition(max_samples=5),
                batch_size=None,
            )

            with ThreadPoolExecutor(max_workers=2) as executor:
                first_future = executor.submit(
                    _run_small_runtime_runner,
                    first_runner,
                    5,
                )
                second_future = executor.submit(
                    _run_small_runtime_runner,
                    second_runner,
                    5,
                )
                first_state = first_future.result(timeout=20.0)
                second_state = second_future.result(timeout=20.0)
            first_records = _accepted_dispatch_records_for_runner(
                first_runner,
                first,
                first_runner,
                first_state,
            )
            second_records = _accepted_dispatch_records_for_runner(
                second_runner,
                second,
                second_runner,
                second_state,
            )

    assert _runner_id(first_runner) != _runner_id(second_runner)
    assert {record.runner_id for record in first_records}.isdisjoint(
        {record.runner_id for record in second_records}
    )
    assert {record.task_id for record in first_records}.isdisjoint(
        {record.task_id for record in second_records}
    )
    assert first_runner.runtime_acceptance_ledger is not (
        second_runner.runtime_acceptance_ledger
    )
    assert set(first_runner.runtime_acceptance_ledger.accepted_task_ids) == {
        record.task_id for record in first_records
    }
    assert set(second_runner.runtime_acceptance_ledger.accepted_task_ids) == {
        record.task_id for record in second_records
    }
    assert int(first_state.num_samples) == 5
    assert int(second_state.num_samples) == 5


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_worker_dispatch_payload_executes_real_model_args_and_ctx_params():
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()
    model = Model(prior_model=runtime_prior_model_with_nested_args)
    args = (
        {
            "prior": {
                "loc": jnp.asarray(0.45),
                "scale": jnp.asarray(0.2),
            },
            "weights": {"likelihood": jnp.asarray(1.5)},
        },
    )
    params = model.init_params(key=jax.random.PRNGKey(23), args=args)

    assert isinstance(params, CtxParams)

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:2"])
        runner = lb.get_nested_sampler(
            model=model,
            args=args,
            params=params,
            collect_phantoms=False,
            sampler=SerializedModelEvaluatingSampler(),
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )

        state = _run_small_runtime_runner(runner, max_samples=4)
        accepted_records = _accepted_dispatch_records_for_runner(
            runner,
            lb,
            runner,
            state,
        )
        raw_records = _accepted_raw_dispatch_records_for_runner(
            runner,
            lb,
            runner,
            state,
        )
        payload = _serialized_problem_payload(raw_records[0], runtime)

    restored_problem = _restore_problem_from_payload(payload, runtime)
    assert isinstance(payload.model_bytes, bytes)
    assert isinstance(payload.args_bytes, bytes)
    assert isinstance(payload.params_bytes, bytes)
    assert len(payload.model_bytes) > 0
    assert len(payload.args_bytes) > 0
    assert len(payload.params_bytes) > 0
    assert accepted_records
    assert restored_problem.model == model
    assert restored_problem.collect_phantoms is False
    assert isinstance(restored_problem.params, CtxParams)
    _assert_trees_equal(restored_problem.args, args)
    _assert_trees_equal(restored_problem.params, params)

    sample = restored_problem.model.sample_U(
        jax.random.PRNGKey(29),
        args=restored_problem.args,
        params=restored_problem.params,
    )
    log_likelihood = restored_problem.model.log_likelihood(
        sample,
        args=restored_problem.args,
        params=restored_problem.params,
        allow_nan=False,
    )
    assert np.isfinite(float(log_likelihood))
    assert int(state.num_samples) == 4


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_accepted_dispatch_records_preserve_runtime_compile_identity():
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()
    model = Model(prior_model=runtime_prior_model_with_nested_args)
    args = (
        {
            "prior": {
                "loc": jnp.asarray(0.45),
                "scale": jnp.asarray(0.2),
            },
            "weights": {"likelihood": jnp.asarray(1.5)},
        },
    )
    params = model.init_params(key=jax.random.PRNGKey(41), args=args)

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1", "gpu:0:1"])
        runner = lb.get_nested_sampler(
            model=model,
            args=args,
            params=params,
            collect_phantoms=False,
            sampler=SerializedModelEvaluatingSampler(),
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )

        state = _run_small_runtime_runner(runner, max_samples=4)
        raw_records = _accepted_raw_dispatch_records_for_runner(
            runner,
            lb,
            runner,
            state,
        )
        record_compile_identities = tuple(
            _record_runtime_compile_identity(record)
            for record in raw_records
        )
        runner_compile_identity = _runtime_compile_identity(runner)
        payloads = tuple(
            _serialized_problem_payload(record, runtime)
            for record in raw_records
        )

    assert record_compile_identities == (
        (runner_compile_identity,) * len(record_compile_identities)
    )
    for payload in payloads:
        restored_problem = _restore_problem_from_payload(payload, runtime)
        assert restored_problem.model == model
        assert restored_problem.collect_phantoms is False
        assert isinstance(restored_problem.params, CtxParams)
        _assert_trees_equal(restored_problem.args, args)
        _assert_trees_equal(restored_problem.params, params)


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_unidim_runtime_dispatch_records_expose_sampler_loop_mode():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        model = make_toy_model()
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=False,
            sampler=_small_runtime_sampler(model, collect_phantoms=False),
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )

        state = _run_small_runtime_runner(runner, max_samples=4)
        raw_records = _accepted_raw_dispatch_records_for_runner(
            runner,
            lb,
            runner,
            state,
        )
        diagnostic_records = _accepted_raw_diagnostic_records_for_runner(
            runner,
            state,
        )

    raw_modes_by_task = {
        str(_record_field(record, "task_id")): _runtime_sampler_loop_mode(
            record
        )
        for record in raw_records
    }
    diagnostic_modes_by_task = {
        str(_record_field(record, "task_id")): _runtime_sampler_loop_mode(
            record
        )
        for record in diagnostic_records
    }
    assert set(raw_modes_by_task.values()) <= {
        "python",
        "fused",
    }
    assert diagnostic_modes_by_task == raw_modes_by_task


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_accepted_dispatch_records_expose_latency_fields_in_diagnostics():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=False,
            sampler=SerializedWorkerOnlySampler(),
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )

        state = _run_small_runtime_runner(runner, max_samples=4)
        raw_records = _accepted_raw_dispatch_records_for_runner(
            runner,
            lb,
            runner,
            state,
        )
        diagnostic_records = _accepted_raw_diagnostic_records_for_runner(
            runner,
            state,
        )

    raw_latency_by_task = {
        str(_record_field(record, "task_id")): _runtime_latency_values(record)
        for record in raw_records
    }
    diagnostic_latency_by_task = {
        str(_record_field(record, "task_id")): _runtime_latency_values(record)
        for record in diagnostic_records
    }
    assert diagnostic_latency_by_task == raw_latency_by_task


def test_multi_client_scheduler_records_client_and_cache_isolation():
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()

    with LoadBalancerClient(address="local") as first:
        with LoadBalancerClient(address="local") as second:
            first.add_workers(["cpu:*:1", "gpu:0:1"])
            first_client_id = first.client_id
            second_client_id = second.client_id
            first_runner = _compile_identity_runner(
                first,
                args_scale=0.75,
                params_scale=0.75,
            )
            second_runner = _compile_identity_runner(
                second,
                args_scale=0.95,
                params_scale=0.95,
            )
            first_compile_identity = _runtime_compile_identity(first_runner)
            second_compile_identity = _runtime_compile_identity(second_runner)

            first_pending_record = _prepare_runtime_dispatch(
                first_runner,
                seed_id="first-seed-000001",
                phantom_cluster_id="first-phantom-000001",
            )
            second_pending_record = _prepare_runtime_dispatch(
                second_runner,
                seed_id="second-seed-000001",
                phantom_cluster_id="second-phantom-000001",
            )
            second_followup_record = _prepare_runtime_dispatch(
                second_runner,
                seed_id="second-seed-000002",
                phantom_cluster_id="second-phantom-000002",
            )
            first_followup_record = _prepare_runtime_dispatch(
                first_runner,
                seed_id="first-seed-000002",
                phantom_cluster_id="first-phantom-000002",
            )

            first_pending = _normalise_lifecycle_record(
                first_pending_record
            )
            second_pending = _normalise_lifecycle_record(
                second_pending_record
            )
            second_followup = _normalise_lifecycle_record(
                second_followup_record
            )
            first_followup = _normalise_lifecycle_record(
                first_followup_record
            )

            first_accepted_record = _complete_runtime_dispatch(
                first_runner,
                first_pending_record,
                _worker_result_for_lifecycle_record(
                    first_pending_record,
                    runtime,
                ),
                current_parent_idx=first_pending.accepted_parent_idx,
                current_log_L_constraint=(
                    first_pending.accepted_log_L_constraint
                ),
            )
            second_accepted_record = _complete_runtime_dispatch(
                second_runner,
                second_pending_record,
                _worker_result_for_lifecycle_record(
                    second_pending_record,
                    runtime,
                ),
                current_parent_idx=second_pending.accepted_parent_idx,
                current_log_L_constraint=(
                    second_pending.accepted_log_L_constraint
                ),
            )
            first_accepted = _normalise_lifecycle_record(
                first_accepted_record
            )
            second_accepted = _normalise_lifecycle_record(
                second_accepted_record
            )

    first_records = (
        first_pending_record,
        first_followup_record,
        first_accepted_record,
    )
    second_records = (
        second_pending_record,
        second_followup_record,
        second_accepted_record,
    )
    first_task_ids = {first_pending.task_id, first_followup.task_id}
    second_task_ids = {second_pending.task_id, second_followup.task_id}

    assert first_compile_identity != second_compile_identity
    assert first_runner.runtime_runner_identity.client_id == first_client_id
    assert second_runner.runtime_runner_identity.client_id == second_client_id
    first_runner_id = first_runner.runtime_runner_identity.runner_id
    second_runner_id = second_runner.runtime_runner_identity.runner_id
    assert first_runner_id != second_runner_id
    assert {record.runner_id for record in first_records} == {
        first_runner_id,
    }
    assert {record.runner_id for record in second_records} == {
        second_runner_id,
    }
    assert first_task_ids.isdisjoint(second_task_ids)
    assert tuple(
        record.sector_id
        for record in (
            first_pending,
            second_pending,
            second_followup,
            first_followup,
        )
    ) == (
        "sector-000001",
        "sector-000002",
        "sector-000001",
        "sector-000002",
    )
    assert {first_pending.sector_id, first_followup.sector_id} == {
        "sector-000001",
        "sector-000002",
    }
    assert {second_pending.sector_id, second_followup.sector_id} == {
        "sector-000001",
        "sector-000002",
    }
    assert first_accepted.status == "accepted"
    assert second_accepted.status == "accepted"
    assert first_accepted.task_id == first_pending.task_id
    assert second_accepted.task_id == second_pending.task_id
    assert first_runner.runtime_acceptance_ledger.accepted_task_ids == (
        first_pending.task_id,
    )
    assert second_runner.runtime_acceptance_ledger.accepted_task_ids == (
        second_pending.task_id,
    )
    assert all(
        _record_client_id(record) == first_client_id
        for record in first_records
    )
    assert all(
        _record_client_id(record) == second_client_id
        for record in second_records
    )
    assert _record_runtime_compile_identity(
        first_accepted_record
    ) == first_compile_identity
    assert _record_runtime_compile_identity(
        second_accepted_record
    ) == second_compile_identity


def test_lifecycle_records_are_owned_by_the_issuing_runner():
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        first_runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        second_runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        first_record = _prepare_runtime_dispatch(first_runner)
        first_pending = _normalise_lifecycle_record(first_record)
        records_before = tuple(lb.coordinator_dispatch_records)

        assert first_pending.runner_id == _runner_id(first_runner)
        assert first_pending.runner_id != _runner_id(second_runner)

        with pytest.raises(ValueError, match="dispatch record owner"):
            _fail_runtime_dispatch(second_runner, first_record)
        with pytest.raises(ValueError, match="dispatch record owner"):
            _retry_runtime_dispatch(second_runner, first_record)
        with pytest.raises(ValueError, match="dispatch record owner"):
            _revoke_runtime_dispatch(second_runner, first_record)
        with pytest.raises(ValueError, match="dispatch record owner"):
            _complete_runtime_dispatch(
                second_runner,
                first_record,
                _worker_result_for_lifecycle_record(first_record, runtime),
                current_parent_idx=first_pending.accepted_parent_idx,
                current_log_L_constraint=(
                    first_pending.accepted_log_L_constraint
                ),
            )

        assert tuple(lb.coordinator_dispatch_records) == records_before
        assert first_runner.runtime_acceptance_ledger.accepted_task_ids == ()
        assert second_runner.runtime_acceptance_ledger.accepted_task_ids == ()

        accepted_record = _complete_runtime_dispatch(
            first_runner,
            first_record,
            _worker_result_for_lifecycle_record(first_record, runtime),
            current_parent_idx=first_pending.accepted_parent_idx,
            current_log_L_constraint=first_pending.accepted_log_L_constraint,
        )
        accepted = _normalise_lifecycle_record(accepted_record)

    assert accepted.status == "accepted"
    assert first_runner.runtime_acceptance_ledger.accepted_task_ids == (
        first_pending.task_id,
    )
    assert second_runner.runtime_acceptance_ledger.accepted_task_ids == ()


def test_local_clients_share_namespace_without_cross_runner_ledger_bleed():
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()

    with LoadBalancerClient(address="local") as first:
        with LoadBalancerClient(address="local") as second:
            assert first.load_balancer_state is second.load_balancer_state
            assert first.client_id != second.client_id

            first.add_workers(["cpu:*:1"])
            assert [
                (sector.sector_id, sector.device_type, sector.device_id)
                for sector in second.compute_sectors
            ] == [("sector-000001", "cpu", "*")]

            second.add_workers(["gpu:0:2"])
            assert first.compute_sectors == second.compute_sectors
            assert [
                (sector.sector_id, sector.device_type, sector.device_id)
                for sector in first.compute_sectors
            ] == [
                ("sector-000001", "cpu", "*"),
                ("sector-000002", "gpu", "0"),
            ]

            first_runner = first.get_nested_sampler(model=make_toy_model())
            second_runner = second.get_nested_sampler(model=make_toy_model())
            first_runner_identity = getattr(
                first_runner,
                "runtime_runner_identity",
            )
            second_runner_identity = getattr(
                second_runner,
                "runtime_runner_identity",
            )
            assert first_runner_identity.client_id == first.client_id
            assert second_runner_identity.client_id == second.client_id
            assert (
                first_runner_identity.runner_id
                != second_runner_identity.runner_id
            )

            first_task = first.allocate_task_identity(
                first_runner_identity.runner_id
            )
            second_task = second.allocate_task_identity(
                second_runner_identity.runner_id
            )
            assert first_task.task_id != second_task.task_id

            first_ledger = getattr(first_runner, "runtime_acceptance_ledger")
            second_ledger = getattr(second_runner, "runtime_acceptance_ledger")
            assert first_ledger is not second_ledger

            first_result = runtime.WorkerResultIdentity(
                task_id=first_task.task_id,
                attempt_id="attempt-1",
                transport_id="transport-1",
            )
            second_result = runtime.WorkerResultIdentity(
                task_id=second_task.task_id,
                attempt_id="attempt-2",
                transport_id="transport-2",
            )
            assert first_ledger.accept(first_result).accepted is True
            assert second_ledger.has_accepted(first_task.task_id) is False
            assert second_ledger.accept(second_result).accepted is True
            assert first_ledger.has_accepted(second_task.task_id) is False

            second_sector_ids_before_shutdown = tuple(
                sector.sector_id for sector in second.compute_sectors
            )
            first.shutdown()
            assert _sector_resource_summary(second) == (("gpu", "0", 2),)
            assert tuple(
                sector.sector_id for sector in second.compute_sectors
            ) == (second_sector_ids_before_shutdown[1],)
            second.add_workers(["cpu:*:3"])
            assert _sector_resource_summary(second) == (
                ("gpu", "0", 2),
                ("cpu", "*", 3),
            )
            assert len({
                sector.sector_id for sector in second.compute_sectors
            }) == 2

    with LoadBalancerClient(address="local") as fresh:
        assert fresh.compute_sectors == ()
        fresh.add_workers(["cpu:*:1"])
        assert [
            (sector.sector_id, sector.device_type, sector.device_id)
            for sector in fresh.compute_sectors
        ] == [("sector-000001", "cpu", "*")]


def test_local_context_exit_unregisters_only_owned_worker_sectors():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as coordinator:
        coordinator.add_workers(["gpu:0:1"])
        with LoadBalancerClient(address="local") as worker_node:
            worker_node.add_workers(["cpu:*:2"])
            assert _sector_resource_summary(coordinator) == (
                ("gpu", "0", 1),
                ("cpu", "*", 2),
            )
            assert coordinator.compute_sectors == worker_node.compute_sectors
            coordinator_sector_id = coordinator.compute_sectors[0].sector_id

        assert _sector_resource_summary(coordinator) == (("gpu", "0", 1),)
        assert tuple(
            sector.sector_id for sector in coordinator.compute_sectors
        ) == (coordinator_sector_id,)

    with LoadBalancerClient(address="local") as fresh:
        assert fresh.compute_sectors == ()
        fresh.add_workers(["cpu:*:1"])
        assert [
            (sector.sector_id, sector.device_type, sector.device_id)
            for sector in fresh.compute_sectors
        ] == [("sector-000001", "cpu", "*")]


def test_owned_worker_cleanup_preserves_duplicate_spec_sibling_sector():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as first_owner_closes:
        with LoadBalancerClient(address="local") as second:
            first_owner_closes.add_workers(["cpu:*:1"])
            second.add_workers(["cpu:*:1"])
            assert _sector_resource_summary(first_owner_closes) == (
                ("cpu", "*", 1),
                ("cpu", "*", 1),
            )
            first_sector_id = first_owner_closes.compute_sectors[0].sector_id
            second_sector_id = second.compute_sectors[1].sector_id
            assert first_sector_id != second_sector_id

            first_owner_closes.shutdown()

            assert _sector_resource_summary(second) == (("cpu", "*", 1),)
            assert tuple(
                sector.sector_id for sector in second.compute_sectors
            ) == (second_sector_id,)

    with LoadBalancerClient(address="local") as first:
        with LoadBalancerClient(address="local") as second_owner_closes:
            first.add_workers(["cpu:*:1"])
            second_owner_closes.add_workers(["cpu:*:1"])
            assert _sector_resource_summary(first) == (
                ("cpu", "*", 1),
                ("cpu", "*", 1),
            )
            first_sector_id = first.compute_sectors[0].sector_id
            second_sector_id = second_owner_closes.compute_sectors[1].sector_id
            assert first_sector_id != second_sector_id

            second_owner_closes.shutdown()

            assert _sector_resource_summary(first) == (("cpu", "*", 1),)
            assert tuple(
                sector.sector_id for sector in first.compute_sectors
            ) == (first_sector_id,)


@PRE_0018_LEGACY_SERIALIZED_WORKER_SKIP
def test_pre_0018_legacy_tcp_clients_share_worker_namespace_and_worker_exit_cleans_sectors():
    LoadBalancerClient = _load_balancer_client()
    address = "tcp://127.0.0.1:19009"

    with LoadBalancerClient(address=address) as coordinator:
        with LoadBalancerClient(address=address) as worker_node:
            worker_node.add_workers(["cpu:*:2"])
            assert (
                coordinator.load_balancer_state
                is worker_node.load_balancer_state
            )
            assert _sector_resource_summary(coordinator) == (
                ("cpu", "*", 2),
            )

            runner = coordinator.get_nested_sampler(
                model=make_toy_model(),
                collect_phantoms=False,
                sampler=SerializedWorkerOnlySampler(),
                target_num_live_points=2,
                max_samples=4,
                shell_size=1,
                termination_condition=TerminationCondition(max_samples=4),
                batch_size=None,
            )
            state = _run_small_runtime_runner(runner, max_samples=4)
            accepted_records = _accepted_dispatch_records_for_runner(
                runner,
                coordinator,
                runner,
                state,
            )

            assert int(state.num_samples) == 4
            assert {
                record.sector_id for record in accepted_records
            } == {
                worker_node.compute_sectors[0].sector_id,
            }
            _assert_accepted_dispatch_is_non_duplicating(
                accepted_records,
                runner,
            )

        assert coordinator.compute_sectors == ()


def test_tcp_worker_namespaces_are_isolated_by_address():
    LoadBalancerClient = _load_balancer_client()
    first_address = "tcp://127.0.0.1:19013"
    second_address = "tcp://127.0.0.1:19014"

    with LoadBalancerClient(address=first_address) as first:
        with LoadBalancerClient(address=second_address) as second:
            first.add_workers(["cpu:*:1"])
            second.add_workers(["gpu:0:2"])

            assert [
                (sector.sector_id, sector.device_type, sector.device_id)
                for sector in first.compute_sectors
            ] == [("sector-000001", "cpu", "*")]
            assert [
                (sector.sector_id, sector.device_type, sector.device_id)
                for sector in second.compute_sectors
            ] == [("sector-000001", "gpu", "0")]
            assert first.load_balancer_state is not second.load_balancer_state
            assert _sector_resource_summary(first) == (("cpu", "*", 1),)
            assert _sector_resource_summary(second) == (("gpu", "0", 2),)


def test_load_balancer_client_surface_does_not_expose_legacy_lease_api():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        for method_name in (
            "acquire_lease",
            "release_lease",
            "request_lease",
            "cancel_lease",
        ):
            assert not hasattr(lb, method_name)


def test_wait_until_shutdown_returns_after_local_context_shutdown():
    LoadBalancerClient = _load_balancer_client()
    errors: queue.Queue[BaseException] = queue.Queue()
    waiting = threading.Event()
    finished = threading.Event()
    lb = LoadBalancerClient(address="local")

    def wait_for_shutdown() -> None:
        waiting.set()
        try:
            lb.wait_until_shutdown()
        except BaseException as exc:  # pragma: no cover - surfaced below.
            errors.put(exc)
        finally:
            finished.set()

    with lb:
        thread = threading.Thread(target=wait_for_shutdown, daemon=True)
        thread.start()
        assert waiting.wait(timeout=1.0)

    assert finished.wait(timeout=2.0)
    thread.join(timeout=1.0)
    assert errors.empty()


def test_worker_waits_for_shared_shutdown_request_then_unregisters_workers():
    LoadBalancerClient = _load_balancer_client()
    address = "tcp://127.0.0.1:19010"
    errors: queue.Queue[BaseException] = queue.Queue()
    waiting = threading.Event()
    finished = threading.Event()

    with LoadBalancerClient(address=address) as coordinator:
        with LoadBalancerClient(address=address) as worker_node:
            worker_node.add_workers(["cpu:*:1"])
            assert _sector_resource_summary(coordinator) == (
                ("cpu", "*", 1),
            )

            request_shutdown = getattr(coordinator, "request_shutdown", None)
            assert callable(request_shutdown), (
                "LoadBalancerClient must expose public request_shutdown()."
            )

            def wait_for_shutdown() -> None:
                waiting.set()
                try:
                    returned = worker_node.wait_until_shutdown()
                    if returned is False:
                        raise AssertionError(
                            "wait_until_shutdown() must return after shared "
                            "shutdown request, not a timeout."
                        )
                except BaseException as exc:  # pragma: no cover - surfaced.
                    errors.put(exc)
                finally:
                    finished.set()

            thread = threading.Thread(target=wait_for_shutdown, daemon=True)
            thread.start()
            assert waiting.wait(timeout=1.0)
            assert not finished.wait(timeout=0.2)

            request_shutdown()

            assert finished.wait(timeout=2.0)
            thread.join(timeout=1.0)
            assert errors.empty()
            assert _sector_resource_summary(coordinator) == (
                ("cpu", "*", 1),
            )

        assert coordinator.compute_sectors == ()


def test_shared_shutdown_request_is_scoped_to_address():
    LoadBalancerClient = _load_balancer_client()
    first_address = "tcp://127.0.0.1:19011"
    second_address = "tcp://127.0.0.1:19012"
    errors: queue.Queue[BaseException] = queue.Queue()
    first_waiting = threading.Event()
    second_waiting = threading.Event()
    first_finished = threading.Event()
    second_finished = threading.Event()

    def wait_for_shutdown(client, waiting, finished) -> None:
        waiting.set()
        try:
            returned = client.wait_until_shutdown()
            if returned is False:
                raise AssertionError(
                    "wait_until_shutdown() must not return by timeout."
                )
        except BaseException as exc:  # pragma: no cover - surfaced below.
            errors.put(exc)
        finally:
            finished.set()

    with LoadBalancerClient(address=first_address) as first:
        with LoadBalancerClient(address=second_address) as second:
            first_thread = threading.Thread(
                target=wait_for_shutdown,
                args=(first, first_waiting, first_finished),
                daemon=True,
            )
            second_thread = threading.Thread(
                target=wait_for_shutdown,
                args=(second, second_waiting, second_finished),
                daemon=True,
            )
            first_thread.start()
            second_thread.start()
            assert first_waiting.wait(timeout=1.0)
            assert second_waiting.wait(timeout=1.0)

            _request_load_balancer_shutdown(first)

            assert first_finished.wait(timeout=2.0)
            assert not second_finished.wait(timeout=0.2)
            assert errors.empty()

            _request_load_balancer_shutdown(second)

            assert second_finished.wait(timeout=2.0)
            first_thread.join(timeout=1.0)
            second_thread.join(timeout=1.0)
            assert errors.empty()


def test_reentered_client_uses_fresh_pool_and_shutdown_event():
    LoadBalancerClient = _load_balancer_client()
    client = LoadBalancerClient(address="local")

    with client:
        client.add_workers(["cpu:*:1"])
        assert _sector_resource_summary(client) == (("cpu", "*", 1),)
        _request_load_balancer_shutdown(client)

    with client:
        assert client.compute_sectors == ()

        errors: queue.Queue[BaseException] = queue.Queue()
        waiting = threading.Event()
        finished = threading.Event()

        def wait_for_shutdown() -> None:
            waiting.set()
            try:
                returned = client.wait_until_shutdown()
                if returned is False:
                    raise AssertionError(
                        "wait_until_shutdown() must not return by timeout."
                    )
            except BaseException as exc:  # pragma: no cover - surfaced below.
                errors.put(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=wait_for_shutdown, daemon=True)
        thread.start()
        assert waiting.wait(timeout=1.0)
        assert not finished.wait(timeout=0.2)

        _request_load_balancer_shutdown(client)

        assert finished.wait(timeout=2.0)
        thread.join(timeout=1.0)
        assert errors.empty()

    with LoadBalancerClient(address="local") as fresh:
        assert fresh.compute_sectors == ()
        fresh.add_workers(["cpu:*:1"])
        assert [
            (sector.sector_id, sector.device_type, sector.device_id)
            for sector in fresh.compute_sectors
        ] == [("sector-000001", "cpu", "*")]

        errors = queue.Queue()
        waiting = threading.Event()
        finished = threading.Event()

        def wait_for_fresh_shutdown() -> None:
            waiting.set()
            try:
                returned = fresh.wait_until_shutdown()
                if returned is False:
                    raise AssertionError(
                        "wait_until_shutdown() must not return by timeout."
                    )
            except BaseException as exc:  # pragma: no cover - surfaced below.
                errors.put(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=wait_for_fresh_shutdown, daemon=True)
        thread.start()
        assert waiting.wait(timeout=1.0)
        assert not finished.wait(timeout=0.2)

        _request_load_balancer_shutdown(fresh)

        assert finished.wait(timeout=2.0)
        thread.join(timeout=1.0)
        assert errors.empty()


def test_recreated_tcp_client_uses_fresh_shutdown_event():
    LoadBalancerClient = _load_balancer_client()
    address = "tcp://127.0.0.1:19015"

    with LoadBalancerClient(address=address) as first:
        _request_load_balancer_shutdown(first)

    with LoadBalancerClient(address=address) as fresh:
        errors: queue.Queue[BaseException] = queue.Queue()
        waiting = threading.Event()
        finished = threading.Event()

        def wait_for_fresh_shutdown() -> None:
            waiting.set()
            try:
                returned = fresh.wait_until_shutdown()
                if returned is False:
                    raise AssertionError(
                        "wait_until_shutdown() must not return by timeout."
                    )
            except BaseException as exc:  # pragma: no cover - surfaced below.
                errors.put(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=wait_for_fresh_shutdown, daemon=True)
        thread.start()
        assert waiting.wait(timeout=1.0)
        assert not finished.wait(timeout=0.2)

        _request_load_balancer_shutdown(fresh)

        assert finished.wait(timeout=2.0)
        thread.join(timeout=1.0)
        assert errors.empty()


def test_runtime_pickle_helpers_round_trip_model_args_and_params():
    runtime = _runtime_module()
    problem = _runtime_problem()

    restored_model = runtime.round_trip_model(problem.model)
    restored_args = runtime.round_trip_args(problem.args)
    restored_params = runtime.round_trip_params(problem.params)

    assert restored_model == problem.model
    _assert_trees_equal(restored_args, problem.args)
    _assert_trees_equal(restored_params, problem.params)


def test_serialized_model_problem_round_trips_real_ctx_params_payload():
    runtime = _runtime_module()
    model = Model(prior_model=runtime_prior_model_with_parameter)
    args = (0.75,)
    params = model.init_params(key=jax.random.PRNGKey(9), args=args)

    assert isinstance(params, CtxParams)

    payload = runtime.SerializedModelProblem.from_problem(
        model=model,
        args=args,
        params=params,
        collect_phantoms=True,
    )
    restored_problem = payload.deserialize_problem()

    assert len(payload.params_bytes) > 0
    assert restored_problem.model == model
    assert restored_problem.collect_phantoms is True
    assert isinstance(restored_problem.params, CtxParams)
    _assert_trees_equal(restored_problem.args, args)
    _assert_trees_equal(restored_problem.params, params)
    _assert_trees_equal(
        runtime.deserialize_params(payload.params_bytes),
        params,
    )


def test_local_load_balancer_state_allocates_deterministic_identities():
    runtime = _runtime_module()
    state = runtime.LocalLoadBalancerState()

    client_id = state.allocate_client_id()
    runner_identity = state.allocate_runner_identity(client_id)
    task_identity = state.allocate_task_identity(runner_identity.runner_id)
    attempt_identity = state.allocate_attempt_identity(task_identity.task_id)
    transport_identity = state.allocate_transport_identity(
        attempt_identity.attempt_id
    )

    assert client_id == "client-000001"
    assert runner_identity == runtime.RunnerIdentity(
        runner_id="runner-000001",
        client_id="client-000001",
    )
    assert task_identity == runtime.TaskIdentity(
        task_id="task-000001",
        runner_id="runner-000001",
    )
    assert attempt_identity == runtime.AttemptIdentity(
        attempt_id="attempt-000001",
        task_id="task-000001",
        attempt_number=1,
    )
    assert transport_identity == runtime.TransportIdentity(
        transport_id="transport-000001",
        attempt_id="attempt-000001",
        delivery_number=1,
    )


def test_attempt_and_delivery_numbers_are_scoped_to_parent_identities():
    runtime = _runtime_module()
    state = runtime.LocalLoadBalancerState()

    first_task = state.allocate_task_identity("runner-a")
    second_task = state.allocate_task_identity("runner-b")

    first_task_attempt = state.allocate_attempt_identity(first_task.task_id)
    second_task_attempt = state.allocate_attempt_identity(second_task.task_id)
    first_task_retry = state.allocate_attempt_identity(first_task.task_id)

    assert first_task_attempt == runtime.AttemptIdentity(
        attempt_id="attempt-000001",
        task_id=first_task.task_id,
        attempt_number=1,
    )
    assert second_task_attempt == runtime.AttemptIdentity(
        attempt_id="attempt-000002",
        task_id=second_task.task_id,
        attempt_number=1,
    )
    assert first_task_retry == runtime.AttemptIdentity(
        attempt_id="attempt-000003",
        task_id=first_task.task_id,
        attempt_number=2,
    )

    first_delivery = state.allocate_transport_identity(
        first_task_attempt.attempt_id
    )
    second_attempt_delivery = state.allocate_transport_identity(
        second_task_attempt.attempt_id
    )
    first_redelivery = state.allocate_transport_identity(
        first_task_attempt.attempt_id
    )

    assert first_delivery == runtime.TransportIdentity(
        transport_id="transport-000001",
        attempt_id=first_task_attempt.attempt_id,
        delivery_number=1,
    )
    assert second_attempt_delivery == runtime.TransportIdentity(
        transport_id="transport-000002",
        attempt_id=second_task_attempt.attempt_id,
        delivery_number=1,
    )
    assert first_redelivery == runtime.TransportIdentity(
        transport_id="transport-000003",
        attempt_id=first_task_attempt.attempt_id,
        delivery_number=2,
    )


def test_acceptance_ledger_accepts_each_task_id_once_across_attempts():
    runtime = _runtime_module()
    ledger = runtime.AcceptanceLedger()
    first_result = runtime.WorkerResultIdentity(
        task_id="task-000001",
        attempt_id="attempt-000001",
        transport_id="transport-000001",
        worker_id="worker-1",
        sector_id="sector-000001",
    )
    replayed_result = runtime.WorkerResultIdentity(
        task_id="task-000001",
        attempt_id="attempt-000002",
        transport_id="transport-000002",
        worker_id="worker-2",
        sector_id="sector-000001",
    )

    first_decision = ledger.accept(first_result)
    replayed_decision = ledger.accept(replayed_result)

    assert first_decision.accepted is True
    assert first_decision.reason == "accepted"
    assert replayed_decision.accepted is False
    assert replayed_decision.reason == "duplicate_task_result"
    assert replayed_decision.accepted_identity == first_result
    assert ledger.accepted_task_ids == ("task-000001",)


def test_pending_runtime_dispatch_is_public_and_non_mutating():
    LoadBalancerClient = _load_balancer_client()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=True,
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        state = _run_small_runtime_runner(runner, max_samples=4)

        state_before = _runtime_state_snapshot(state)
        accepted_task_ids_before = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        pending_record = _prepare_runtime_dispatch(runner)
        state_after = _runtime_state_snapshot(state)

        pending = _normalise_lifecycle_record(pending_record)
        task_records = _lifecycle_records_for_task(
            pending.task_id,
            lb,
            runner,
        )

    assert pending.status == "pending"
    assert pending.identity_owner == "coordinator"
    assert pending.runner_id == _runner_id(runner)
    assert pending.attempt_number == 1
    assert pending.delivery_number == 1
    assert pending.requested_parent_idx == 0
    assert pending.effective_parent_idx == 0
    assert pending.accepted_parent_idx == 0
    assert pending.effective_log_L_constraint == -0.5
    assert pending.accepted_log_L_constraint == -0.5
    assert pending.seed_id == "seed-000001"
    assert pending.phantom_cluster_id == "phantom-cluster-000001"
    assert pending.worker_id
    assert pending.sector_id == "sector-000001"
    assert [record.status for record in task_records] == ["pending"]
    assert (
        runner.runtime_acceptance_ledger.accepted_task_ids
        == accepted_task_ids_before
    )
    _assert_runtime_state_snapshot_equal(state_before, state_after)


def test_pending_runtime_dispatch_uses_explicit_sentinel_parent_metadata():
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()
    sentinel_parent_idx = int(runtime.SENTINEL_PARENT_IDX)

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=True,
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        state = _run_small_runtime_runner(runner, max_samples=4)

        state_before = _runtime_state_snapshot(state)
        accepted_task_ids_before = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        pending_record = _prepare_runtime_dispatch(
            runner,
            requested_parent_idx=sentinel_parent_idx,
            effective_parent_idx=sentinel_parent_idx,
            accepted_parent_idx=sentinel_parent_idx,
            effective_log_L_constraint=-np.inf,
            accepted_log_L_constraint=-np.inf,
            seed_id="seed-sentinel-unavailable",
            phantom_cluster_id="phantom-cluster-sentinel",
        )
        state_after = _runtime_state_snapshot(state)

        pending = _normalise_lifecycle_record(pending_record)
        task_records = _lifecycle_records_for_task(
            pending.task_id,
            lb,
            runner,
        )

    assert sentinel_parent_idx == -1
    assert pending.status == "pending"
    assert pending.requested_parent_idx == sentinel_parent_idx
    assert pending.effective_parent_idx == sentinel_parent_idx
    assert pending.accepted_parent_idx == sentinel_parent_idx
    assert np.isneginf(pending.effective_log_L_constraint)
    assert np.isneginf(pending.accepted_log_L_constraint)
    assert pending.seed_id == "seed-sentinel-unavailable"
    assert pending.phantom_cluster_id == "phantom-cluster-sentinel"
    assert [record.status for record in task_records] == ["pending"]
    assert (
        runner.runtime_acceptance_ledger.accepted_task_ids
        == accepted_task_ids_before
    )
    _assert_runtime_state_snapshot_equal(state_before, state_after)


def test_failed_revoked_retried_and_stale_dispatch_lifecycle_records():
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=True,
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        state = _run_small_runtime_runner(runner, max_samples=4)

        state_before = _runtime_state_snapshot(state)
        accepted_task_ids_before = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        pending_record = _prepare_runtime_dispatch(
            runner,
            requested_parent_idx=2,
            effective_parent_idx=1,
            accepted_parent_idx=1,
            effective_log_L_constraint=-0.75,
            accepted_log_L_constraint=-0.75,
            seed_id="seed-lifecycle-parent",
            phantom_cluster_id="phantom-cluster-preserved",
        )
        failed_record = _fail_runtime_dispatch(runner, pending_record)
        retry_after_failure_record = _retry_runtime_dispatch(
            runner,
            failed_record,
            reason="worker failure",
        )
        revoked_record = _revoke_runtime_dispatch(
            runner,
            retry_after_failure_record,
        )
        retry_after_revoke_record = _retry_runtime_dispatch(
            runner,
            revoked_record,
            reason="revoked delivery should be reissued",
        )
        active_after_supersede_record = _retry_runtime_dispatch(
            runner,
            retry_after_revoke_record,
            reason="attempt timed out before completion arrived",
        )

        pending = _normalise_lifecycle_record(pending_record)
        failed = _normalise_lifecycle_record(failed_record)
        retry_after_failure = _normalise_lifecycle_record(
            retry_after_failure_record
        )
        revoked = _normalise_lifecycle_record(revoked_record)
        retry_after_revoke = _normalise_lifecycle_record(
            retry_after_revoke_record
        )
        active_after_supersede = _normalise_lifecycle_record(
            active_after_supersede_record
        )

        state_before_rejected_completions = _runtime_state_snapshot(state)
        failed_completion_record = _complete_runtime_dispatch(
            runner,
            pending_record,
            _worker_result_for_lifecycle_record(pending_record, runtime),
            current_parent_idx=pending.accepted_parent_idx,
            current_log_L_constraint=pending.accepted_log_L_constraint,
        )
        revoked_completion_record = _complete_runtime_dispatch(
            runner,
            retry_after_failure_record,
            _worker_result_for_lifecycle_record(
                retry_after_failure_record,
                runtime,
            ),
            current_parent_idx=retry_after_failure.accepted_parent_idx,
            current_log_L_constraint=(
                retry_after_failure.accepted_log_L_constraint
            ),
        )
        accepted_task_ids_after_rejected_completions = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        state_after_rejected_completions = _runtime_state_snapshot(state)

        active_completion_record = _complete_runtime_dispatch(
            runner,
            active_after_supersede_record,
            _worker_result_for_lifecycle_record(
                active_after_supersede_record,
                runtime,
            ),
            current_parent_idx=active_after_supersede.accepted_parent_idx,
            current_log_L_constraint=(
                active_after_supersede.accepted_log_L_constraint
            ),
        )
        accepted_task_ids_after_active_completion = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        state_after_active_completion = _runtime_state_snapshot(state)

        superseded_completion_record = _complete_runtime_dispatch(
            runner,
            retry_after_revoke_record,
            _worker_result_for_lifecycle_record(
                retry_after_revoke_record,
                runtime,
            ),
            current_parent_idx=retry_after_revoke.accepted_parent_idx,
            current_log_L_constraint=(
                retry_after_revoke.accepted_log_L_constraint
            ),
        )
        accepted_task_ids_after_superseded_completion = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        state_after_superseded_completion = _runtime_state_snapshot(state)

        failed_completion = _normalise_lifecycle_record(
            failed_completion_record
        )
        revoked_completion = _normalise_lifecycle_record(
            revoked_completion_record
        )
        active_completion = _normalise_lifecycle_record(
            active_completion_record
        )
        superseded_completion = _normalise_lifecycle_record(
            superseded_completion_record
        )
        task_records = _lifecycle_records_for_task(
            pending.task_id,
            lb,
            runner,
        )

    assert [record.status for record in task_records] == [
        "pending",
        "failed",
        "retried",
        "revoked",
        "retried",
        "retried",
        "failed",
        "revoked",
        "accepted",
        "stale_task_result",
    ]
    assert {
        pending.task_id,
        failed.task_id,
        retry_after_failure.task_id,
        revoked.task_id,
        retry_after_revoke.task_id,
        active_after_supersede.task_id,
    } == {pending.task_id}
    assert failed.attempt_id == pending.attempt_id
    assert failed.transport_id == pending.transport_id
    assert retry_after_failure.attempt_id != pending.attempt_id
    assert retry_after_failure.transport_id != pending.transport_id
    assert retry_after_failure.attempt_number == 2
    assert retry_after_failure.delivery_number == 1
    assert revoked.attempt_id == retry_after_failure.attempt_id
    assert revoked.transport_id == retry_after_failure.transport_id
    assert retry_after_revoke.attempt_id not in {
        pending.attempt_id,
        retry_after_failure.attempt_id,
    }
    assert retry_after_revoke.transport_id not in {
        pending.transport_id,
        retry_after_failure.transport_id,
    }
    assert retry_after_revoke.attempt_number == 3
    assert retry_after_revoke.delivery_number == 1
    assert active_after_supersede.attempt_id not in {
        pending.attempt_id,
        retry_after_failure.attempt_id,
        retry_after_revoke.attempt_id,
    }
    assert active_after_supersede.transport_id not in {
        pending.transport_id,
        retry_after_failure.transport_id,
        retry_after_revoke.transport_id,
    }
    assert active_after_supersede.attempt_number == 4
    assert active_after_supersede.delivery_number == 1

    _assert_lifecycle_parent_metadata_preserved(
        pending,
        failed,
        retry_after_failure,
        revoked,
        retry_after_revoke,
        active_after_supersede,
        failed_completion,
        revoked_completion,
        active_completion,
        superseded_completion,
    )
    assert failed_completion.status == "failed"
    assert failed_completion.task_id == pending.task_id
    assert failed_completion.attempt_id == failed.attempt_id
    assert failed_completion.transport_id == failed.transport_id
    assert revoked_completion.status == "revoked"
    assert revoked_completion.task_id == pending.task_id
    assert revoked_completion.attempt_id == revoked.attempt_id
    assert revoked_completion.transport_id == revoked.transport_id
    assert active_completion.status == "accepted"
    assert active_completion.task_id == pending.task_id
    assert (
        active_completion.attempt_id
        == active_after_supersede.attempt_id
    )
    assert (
        active_completion.transport_id
        == active_after_supersede.transport_id
    )
    assert superseded_completion.status == "stale_task_result"
    assert superseded_completion.task_id == pending.task_id
    assert superseded_completion.attempt_id == retry_after_revoke.attempt_id
    assert (
        superseded_completion.transport_id
        == retry_after_revoke.transport_id
    )
    assert (
        accepted_task_ids_after_rejected_completions
        == accepted_task_ids_before
    )
    assert len(accepted_task_ids_after_active_completion) == (
        len(accepted_task_ids_before) + 1
    )
    assert set(accepted_task_ids_after_active_completion) == (
        set(accepted_task_ids_before) | {pending.task_id}
    )
    assert accepted_task_ids_after_active_completion.count(
        pending.task_id
    ) == 1
    assert (
        accepted_task_ids_after_superseded_completion
        == accepted_task_ids_after_active_completion
    )
    _assert_runtime_state_snapshot_equal(
        state_before,
        state_before_rejected_completions,
    )
    _assert_runtime_state_snapshot_equal(
        state_before_rejected_completions,
        state_after_rejected_completions,
    )
    _assert_runtime_state_snapshot_equal(
        state_after_active_completion,
        state_after_superseded_completion,
    )


def test_runtime_completion_rejects_superseded_attempt_then_accepts_live_retry():
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        state = _run_small_runtime_runner(runner, max_samples=4)

        pending_record = _prepare_runtime_dispatch(runner)
        later_attempt_record = _retry_runtime_dispatch(
            runner,
            pending_record,
            reason="timeout before original completion arrived",
        )
        pending = _normalise_lifecycle_record(pending_record)
        later_attempt = _normalise_lifecycle_record(later_attempt_record)
        first_result = _worker_result_for_lifecycle_record(
            pending_record,
            runtime,
        )
        later_result = _worker_result_for_lifecycle_record(
            later_attempt_record,
            runtime,
        )

        accepted_task_ids_before = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        superseded_record = _complete_runtime_dispatch(
            runner,
            pending_record,
            first_result,
            current_parent_idx=pending.accepted_parent_idx,
            current_log_L_constraint=pending.accepted_log_L_constraint,
        )
        accepted_task_ids_after_superseded = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        state_before_replay = _runtime_state_snapshot(state)
        replayed_record = _complete_runtime_dispatch(
            runner,
            pending_record,
            first_result,
            current_parent_idx=pending.accepted_parent_idx,
            current_log_L_constraint=pending.accepted_log_L_constraint,
        )
        state_after_replay = _runtime_state_snapshot(state)
        accepted_task_ids_after_replay = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        state_before_live_completion = _runtime_state_snapshot(state)
        later_completion_record = _complete_runtime_dispatch(
            runner,
            later_attempt_record,
            later_result,
            current_parent_idx=later_attempt.accepted_parent_idx,
            current_log_L_constraint=(
                later_attempt.accepted_log_L_constraint
            ),
        )
        state_after_live_completion = _runtime_state_snapshot(state)
        accepted_task_ids_after_later_completion = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )

    superseded = _normalise_lifecycle_record(superseded_record)
    replayed = _normalise_lifecycle_record(replayed_record)
    later_completion = _normalise_lifecycle_record(later_completion_record)

    assert superseded.status == "stale_task_result"
    assert replayed.status == "stale_task_result"
    assert later_completion.status == "accepted"
    assert pending.task_id not in accepted_task_ids_before
    assert superseded.task_id == pending.task_id
    assert replayed.task_id == pending.task_id
    assert later_completion.task_id == pending.task_id
    assert later_completion.attempt_id == later_attempt.attempt_id
    assert later_completion.attempt_number == 2
    assert accepted_task_ids_after_superseded == accepted_task_ids_before
    assert accepted_task_ids_after_replay == accepted_task_ids_before
    assert len(accepted_task_ids_after_later_completion) == (
        len(accepted_task_ids_before) + 1
    )
    assert set(accepted_task_ids_after_later_completion) == (
        set(accepted_task_ids_before) | {pending.task_id}
    )
    assert accepted_task_ids_after_later_completion.count(pending.task_id) == 1
    _assert_runtime_state_snapshot_equal(
        state_before_replay,
        state_after_replay,
    )
    _assert_runtime_state_snapshot_equal(
        state_before_live_completion,
        state_after_live_completion,
    )


def test_runtime_completion_accepts_later_attempt_first_then_stales_original():
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        state = _run_small_runtime_runner(runner, max_samples=4)

        pending_record = _prepare_runtime_dispatch(runner)
        later_attempt_record = _retry_runtime_dispatch(
            runner,
            pending_record,
            reason="timeout before original completion arrived",
        )
        pending = _normalise_lifecycle_record(pending_record)
        later_attempt = _normalise_lifecycle_record(later_attempt_record)
        original_result = _worker_result_for_lifecycle_record(
            pending_record,
            runtime,
        )
        later_result = _worker_result_for_lifecycle_record(
            later_attempt_record,
            runtime,
        )

        accepted_task_ids_before = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        later_accepted_record = _complete_runtime_dispatch(
            runner,
            later_attempt_record,
            later_result,
            current_parent_idx=later_attempt.accepted_parent_idx,
            current_log_L_constraint=(
                later_attempt.accepted_log_L_constraint
            ),
        )
        accepted_task_ids_after_later = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        state_before_original_completion = _runtime_state_snapshot(state)
        original_completion_record = _complete_runtime_dispatch(
            runner,
            pending_record,
            original_result,
            current_parent_idx=pending.accepted_parent_idx,
            current_log_L_constraint=pending.accepted_log_L_constraint,
        )
        state_after_original_completion = _runtime_state_snapshot(state)
        accepted_task_ids_after_original = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        task_records = _lifecycle_records_for_task(
            pending.task_id,
            lb,
            runner,
        )

    later_accepted = _normalise_lifecycle_record(later_accepted_record)
    original_completion = _normalise_lifecycle_record(
        original_completion_record
    )

    assert [record.status for record in task_records] == [
        "pending",
        "retried",
        "accepted",
        "stale_task_result",
    ]
    assert pending.task_id not in accepted_task_ids_before
    assert later_accepted.status == "accepted"
    assert later_accepted.task_id == pending.task_id
    assert later_accepted.attempt_id == later_attempt.attempt_id
    assert later_accepted.attempt_number == 2
    assert original_completion.status == "stale_task_result"
    assert original_completion.task_id == pending.task_id
    assert original_completion.attempt_id == pending.attempt_id
    assert original_completion.attempt_number == 1
    assert len(accepted_task_ids_after_later) == (
        len(accepted_task_ids_before) + 1
    )
    assert set(accepted_task_ids_after_later) == (
        set(accepted_task_ids_before) | {pending.task_id}
    )
    assert accepted_task_ids_after_later.count(pending.task_id) == 1
    assert accepted_task_ids_after_original == accepted_task_ids_after_later
    _assert_runtime_state_snapshot_equal(
        state_before_original_completion,
        state_after_original_completion,
    )


@pytest.mark.parametrize(
    "mismatch_field",
    ("task_id", "attempt_id", "transport_id"),
)
def test_runtime_completion_rejects_mismatched_result_identity(
        mismatch_field: str,
):
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=True,
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        state = _run_small_runtime_runner(runner, max_samples=4)

        state_before = _runtime_state_snapshot(state)
        accepted_task_ids_before = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        pending_record = _prepare_runtime_dispatch(runner)
        pending = _normalise_lifecycle_record(pending_record)
        mismatched_result = _mismatched_worker_result_for_lifecycle_record(
            pending_record,
            runtime,
            mismatch_field=mismatch_field,
        )

        assert (
            mismatched_result.identity.task_id != pending.task_id
        ) == (mismatch_field == "task_id")
        assert (
            mismatched_result.identity.attempt_id != pending.attempt_id
        ) == (mismatch_field == "attempt_id")
        assert (
            mismatched_result.identity.transport_id
            != pending.transport_id
        ) == (mismatch_field == "transport_id")

        rejected_record = _complete_runtime_dispatch(
            runner,
            pending_record,
            mismatched_result,
            current_parent_idx=pending.accepted_parent_idx,
            current_log_L_constraint=pending.accepted_log_L_constraint,
        )
        state_after = _runtime_state_snapshot(state)
        accepted_task_ids_after = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        task_records = _lifecycle_records_for_task(
            pending.task_id,
            lb,
            runner,
        )

    rejected = _normalise_lifecycle_record(rejected_record)

    assert [record.status for record in task_records] == [
        "pending",
        "mismatched_result_identity",
    ]
    assert rejected.status == "mismatched_result_identity"
    assert rejected.task_id == pending.task_id
    assert rejected.attempt_id == pending.attempt_id
    assert rejected.transport_id == pending.transport_id
    _assert_lifecycle_parent_metadata_preserved(pending, rejected)
    assert accepted_task_ids_after == accepted_task_ids_before
    _assert_runtime_state_snapshot_equal(state_before, state_after)


def test_stale_parent_target_completion_is_rejected_without_ledger_mutation():
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=True,
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        state = _run_small_runtime_runner(runner, max_samples=4)

        state_before = _runtime_state_snapshot(state)
        accepted_task_ids_before = (
            runner.runtime_acceptance_ledger.accepted_task_ids
        )
        pending_record = _prepare_runtime_dispatch(runner)
        pending = _normalise_lifecycle_record(pending_record)
        worker_result = _worker_result_with_runtime_stats_for_lifecycle_record(
            pending_record,
            runtime,
            dispatch_latency_seconds=1.75,
            payload_cache_latency_seconds=0.25,
            sampler_execution_latency_seconds=1.5,
        )
        changed_parent_idx = pending.accepted_parent_idx + 1
        changed_log_L_constraint = pending.accepted_log_L_constraint + 0.25

        stale_record = _complete_runtime_dispatch(
            runner,
            pending_record,
            worker_result,
            current_parent_idx=changed_parent_idx,
            current_log_L_constraint=changed_log_L_constraint,
        )
        state_after = _runtime_state_snapshot(state)

    stale = _normalise_lifecycle_record(stale_record)

    assert pending.requested_parent_idx == 0
    assert pending.effective_parent_idx == 0
    assert pending.accepted_parent_idx == 0
    assert pending.effective_log_L_constraint == -0.5
    assert pending.accepted_log_L_constraint == -0.5
    assert stale.status == "stale_parent_target"
    assert stale.task_id == pending.task_id
    assert stale.attempt_id == pending.attempt_id
    assert _runtime_latency_values(stale_record) == (1.75, 0.25, 1.5)
    _assert_lifecycle_parent_metadata_preserved(pending, stale)
    assert int(_record_field(
        stale_record,
        "current_parent_idx",
        "observed_parent_idx",
    )) == changed_parent_idx
    assert float(_record_field(
        stale_record,
        "current_effective_log_L_constraint",
        "current_strict_contour",
    )) == changed_log_L_constraint
    assert (
        runner.runtime_acceptance_ledger.accepted_task_ids
        == accepted_task_ids_before
    )
    _assert_runtime_state_snapshot_equal(state_before, state_after)


def test_terminal_late_completion_preserves_worker_latency_fields():
    LoadBalancerClient = _load_balancer_client()
    runtime = _runtime_module()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:1"])
        runner = lb.get_nested_sampler(
            model=make_toy_model(),
            collect_phantoms=False,
            target_num_live_points=2,
            max_samples=4,
            shell_size=1,
            termination_condition=TerminationCondition(max_samples=4),
            batch_size=None,
        )
        pending_record = _prepare_runtime_dispatch(runner)
        failed_record = _fail_runtime_dispatch(runner, pending_record)
        pending = _normalise_lifecycle_record(pending_record)
        failed_completion_record = _complete_runtime_dispatch(
            runner,
            pending_record,
            _worker_result_with_runtime_stats_for_lifecycle_record(
                pending_record,
                runtime,
                dispatch_latency_seconds=2.5,
                payload_cache_latency_seconds=0.5,
                sampler_execution_latency_seconds=2.0,
            ),
            current_parent_idx=pending.accepted_parent_idx,
            current_log_L_constraint=pending.accepted_log_L_constraint,
        )

    failed_completion = _normalise_lifecycle_record(failed_completion_record)
    failed = _normalise_lifecycle_record(failed_record)

    assert failed_completion.status == "failed"
    assert failed_completion.task_id == failed.task_id
    assert failed_completion.attempt_id == failed.attempt_id
    assert failed_completion.transport_id == failed.transport_id
    assert _runtime_latency_values(failed_completion_record) == (
        2.5,
        0.5,
        2.0,
    )
