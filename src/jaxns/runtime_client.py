"""Local scientific client for the JAXNS worker-pool coordinator."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
import sys
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from jaxns.runtime_config import RuntimeConfig, load_runtime_config
from jaxns.runtime_protocol import (
    ACK,
    CAPACITY,
    ERROR,
    PING,
    REGISTER,
    REGISTERED,
    RELEASE,
    RELEASED,
    RESULT,
    SHUTDOWN,
    STATUS,
    TASK,
    decode_header,
    encode_header,
)

if TYPE_CHECKING:
    from jaxns.constrained_sampler import (
        ConstrainedSampleBatch,
        ConstrainedSampleRequest,
    )
    from jaxns.distributed_core import WorkerSession
    from jaxns.multi_ellipsoid_utils import SamplerData


class RuntimeUnavailableError(RuntimeError):
    """The configured local supervisor did not answer before the deadline."""


def _sampler_batch_group(sampler_data: SamplerData | None) -> str:
    """Fingerprint direction state while excluding observational counters.

    GMM fit bookkeeping and completed-chain counters travel in ``State`` for
    diagnostics and warm-refit policy, but workers do not read them when they
    draw a direction. Including those counters would make otherwise identical
    refills incompatible and silently collapse a worker's configured vmap.
    """
    execution_state = (
        None
        if sampler_data is None
        else (
            sampler_data.mixture,
            sampler_data.centres,
            sampler_data.radii,
            sampler_data.rotations,
            sampler_data.log_volumes,
            sampler_data.log_L_max,
            sampler_data.valid,
        )
    )
    return hashlib.sha256(pickle.dumps(
        execution_state,
        protocol=pickle.HIGHEST_PROTOCOL,
    )).hexdigest()


class SupervisorClient:
    """Connect one scientific session to its local coordinator.

    Model code enters the coordinator only through same-user IPC. The
    coordinator may forward it to explicitly authorized CurveZMQ workers, but
    this code-executing boundary is never exposed as unauthenticated TCP.
    """

    def __init__(
            self,
            endpoint: str,
            *,
            default_timeout_s: float,
            config_fingerprint: str,
            max_payload_bytes: int = 536_870_912,
    ):
        if not endpoint.startswith("ipc://"):
            raise ValueError("The local runtime accepts only ipc:// endpoints.")
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise ImportError(
                "Distributed execution requires `pip install jaxns[distributed]`."
            ) from exc

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(
            zmq.IDENTITY,
            f"client:{uuid4().hex}".encode(),
        )
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        self._default_timeout_s = default_timeout_s
        self._config_fingerprint = config_fingerprint
        self._max_payload_bytes = max_payload_bytes
        self._results: deque[
            tuple[dict[str, object], tuple[bytes, ...]]
        ] = deque()
        self._decoded_results: deque[
            tuple[int, ConstrainedSampleBatch]
        ] = deque()

    @classmethod
    def from_config(
            cls,
            path: str | Path,
            *,
            use_manifest: bool = False,
    ) -> SupervisorClient:
        config = load_runtime_config(path)
        endpoint = _manifest_endpoint(config) if use_manifest else config.endpoint
        return cls(
            endpoint,
            default_timeout_s=config.startup_timeout_s,
            config_fingerprint=config.fingerprint,
            max_payload_bytes=config.max_payload_bytes,
        )

    def close(self) -> None:
        self._poller.unregister(self._socket)
        self._socket.close(linger=0)
        self._context.term()

    def __enter__(self) -> SupervisorClient:  # noqa: PYI034
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def ping(self, timeout_s: float | None = None) -> dict[str, object]:
        return self._round_trip(PING, timeout_s=timeout_s)

    def status(self, timeout_s: float | None = None) -> dict[str, object]:
        return self._round_trip(STATUS, timeout_s=timeout_s)

    def capacity(
            self,
            session_id: str | None = None,
            timeout_s: float | None = None,
    ) -> int:
        """Return live lanes able to execute the selected session."""
        header = self._round_trip(
            CAPACITY,
            timeout_s=timeout_s,
            session_id=session_id,
        )
        lanes = header.get("lanes")
        if type(lanes) is not int or lanes < 0:
            raise RuntimeError("Coordinator returned invalid lane capacity.")
        return lanes

    def shutdown(self, timeout_s: float | None = None) -> dict[str, object]:
        return self._round_trip(SHUTDOWN, timeout_s=timeout_s)

    def register(
            self,
            session_id: str,
            session: WorkerSession,
    ) -> tuple[int, ...]:
        """Register immutable model data once and return worker capacities."""
        try:
            import cloudpickle
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise ImportError(
                "Distributed model registration requires "
                "`pip install jaxns[distributed]`."
            ) from exc
        # Model definitions commonly originate in notebooks, scripts, or
        # closures. Cloudpickle preserves that Python code boundary; the
        # Loading this payload can execute code. It enters through local IPC
        # and reaches only worker public keys authorized by the coordinator.
        payload = cloudpickle.dumps(
            session,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        self._check_payload(payload)
        self._socket.send_multipart([
            encode_header(
                REGISTER,
                session_id=session_id,
                config_fingerprint=self._config_fingerprint,
                capabilities=_scientific_capabilities(),
            ),
            payload,
        ])
        deadline = time.monotonic() + self._default_timeout_s
        while True:
            header, result_payloads = self._receive_until(deadline)
            command = header["command"]
            if command == RESULT:
                self._results.append((header, result_payloads))
                continue
            if command == ERROR:
                _raise_runtime_error(header)
            if command != REGISTERED:
                raise RuntimeError(
                    f"Expected worker registration, received {command!r}."
                )
            if header.get("session_id") != session_id:
                raise RuntimeError("Supervisor registered an unexpected session.")
            capacities = header.get("capacities")
            if not isinstance(capacities, list) or not all(
                type(value) is int and value > 0 for value in capacities
            ):
                raise RuntimeError("Supervisor returned invalid worker capacities.")
            return tuple(capacities)

    def release(self, session_id: str) -> None:
        """Release one drained session and its worker-side compiled programs."""
        self._socket.send_multipart([
            encode_header(RELEASE, session_id=session_id),
            b"",
        ])
        header, payloads = self._receive_until(
            time.monotonic() + self._default_timeout_s
        )
        del payloads
        if header["command"] == ERROR:
            _raise_runtime_error(header)
        if header["command"] != RELEASED:
            raise RuntimeError(
                f"Expected session release, received {header['command']!r}."
            )

    def submit(
            self,
            session_id: str,
            task_id: int,
            request: ConstrainedSampleRequest,
    ) -> None:
        """Submit one retry-stable logical lineage thread."""
        self.submit_many(session_id, ((task_id, request),))

    def submit_many(
            self,
            session_id: str,
            tasks: tuple[tuple[int, ConstrainedSampleRequest], ...],
    ) -> None:
        """Atomically queue one transport group of scalar scientific tasks."""
        if not tasks:
            raise ValueError("At least one distributed task is required.")
        task_ids = []
        batch_groups = []
        payloads = []
        for task_id, request in tasks:
            width = int(request.log_L_constraints.shape[0])
            if width != 1:
                raise ValueError(
                    "Distributed tasks are scalar logical threads; worker-local "
                    "batch_size controls vmap grouping."
                )
            task_ids.append(task_id)
            batch_groups.append(_sampler_batch_group(request.sampler_data))
            payloads.append(pickle.dumps(
                request,
                protocol=pickle.HIGHEST_PROTOCOL,
            ))
        self._check_payloads(payloads)
        self._socket.send_multipart([
            encode_header(
                TASK,
                session_id=session_id,
                task_ids=task_ids,
                batch_groups=batch_groups,
            ),
            *payloads,
        ])

    def receive(
            self,
            session_id: str,
            *,
            timeout_s: float | None = None,
    ) -> tuple[int, ConstrainedSampleBatch]:
        """Receive whichever task completes first for the registered session."""
        if not self._decoded_results:
            group = self.receive_group(session_id, timeout_s=timeout_s)
            self._decoded_results.extend(group)
        return self._decoded_results.popleft()

    def receive_group(
            self,
            session_id: str,
            *,
            timeout_s: float | None = None,
    ) -> tuple[tuple[int, ConstrainedSampleBatch], ...]:
        """Receive one complete worker assignment without a lane barrier leak."""
        if self._decoded_results:
            result = tuple(self._decoded_results)
            self._decoded_results.clear()
            return result
        if self._results:
            header, payloads = self._results.popleft()
        else:
            timeout = self._default_timeout_s if timeout_s is None else timeout_s
            header, payloads = self._receive_until(time.monotonic() + timeout)
        command = header["command"]
        if command == ERROR:
            _raise_runtime_error(header)
        if command != RESULT:
            raise RuntimeError(f"Expected a worker result, received {command!r}.")
        if header.get("session_id") != session_id:
            raise RuntimeError("Supervisor returned a result for another session.")
        task_ids = header.get("task_ids")
        if (
            not isinstance(task_ids, list)
            or not task_ids
            or not all(type(task_id) is int for task_id in task_ids)
        ):
            raise RuntimeError("Worker result has no integer task IDs.")
        if len(task_ids) != len(payloads):
            raise RuntimeError("Worker result IDs and payloads disagree.")
        results = []
        try:
            self._check_payloads(payloads)
            for task_id, payload in zip(task_ids, payloads, strict=True):
                results.append((task_id, pickle.loads(payload)))
        except Exception as exc:
            raise RuntimeError("Worker result payload could not be decoded.") from exc
        return tuple(results)

    def acknowledge(self, session_id: str, task_id: int) -> None:
        """Allow the supervisor to release one committed result payload."""
        self._socket.send_multipart([
            encode_header(ACK, session_id=session_id, task_id=task_id),
            b"",
        ])

    def _round_trip(
            self,
            command: str,
            *,
            timeout_s: float | None,
            **fields: object,
    ) -> dict[str, object]:
        self._socket.send_multipart([encode_header(command, **fields), b""])
        timeout = self._default_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout
        while True:
            header, payloads = self._receive_until(deadline)
            observed = header["command"]
            if observed == RESULT:
                # Capacity/status requests can race task completions on the
                # same DEALER socket. Preserve those results for receive().
                self._results.append((header, payloads))
                continue
            if observed == ERROR:
                _raise_runtime_error(header)
            if observed != command:
                raise RuntimeError(
                    f"Expected {command!r}, received {observed!r}."
                )
            return header

    def _receive_until(
            self,
            deadline: float,
    ) -> tuple[dict[str, object], tuple[bytes, ...]]:
        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
        events = dict(self._poller.poll(remaining_ms))
        if self._socket not in events:
            raise RuntimeUnavailableError(
                "The local JAXNS supervisor did not answer before the timeout."
            )
        frames = self._socket.recv_multipart()
        if len(frames) < 2:
            raise RuntimeError("Supervisor returned an invalid frame count.")
        return decode_header(frames[0]), tuple(frames[1:])

    def _check_payload(self, payload: bytes) -> None:
        self._check_payloads((payload,))

    def _check_payloads(
            self,
            payloads: tuple[bytes, ...] | list[bytes],
    ) -> None:
        size = sum(len(payload) for payload in payloads)
        if size > self._max_payload_bytes:
            raise ValueError(
                f"Runtime payload is {size} bytes; configured maximum "
                f"is {self._max_payload_bytes} bytes."
            )


def _manifest_endpoint(config: RuntimeConfig) -> str:
    """Find a running stack after its config path or runtime directory moved."""
    try:
        document = json.loads(config.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return config.endpoint
    if document.get("stack_id") != config.stack_id:
        return config.endpoint
    endpoint = document.get("endpoint")
    if type(endpoint) is str and endpoint.startswith("ipc://"):
        return endpoint
    return config.endpoint


def _scientific_capabilities() -> dict[str, object]:
    """Describe execution semantics without constraining device platform."""
    import jax
    import jax.numpy as jnp

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "jaxns": importlib.metadata.version("jaxns"),
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "x64": bool(jax.config.x64_enabled),
        "measure_dtype": str(jnp.dtype(jnp.float64)),
    }


def _raise_runtime_error(header: dict[str, object]) -> None:
    message = header.get("error", "The distributed runtime reported an error.")
    remote_traceback = header.get("traceback")
    if type(remote_traceback) is str and remote_traceback:
        message = f"{message}\nRemote traceback:\n{remote_traceback}"
    raise RuntimeError(str(message))
