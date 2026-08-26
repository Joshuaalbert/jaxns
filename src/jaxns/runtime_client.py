"""Trusted local client for the JAXNS constrained-sampling supervisor."""

from __future__ import annotations

import json
import pickle
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from jaxns.runtime_config import RuntimeConfig, load_runtime_config
from jaxns.runtime_protocol import (
    ACK,
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


class RuntimeUnavailableError(RuntimeError):
    """The configured local supervisor did not answer before the deadline."""


class SupervisorClient:
    """Connect one scientific session to a trusted local worker stack.

    The transport deliberately uses pickle only inside a same-user IPC
    directory. Pickle is code-executing input, so this client rejects non-IPC
    endpoints and must not be repurposed as a network protocol.
    """

    def __init__(
            self,
            endpoint: str,
            *,
            default_timeout_s: float,
            config_fingerprint: str,
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
        self._results: deque[tuple[dict[str, object], bytes]] = deque()

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
        # trusted-local restriction remains mandatory because loading this
        # payload can execute code.
        payload = cloudpickle.dumps(
            session,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        self._socket.send_multipart([
            encode_header(
                REGISTER,
                session_id=session_id,
                config_fingerprint=self._config_fingerprint,
            ),
            payload,
        ])
        deadline = time.monotonic() + self._default_timeout_s
        while True:
            header, result_payload = self._receive_until(deadline)
            command = header["command"]
            if command == RESULT:
                self._results.append((header, result_payload))
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
        header, payload = self._receive_until(
            time.monotonic() + self._default_timeout_s
        )
        del payload
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
        """Submit one retry-stable request without waiting for a barrier."""
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        batch_size = int(request.log_L_constraints.shape[0])
        self._socket.send_multipart([
            encode_header(
                TASK,
                session_id=session_id,
                task_id=task_id,
                batch_size=batch_size,
            ),
            payload,
        ])

    def receive(
            self,
            session_id: str,
            *,
            timeout_s: float | None = None,
    ) -> tuple[int, ConstrainedSampleBatch]:
        """Receive whichever task completes first for the registered session."""
        if self._results:
            header, payload = self._results.popleft()
        else:
            timeout = self._default_timeout_s if timeout_s is None else timeout_s
            header, payload = self._receive_until(time.monotonic() + timeout)
        command = header["command"]
        if command == ERROR:
            _raise_runtime_error(header)
        if command != RESULT:
            raise RuntimeError(f"Expected a worker result, received {command!r}.")
        if header.get("session_id") != session_id:
            raise RuntimeError("Supervisor returned a result for another session.")
        task_id = header.get("task_id")
        if type(task_id) is not int:
            raise RuntimeError("Worker result has no integer task ID.")
        try:
            batch = pickle.loads(payload)
        except Exception as exc:
            raise RuntimeError("Worker result payload could not be decoded.") from exc
        return task_id, batch

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
    ) -> dict[str, object]:
        self._socket.send_multipart([encode_header(command), b""])
        timeout = self._default_timeout_s if timeout_s is None else timeout_s
        header, payload = self._receive_until(time.monotonic() + timeout)
        del payload
        if header["command"] == ERROR:
            _raise_runtime_error(header)
        if header["command"] != command:
            raise RuntimeError(
                f"Expected {command!r}, received {header['command']!r}."
            )
        return header

    def _receive_until(
            self,
            deadline: float,
    ) -> tuple[dict[str, object], bytes]:
        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
        events = dict(self._poller.poll(remaining_ms))
        if self._socket not in events:
            raise RuntimeUnavailableError(
                "The local JAXNS supervisor did not answer before the timeout."
            )
        frames = self._socket.recv_multipart()
        if len(frames) != 2:
            raise RuntimeError("Supervisor returned an invalid frame count.")
        return decode_header(frames[0]), frames[1]


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


def _raise_runtime_error(header: dict[str, object]) -> None:
    message = header.get("error", "The distributed runtime reported an error.")
    remote_traceback = header.get("traceback")
    if type(remote_traceback) is str and remote_traceback:
        message = f"{message}\nRemote traceback:\n{remote_traceback}"
    raise RuntimeError(str(message))
