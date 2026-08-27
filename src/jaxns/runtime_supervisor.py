"""JAX-free coordinator for an asynchronous multi-node worker pool."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import fcntl
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import zmq

from jaxns.runtime_config import (
    RuntimeConfig,
    WorkerConfig,
    load_runtime_config,
    worker_name,
)
from jaxns.runtime_protocol import (
    ACK,
    CAPACITY,
    DRAIN,
    DRAINED,
    ERROR,
    EVALUATE,
    HEARTBEAT,
    HEARTBEAT_ACK,
    LEASED,
    NODE_HEARTBEAT,
    NODE_HEARTBEAT_ACK,
    NODE_RESTARTED,
    NODE_STATUS,
    NODE_STOPPED,
    PING,
    PROTOCOL_VERSION,
    READY,
    REGISTER,
    REGISTERED,
    RELEASE,
    RELEASED,
    RESULT,
    SAMPLE,
    SHUTDOWN,
    STATUS,
    STOP,
    STOPPED,
    TASK,
    decode_header,
    encode_header,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class RestartRequest:
    """Idempotent request to replace one dropped remote worker instance."""

    request_id: str
    worker_name: str
    instance_id: str
    lease_generation: int
    reason: str


class WorkerRecord:
    """Observed process, lease, and protocol state for one worker."""

    __slots__ = (
        "assignment_id",
        "capabilities",
        "compile_s",
        "config",
        "device",
        "draining",
        "dropped",
        "elapsed_s",
        "exit_code",
        "identity",
        "instance_id",
        "last_heartbeat_s",
        "lease_generation",
        "lease_id",
        "log_file",
        "node_id",
        "peak_rss_kib",
        "platform",
        "process",
        "ready",
        "registered",
        "reported_process_id",
        "restart_at_s",
        "restart_count",
        "router",
        "started_s",
        "task",
        "worker_id",
    )

    def __init__(
            self,
            config: WorkerConfig,
            process: subprocess.Popen | None,
            log_file: object | None,
            identity: bytes,
            *,
            node_id: str = "",
            instance_id: str = "",
            router=None,
    ) -> None:
        self.config = config
        self.capabilities: dict[str, object] = {}
        self.process = process
        self.log_file = log_file
        self.identity = identity
        self.node_id = node_id
        name = worker_name(config)
        self.worker_id = f"{node_id}/{name}" if node_id else name
        self.instance_id = instance_id
        self.router = router
        self.lease_id = ""
        self.lease_generation = 0
        self.ready = False
        self.draining = False
        self.dropped = False
        self.registered: set[str] = set()
        self.reported_process_id = 0
        self.restart_at_s = 0.0
        self.restart_count = 0
        self.task: tuple[tuple[str, int], ...] | None = None
        self.assignment_id = ""
        self.started_s = 0.0
        self.last_heartbeat_s = 0.0
        self.platform = ""
        self.device = ""
        self.peak_rss_kib = 0
        self.compile_s = 0.0
        self.elapsed_s = 0.0
        self.exit_code: int | None = None


class TaskRecord:
    """Retry-stable worker operation retained until client acknowledgement."""

    __slots__ = (
        "batch_group",
        "completion_group",
        "fingerprint",
        "operation",
        "payload",
        "queued_s",
        "result_header",
        "result_payload",
        "state",
        "task_id",
        "worker",
    )

    def __init__(
            self,
            task_id: int,
            batch_group: str,
            operation: str,
            fingerprint: str,
            payload: bytes,
    ) -> None:
        self.task_id = task_id
        self.batch_group = batch_group
        self.operation = operation
        self.completion_group = f"task:{task_id}"
        self.fingerprint = fingerprint
        self.payload = payload
        self.queued_s = time.monotonic()
        self.state = "queued"
        self.worker: bytes | None = None
        self.result_header = b""
        self.result_payload = b""


class SessionRecord:
    """One scientific client registration and its fair task queue."""

    __slots__ = (
        "capabilities",
        "client",
        "fingerprint",
        "payload",
        "queue",
        "registered",
        "registration_notified",
        "session_id",
        "tasks",
    )

    def __init__(
            self,
            session_id: str,
            fingerprint: str,
            payload: bytes,
            client: bytes,
            capabilities: dict[str, object] | None = None,
    ) -> None:
        self.session_id = session_id
        self.fingerprint = fingerprint
        self.payload = payload
        self.client = client
        self.capabilities = {} if capabilities is None else capabilities
        self.registered: set[bytes] = set()
        self.registration_notified = False
        self.tasks: dict[int, TaskRecord] = {}
        self.queue: collections.deque[int] = collections.deque()


class Supervisor:
    """Route opaque scalar operations across local and remote workers."""

    def __init__(self, config: RuntimeConfig):
        if config.role != "coordinator":
            raise ValueError("runtime_supervisor requires a coordinator config.")
        self.config = config
        self.context = zmq.Context()
        self.socket = self._router()
        self.network_socket = self._router()
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.poller.register(self.network_socket, zmq.POLLIN)
        self.workers: dict[bytes, WorkerRecord] = {}
        self.worker_ids: dict[str, bytes] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self.session_order: collections.deque[str] = collections.deque()
        self.drains: dict[str, tuple[object, bytes]] = {}
        self.restart_requests: dict[str, dict[str, RestartRequest]] = {}
        self.stopping = False
        self.lock_file = None

    def _router(self):
        router = self.context.socket(zmq.ROUTER)
        router.setsockopt(zmq.LINGER, 0)
        return router

    def run(self) -> None:
        self._prepare_runtime()
        self._acquire_ownership()
        self.socket.bind(self.config.endpoint)
        network = self.config.network
        if network.port is None:  # pragma: no cover - validated by role
            raise ValueError("A coordinator requires a TCP port.")
        self.network_socket.bind(f"tcp://0.0.0.0:{network.port}")
        self._write_manifest()
        self._start_workers()
        logger.info(
            "Coordinator %s listening locally at %s and on TCP port %d",
            self.config.stack_id,
            self.config.endpoint,
            network.port,
        )
        try:
            while not self.stopping:
                # A short poll bound makes batch_wait_s a real bound for a
                # partially populated worker group rather than adding a hidden
                # 100 ms event-loop delay.
                poll_ms = max(1, min(100, int(self.config.batch_wait_s * 1000) + 1))
                events = dict(self.poller.poll(poll_ms))
                if self.socket in events:
                    self._receive(self.socket)
                if self.network_socket in events:
                    self._receive(self.network_socket)
                self._reap_workers()
                self._restart_workers()
                self._expire_workers()
                self._expire_tasks()
                self._dispatch()
                self._finish_drains()
        finally:
            self._shutdown_workers()
            self._cleanup()

    def request_stop(self, signum=None, frame=None) -> None:
        del signum, frame
        self.stopping = True

    def _prepare_runtime(self) -> None:
        scientific_endpoint = Path(
            self.config.endpoint.removeprefix("ipc://")
        ).parent
        for directory in (
            self.config.runtime_dir,
            self.config.log_dir,
            self.config.lock.parent,
            scientific_endpoint,
        ):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)

    def _acquire_ownership(self) -> None:
        self.lock_file = self.config.lock.open("a+b")
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Stack {self.config.stack_id!r} already has an owner on node "
                f"{self.config.node_id!r}."
            ) from exc

    def _write_manifest(self) -> None:
        document = {
            "schema_version": 2,
            "protocol_version": PROTOCOL_VERSION,
            "stack_id": self.config.stack_id,
            "node_id": self.config.node_id,
            "role": self.config.role,
            "config_fingerprint": self.config.fingerprint,
            "process_id": os.getpid(),
            "endpoint": self.config.endpoint,
            "port": self.config.network.port,
            "transports": ["ipc", "tcp"],
            "started_ns": time.time_ns(),
        }
        temporary = self.config.manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.config.manifest)

    def _start_workers(self) -> None:
        for worker in self.config.workers:
            instance_id = uuid4().hex
            name = worker_name(worker)
            identity = (
                f"worker:{self.config.node_id}/{name}:{instance_id}"
            ).encode()
            process, log_file = _start_worker_process(
                self.config,
                worker,
                self.config.endpoint,
                instance_id,
            )
            record = WorkerRecord(
                worker,
                process,
                log_file,
                identity,
                node_id=self.config.node_id,
                instance_id=instance_id,
                router=self.socket,
            )
            self.workers[identity] = record
            self.worker_ids[record.worker_id] = identity

    def _receive(self, router) -> None:
        frames = router.recv_multipart()
        if len(frames) < 3:
            logger.warning("Dropping message with %d frames", len(frames))
            return
        identity, raw_header, *payloads = frames
        try:
            header = decode_header(raw_header)
            command = header["command"]
            if command == HEARTBEAT:
                self._receive_heartbeat(router, identity, header)
            elif identity.startswith(b"worker:"):
                self._receive_worker(router, identity, command, header, payloads)
            elif identity.startswith(b"node:"):
                self._receive_node(router, identity, command, header)
            elif router is self.socket:
                self._receive_client(identity, command, header, payloads)
            else:
                raise ValueError("Scientific clients must use local IPC.")
        except Exception as exc:
            logger.exception("Protocol handling failed")
            self._send(
                identity,
                encode_header(ERROR, error=f"{type(exc).__name__}: {exc}"),
                router=router,
            )

    def _receive_client(
            self,
            identity: bytes,
            command: str,
            header: dict[str, object],
            payloads: list[bytes],
    ) -> None:
        if command == PING:
            self._send(identity, encode_header(
                PING,
                stack_id=self.config.stack_id,
                node_id=self.config.node_id,
                config_fingerprint=self.config.fingerprint,
                process_id=os.getpid(),
            ))
        elif command == STATUS:
            self._send(identity, encode_header(STATUS, **self._status()))
        elif command == CAPACITY:
            session_id = header.get("session_id")
            if session_id is not None and type(session_id) is not str:
                raise ValueError("Capacity session ID must be a string or null.")
            ready = [
                worker
                for worker in self.workers.values()
                if worker.ready and not worker.draining
                and (
                    session_id is None
                    or session_id in worker.registered
                )
            ]
            self._send(identity, encode_header(
                CAPACITY,
                lanes=sum(worker.config.batch_size for worker in ready),
                workers=len(ready),
            ))
        elif command == SHUTDOWN:
            self._send(identity, encode_header(SHUTDOWN, accepted=True))
            self.stopping = True
        elif command == REGISTER:
            self._register_client(
                identity,
                header,
                _single_payload(payloads, REGISTER),
            )
        elif command == RELEASE:
            self._release_client(identity, header)
        elif command == TASK:
            self._queue_tasks(identity, header, payloads)
        elif command == ACK:
            self._acknowledge(identity, header)
        else:
            raise ValueError(f"Unsupported client command {command!r}.")

    def _receive_worker(
            self,
            router,
            identity: bytes,
            command: str,
            header: dict[str, object],
            payloads: list[bytes],
    ) -> None:
        if command == READY:
            self._lease_worker(router, identity, header)
            return
        worker = self.workers.get(identity)
        if worker is None or not self._lease_matches(worker, header):
            raise ValueError("Unknown or stale worker lease.")
        if command == REGISTERED:
            session_id = _string_field(header, "session_id")
            session = self.sessions.get(session_id)
            if session is not None:
                worker.registered.add(session_id)
                session.registered.add(identity)
                self._complete_registration(session)
        elif command == RESULT:
            self._complete_tasks(worker, header, payloads)
        elif command == ERROR:
            self._worker_error(worker, header)
        elif command == STOPPED:
            self._drop_worker(worker, "worker stopped")
        else:
            raise ValueError(f"Unsupported worker command {command!r}.")

    def _lease_worker(self, router, identity: bytes, header: dict[str, object]) -> None:
        had_capacity = any(
            candidate.ready and not candidate.draining
            for candidate in self.workers.values()
        )
        worker_id = _string_field(header, "worker_id")
        node_id = _string_field(header, "node_id")
        configured_platform = _string_field(header, "configured_platform")
        configured_device = _string_field(header, "configured_device")
        instance_id = _string_field(header, "instance_id")
        batch_size = _integer_field(header, "batch_size")
        config = WorkerConfig(
            platform=configured_platform,
            device=configured_device,
            batch_size=batch_size,
        )
        if worker_id != f"{node_id}/{worker_name(config)}":
            raise ValueError("Worker identity fields are inconsistent.")
        existing_identity = self.worker_ids.get(worker_id)
        previous_generation = 0
        worker = self.workers.get(identity)
        if worker is None:
            if existing_identity is not None:
                existing = self.workers[existing_identity]
                previous_generation = existing.lease_generation
                # A node can disappear without completing its shutdown
                # handshake and then be restarted by an operator much later.
                # The new process is authoritative for this logical device;
                # fencing the old lease also requeues every unfinished task.
                if existing.process is not None and existing.process.poll() is None:
                    existing.process.terminate()
                self._drop_worker(
                    existing,
                    "superseded by a fresh worker instance",
                    request_restart=False,
                    log_starvation=False,
                )
                del self.workers[existing_identity]
            worker = WorkerRecord(
                config,
                None,
                None,
                identity,
                node_id=node_id,
                instance_id=instance_id,
                router=router,
            )
            self.workers[identity] = worker
            self.worker_ids[worker_id] = identity
        elif worker.instance_id != instance_id:
            raise ValueError("Configured worker started with an unexpected instance.")
        if worker.config.batch_size != batch_size:
            raise ValueError("Worker reported a different batch capacity.")
        worker.router = router
        worker.platform = _string_field(header, "platform")
        worker.device = _string_field(header, "device")
        worker.capabilities = _dictionary_field(header, "capabilities")
        worker.reported_process_id = _integer_field(header, "process_id")
        worker.lease_id = uuid4().hex
        worker.lease_generation = max(
            worker.lease_generation,
            previous_generation,
        ) + 1
        worker.ready = True
        worker.draining = False
        worker.dropped = False
        worker.restart_count = 0
        worker.last_heartbeat_s = time.monotonic()
        # A fresh lease proves that this logical device has already been
        # replaced. Retire any at-least-once request targeting an older
        # instance, even if its explicit acknowledgement was lost.
        self._clear_restart_requests(
            node_id,
            worker_name(config),
            current_instance_id=instance_id,
        )
        self._send(
            identity,
            encode_header(
                LEASED,
                lease_id=worker.lease_id,
                lease_generation=worker.lease_generation,
                heartbeat_interval_s=self.config.heartbeat_interval_s,
                missed_heartbeats=self.config.missed_heartbeats,
            ),
            router=router,
        )
        for session in self.sessions.values():
            self._send_registration(worker, session)
        logger.info("Leased worker %s with capacity %d", worker_id, batch_size)
        if not had_capacity:
            logger.info("Distributed worker capacity is available again")

    def _receive_heartbeat(
            self,
            router,
            identity: bytes,
            header: dict[str, object],
    ) -> None:
        worker_id = _string_field(header, "worker_id")
        worker_identity = self.worker_ids.get(worker_id)
        worker = self.workers.get(worker_identity) if worker_identity else None
        accepted = (
            worker is not None
            and worker.instance_id == header.get("instance_id")
            and worker.lease_id == header.get("lease_id")
            and worker.lease_generation == header.get("lease_generation")
            and worker.ready
            and not worker.dropped
        )
        if accepted:
            worker.last_heartbeat_s = time.monotonic()
        self._send(
            identity,
            encode_header(HEARTBEAT_ACK, accepted=accepted),
            router=router,
        )

    def _receive_node(
            self,
            router,
            identity: bytes,
            command: str,
            header: dict[str, object],
    ) -> None:
        node_id = _string_field(header, "node_id")
        if command == DRAIN:
            for worker in self.workers.values():
                if worker.node_id == node_id and worker.ready:
                    worker.draining = True
            self.drains[node_id] = (router, identity)
            self._finish_drains()
        elif command == NODE_STATUS:
            self._send(
                identity,
                encode_header(
                    NODE_STATUS,
                    node_id=node_id,
                    workers=[
                        self._worker_status(worker)
                        for worker in self.workers.values()
                        if worker.node_id == node_id
                    ],
                ),
                router=router,
            )
        elif command == NODE_HEARTBEAT:
            # The node initiates this bidirectional control connection, so no
            # inbound worker-node port is required. Restart requests remain in
            # coordinator memory until acknowledged or made obsolete by a
            # fresh lease, and are therefore replayed after a partition.
            requests = self.restart_requests.get(node_id, {})
            self._send(
                identity,
                encode_header(
                    NODE_HEARTBEAT_ACK,
                    node_id=node_id,
                    restart_requests=[
                        {
                            "request_id": request.request_id,
                            "worker_name": request.worker_name,
                            "instance_id": request.instance_id,
                            "lease_generation": request.lease_generation,
                            "reason": request.reason,
                        }
                        for request in requests.values()
                    ],
                ),
                router=router,
            )
        elif command == NODE_RESTARTED:
            request_ids = _string_list(header, "request_ids")
            requests = self.restart_requests.get(node_id, {})
            for request_id in request_ids:
                requests.pop(request_id, None)
            if not requests:
                self.restart_requests.pop(node_id, None)
            self._send(
                identity,
                encode_header(
                    ACK,
                    node_id=node_id,
                    request_ids=request_ids,
                ),
                router=router,
            )
        elif command == NODE_STOPPED:
            for worker in tuple(self.workers.values()):
                if worker.node_id == node_id and worker.ready:
                    self._drop_worker(
                        worker,
                        "node stopped",
                        request_restart=False,
                    )
            self._send(
                identity,
                encode_header(ACK, node_id=node_id),
                router=router,
            )
        else:
            raise ValueError(f"Unsupported node command {command!r}.")

    def _register_client(
            self,
            identity: bytes,
            header: dict[str, object],
            payload: bytes,
    ) -> None:
        session_id = _string_field(header, "session_id")
        fingerprint = hashlib.sha256(payload).hexdigest()
        capabilities = _dictionary_field(header, "capabilities")
        session = self.sessions.get(session_id)
        if session is None:
            session = SessionRecord(
                session_id,
                fingerprint,
                payload,
                identity,
                capabilities=capabilities,
            )
            self.sessions[session_id] = session
            self.session_order.append(session_id)
        elif session.fingerprint != fingerprint:
            raise ValueError(f"Session {session_id!r} changed its model data.")
        elif session.capabilities != capabilities:
            raise ValueError(
                f"Session {session_id!r} changed its execution capabilities."
            )
        else:
            session.client = identity
            session.registration_notified = False
        compatible = [
            worker
            for worker in self.workers.values()
            if worker.ready
            if self._compatible(worker, session)
        ]
        # No compatible worker is a valid dynamic-pool state. The session
        # remains registered indefinitely so a repaired or newly added node
        # can complete the handshake even days into the scientific run.
        for worker in compatible:
            if session_id not in worker.registered:
                self._send_registration(worker, session)
        self._complete_registration(session)
        self._replay_completed(
            session,
            tuple(
                task
                for task in session.tasks.values()
                if task.state == "completed"
            ),
        )

    def _release_client(self, identity: bytes, header: dict[str, object]) -> None:
        session_id = _string_field(header, "session_id")
        session = self.sessions.get(session_id)
        if session is None:
            self._send(identity, encode_header(RELEASED, session_id=session_id))
            return
        if session.client != identity:
            raise ValueError(f"Client does not own session {session_id!r}.")
        if session.tasks:
            raise RuntimeError(f"Session {session_id!r} has unacknowledged tasks.")
        for worker in self.workers.values():
            worker.registered.discard(session_id)
            if worker.ready:
                self._send(
                    worker.identity,
                    encode_header(RELEASE, session_id=session_id),
                    router=worker.router,
                )
        self.session_order.remove(session_id)
        del self.sessions[session_id]
        self._send(identity, encode_header(RELEASED, session_id=session_id))

    def _send_registration(self, worker: WorkerRecord, session: SessionRecord) -> None:
        if not self._compatible(worker, session):
            return
        self._send(
            worker.identity,
            encode_header(
                REGISTER,
                session_id=session.session_id,
                fingerprint=session.fingerprint,
            ),
            session.payload,
            router=worker.router,
        )

    def _complete_registration(self, session: SessionRecord) -> None:
        if session.registration_notified:
            return
        registered = [
            worker
            for worker in self.workers.values()
            if worker.ready and session.session_id in worker.registered
        ]
        if not registered:
            return
        self._send(
            session.client,
            encode_header(
                REGISTERED,
                session_id=session.session_id,
                capacities=[worker.config.batch_size for worker in registered],
                workers=[self._worker_status(worker) for worker in registered],
            ),
        )
        session.registration_notified = True

    def _queue_tasks(
            self,
            identity: bytes,
            header: dict[str, object],
            payloads: list[bytes],
    ) -> None:
        session_id = _string_field(header, "session_id")
        task_ids = _integer_list(header, "task_ids")
        batch_groups = _string_list(header, "batch_groups")
        operations = _string_list(header, "operations")
        if (
            len(task_ids) != len(batch_groups)
            or len(task_ids) != len(operations)
            or len(task_ids) != len(payloads)
        ):
            raise ValueError(
                "Task IDs, batch groups, operations, and payloads disagree."
            )
        if any(operation not in (SAMPLE, EVALUATE) for operation in operations):
            raise ValueError("Distributed task operation is unsupported.")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Task IDs within one atomic submission must be unique.")
        session = self._session_owned(identity, session_id)
        prepared = []
        for task_id, batch_group, operation, payload in zip(
                task_ids,
                batch_groups,
                operations,
                payloads,
                strict=True,
        ):
            fingerprint = hashlib.sha256(payload).hexdigest()
            existing = session.tasks.get(task_id)
            if existing is not None and existing.fingerprint != fingerprint:
                raise ValueError(f"Task {task_id} changed its payload.")
            if existing is not None and (
                existing.batch_group != batch_group
                or existing.operation != operation
            ):
                raise ValueError(f"Task {task_id} changed its execution group.")
            prepared.append((
                task_id,
                batch_group,
                operation,
                fingerprint,
                payload,
                existing,
            ))

        # Validate the whole transport batch before mutating the retry table.
        # A malformed sibling cannot leave a partially queued reservation.
        completed = []
        for (
            task_id,
            batch_group,
            operation,
            fingerprint,
            payload,
            existing,
        ) in prepared:
            if existing is not None:
                if existing.state == "completed":
                    completed.append(existing)
                elif existing.state == "failed":
                    self._send(
                        identity,
                        existing.result_header,
                        existing.result_payload,
                    )
                continue
            task = TaskRecord(
                task_id,
                batch_group,
                operation,
                fingerprint,
                payload,
            )
            session.tasks[task_id] = task
            session.queue.append(task_id)
        self._replay_completed(session, tuple(completed))

    def _replay_completed(
            self,
            session: SessionRecord,
            tasks: tuple[TaskRecord, ...],
    ) -> None:
        """Replay completed siblings as one coordinator-to-client message."""
        if not tasks:
            return
        grouped = {}
        for task in sorted(tasks, key=lambda value: value.task_id):
            grouped.setdefault(task.completion_group, []).append(task)
        for group in grouped.values():
            self._send(
                session.client,
                encode_header(
                    RESULT,
                    session_id=session.session_id,
                    task_ids=[task.task_id for task in group],
                    replayed=True,
                ),
                *[task.result_payload for task in group],
            )

    def _acknowledge(self, identity: bytes, header: dict[str, object]) -> None:
        session = self._session_owned(
            identity,
            _string_field(header, "session_id"),
        )
        task_id = _integer_field(header, "task_id")
        task = session.tasks.get(task_id)
        if task is not None and task.state == "completed":
            del session.tasks[task_id]

    def _dispatch(self) -> None:
        for worker in self.workers.values():
            if (
                not worker.ready
                or worker.draining
                or worker.task is not None
            ):
                continue
            selected = self._next_tasks(worker)
            if selected is None:
                continue
            session, tasks = selected
            assignment_id = uuid4().hex
            worker.task = tuple(
                (session.session_id, task.task_id) for task in tasks
            )
            worker.assignment_id = assignment_id
            worker.started_s = time.monotonic()
            for task in tasks:
                task.state = "running"
                task.worker = worker.identity
            self._send(
                worker.identity,
                encode_header(
                    TASK,
                    session_id=session.session_id,
                    task_ids=[task.task_id for task in tasks],
                    operation=tasks[0].operation,
                    assignment_id=assignment_id,
                    lease_id=worker.lease_id,
                ),
                *[task.payload for task in tasks],
                router=worker.router,
            )

    def _next_tasks(
            self,
            worker: WorkerRecord,
    ) -> tuple[SessionRecord, tuple[TaskRecord, ...]] | None:
        now = time.monotonic()
        for _ in range(len(self.session_order)):
            session_id = self.session_order[0]
            self.session_order.rotate(-1)
            session = self.sessions[session_id]
            if session_id not in worker.registered:
                continue
            queued = [
                session.tasks[task_id]
                for task_id in tuple(session.queue)
                if task_id in session.tasks
                and session.tasks[task_id].state == "queued"
            ]
            if not queued:
                continue
            group = queued[0].batch_group
            operation = queued[0].operation
            compatible = [
                task
                for task in queued
                if task.batch_group == group and task.operation == operation
            ]
            width = min(worker.config.batch_size, len(compatible))
            if (
                width < worker.config.batch_size
                and now - compatible[0].queued_s < self.config.batch_wait_s
            ):
                continue
            selected = tuple(compatible[:width])
            selected_ids = {task.task_id for task in selected}
            session.queue = collections.deque(
                task_id for task_id in session.queue if task_id not in selected_ids
            )
            return session, selected
        return None

    def _complete_tasks(
            self,
            worker: WorkerRecord,
            header: dict[str, object],
            payloads: list[bytes],
    ) -> None:
        task_ids = _integer_list(header, "task_ids")
        assignment_id = _string_field(header, "assignment_id")
        expected = worker.task
        if (
            expected is None
            or assignment_id != worker.assignment_id
            or [task_id for _, task_id in expected] != task_ids
        ):
            logger.warning("Ignoring late assignment %s from %s", assignment_id, worker.worker_id)
            return
        if len(payloads) != len(task_ids):
            raise ValueError("Worker result payload count does not match task IDs.")
        session_id = expected[0][0]
        session = self.sessions[session_id]
        for task_id, payload in zip(task_ids, payloads, strict=True):
            task = session.tasks[task_id]
            task.state = "completed"
            task.completion_group = assignment_id
            task.worker = None
            task.result_header = encode_header(
                RESULT,
                session_id=session_id,
                task_ids=[task_id],
                worker_id=worker.worker_id,
                elapsed_s=header.get("elapsed_s", 0.0),
                compile_s=header.get("compile_s", 0.0),
                peak_rss_kib=header.get("peak_rss_kib", 0),
            )
            task.result_payload = payload
        # A worker's lanes already synchronize inside one compiled vmap. Send
        # that assignment to the scientific process atomically so result byte
        # size cannot change which subset is committed and refilled together.
        self._send(
            session.client,
            encode_header(
                RESULT,
                session_id=session_id,
                task_ids=task_ids,
                worker_id=worker.worker_id,
                elapsed_s=header.get("elapsed_s", 0.0),
                compile_s=header.get("compile_s", 0.0),
                peak_rss_kib=header.get("peak_rss_kib", 0),
            ),
            *payloads,
        )
        worker.task = None
        worker.assignment_id = ""
        worker.started_s = 0.0
        worker.peak_rss_kib = int(header.get("peak_rss_kib", 0))
        worker.compile_s += float(header.get("compile_s", 0.0))
        worker.elapsed_s += float(header.get("elapsed_s", 0.0))

    def _worker_error(self, worker: WorkerRecord, header: dict[str, object]) -> None:
        task_ids = header.get("task_ids")
        session_id = header.get("session_id")
        if type(session_id) is str and type(task_ids) is list:
            session = self.sessions.get(session_id)
            if session is not None:
                for task_id in task_ids:
                    if type(task_id) is not int:
                        continue
                    response = encode_header(
                        ERROR,
                        session_id=session_id,
                        task_id=task_id,
                        error=header.get("error", "worker task failed"),
                        traceback=header.get("traceback", ""),
                    )
                    task = session.tasks.get(task_id)
                    if task is not None:
                        task.state = "failed"
                        task.result_header = response
                    self._send(session.client, response)
            worker.task = None
            worker.assignment_id = ""
            worker.started_s = 0.0
            return
        if type(session_id) is str:
            session = self.sessions.get(session_id)
            if session is not None:
                self._send(session.client, encode_header(
                    ERROR,
                    session_id=session_id,
                    error=header.get("error", "worker registration failed"),
                    traceback=header.get("traceback", ""),
                ))

    def _reap_workers(self) -> None:
        for worker in self.workers.values():
            if worker.process is None:
                continue
            exit_code = worker.process.poll()
            if exit_code is None or worker.exit_code is not None:
                continue
            worker.exit_code = exit_code
            if worker.ready:
                self._drop_worker(worker, f"process exited with code {exit_code}")
            if worker.log_file is not None:
                worker.log_file.close()
                worker.log_file = None
            worker.restart_count += 1
            delay_s = min(30.0, 0.25 * 2 ** min(worker.restart_count, 7))
            worker.restart_at_s = time.monotonic() + delay_s

    def _restart_workers(self) -> None:
        """Replace configured local processes under a fresh worker lease."""
        now = time.monotonic()
        for identity, worker in tuple(self.workers.items()):
            if (
                worker.process is None
                or worker.exit_code is None
                or now < worker.restart_at_s
            ):
                continue
            instance_id = uuid4().hex
            name = worker_name(worker.config)
            next_identity = (
                f"worker:{self.config.node_id}/{name}:{instance_id}"
            ).encode()
            process, log_file = _start_worker_process(
                self.config,
                worker.config,
                self.config.endpoint,
                instance_id,
            )
            replacement = WorkerRecord(
                worker.config,
                process,
                log_file,
                next_identity,
                node_id=self.config.node_id,
                instance_id=instance_id,
                router=self.socket,
            )
            replacement.lease_generation = worker.lease_generation
            replacement.restart_count = worker.restart_count
            del self.workers[identity]
            self.workers[next_identity] = replacement
            self.worker_ids[replacement.worker_id] = next_identity
            logger.info("Restarted worker %s as a fresh instance", name)

    def _expire_workers(self) -> None:
        deadline = self.config.heartbeat_interval_s * self.config.missed_heartbeats
        now = time.monotonic()
        for worker in self.workers.values():
            if worker.ready and now - worker.last_heartbeat_s > deadline:
                if worker.process is not None and worker.process.poll() is None:
                    worker.process.terminate()
                self._drop_worker(worker, "missed heartbeat lease")

    def _expire_tasks(self) -> None:
        now = time.monotonic()
        for worker in self.workers.values():
            if worker.task is None or now - worker.started_s <= self.config.task_timeout_s:
                continue
            if worker.process is not None and worker.process.poll() is None:
                worker.process.terminate()
            self._drop_worker(worker, "task timeout")

    def _drop_worker(
            self,
            worker: WorkerRecord,
            reason: str,
            *,
            request_restart: bool = True,
            log_starvation: bool = True,
    ) -> None:
        if worker.dropped:
            return
        logger.warning("Dropping worker %s: %s", worker.worker_id, reason)
        self._requeue(worker)
        worker.ready = False
        worker.dropped = True
        worker.draining = False
        worker.registered.clear()
        for session in self.sessions.values():
            session.registered.discard(worker.identity)
        if request_restart and worker.process is None and worker.node_id:
            self._queue_restart_request(worker, reason)
        if log_starvation and not any(
            candidate.ready and not candidate.draining
            for candidate in self.workers.values()
        ):
            logger.warning(
                "Distributed worker pool has no live capacity; queued work "
                "will wait for a worker to recover or join"
            )

    def _queue_restart_request(
            self,
            worker: WorkerRecord,
            reason: str,
    ) -> None:
        """Retain one restart request until its remote node acknowledges it."""
        requests = self.restart_requests.setdefault(worker.node_id, {})
        for request in requests.values():
            if (
                request.worker_name == worker_name(worker.config)
                and request.instance_id == worker.instance_id
            ):
                return
        request = RestartRequest(
            request_id=uuid4().hex,
            worker_name=worker_name(worker.config),
            instance_id=worker.instance_id,
            lease_generation=worker.lease_generation,
            reason=reason,
        )
        requests[request.request_id] = request
        logger.info(
            "Queued restart request for remote worker %s instance %s",
            worker.worker_id,
            worker.instance_id,
        )

    def _clear_restart_requests(
            self,
            node_id: str,
            name: str,
            *,
            current_instance_id: str,
    ) -> None:
        """Retire requests made obsolete by a fresh worker lease."""
        requests = self.restart_requests.get(node_id)
        if requests is None:
            return
        obsolete = [
            request_id
            for request_id, request in requests.items()
            if request.worker_name == name
            and request.instance_id != current_instance_id
        ]
        for request_id in obsolete:
            del requests[request_id]
        if not requests:
            del self.restart_requests[node_id]

    def _requeue(self, worker: WorkerRecord) -> None:
        if worker.task is None:
            return
        for session_id, task_id in reversed(worker.task):
            session = self.sessions.get(session_id)
            if session is None:
                continue
            task = session.tasks.get(task_id)
            if task is None:
                continue
            task.state = "queued"
            task.worker = None
            task.queued_s = time.monotonic()
            session.queue.appendleft(task_id)
        worker.task = None
        worker.assignment_id = ""
        worker.started_s = 0.0

    def _finish_drains(self) -> None:
        for node_id, (router, identity) in tuple(self.drains.items()):
            active = any(
                worker.node_id == node_id and worker.task is not None
                for worker in self.workers.values()
            )
            if active:
                continue
            self._send(
                identity,
                encode_header(DRAINED, node_id=node_id),
                router=router,
            )
            del self.drains[node_id]

    def _session_owned(self, identity: bytes, session_id: str) -> SessionRecord:
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session {session_id!r}.")
        if session.client != identity:
            raise ValueError(f"Client does not own session {session_id!r}.")
        return session

    def _status(self) -> dict[str, object]:
        return {
            "stack_id": self.config.stack_id,
            "node_id": self.config.node_id,
            "config_fingerprint": self.config.fingerprint,
            "process_id": os.getpid(),
            "state": "stopping" if self.stopping else "ready",
            "workers": [self._worker_status(worker) for worker in self.workers.values()],
            "sessions": len(self.sessions),
            "queued": sum(
                sum(task.state == "queued" for task in session.tasks.values())
                for session in self.sessions.values()
            ),
            "running": sum(
                0 if worker.task is None else len(worker.task)
                for worker in self.workers.values()
            ),
            "completed_unacknowledged": sum(
                sum(task.state == "completed" for task in session.tasks.values())
                for session in self.sessions.values()
            ),
            "pending_restarts": sum(
                len(requests)
                for requests in self.restart_requests.values()
            ),
        }

    def _worker_status(self, worker: WorkerRecord) -> dict[str, object]:
        if worker.dropped:
            state = "dropped"
        elif worker.draining:
            state = "draining"
        elif worker.ready and worker.task is not None:
            state = "busy"
        elif worker.ready:
            state = "ready"
        else:
            state = "connecting"
        return {
            "name": worker_name(worker.config),
            "worker_id": worker.worker_id,
            "node_id": worker.node_id,
            "configured_platform": worker.config.platform,
            "platform": worker.platform,
            "device": worker.device,
            "batch_size": worker.config.batch_size,
            "process_id": (
                worker.reported_process_id
                if worker.process is None
                else worker.process.pid
            ),
            "ready": worker.ready,
            "busy": worker.task is not None,
            "draining": worker.draining,
            "lease_generation": worker.lease_generation,
            "heartbeat_age_s": (
                None
                if worker.last_heartbeat_s == 0.0
                else max(0.0, time.monotonic() - worker.last_heartbeat_s)
            ),
            "state": state,
            "exit_code": worker.exit_code,
            "compile_s": worker.compile_s,
            "elapsed_s": worker.elapsed_s,
            "peak_rss_kib": worker.peak_rss_kib,
            "capabilities": worker.capabilities,
        }

    def _compatible(self, worker: WorkerRecord, session: SessionRecord) -> bool:
        fields = ("python", "jaxns", "jax", "jaxlib", "x64", "measure_dtype")
        return all(
            worker.capabilities.get(field) == session.capabilities.get(field)
            for field in fields
        )

    def _shutdown_workers(self) -> None:
        deadline = time.monotonic() + self.config.shutdown_timeout_s
        for worker in self.workers.values():
            if worker.ready:
                self._send(
                    worker.identity,
                    encode_header(STOP, lease_id=worker.lease_id),
                    router=worker.router,
                )
        for worker in self.workers.values():
            if worker.process is None or worker.process.poll() is not None:
                continue
            try:
                worker.process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                worker.process.terminate()
        for worker in self.workers.values():
            if worker.process is not None and worker.process.poll() is None:
                try:
                    worker.process.wait(timeout=self.config.shutdown_timeout_s)
                except subprocess.TimeoutExpired:
                    worker.process.kill()
                    worker.process.wait()
            if worker.log_file is not None:
                worker.log_file.close()

    def _cleanup(self) -> None:
        for router in (self.socket, self.network_socket):
            try:
                self.poller.unregister(router)
            except KeyError:
                pass
            router.close(linger=0)
        self.context.term()
        try:
            self.config.manifest.unlink()
        except FileNotFoundError:
            pass
        try:
            Path(self.config.endpoint.removeprefix("ipc://")).unlink()
        except FileNotFoundError:
            pass
        if self.lock_file is not None:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()

    def _lease_matches(self, worker: WorkerRecord, header: dict[str, object]) -> bool:
        return (
            worker.lease_id == header.get("lease_id")
            and worker.lease_generation == header.get("lease_generation")
            and worker.ready
        )

    def _send(
            self,
            identity: bytes,
            header: bytes,
            *payloads: bytes,
            router=None,
    ) -> None:
        target = self.socket if router is None else router
        target.send_multipart([identity, header, *(payloads or (b"",))])


def _start_worker_process(
        config: RuntimeConfig,
        worker: WorkerConfig,
        endpoint: str,
        instance_id: str,
) -> tuple[subprocess.Popen, object]:
    name = worker_name(worker)
    log_path = config.log_dir / f"{name}.log"
    log_file = log_path.open("ab", buffering=0)
    environment = os.environ.copy()
    # Workers are intentionally multi-process and may share an accelerator.
    # Each process must claim only the memory its program actually needs.
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if worker.platform == "gpu":
        environment["CUDA_VISIBLE_DEVICES"] = worker.device
        environment["JAX_PLATFORMS"] = "cuda"
        device_index = 0
    else:
        environment["JAX_PLATFORMS"] = worker.platform
        device_index = int(worker.device)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "jaxns.runtime_worker",
            "--config",
            str(config.source),
            "--endpoint",
            endpoint,
            "--platform",
            worker.platform,
            "--configured-device",
            worker.device,
            "--batch-size",
            str(worker.batch_size),
            "--device-index",
            str(device_index),
            "--program-cache-size",
            str(config.program_cache_size),
            "--instance-id",
            instance_id,
        ],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=log_file,
        start_new_session=False,
    )
    return process, log_file


def _string_field(header: dict[str, object], name: str) -> str:
    value = header.get(name)
    if type(value) is not str:
        raise ValueError(f"Protocol field {name!r} must be a string.")
    return value


def _integer_field(header: dict[str, object], name: str) -> int:
    value = header.get(name)
    if type(value) is not int:
        raise ValueError(f"Protocol field {name!r} must be an integer.")
    return value


def _integer_list(header: dict[str, object], name: str) -> list[int]:
    value = header.get(name)
    if type(value) is not list or not value or not all(
        type(item) is int for item in value
    ):
        raise ValueError(f"Protocol field {name!r} must be an integer list.")
    return value


def _string_list(header: dict[str, object], name: str) -> list[str]:
    value = header.get(name)
    if type(value) is not list or not value or not all(
        type(item) is str for item in value
    ):
        raise ValueError(f"Protocol field {name!r} must be a string list.")
    return value


def _single_payload(payloads: list[bytes], command: str) -> bytes:
    if len(payloads) != 1:
        raise ValueError(f"{command} requires exactly one payload.")
    return payloads[0]


def _dictionary_field(
        header: dict[str, object],
        name: str,
) -> dict[str, object]:
    value = header.get(name)
    if type(value) is not dict:
        raise ValueError(f"Protocol field {name!r} must be an object.")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_runtime_config(args.config)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=config.log_dir / "supervisor.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    supervisor = Supervisor(config)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    supervisor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
