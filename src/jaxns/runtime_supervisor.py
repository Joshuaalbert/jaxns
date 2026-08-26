"""JAX-free local supervisor for asynchronous constrained-sampling workers."""

from __future__ import annotations

import argparse
import collections
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

import zmq

from jaxns.runtime_config import RuntimeConfig, WorkerConfig, load_runtime_config
from jaxns.runtime_protocol import (
    ACK,
    ERROR,
    PING,
    READY,
    REGISTER,
    REGISTERED,
    RELEASE,
    RELEASED,
    RESULT,
    SHUTDOWN,
    STATUS,
    STOP,
    STOPPED,
    TASK,
    decode_header,
    encode_header,
)

logger = logging.getLogger(__name__)


class WorkerRecord:
    """Observed process and protocol state for one configured worker."""

    __slots__ = (
        "compile_s",
        "config",
        "device",
        "elapsed_s",
        "exit_code",
        "identity",
        "log_file",
        "peak_rss_kib",
        "platform",
        "process",
        "ready",
        "registered",
        "started_s",
        "task",
    )

    def __init__(
            self,
            config: WorkerConfig,
            process: subprocess.Popen,
            log_file: object,
            identity: bytes,
    ) -> None:
        self.config = config
        self.process = process
        self.log_file = log_file
        self.identity = identity
        self.ready = False
        self.registered: set[str] = set()
        self.task: tuple[str, int] | None = None
        self.started_s = 0.0
        self.platform = ""
        self.device = ""
        self.peak_rss_kib = 0
        self.compile_s = 0.0
        self.elapsed_s = 0.0
        self.exit_code: int | None = None


class TaskRecord:
    """Retry-stable opaque task retained until client acknowledgement."""

    __slots__ = (
        "batch_size",
        "fingerprint",
        "payload",
        "result_header",
        "result_payload",
        "state",
        "task_id",
        "worker",
    )

    def __init__(
            self,
            task_id: int,
            batch_size: int,
            fingerprint: str,
            payload: bytes,
    ) -> None:
        self.task_id = task_id
        self.batch_size = batch_size
        self.fingerprint = fingerprint
        self.payload = payload
        self.state = "queued"
        self.worker: bytes | None = None
        self.result_header = b""
        self.result_payload = b""


class SessionRecord:
    """One scientific client registration and its fair task queue."""

    __slots__ = (
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
    ) -> None:
        self.session_id = session_id
        self.fingerprint = fingerprint
        self.payload = payload
        self.client = client
        self.registered: set[bytes] = set()
        self.registration_notified = False
        self.tasks: dict[int, TaskRecord] = {}
        self.queue: collections.deque[int] = collections.deque()


class Supervisor:
    """Own worker processes and route opaque tasks without importing JAX."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.workers: dict[bytes, WorkerRecord] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self.session_order: collections.deque[str] = collections.deque()
        self.stopping = False
        self.lock_file = None

    def run(self) -> None:
        self._prepare_runtime()
        self._acquire_ownership()
        self.socket.bind(self.config.endpoint)
        self._write_manifest()
        self._start_workers()
        logger.info(
            "Supervisor %s listening at %s with %d workers",
            self.config.stack_id,
            self.config.endpoint,
            len(self.workers),
        )
        try:
            while not self.stopping:
                events = dict(self.poller.poll(100))
                if self.socket in events:
                    self._receive()
                self._reap_workers()
                self._expire_tasks()
                self._dispatch()
        finally:
            self._shutdown_workers()
            self._cleanup()

    def request_stop(self, signum=None, frame=None) -> None:
        del signum, frame
        self.stopping = True

    def _prepare_runtime(self) -> None:
        for directory in (
            self.config.runtime_dir,
            self.config.log_dir,
            self.config.lock.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)

    def _acquire_ownership(self) -> None:
        self.lock_file = self.config.lock.open("a+b")
        try:
            fcntl.flock(
                self.lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Stack {self.config.stack_id!r} already has an owner."
            ) from exc

    def _write_manifest(self) -> None:
        document = {
            "schema_version": 1,
            "stack_id": self.config.stack_id,
            "config_fingerprint": self.config.fingerprint,
            "process_id": os.getpid(),
            "endpoint": self.config.endpoint,
            "started_ns": time.time_ns(),
        }
        temporary = self.config.manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.config.manifest)

    def _start_workers(self) -> None:
        for worker in self.config.workers:
            log_path = self.config.log_dir / f"{worker.name}.log"
            log_file = log_path.open("ab", buffering=0)
            environment = os.environ.copy()
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
                    "--endpoint",
                    self.config.endpoint,
                    "--name",
                    worker.name,
                    "--batch-size",
                    str(worker.batch_size),
                    "--device-index",
                    str(device_index),
                    "--program-cache-size",
                    str(self.config.program_cache_size),
                ],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                start_new_session=False,
            )
            identity = f"worker:{worker.name}".encode()
            self.workers[identity] = WorkerRecord(
                config=worker,
                process=process,
                log_file=log_file,
                identity=identity,
            )

    def _receive(self) -> None:
        frames = self.socket.recv_multipart()
        if len(frames) != 3:
            logger.warning("Dropping message with %d frames", len(frames))
            return
        identity, raw_header, payload = frames
        try:
            header = decode_header(raw_header)
            command = header["command"]
            if identity.startswith(b"worker:"):
                self._receive_worker(identity, command, header, payload)
            else:
                self._receive_client(identity, command, header, payload)
        except Exception as exc:
            # Protocol isolation is the supervisor's outer fault boundary.
            # One malformed client or worker message must not kill the pool.
            logger.exception("Protocol handling failed")
            self._send(
                identity,
                encode_header(ERROR, error=f"{type(exc).__name__}: {exc}"),
            )

    def _receive_client(
            self,
            identity: bytes,
            command: str,
            header: dict[str, object],
            payload: bytes,
    ) -> None:
        if command == PING:
            self._send(
                identity,
                encode_header(
                    PING,
                    stack_id=self.config.stack_id,
                    config_fingerprint=self.config.fingerprint,
                    process_id=os.getpid(),
                ),
            )
            return
        if command == STATUS:
            self._send(identity, encode_header(STATUS, **self._status()))
            return
        if command == SHUTDOWN:
            self._send(identity, encode_header(SHUTDOWN, accepted=True))
            self.stopping = True
            return
        if command == REGISTER:
            self._register_client(identity, header, payload)
            return
        if command == RELEASE:
            self._release_client(identity, header)
            return
        if command == TASK:
            self._queue_task(identity, header, payload)
            return
        if command == ACK:
            self._acknowledge(identity, header)
            return
        raise ValueError(f"Unsupported client command {command!r}.")

    def _receive_worker(
            self,
            identity: bytes,
            command: str,
            header: dict[str, object],
            payload: bytes,
    ) -> None:
        worker = self.workers.get(identity)
        if worker is None:
            raise ValueError(f"Unknown worker identity {identity!r}.")
        if command == READY:
            worker.ready = True
            worker.platform = _string_field(header, "platform")
            worker.device = _string_field(header, "device")
            for session in self.sessions.values():
                self._send_registration(worker, session)
            return
        if command == REGISTERED:
            session_id = _string_field(header, "session_id")
            session = self.sessions.get(session_id)
            if session is None:
                return
            worker.registered.add(session_id)
            session.registered.add(identity)
            self._complete_registration(session)
            return
        if command == RESULT:
            self._complete_task(worker, header, payload)
            return
        if command == ERROR:
            self._worker_error(worker, header)
            return
        if command == STOPPED:
            worker.ready = False
            return
        raise ValueError(f"Unsupported worker command {command!r}.")

    def _register_client(
            self,
            identity: bytes,
            header: dict[str, object],
            payload: bytes,
    ) -> None:
        session_id = _string_field(header, "session_id")
        config_fingerprint = _string_field(header, "config_fingerprint")
        if config_fingerprint != self.config.fingerprint:
            raise ValueError(
                "Scientific client configuration does not match the running "
                "worker stack."
            )
        fingerprint = hashlib.sha256(payload).hexdigest()
        # Reconnection may repeat registration after the client lost its
        # reply. Exact bytes prove the session still names the same model;
        # accepting drift here could apply retained task results to new code.
        session = self.sessions.get(session_id)
        if session is None:
            session = SessionRecord(
                session_id=session_id,
                fingerprint=fingerprint,
                payload=payload,
                client=identity,
            )
            self.sessions[session_id] = session
            self.session_order.append(session_id)
        elif session.fingerprint != fingerprint:
            raise ValueError(
                f"Session {session_id!r} reconnected with different model data."
            )
        else:
            session.client = identity
            session.registration_notified = False

        ready = [worker for worker in self.workers.values() if worker.ready]
        if not ready:
            raise RuntimeError("No worker has acknowledged startup.")
        for worker in ready:
            if session_id not in worker.registered:
                self._send_registration(worker, session)
        self._complete_registration(session)
        for task in session.tasks.values():
            if task.state == "completed":
                self._send(identity, task.result_header, task.result_payload)

    def _release_client(
            self,
            identity: bytes,
            header: dict[str, object],
    ) -> None:
        session_id = _string_field(header, "session_id")
        session = self.sessions.get(session_id)
        if session is None:
            self._send(
                identity,
                encode_header(RELEASED, session_id=session_id),
            )
            return
        if session.client != identity:
            raise ValueError(f"Client does not own session {session_id!r}.")
        if session.tasks:
            raise RuntimeError(
                f"Session {session_id!r} still has unacknowledged tasks."
            )
        for worker in self.workers.values():
            worker.registered.discard(session_id)
            if worker.ready:
                self._send(
                    worker.identity,
                    encode_header(RELEASE, session_id=session_id),
                )
        self.session_order.remove(session_id)
        del self.sessions[session_id]
        self._send(
            identity,
            encode_header(RELEASED, session_id=session_id),
        )

    def _send_registration(
            self,
            worker: WorkerRecord,
            session: SessionRecord,
    ) -> None:
        self._send(
            worker.identity,
            encode_header(
                REGISTER,
                session_id=session.session_id,
                fingerprint=session.fingerprint,
            ),
            session.payload,
        )

    def _complete_registration(self, session: SessionRecord) -> None:
        if session.registration_notified:
            return
        ready = tuple(
            worker
            for worker in self.workers.values()
            if worker.ready
        )
        if not ready or any(
            session.session_id not in worker.registered
            for worker in ready
        ):
            return
        self._send(
            session.client,
            encode_header(
                REGISTERED,
                session_id=session.session_id,
                capacities=[worker.config.batch_size for worker in ready],
                workers=[
                    {
                        "name": worker.config.name,
                        "batch_size": worker.config.batch_size,
                        "platform": worker.platform,
                        "device": worker.device,
                    }
                    for worker in ready
                ],
            ),
        )
        session.registration_notified = True

    def _queue_task(
            self,
            identity: bytes,
            header: dict[str, object],
            payload: bytes,
    ) -> None:
        session_id = _string_field(header, "session_id")
        task_id = _integer_field(header, "task_id")
        batch_size = _integer_field(header, "batch_size")
        session = self._session_owned(identity, session_id)
        fingerprint = hashlib.sha256(payload).hexdigest()
        existing = session.tasks.get(task_id)
        if existing is not None:
            # Task identity is the exactly-once boundary. A duplicate request
            # either observes the retained result or remains represented by
            # the one queued/running record; it never creates a second draw.
            if existing.fingerprint != fingerprint:
                raise ValueError(
                    f"Task {task_id} was resubmitted with different payload."
                )
            if existing.state in ("completed", "failed"):
                self._send(identity, existing.result_header, existing.result_payload)
            return
        task = TaskRecord(
            task_id=task_id,
            batch_size=batch_size,
            fingerprint=fingerprint,
            payload=payload,
        )
        session.tasks[task_id] = task
        session.queue.append(task_id)
        self._report_missing_capacity(batch_size)

    def _acknowledge(
            self,
            identity: bytes,
            header: dict[str, object],
    ) -> None:
        session_id = _string_field(header, "session_id")
        task_id = _integer_field(header, "task_id")
        session = self._session_owned(identity, session_id)
        task = session.tasks.get(task_id)
        if task is not None and task.state == "completed":
            del session.tasks[task_id]

    def _complete_task(
            self,
            worker: WorkerRecord,
            header: dict[str, object],
            payload: bytes,
    ) -> None:
        session_id = _string_field(header, "session_id")
        task_id = _integer_field(header, "task_id")
        if worker.task != (session_id, task_id):
            logger.warning(
                "Ignoring late result %s/%s from %s",
                session_id,
                task_id,
                worker.config.name,
            )
            return
        session = self.sessions[session_id]
        task = session.tasks[task_id]
        task.state = "completed"
        task.worker = None
        task.result_header = encode_header(
            RESULT,
            session_id=session_id,
            task_id=task_id,
            name=worker.config.name,
            elapsed_s=header.get("elapsed_s", 0.0),
            compile_s=header.get("compile_s", 0.0),
            peak_rss_kib=header.get("peak_rss_kib", 0),
        )
        task.result_payload = payload
        worker.task = None
        worker.started_s = 0.0
        worker.peak_rss_kib = int(header.get("peak_rss_kib", 0))
        worker.compile_s += float(header.get("compile_s", 0.0))
        worker.elapsed_s += float(header.get("elapsed_s", 0.0))
        self._send(session.client, task.result_header, payload)

    def _worker_error(
            self,
            worker: WorkerRecord,
            header: dict[str, object],
    ) -> None:
        session_id = header.get("session_id")
        task_id = header.get("task_id")
        worker.task = None
        worker.started_s = 0.0
        if type(session_id) is str and type(task_id) is int:
            session = self.sessions.get(session_id)
            if session is not None:
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
                    task.result_payload = b""
                self._send(
                    session.client,
                    response,
                )
            return
        if type(session_id) is str:
            session = self.sessions.get(session_id)
            if session is not None:
                self._send(
                    session.client,
                    encode_header(
                        ERROR,
                        session_id=session_id,
                        error=header.get("error", "worker registration failed"),
                        traceback=header.get("traceback", ""),
                    ),
                )

    def _dispatch(self) -> None:
        # Scan every idle worker on each event-loop turn. Faster workers
        # naturally return to this pool sooner, while _next_task rotates the
        # client order so throughput does not turn into session starvation.
        for worker in self.workers.values():
            if not worker.ready or worker.task is not None:
                continue
            selected = self._next_task(worker)
            if selected is None:
                continue
            session, task = selected
            task.state = "running"
            task.worker = worker.identity
            worker.task = (session.session_id, task.task_id)
            worker.started_s = time.monotonic()
            self._send(
                worker.identity,
                encode_header(
                    TASK,
                    session_id=session.session_id,
                    task_id=task.task_id,
                ),
                task.payload,
            )

    def _next_task(
            self,
            worker: WorkerRecord,
    ) -> tuple[SessionRecord, TaskRecord] | None:
        for _ in range(len(self.session_order)):
            session_id = self.session_order[0]
            self.session_order.rotate(-1)
            session = self.sessions[session_id]
            if session_id not in worker.registered:
                continue
            for _ in range(len(session.queue)):
                task_id = session.queue.popleft()
                task = session.tasks.get(task_id)
                if task is None or task.state != "queued":
                    continue
                if task.batch_size != worker.config.batch_size:
                    session.queue.append(task_id)
                    continue
                return session, task
        return None

    def _reap_workers(self) -> None:
        for worker in self.workers.values():
            exit_code = worker.process.poll()
            if exit_code is None or worker.exit_code is not None:
                continue
            was_ready = worker.ready
            worker.ready = False
            worker.exit_code = exit_code
            logger.error(
                "Worker %s exited with code %s",
                worker.config.name,
                exit_code,
            )
            if was_ready:
                self._requeue(worker)
                self._report_missing_capacity(worker.config.batch_size)
                for session in self.sessions.values():
                    self._complete_registration(session)

    def _expire_tasks(self) -> None:
        now = time.monotonic()
        for worker in self.workers.values():
            if (
                worker.task is None
                or now - worker.started_s <= self.config.task_timeout_s
            ):
                continue
            logger.error(
                "Worker %s exceeded task timeout",
                worker.config.name,
            )
            worker.process.terminate()
            try:
                worker.process.wait(timeout=self.config.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait()
            worker.ready = False
            worker.exit_code = worker.process.returncode
            self._requeue(worker)
            self._report_missing_capacity(worker.config.batch_size)

    def _requeue(self, worker: WorkerRecord) -> None:
        if worker.task is None:
            return
        session_id, task_id = worker.task
        session = self.sessions[session_id]
        task = session.tasks[task_id]
        # Keep the original payload, ID, and random keys. Requeue changes only
        # transport ownership; treating failure as a new draw would bias the
        # scientific stream and leave the old reservation unmatched.
        task.state = "queued"
        task.worker = None
        session.queue.appendleft(task_id)
        worker.task = None
        worker.started_s = 0.0

    def _report_missing_capacity(self, batch_size: int) -> None:
        if any(
            worker.ready and worker.config.batch_size == batch_size
            for worker in self.workers.values()
        ):
            return
        for session in self.sessions.values():
            blocked = [
                task.task_id
                for task in session.tasks.values()
                if task.state == "queued" and task.batch_size == batch_size
            ]
            if blocked:
                self._send(
                    session.client,
                    encode_header(
                        ERROR,
                        session_id=session.session_id,
                        error=(
                            f"No ready worker supports batch size {batch_size}; "
                            f"blocked tasks: {blocked}."
                        ),
                    ),
                )

    def _session_owned(
            self,
            identity: bytes,
            session_id: str,
    ) -> SessionRecord:
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session {session_id!r}.")
        if session.client != identity:
            raise ValueError(f"Client does not own session {session_id!r}.")
        return session

    def _status(self) -> dict[str, object]:
        return {
            "stack_id": self.config.stack_id,
            "config_fingerprint": self.config.fingerprint,
            "process_id": os.getpid(),
            "state": "stopping" if self.stopping else "ready",
            "workers": [
                {
                    "name": worker.config.name,
                    "configured_platform": worker.config.platform,
                    "platform": worker.platform,
                    "device": worker.device,
                    "batch_size": worker.config.batch_size,
                    "process_id": worker.process.pid,
                    "ready": worker.ready,
                    "busy": worker.task is not None,
                    "exit_code": worker.exit_code,
                    "compile_s": worker.compile_s,
                    "elapsed_s": worker.elapsed_s,
                    "peak_rss_kib": worker.peak_rss_kib,
                }
                for worker in self.workers.values()
            ],
            "sessions": len(self.sessions),
            "queued": sum(
                sum(task.state == "queued" for task in session.tasks.values())
                for session in self.sessions.values()
            ),
            "running": sum(worker.task is not None for worker in self.workers.values()),
            "completed_unacknowledged": sum(
                sum(task.state == "completed" for task in session.tasks.values())
                for session in self.sessions.values()
            ),
        }

    def _shutdown_workers(self) -> None:
        # Cooperative STOP preserves clean worker logs. The bounded process
        # escalation prevents a data-dependent JAX loop from making `down`
        # hang indefinitely during operator or test teardown.
        deadline = time.monotonic() + self.config.shutdown_timeout_s
        for worker in self.workers.values():
            if worker.process.poll() is None and worker.ready:
                self._send(worker.identity, encode_header(STOP))
        for worker in self.workers.values():
            if worker.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                worker.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                worker.process.terminate()
        for worker in self.workers.values():
            if worker.process.poll() is None:
                try:
                    worker.process.wait(timeout=self.config.shutdown_timeout_s)
                except subprocess.TimeoutExpired:
                    worker.process.kill()
                    worker.process.wait()
            worker.log_file.close()

    def _cleanup(self) -> None:
        try:
            self.poller.unregister(self.socket)
        except KeyError:
            pass
        self.socket.close(linger=0)
        self.context.term()
        try:
            self.config.manifest.unlink()
        except FileNotFoundError:
            pass
        try:
            endpoint_path = Path(self.config.endpoint.removeprefix("ipc://"))
            endpoint_path.unlink()
        except FileNotFoundError:
            pass
        if self.lock_file is not None:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()

    def _send(
            self,
            identity: bytes,
            header: bytes,
            payload: bytes = b"",
    ) -> None:
        self.socket.send_multipart([identity, header, payload])


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
