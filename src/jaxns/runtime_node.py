"""JAX-free lifecycle owner for workers on one remote machine."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import logging
import os
import signal
import subprocess
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
    DRAIN,
    DRAINED,
    ERROR,
    NODE_HEARTBEAT,
    NODE_HEARTBEAT_ACK,
    NODE_RESTARTED,
    NODE_STATUS,
    NODE_STOPPED,
    PING,
    SHUTDOWN,
    STATUS,
    decode_header,
    encode_header,
)
from jaxns.runtime_supervisor import _start_worker_process

logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class NodeWorker:
    """One configured process that is replaced after lease loss or exit."""

    config: WorkerConfig
    process: subprocess.Popen | None = None
    log_file: object | None = None
    started_s: float = 0.0
    restart_at_s: float = 0.0
    restart_count: int = 0
    exit_code: int | None = None
    instance_id: str = ""


class NodeSupervisor:
    """Own local device processes while the central coordinator owns work."""

    def __init__(self, config: RuntimeConfig):
        if config.role != "node":
            raise ValueError("runtime_node requires a remote worker-node config.")
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.coordinator = self.context.socket(zmq.DEALER)
        self.coordinator.setsockopt(
            zmq.IDENTITY,
            f"node:{self.config.node_id}:heartbeat".encode(),
        )
        self.coordinator.setsockopt(zmq.LINGER, 0)
        # Do not accumulate days of stale heartbeat messages through a
        # partition. The next periodic heartbeat is sufficient to recover the
        # bidirectional control channel after ZeroMQ reconnects.
        self.coordinator.setsockopt(zmq.IMMEDIATE, 1)
        self.coordinator.setsockopt(zmq.SNDHWM, 1)
        self.coordinator.connect(self.config.worker_endpoint)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.poller.register(self.coordinator, zmq.POLLIN)
        self.workers: list[NodeWorker] = []
        self.next_heartbeat_s = 0.0
        self.leased = False
        self.stopping = False
        self.lock_file = None

    def run(self) -> None:
        self._prepare_runtime()
        self._acquire_ownership()
        self.socket.bind(self.config.endpoint)
        self._write_manifest()
        self._start_workers()
        try:
            while not self.stopping:
                self._send_heartbeat_if_due()
                events = dict(self.poller.poll(100))
                if self.socket in events:
                    self._receive()
                if self.coordinator in events:
                    self._receive_coordinator()
                self._observe_processes()
        finally:
            self._drain(
                self.config.shutdown_timeout_s if self.leased else 1.0
            )
            self._stop_workers()
            self._notify_stopped()
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
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Node {self.config.node_id!r} already has a lifecycle owner."
            ) from exc

    def _write_manifest(self) -> None:
        document = {
            "schema_version": 2,
            "stack_id": self.config.stack_id,
            "node_id": self.config.node_id,
            "role": self.config.role,
            "config_fingerprint": self.config.fingerprint,
            "process_id": os.getpid(),
            "endpoint": self.config.endpoint,
            "coordinator": self.config.worker_endpoint,
            "started_ns": time.time_ns(),
        }
        temporary = self.config.manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.config.manifest)

    def _start_workers(self) -> None:
        for config in self.config.workers:
            worker = NodeWorker(config=config)
            self.workers.append(worker)
            self._start_worker(worker)

    def _start_worker(self, worker: NodeWorker) -> None:
        instance_id = uuid4().hex
        process, log_file = _start_worker_process(
            self.config,
            worker.config,
            self.config.worker_endpoint,
            instance_id,
        )
        worker.process = process
        worker.log_file = log_file
        worker.instance_id = instance_id
        worker.started_s = time.monotonic()
        worker.restart_at_s = 0.0
        worker.exit_code = None

    def _receive(self) -> None:
        frames = self.socket.recv_multipart()
        if len(frames) != 3:
            return
        identity, raw_header, _ = frames
        header = decode_header(raw_header)
        command = header["command"]
        if command == PING:
            response = encode_header(
                PING,
                stack_id=self.config.stack_id,
                node_id=self.config.node_id,
                config_fingerprint=self.config.fingerprint,
                process_id=os.getpid(),
            )
        elif command == STATUS:
            response = encode_header(STATUS, **self._status())
        elif command == SHUTDOWN:
            response = encode_header(SHUTDOWN, accepted=True)
            self.stopping = True
        else:
            raise ValueError(f"Unsupported node command {command!r}.")
        self.socket.send_multipart([identity, response, b""])

    def _send_heartbeat_if_due(self) -> None:
        """Offer one bounded control heartbeat to the central coordinator."""
        now = time.monotonic()
        if now < self.next_heartbeat_s:
            return
        self.next_heartbeat_s = now + self.config.heartbeat_interval_s
        try:
            self.coordinator.send_multipart(
                [
                    encode_header(
                        NODE_HEARTBEAT,
                        node_id=self.config.node_id,
                    ),
                    b"",
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            # IMMEDIATE plus a one-message HWM makes an unreachable
            # coordinator a quiet retry state rather than an unbounded queue.
            return

    def _receive_coordinator(self) -> None:
        """Apply commands returned over the node-initiated control socket."""
        frames = self.coordinator.recv_multipart()
        if len(frames) != 2:
            logger.warning(
                "Coordinator returned %d node-control frames",
                len(frames),
            )
            return
        header = decode_header(frames[0])
        command = header["command"]
        if command == NODE_HEARTBEAT_ACK:
            self._handle_restart_requests(header.get("restart_requests"))
        elif command == ACK:
            return
        elif command == ERROR:
            logger.warning(
                "Coordinator rejected node control message: %s",
                header.get("error", "unknown error"),
            )
        else:
            logger.warning("Unexpected coordinator command %r", command)

    def _handle_restart_requests(self, values) -> None:
        """Fence only the exact remote instances named by the coordinator."""
        if type(values) is not list:
            logger.warning("Coordinator returned invalid restart requests")
            return
        workers = {
            worker_name(worker.config): worker
            for worker in self.workers
        }
        handled = []
        for value in values:
            if type(value) is not dict:
                logger.warning("Coordinator returned an invalid restart request")
                continue
            request_id = value.get("request_id")
            name = value.get("worker_name")
            instance_id = value.get("instance_id")
            generation = value.get("lease_generation")
            reason = value.get("reason")
            if (
                type(request_id) is not str
                or type(name) is not str
                or type(instance_id) is not str
                or type(generation) is not int
                or type(reason) is not str
            ):
                logger.warning("Coordinator returned an invalid restart request")
                continue
            worker = workers.get(name)
            if worker is None or worker.instance_id != instance_id:
                # Missing configuration or a different current instance means
                # this at-least-once request is already obsolete.
                handled.append(request_id)
                continue
            process = worker.process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    logger.exception(
                        "Could not fence worker %s instance %s",
                        name,
                        instance_id,
                    )
                    continue
                logger.warning(
                    "Coordinator fenced worker %s lease %d: %s",
                    name,
                    generation,
                    reason,
                )
            handled.append(request_id)
        if handled:
            self.coordinator.send_multipart([
                encode_header(
                    NODE_RESTARTED,
                    node_id=self.config.node_id,
                    request_ids=handled,
                ),
                b"",
            ])

    def _observe_processes(self) -> None:
        now = time.monotonic()
        healthy_s = max(
            10.0,
            2.0
            * self.config.heartbeat_interval_s
            * self.config.missed_heartbeats,
        )
        for worker in self.workers:
            process = worker.process
            if process is not None and process.poll() is None:
                if now - worker.started_s >= healthy_s:
                    worker.restart_count = 0
                continue
            if process is not None:
                worker.exit_code = process.returncode
                logger.warning(
                    "Worker %s exited with code %s; scheduling replacement",
                    worker_name(worker.config),
                    process.returncode,
                )
                if worker.log_file is not None:
                    worker.log_file.close()
                worker.process = None
                worker.log_file = None
                worker.restart_count += 1
                # A partition can repeatedly fence a newly started process.
                # Backoff prevents a hot restart loop while remaining
                # unbounded so the node heals whenever connectivity returns.
                delay_s = min(30.0, 0.25 * 2 ** min(worker.restart_count, 7))
                worker.restart_at_s = now + delay_s
            if not self.stopping and now >= worker.restart_at_s:
                self._start_worker(worker)

    def _status(self) -> dict[str, object]:
        remote = self._coordinator_status(timeout_s=1.0)
        remote_workers = {
            worker["name"]: worker
            for worker in remote.get("workers", [])
            if type(worker) is dict and type(worker.get("name")) is str
        }
        if any(worker.get("ready") is True for worker in remote_workers.values()):
            self.leased = True
        return {
            "stack_id": self.config.stack_id,
            "node_id": self.config.node_id,
            "config_fingerprint": self.config.fingerprint,
            "process_id": os.getpid(),
            "state": "stopping" if self.stopping else "ready",
            "coordinator": self.config.worker_endpoint,
            "workers": [
                {
                    "name": worker_name(worker.config),
                    "process_id": (
                        None if worker.process is None else worker.process.pid
                    ),
                    "ready": (
                        worker.process is not None
                        and worker.process.poll() is None
                        and remote_workers.get(
                            worker_name(worker.config), {}
                        ).get("ready") is True
                    ),
                    "state": remote_workers.get(
                        worker_name(worker.config), {}
                    ).get(
                        "state",
                        (
                            "connecting"
                            if worker.process is not None
                            and worker.process.poll() is None
                            else "restarting"
                        ),
                    ),
                    "exit_code": worker.exit_code,
                    "restart_count": worker.restart_count,
                }
                for worker in self.workers
            ],
        }

    def _coordinator_status(self, timeout_s: float) -> dict[str, object]:
        manager = self._manager()
        try:
            manager.send_multipart([
                encode_header(NODE_STATUS, node_id=self.config.node_id),
                b"",
            ])
            if not manager.poll(timeout=max(1, int(timeout_s * 1000))):
                return {"workers": []}
            header = decode_header(manager.recv_multipart()[0])
            if header["command"] != NODE_STATUS:
                return {"workers": []}
            return header
        finally:
            manager.close(linger=0)

    def _manager(self):
        socket = self.context.socket(zmq.DEALER)
        socket.setsockopt(
            zmq.IDENTITY,
            f"node:{self.config.node_id}:{uuid4().hex}".encode(),
        )
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.config.worker_endpoint)
        return socket

    def _drain(self, timeout_s: float) -> None:
        manager = self._manager()
        try:
            manager.send_multipart([
                encode_header(DRAIN, node_id=self.config.node_id),
                b"",
            ])
            if manager.poll(timeout=max(1, int(timeout_s * 1000))):
                header = decode_header(manager.recv_multipart()[0])
                if header["command"] != DRAINED:
                    logger.warning("Coordinator rejected node drain: %s", header)
            else:
                logger.warning("Coordinator did not acknowledge node drain")
        finally:
            manager.close(linger=0)

    def _stop_workers(self) -> None:
        deadline = time.monotonic() + self.config.shutdown_timeout_s
        for worker in self.workers:
            if worker.process is not None and worker.process.poll() is None:
                worker.process.terminate()
        for worker in self.workers:
            process = worker.process
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if worker.log_file is not None:
                worker.log_file.close()

    def _notify_stopped(self) -> None:
        manager = self._manager()
        try:
            manager.send_multipart([
                encode_header(NODE_STOPPED, node_id=self.config.node_id),
                b"",
            ])
            if manager.poll(timeout=1000):
                header = decode_header(manager.recv_multipart()[0])
                if header["command"] != ACK:
                    logger.warning("Unexpected node-stop response: %s", header)
        finally:
            manager.close(linger=0)

    def _cleanup(self) -> None:
        self.poller.unregister(self.socket)
        self.poller.unregister(self.coordinator)
        self.socket.close(linger=0)
        self.coordinator.close(linger=0)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_runtime_config(args.config)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=config.log_dir / "node.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    node = NodeSupervisor(config)
    signal.signal(signal.SIGINT, node.request_stop)
    signal.signal(signal.SIGTERM, node.request_stop)
    node.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
