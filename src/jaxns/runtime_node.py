"""JAX-free lifecycle owner for workers on one remote machine."""

from __future__ import annotations

import argparse
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

from jaxns.runtime_config import RuntimeConfig, load_runtime_config
from jaxns.runtime_protocol import (
    ACK,
    DRAIN,
    DRAINED,
    NODE_STATUS,
    NODE_STOPPED,
    PING,
    SHUTDOWN,
    STATUS,
    decode_header,
    encode_header,
)
from jaxns.runtime_supervisor import _start_worker_process
from jaxns.runtime_transport import configure_curve_client

logger = logging.getLogger(__name__)


class NodeSupervisor:
    """Own local device processes while the central coordinator owns work."""

    def __init__(self, config: RuntimeConfig):
        if config.role != "node" or config.network is None:
            raise ValueError("runtime_node requires a remote worker-node config.")
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.processes: list[tuple[object, object, str]] = []
        self.reported_exits: set[int] = set()
        self.leased = False
        self.stopping = False
        self.lock_file = None

    def run(self) -> None:
        self._prepare_runtime()
        self._acquire_ownership()
        self.socket.bind(self.config.endpoint)
        self._write_manifest()
        self._start_workers()
        self._wait_until_leased()
        self.leased = True
        try:
            while not self.stopping:
                events = dict(self.poller.poll(100))
                if self.socket in events:
                    self._receive()
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
        for worker in self.config.workers:
            process, log_file = _start_worker_process(
                self.config,
                worker,
                self.config.worker_endpoint,
                uuid4().hex,
            )
            self.processes.append((process, log_file, worker.name))

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

    def _observe_processes(self) -> None:
        for process, _, name in self.processes:
            if process.poll() is not None and process.pid not in self.reported_exits:
                logger.error("Worker %s exited with code %s", name, process.returncode)
                self.reported_exits.add(process.pid)

    def _status(self) -> dict[str, object]:
        remote = self._coordinator_status(timeout_s=1.0)
        remote_workers = {
            worker["name"]: worker
            for worker in remote.get("workers", [])
            if isinstance(worker, dict) and type(worker.get("name")) is str
        }
        return {
            "stack_id": self.config.stack_id,
            "node_id": self.config.node_id,
            "config_fingerprint": self.config.fingerprint,
            "process_id": os.getpid(),
            "state": "stopping" if self.stopping else "ready",
            "coordinator": self.config.worker_endpoint,
            "workers": [
                {
                    "name": name,
                    "process_id": process.pid,
                    "ready": (
                        process.poll() is None
                        and remote_workers.get(name, {}).get("ready") is True
                    ),
                    "state": remote_workers.get(name, {}).get(
                        "state",
                        "disconnected" if process.poll() is None else "exited",
                    ),
                    "exit_code": process.poll(),
                }
                for process, _, name in self.processes
            ],
        }

    def _wait_until_leased(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_s
        expected = {worker.name for worker in self.config.workers}
        while time.monotonic() < deadline:
            for process, _, name in self.processes:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Worker {name!r} exited during node startup with "
                        f"code {process.returncode}."
                    )
            status = self._coordinator_status(timeout_s=0.5)
            ready = {
                worker["name"]
                for worker in status.get("workers", [])
                if isinstance(worker, dict) and worker.get("ready") is True
            }
            if expected <= ready:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"Workers on node {self.config.node_id!r} did not receive "
            "coordinator leases before startup timeout."
        )

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
        configure_curve_client(socket, self.config.network)
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
        for process, _, _ in self.processes:
            if process.poll() is None:
                process.terminate()
        for process, log_file, _ in self.processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log_file.close()

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
        self.socket.close(linger=0)
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
