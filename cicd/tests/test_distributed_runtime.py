"""Local-process lifecycle and real constrained-sampling integration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import pytest
from jax import numpy as jnp

from cicd.tests.distributed_support import make_periodic_model, make_toy_model
from jaxns.cli import _stop_started_process
from jaxns.constrained_sampler import (
    ConstrainedSampleRequest,
    EllipsoidalDirection,
    UniDimSliceSampler,
    sample_request,
)
from jaxns.distributed_core import (
    DistributedNestedSampler,
)
from jaxns.multi_ellipsoid_utils import empty_sampler_data
from jaxns.runtime.client import SupervisorClient, _sampler_batch_group
from jaxns.runtime.config import (
    WorkerConfig,
    load_runtime_config,
    worker_name,
)
from jaxns.runtime.coordinator import (
    SessionRecord,
    Supervisor,
    TaskRecord,
    WorkerRecord,
)
from jaxns.runtime.node import NodeSupervisor, NodeWorker
from jaxns.runtime.protocol import (
    ACK,
    CAPACITY,
    MAX_HEADER_BYTES,
    NODE_HEARTBEAT,
    NODE_HEARTBEAT_ACK,
    NODE_RESTARTED,
    PROTOCOL_VERSION,
    SAMPLE,
    decode_header,
)
from jaxns.runtime.session import WorkerSession
from jaxns.runtime.worker import _fence_process
from jaxns.samples import SeedPoint
from jaxns.termination_condition import TerminationCondition

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_batch_group_ignores_direction_diagnostics_but_not_geometry():
    data = empty_sampler_data(num_components=2, dimension=1)
    diagnostics = dataclasses.replace(
        data,
        num_samples=jnp.asarray(100),
        num_attempted=jnp.asarray(90),
        num_updates=jnp.asarray(4),
        num_directions=jnp.asarray(500),
        num_isotropic=jnp.asarray(5),
    )
    geometry = dataclasses.replace(
        data,
        radii=jnp.ones_like(data.radii),
    )

    assert _sampler_batch_group(data) == _sampler_batch_group(diagnostics)
    assert _sampler_batch_group(data) != _sampler_batch_group(geometry)

    # Protect the premise behind that grouping rule: diagnostic fields must
    # remain observational and cannot enter the worker's direction law.
    model = make_toy_model()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=2,
        direction=EllipsoidalDirection(num_components=2),
    )
    request = ConstrainedSampleRequest(
        keys=jax.random.split(jax.random.PRNGKey(4), 1),
        valid=jnp.asarray([True]),
        log_L_constraints=jnp.asarray([-1.0]),
        seed_points=SeedPoint(
            U0=jnp.asarray([0.5]),
            log_L0=jnp.asarray([-0.0625]),
        ),
        sampler_data=data,
    )
    reference = sample_request(sampler, request)
    observed = sample_request(
        sampler,
        dataclasses.replace(request, sampler_data=diagnostics),
    )
    assert all(
        bool(jnp.array_equal(left, right))
        for left, right in zip(
            jax.tree.leaves(reference),
            jax.tree.leaves(observed),
            strict=True,
        )
    )


def _free_tcp_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _config_port(path: Path) -> int:
    if path.exists():
        port = load_runtime_config(path).network.port
        if port is not None:
            return port
    return _free_tcp_port()


def _write_config(path: Path, *, task_timeout_s: float = 30.0) -> None:
    port = _config_port(path)
    path.write_text(
        f"""
[runtime]
stack_id = "distributed-test"
runtime_dir = "runtime"
log_dir = "logs"
startup_timeout_s = 30
shutdown_timeout_s = 15
task_timeout_s = {task_timeout_s}

[network]
port = {port}

[[workers]]
platform = "cpu"
device = 0
batch_size = 2
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_batch_config(path: Path) -> None:
    """Write one deterministic vmap worker for trajectory invariance tests."""
    port = _config_port(path)
    path.write_text(
        f"""
[runtime]
stack_id = "distributed-batch-test"
runtime_dir = "runtime"
log_dir = "logs"
startup_timeout_s = 30
shutdown_timeout_s = 15
task_timeout_s = 30

[network]
port = {port}

[[workers]]
platform = "cpu"
device = 0
batch_size = 2
""".strip() + "\n",
        encoding="utf-8",
    )


def _cli(config: Path, *args: str, check: bool = True):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "jaxns.cli",
            "--config",
            str(config),
            *args,
        ],
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
        timeout=45,
    )


def _status(config: Path) -> dict[str, object]:
    return json.loads(_cli(config, "status").stdout)


def _write_network_configs(directory: Path) -> tuple[Path, Path]:
    port = _free_tcp_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    coordinator = directory / "coordinator.toml"
    coordinator.write_text(
        f"""
[runtime]
stack_id = "network-test"
node_id = "main"
runtime_dir = "runtime"
log_dir = "logs"
startup_timeout_s = 30
shutdown_timeout_s = 15
task_timeout_s = 30
heartbeat_interval_s = 0.25
missed_heartbeats = 2

[network]
port = {port}
""".strip() + "\n",
        encoding="utf-8",
    )
    node = directory / "node.toml"
    node.write_text(
        f"""
[runtime]
stack_id = "network-test"
node_id = "remote"
runtime_dir = "runtime"
log_dir = "logs"
startup_timeout_s = 30
shutdown_timeout_s = 15
task_timeout_s = 30
heartbeat_interval_s = 0.25
missed_heartbeats = 2

[network]
coordinator = "{endpoint}"

[[workers]]
platform = "cpu"
device = 0
batch_size = 2
""".strip() + "\n",
        encoding="utf-8",
    )
    return coordinator, node


def test_config_expands_workers_and_fingerprints_resolved_behavior(tmp_path):
    config_path = tmp_path / "workers.toml"
    _write_config(config_path)
    config = load_runtime_config(config_path)

    assert config.stack_id == "distributed-test"
    assert [worker_name(worker) for worker in config.workers] == ["cpu-0"]
    assert [worker.batch_size for worker in config.workers] == [2]
    assert config.endpoint.startswith("ipc://")
    validated = json.loads(_cli(config_path, "config", "validate").stdout)
    assert validated["config_fingerprint"] == config.fingerprint

    equivalent_path = tmp_path / "equivalent.toml"
    equivalent_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "task_timeout_s = 30.0",
            "task_timeout_s=30",
        ),
        encoding="utf-8",
    )
    equivalent = load_runtime_config(equivalent_path)
    # Default paths resolve relative to different config files only through
    # their common parent, so formatting changes cannot create false drift.
    assert equivalent.fingerprint == config.fingerprint

    malformed = tmp_path / "malformed.toml"
    malformed.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "device = 0",
            'device = "../device"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        load_runtime_config(malformed)

    with pytest.raises(ValueError, match="header exceeds"):
        decode_header(b" " * (MAX_HEADER_BYTES + 1))


def test_supervisor_and_config_validation_avoid_scientific_imports(tmp_path):
    config_path = tmp_path / "workers.toml"
    _write_config(config_path)
    supervisor = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import jaxns.runtime.coordinator; "
                "assert 'jax' not in sys.modules"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert supervisor.stderr == ""
    validation = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from jaxns.cli import main; "
                f"assert main(['--config', {str(config_path)!r}, "
                "'config', 'validate']) == 0; "
                "assert 'jax' not in sys.modules; "
                "assert 'zmq' not in sys.modules; "
                "assert 'cloudpickle' not in sys.modules"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"stack_id": "distributed-test"' in validation.stdout


def test_failed_cli_startup_forcibly_cleans_up_owned_supervisor():
    class Process:
        def __init__(self):
            self.returncode = None
            self.terminated = False
            self.killed = False
            self.waits = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            del timeout
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("supervisor", 0.1)
            self.returncode = -signal.SIGKILL
            return self.returncode

        def kill(self):
            self.killed = True

    process = Process()
    _stop_started_process(
        SimpleNamespace(shutdown_timeout_s=0.1),
        process,
    )

    assert process.terminated
    assert process.killed
    assert process.waits == 2


def test_supervisor_deduplicates_task_identity_until_acknowledgement():
    supervisor = object.__new__(Supervisor)
    session = SessionRecord("session", "registration", b"model", b"client")
    supervisor.sessions = {"session": session}
    sent = []
    supervisor._send = lambda *frames: sent.append(frames)
    header = {
        "session_id": "session",
        "task_ids": [5, 6],
        "batch_groups": ["direction-state", "direction-state"],
        "operations": [SAMPLE, SAMPLE],
    }
    payloads = [b"request-five", b"request-six"]

    supervisor._queue_tasks(b"client", header, payloads)
    supervisor._queue_tasks(b"client", header, payloads)
    assert list(session.queue) == [5, 6]
    assert list(session.tasks) == [5, 6]

    for task_id in (5, 6):
        task = session.tasks[task_id]
        task.state = "completed"
        task.completion_group = "assignment"
        task.result_header = b"result-header"
        task.result_payload = f"result-{task_id}".encode()
    supervisor._queue_tasks(b"client", header, payloads)
    assert len(sent) == 1
    assert decode_header(sent[0][1])["task_ids"] == [5, 6]
    assert sent[0][2:] == (b"result-5", b"result-6")
    for task_id in (5, 6):
        supervisor._acknowledge(
            b"client",
            {"session_id": "session", "task_id": task_id},
        )
    assert not session.tasks


def test_supervisor_validates_atomic_task_group_before_queue_mutation():
    supervisor = object.__new__(Supervisor)
    session = SessionRecord("session", "registration", b"model", b"client")
    supervisor.sessions = {"session": session}
    supervisor._send = lambda *frames: None
    header = {
        "session_id": "session",
        "task_ids": [1, 2],
        "batch_groups": ["direction-state"],
        "operations": [SAMPLE, SAMPLE],
    }

    with pytest.raises(ValueError, match="disagree"):
        supervisor._queue_tasks(b"client", header, [b"one", b"two"])

    assert not session.tasks
    assert not session.queue

    header["batch_groups"] = ["direction-state", "direction-state"]
    header["task_ids"] = [1, 1]
    with pytest.raises(ValueError, match="unique"):
        supervisor._queue_tasks(b"client", header, [b"one", b"one"])

    assert not session.tasks
    assert not session.queue


def test_task_timeout_requeues_byte_identical_payload_on_compatible_worker():
    class Process:
        def __init__(self):
            self.returncode = None

        def terminate(self):
            self.returncode = -signal.SIGTERM

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            return self.returncode

        def kill(self):
            self.returncode = -signal.SIGKILL

    config = WorkerConfig("cpu", "0", 1)
    active = WorkerRecord(config, Process(), None, b"worker:first")
    active.ready = True
    active.task = (("session", 6),)
    active.started_s = time.monotonic() - 2.0
    spare_config = WorkerConfig("gpu", "0", 1)
    spare = WorkerRecord(spare_config, Process(), None, b"worker:second")
    spare.ready = True
    payload = b"scientific-request"
    task = TaskRecord(
        6,
        "direction-state",
        SAMPLE,
        hashlib.sha256(payload).hexdigest(),
        payload,
    )
    task.state = "running"
    task.worker = active.identity
    session = SessionRecord("session", "registration", b"model", b"client")
    session.tasks[6] = task

    supervisor = object.__new__(Supervisor)
    supervisor.config = SimpleNamespace(
        task_timeout_s=0.1,
    )
    supervisor.workers = {
        active.identity: active,
        spare.identity: spare,
    }
    supervisor.sessions = {"session": session}
    supervisor._expire_tasks()

    assert task.state == "queued"
    assert task.payload is payload
    assert task.fingerprint == hashlib.sha256(payload).hexdigest()
    assert list(session.queue) == [6]
    assert active.task is None
    assert active.dropped


def test_two_missed_heartbeats_fence_lease_and_requeue():
    worker_config = WorkerConfig("cpu", "0", 1)
    worker = WorkerRecord(worker_config, None, None, b"worker")
    worker.ready = True
    worker.last_heartbeat_s = time.monotonic() - 0.21
    worker.task = (("session", 4),)
    payload = b"retry-stable"
    task = TaskRecord(
        4,
        "direction-state",
        SAMPLE,
        hashlib.sha256(payload).hexdigest(),
        payload,
    )
    task.state = "running"
    session = SessionRecord("session", "registration", b"model", b"client")
    session.tasks[4] = task

    supervisor = object.__new__(Supervisor)
    supervisor.config = SimpleNamespace(
        heartbeat_interval_s=0.1,
        missed_heartbeats=2,
    )
    supervisor.workers = {worker.identity: worker}
    supervisor.sessions = {session.session_id: session}
    supervisor.restart_requests = {}
    supervisor._expire_workers()

    assert worker.dropped
    assert not worker.ready
    assert list(session.queue) == [4]
    assert task.payload == payload


def test_remote_restart_request_waits_for_node_heartbeat_and_acknowledgement():
    worker = WorkerRecord(
        WorkerConfig("cpu", "0", 1),
        None,
        None,
        b"worker:remote/cpu-0:old",
        node_id="remote",
        instance_id="old",
    )
    worker.ready = True
    worker.lease_generation = 3
    supervisor = object.__new__(Supervisor)
    supervisor.workers = {worker.identity: worker}
    supervisor.sessions = {}
    supervisor.restart_requests = {}
    sent = []
    supervisor._send = lambda *args, **kwargs: sent.append((args, kwargs))

    supervisor._drop_worker(worker, "missed heartbeat lease")

    requests = supervisor.restart_requests["remote"]
    assert len(requests) == 1
    request_id = next(iter(requests))
    supervisor._receive_node(
        object(),
        b"node:remote:heartbeat",
        NODE_HEARTBEAT,
        {"node_id": "remote"},
    )
    heartbeat = decode_header(sent[-1][0][1])
    assert heartbeat["command"] == NODE_HEARTBEAT_ACK
    assert heartbeat["restart_requests"] == [{
        "instance_id": "old",
        "lease_generation": 3,
        "reason": "missed heartbeat lease",
        "request_id": request_id,
        "worker_name": "cpu-0",
    }]

    supervisor._receive_node(
        object(),
        b"node:remote:heartbeat",
        NODE_RESTARTED,
        {"node_id": "remote", "request_ids": [request_id]},
    )
    acknowledged = decode_header(sent[-1][0][1])
    assert acknowledged["command"] == ACK
    assert supervisor.restart_requests == {}


def test_node_restart_request_fences_only_the_named_worker_instance():
    class Process:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return -signal.SIGTERM if self.terminated else None

        def terminate(self):
            self.terminated = True

    class Coordinator:
        def __init__(self):
            self.messages = []

        def send_multipart(self, frames):
            self.messages.append(frames)

    process = Process()
    worker = NodeWorker(
        config=WorkerConfig("cpu", "0", 1),
        process=process,
        instance_id="old",
    )
    node = object.__new__(NodeSupervisor)
    node.config = SimpleNamespace(node_id="remote")
    node.workers = [worker]
    node.coordinator = Coordinator()
    request = {
        "request_id": "restart-1",
        "worker_name": "cpu-0",
        "instance_id": "old",
        "lease_generation": 3,
        "reason": "missed heartbeat lease",
    }

    node._handle_restart_requests([request])

    assert process.terminated
    acknowledgement = decode_header(node.coordinator.messages[-1][0])
    assert acknowledgement["command"] == NODE_RESTARTED
    assert acknowledgement["request_ids"] == ["restart-1"]

    # An acknowledgement may be lost. Replaying the old request after the
    # replacement has a new instance ID must not kill that replacement.
    replacement = Process()
    worker.process = replacement
    worker.instance_id = "new"
    node._handle_restart_requests([request])
    assert not replacement.terminated


def test_fenced_worker_forces_process_exit_for_supervisor_restart():
    alive = threading.Event()
    alive.set()
    stopping = threading.Event()
    signals = []
    with patch(
        "jaxns.runtime.worker.os.kill",
        lambda process_id, process_signal: signals.append((
            process_id,
            process_signal,
        )),
    ):
        _fence_process(alive, stopping)

    assert not alive.is_set()
    assert signals == [(os.getpid(), signal.SIGTERM)]

    # An ordinary shutdown has already arranged process termination and must
    # not race it with a second signal from the heartbeat thread.
    alive.set()
    stopping.set()
    signals.clear()
    with patch("jaxns.runtime.worker.os.kill"):
        _fence_process(alive, stopping)
    assert not alive.is_set()
    assert signals == []


def test_fresh_worker_instance_supersedes_live_stale_lease():
    config = WorkerConfig("cpu", "0", 1)
    old = WorkerRecord(
        config,
        None,
        None,
        b"worker:node/cpu-0:old",
        node_id="node",
        instance_id="old",
    )
    old.ready = True
    old.lease_generation = 7
    old.task = (("session", 9),)
    task = TaskRecord(
        9,
        "direction-state",
        SAMPLE,
        "fingerprint",
        b"payload",
    )
    task.state = "running"
    task.worker = old.identity
    session = SessionRecord("session", "model", b"model", b"client")
    session.tasks[9] = task

    supervisor = object.__new__(Supervisor)
    supervisor.config = SimpleNamespace(
        heartbeat_interval_s=1.0,
        missed_heartbeats=2,
    )
    supervisor.workers = {old.identity: old}
    supervisor.worker_ids = {old.worker_id: old.identity}
    supervisor.sessions = {session.session_id: session}
    supervisor.restart_requests = {}
    supervisor._send = lambda *args, **kwargs: None

    new_identity = b"worker:node/cpu-0:new"
    supervisor._lease_worker(
        object(),
        new_identity,
        {
            "worker_id": "node/cpu-0",
            "node_id": "node",
            "configured_platform": "cpu",
            "configured_device": "0",
            "instance_id": "new",
            "batch_size": 1,
            "platform": "cpu",
            "device": "TFRT_CPU_0",
            "capabilities": {},
            "process_id": 123,
        },
    )

    replacement = supervisor.workers[new_identity]
    assert old.identity not in supervisor.workers
    assert replacement.ready
    assert replacement.lease_generation == 8
    assert supervisor.worker_ids[replacement.worker_id] == new_identity
    assert task.state == "queued"
    assert task.worker is None
    assert list(session.queue) == [9]


def test_supervisor_rotates_dispatch_fairly_between_sessions():
    config = WorkerConfig("cpu", "0", 1)
    worker = WorkerRecord(config, object(), None, b"worker")
    worker.ready = True
    worker.registered.update(("first", "second"))
    first = SessionRecord("first", "one", b"model", b"client-one")
    second = SessionRecord("second", "two", b"model", b"client-two")
    for session, task_id in ((first, 1), (second, 2)):
        task = TaskRecord(
            task_id,
            "direction-state",
            SAMPLE,
            str(task_id),
            b"payload",
        )
        session.tasks[task_id] = task
        session.queue.append(task_id)

    supervisor = object.__new__(Supervisor)
    supervisor.session_order = deque(("first", "second"))
    supervisor.sessions = {"first": first, "second": second}
    supervisor.config = SimpleNamespace(batch_wait_s=0.0)
    selected_first = supervisor._next_tasks(worker)
    selected_second = supervisor._next_tasks(worker)

    assert selected_first[0].session_id == "first"
    assert selected_second[0].session_id == "second"


def test_incompatible_worker_is_rejected_before_model_registration():
    worker = WorkerRecord(
        WorkerConfig("cpu", "0", 1),
        None,
        None,
        b"worker",
    )
    worker.ready = True
    worker.capabilities = {
        "python": "3.12",
        "jaxns": "3.0.0",
        "jax": "0.10.0",
        "jaxlib": "0.10.0",
        "x64": True,
        "measure_dtype": "float64",
    }
    session = SessionRecord(
        "session",
        "registration",
        b"model",
        b"client",
        capabilities={**worker.capabilities, "jax": "0.11.0"},
    )
    supervisor = object.__new__(Supervisor)
    sent = []
    supervisor._send = lambda *args, **kwargs: sent.append((args, kwargs))

    assert not supervisor._compatible(worker, session)
    supervisor._send_registration(worker, session)
    assert sent == []

    # No worker is different from an incompatible worker: preserve a waiting
    # session so a node that joins later can complete registration.
    supervisor.config = SimpleNamespace(fingerprint="config")
    supervisor.sessions = {}
    supervisor.session_order = deque()
    supervisor.workers = {}
    supervisor._register_client(
        b"client",
        {
            "session_id": "waiting",
            "config_fingerprint": "config",
            "capabilities": session.capabilities,
        },
        b"model",
    )
    assert "waiting" in supervisor.sessions
    assert sent == []


def test_capacity_counts_only_workers_registered_for_session():
    compatible = WorkerRecord(
        WorkerConfig("cpu", "0", 2),
        None,
        None,
        b"compatible",
    )
    compatible.ready = True
    compatible.registered.add("session")
    incompatible = WorkerRecord(
        WorkerConfig("gpu", "0", 8),
        None,
        None,
        b"incompatible",
    )
    incompatible.ready = True
    supervisor = object.__new__(Supervisor)
    supervisor.workers = {
        compatible.identity: compatible,
        incompatible.identity: incompatible,
    }
    sent = []
    supervisor._send = lambda *args, **kwargs: sent.append((args, kwargs))

    supervisor._receive_client(
        b"client",
        CAPACITY,
        {"session_id": "session"},
        b"",
    )

    header = decode_header(sent[0][0][1])
    assert header["lanes"] == 2
    assert header["workers"] == 1


def test_phantom_payload_does_not_change_vector_worker_trajectory(tmp_path):
    config_path = tmp_path / "batch-workers.toml"
    _write_batch_config(config_path)
    _cli(config_path, "up")
    try:
        model = make_toy_model()

        def run(collect_phantoms):
            sampler = UniDimSliceSampler(
                model=model,
                num_slices=2,
                collect_phantom_samples=collect_phantoms,
                phantom_burn_in=0,
            )
            runner = DistributedNestedSampler(
                model=model,
                coordinator_port=load_runtime_config(
                    config_path
                ).network.port,
                root_allocation_degree=4,
                delta_K=4,
                max_samples=16,
                initial_capacity=8,
                collect_phantom_samples=collect_phantoms,
                sampler=sampler,
            )
            return runner.run_until_goal(
                lambda state: int(state.goal_loop_iter) >= 2,
                depth_cond=TerminationCondition(dlogZ=jnp.asarray(0.5)),
                key=jax.random.PRNGKey(52),
            ).state

        classic = run(False)
        phantom = run(True)
        classic_count = int(classic.num_samples)
        phantom_count = int(phantom.num_samples)

        # Phantom collection changes only retained intermediate states. The
        # same stable task IDs and random streams must create the same race.
        assert classic_count == phantom_count
        assert int(classic.root_out_degree) == int(phantom.root_out_degree)
        paired_fields = (
            (
                "log_L_constraints",
                classic.samples.log_L_constraints,
                phantom.samples.log_L_constraints,
            ),
            (
                "log_likelihoods",
                classic.samples.log_likelihoods,
                phantom.samples.log_likelihoods,
            ),
            (
                "U_samples",
                classic.samples.U_samples,
                phantom.samples.U_samples,
            ),
            (
                "out_degree",
                classic.samples.out_degree,
                phantom.samples.out_degree,
            ),
            (
                "num_likelihood_evaluations",
                classic.samples.num_likelihood_evaluations,
                phantom.samples.num_likelihood_evaluations,
            ),
        )
        for field, left, right in paired_fields:
            assert bool(jnp.array_equal(
                left[:classic_count],
                right[:phantom_count],
            )), field
        assert not bool(jnp.any(
            classic.samples.phantom_samples.valid_mask[:classic_count]
        ))
        assert bool(jnp.any(
            phantom.samples.phantom_samples.valid_mask[:phantom_count]
        ))
    finally:
        stopped = _cli(config_path, "down", check=False)
        assert stopped.returncode == 0, stopped.stderr


def test_periodic_sampler_executes_in_real_worker_processes(tmp_path):
    """Wire serialisation retains topology and canonical worker outputs."""
    config_path = tmp_path / "periodic-workers.toml"
    _write_batch_config(config_path)
    _cli(config_path, "up")
    try:
        model = make_periodic_model()
        runner = DistributedNestedSampler(
            model=model,
            coordinator_port=load_runtime_config(config_path).network.port,
            root_allocation_degree=4,
            delta_K=4,
            max_samples=12,
            initial_capacity=8,
            sampler=UniDimSliceSampler(model=model, num_slices=4),
        )

        checkpoint = runner.run_until_goal(
            lambda state: int(state.goal_loop_iter) >= 1,
            depth_cond=TerminationCondition(dlogZ=jnp.asarray(0.5)),
            key=jax.random.PRNGKey(275),
        )

        assert runner.sampler._periodic == (True,)
        assert int(checkpoint.state.num_samples) > 4
        num_samples = int(checkpoint.state.num_samples)
        for value in jax.tree.leaves(checkpoint.state.samples.U_samples):
            # [N, ...] workers always return the canonical half-open chart.
            value = value[:num_samples]
            assert jnp.all(value >= 0.0)
            assert jnp.all(value < 1.0)
    finally:
        stopped = _cli(config_path, "down", check=False)
        assert stopped.returncode == 0, stopped.stderr


def test_real_pool_runs_scalar_vmap_retries_and_cli_lifecycle(tmp_path):
    config_path = tmp_path / "workers.toml"
    _write_config(config_path)
    _cli(config_path, "up")
    try:
        config = load_runtime_config(config_path)
        manifest = json.loads(config.manifest.read_text(encoding="utf-8"))
        assert manifest["protocol_version"] == PROTOCOL_VERSION
        assert manifest["transports"] == ["ipc", "tcp"]
        idempotent = json.loads(_cli(config_path, "up").stdout)
        assert idempotent["idempotent"] is True
        assert all(worker["ready"] for worker in idempotent["workers"])

        model = make_toy_model()
        sampler = UniDimSliceSampler(
            model=model,
            num_slices=2,
            collect_phantom_samples=True,
            phantom_burn_in=0,
        )
        distributed = DistributedNestedSampler(
            model=model,
            coordinator_port=config.network.port,
            root_allocation_degree=4,
            delta_K=4,
            max_samples=16,
            initial_capacity=8,
            sampler=sampler,
        )
        checkpoint_dir = tmp_path / "distributed-checkpoint"
        first_checkpoint = distributed.run_until_goal(
            lambda state: int(state.goal_loop_iter) >= 1,
            depth_cond=TerminationCondition(dlogZ=jnp.asarray(0.5)),
            key=jax.random.PRNGKey(21),
            checkpoint_dir=checkpoint_dir,
        )
        checkpoint = distributed.run_until_goal(
            lambda state: int(state.goal_loop_iter) >= 2,
            depth_cond=TerminationCondition(dlogZ=jnp.asarray(0.5)),
            # The persisted state owns the exact continuation stream.
            key=jax.random.PRNGKey(999),
            checkpoint_dir=checkpoint_dir,
        )

        assert int(first_checkpoint.state.goal_loop_iter) >= 1
        assert not checkpoint.pending
        assert int(checkpoint.reservations.num_reserved) == 0
        state = checkpoint.state
        assert int(state.num_samples) > 4
        assert int(state.root_out_degree) + int(jnp.sum(
            state.samples.out_degree[:state.num_samples]
        )) == int(state.num_samples)
        assert bool(jnp.any(
            state.samples.phantom_samples.valid_mask[4:state.num_samples]
        ))

        # Exercise worker-local vmap explicitly. A partially filled worker
        # takes the scalar path after batch_wait_s, while two compatible queued
        # tasks use its configured two-lane vmap.
        conformance_id = "worker-programs"
        # The local lambda proves one-time model/session registration supports
        # notebook and closure code rather than only importable module symbols.
        conformance = WorkerSession(
            model=model,
            sampler=sampler,
            args=(lambda value: value,),
            params=None,
        )
        requests = {}
        with SupervisorClient.from_config(config_path) as client:
            capacities = client.register(conformance_id, conformance)
            # Registration returns as soon as one worker is usable; later
            # workers register the same session without a pool-wide barrier.
            assert capacities
            for task_id in range(1, 7):
                request = ConstrainedSampleRequest(
                    keys=jax.random.split(jax.random.PRNGKey(30 + task_id), 1),
                    valid=jnp.ones((1,), dtype=bool),
                    log_L_constraints=jnp.full((1,), -1.0),
                    seed_points=SeedPoint(
                        U0=jnp.full((1,), 0.5),
                        log_L0=jnp.full((1,), -0.0625),
                    ),
                    sampler_data=None,
                )
                requests[task_id] = request
                client.submit(conformance_id, task_id, request)
            completed = {
                client.receive(conformance_id, timeout_s=30.0)[0]
                for _ in requests
            }
            assert completed == set(requests)
            for task_id in completed:
                client.acknowledge(conformance_id, task_id)
            client.release(conformance_id)

        workers = _status(config_path)["workers"]
        assert workers[0]["compile_s"] > 0.0
        compile_before_resume = {
            worker["name"]: worker["compile_s"] for worker in workers
        }
        with SupervisorClient.from_config(config_path) as client:
            client.register(conformance_id, conformance)
            for task_id, request in requests.items():
                client.submit(conformance_id, task_id, request)
            resumed = {
                client.receive(conformance_id, timeout_s=30.0)[0]
                for _ in requests
            }
            for task_id in resumed:
                client.acknowledge(conformance_id, task_id)
            client.release(conformance_id)
        compile_after_resume = {
            worker["name"]: worker["compile_s"]
            for worker in _status(config_path)["workers"]
        }
        assert compile_after_resume == compile_before_resume

        # Register a new uncompiled session, kill its active worker, and
        # observe the unchanged task complete after an automatic replacement.
        session_id = "worker-loss"
        loss_sampler = UniDimSliceSampler(model=model, num_slices=50)
        session = WorkerSession(
            model=model,
            sampler=loss_sampler,
            args=(),
            params=None,
        )
        request = ConstrainedSampleRequest(
            keys=jax.random.split(jax.random.PRNGKey(22), 1),
            valid=jnp.asarray([True]),
            log_L_constraints=jnp.asarray([-1.0]),
            seed_points=SeedPoint(
                U0=jnp.asarray([0.5]),
                log_L0=jnp.asarray([-0.0625]),
            ),
            sampler_data=None,
        )
        with SupervisorClient.from_config(config_path) as client:
            capacities = client.register(session_id, session)
            assert capacities
            client.submit(session_id, 91, request)
            busy = None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and busy is None:
                for worker in _status(config_path)["workers"]:
                    if worker["busy"]:
                        busy = worker
                        break
                if busy is None:
                    time.sleep(0.01)
            assert busy is not None, "Task completed before worker loss was observed."
            os.kill(busy["process_id"], signal.SIGTERM)
            task_id, batch = client.receive(session_id, timeout_s=30.0)
            assert task_id == 91
            assert float(batch.log_likelihoods[0]) > -1.0
            client.acknowledge(session_id, task_id)
            client.release(session_id)

        healed = _status(config_path)
        assert len(healed["workers"]) == 1
        assert healed["workers"][0]["ready"]
        assert healed["workers"][0]["process_id"] != busy["process_id"]

        # A live stack rejects changed behavior rather than spawning another
        # owner, while `down` still finds the old endpoint through its manifest.
        _write_config(config_path, task_timeout_s=31.0)
        drift = _cli(config_path, "up", check=False)
        assert drift.returncode == 1
        assert "different configuration" in drift.stderr
    finally:
        stopped = _cli(config_path, "down", check=False)
        assert stopped.returncode == 0, stopped.stderr

    second_down = json.loads(_cli(config_path, "down").stdout)
    assert second_down["idempotent"] is True


def test_node_joins_runs_restarts_and_drains_over_tcp(tmp_path):
    coordinator, node = _write_network_configs(tmp_path)
    _cli(coordinator, "up")
    try:
        assert _status(coordinator)["workers"] == []
        _cli(node, "up")
        try:
            deadline = time.monotonic() + 15.0
            remote = None
            while time.monotonic() < deadline:
                workers = _status(coordinator)["workers"]
                remote = next(
                    (
                        worker for worker in workers
                        if worker["worker_id"] == "remote/cpu-0"
                        and worker["ready"]
                    ),
                    None,
                )
                if remote is not None:
                    break
                time.sleep(0.05)
            assert remote is not None

            # The node supervisor owns desired device state. An abrupt worker
            # loss creates a fresh instance with the same logical name, and
            # the coordinator advances the lease generation rather than
            # permanently starving that device from the pool.
            old_process_id = remote["process_id"]
            old_generation = remote["lease_generation"]
            os.kill(old_process_id, signal.SIGKILL)
            deadline = time.monotonic() + 15.0
            replacement = None
            while time.monotonic() < deadline:
                replacement = next(
                    (
                        worker
                        for worker in _status(coordinator)["workers"]
                        if worker["worker_id"] == "remote/cpu-0"
                        and worker["ready"]
                        and worker["process_id"] != old_process_id
                    ),
                    None,
                )
                if replacement is not None:
                    break
                time.sleep(0.05)
            assert replacement is not None
            assert replacement["lease_generation"] > old_generation

            model = make_toy_model()
            sampler = UniDimSliceSampler(model=model, num_slices=2)
            distributed = DistributedNestedSampler(
                model=model,
                coordinator_port=load_runtime_config(
                    coordinator
                ).network.port,
                root_allocation_degree=3,
                delta_K=3,
                max_samples=12,
                initial_capacity=6,
                sampler=sampler,
            )
            checkpoint = distributed.run_until_goal(
                lambda state: int(state.goal_loop_iter) >= 1,
                depth_cond=TerminationCondition(dlogZ=jnp.asarray(0.5)),
                key=jax.random.PRNGKey(77),
            )
            assert not checkpoint.pending
            assert int(checkpoint.state.num_samples) > 3

            # Removing the only node is an operational pause, not a failed
            # scientific run. Keep the same thread and checkpoint alive beyond
            # its coordinator-health timeout, then let a fresh node instance
            # register and consume the coordinator's queued work.
            slow_sampler = UniDimSliceSampler(model=model, num_slices=100)
            starved = DistributedNestedSampler(
                model=model,
                coordinator_port=load_runtime_config(
                    coordinator
                ).network.port,
                root_allocation_degree=3,
                delta_K=3,
                max_samples=60,
                initial_capacity=12,
                sampler=slow_sampler,
                receive_timeout_s=0.1,
            )
            outcome = {}

            def run_while_capacity_changes():
                try:
                    outcome["checkpoint"] = starved.run_until_goal(
                        lambda state: int(state.goal_loop_iter) >= 3,
                        depth_cond=TerminationCondition(
                            dlogZ=jnp.asarray(0.5)
                        ),
                        key=jax.random.PRNGKey(78),
                    )
                except BaseException as exc:  # noqa: BLE001
                    outcome["error"] = exc

            science = threading.Thread(
                target=run_while_capacity_changes,
                name="jaxns-starvation-test",
            )
            science.start()
            deadline = time.monotonic() + 15.0
            busy = False
            while time.monotonic() < deadline and not busy:
                busy = any(
                    worker["worker_id"] == "remote/cpu-0"
                    and worker["busy"]
                    for worker in _status(coordinator)["workers"]
                )
                if not busy:
                    time.sleep(0.02)
            assert busy, "Scientific work never reached the only remote node."

            stopped = _cli(node, "down", check=False)
            assert stopped.returncode == 0, stopped.stderr
            # This exceeds receive_timeout_s by a wide margin. The old path
            # raised here; the new path must retain and wait on queued tasks.
            time.sleep(0.5)
            assert science.is_alive()
            _cli(node, "up")
            science.join(timeout=45.0)
            assert not science.is_alive()
            assert "error" not in outcome
            recovered = outcome["checkpoint"]
            assert not recovered.pending
            assert int(recovered.state.goal_loop_iter) >= 3
        finally:
            stopped = _cli(node, "down", check=False)
            assert stopped.returncode == 0, stopped.stderr

        deadline = time.monotonic() + 5.0
        state = None
        while time.monotonic() < deadline:
            workers = _status(coordinator)["workers"]
            state = next(
                worker["state"]
                for worker in workers
                if worker["worker_id"] == "remote/cpu-0"
            )
            if state == "dropped":
                break
            time.sleep(0.05)
        assert state == "dropped"

    finally:
        stopped = _cli(coordinator, "down", check=False)
        assert stopped.returncode == 0, stopped.stderr
