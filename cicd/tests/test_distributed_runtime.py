"""Local-process lifecycle and real constrained-sampling integration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import jax
import pytest
from jax import numpy as jnp

from cicd.tests.distributed_support import make_toy_model
from jaxns.cli import _stop_started_process
from jaxns.constrained_sampler import (
    ConstrainedSampleRequest,
    UniDimSliceSampler,
)
from jaxns.distributed_core import (
    DistributedNestedSampler,
    WorkerSession,
)
from jaxns.runtime_client import SupervisorClient
from jaxns.runtime_config import WorkerConfig, load_runtime_config
from jaxns.runtime_protocol import (
    CAPACITY,
    MAX_HEADER_BYTES,
    PROTOCOL_VERSION,
    decode_header,
)
from jaxns.runtime_supervisor import (
    SessionRecord,
    Supervisor,
    TaskRecord,
    WorkerRecord,
)
from jaxns.runtime_transport import create_curve_certificate
from jaxns.samples import SeedPoint
from jaxns.termination_condition import TerminationCondition

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_config(path: Path, *, task_timeout_s: float = 30.0) -> None:
    path.write_text(
        f"""
[runtime]
stack_id = "distributed-test"
runtime_dir = "runtime"
log_dir = "logs"
startup_timeout_s = 30
shutdown_timeout_s = 15
task_timeout_s = {task_timeout_s}

[[workers]]
name = "vector"
platform = "cpu"
device = 0
batch_size = 2

[[workers]]
name = "scalar"
platform = "cpu"
device = 0
batch_size = 1
count = 2
""".strip()
        + "\n",
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


def _free_tcp_endpoint() -> str:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _write_network_configs(directory: Path) -> tuple[Path, Path]:
    endpoint = _free_tcp_endpoint()
    server_public, server_secret = create_curve_certificate(
        directory / "server-keys",
        "coordinator",
    )
    node_public, node_secret = create_curve_certificate(
        directory / "node-keys",
        "worker-node",
    )
    authorized = directory / "authorized"
    authorized.mkdir()
    shutil.copy2(node_public, authorized / node_public.name)
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
listen = "{endpoint}"
advertise = "{endpoint}"
server_public_key = "{server_public}"
server_secret_key = "{server_secret}"
authorized_clients = "{authorized}"
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
server_public_key = "{server_public}"
client_public_key = "{node_public}"
client_secret_key = "{node_secret}"

[[workers]]
name = "cpu"
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
    assert [worker.name for worker in config.workers] == [
        "vector",
        "scalar-0",
        "scalar-1",
    ]
    assert [worker.batch_size for worker in config.workers] == [2, 1, 1]
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
            'name = "vector"',
            'name = "../vector"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="may contain only"):
        load_runtime_config(malformed)

    bounded = object.__new__(SupervisorClient)
    bounded._max_payload_bytes = 2
    with pytest.raises(ValueError, match="configured maximum"):
        bounded._check_payload(b"too large")
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
                "import sys; import jaxns.runtime_supervisor; "
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
    supervisor._report_missing_capacity = lambda batch_size: None
    header = {
        "session_id": "session",
        "task_id": 5,
        "batch_group": "direction-state",
    }

    supervisor._queue_task(b"client", header, b"request")
    supervisor._queue_task(b"client", header, b"request")
    assert list(session.queue) == [5]
    assert list(session.tasks) == [5]

    task = session.tasks[5]
    task.state = "completed"
    task.result_header = b"result-header"
    task.result_payload = b"result-payload"
    supervisor._queue_task(b"client", header, b"request")
    assert sent == [(b"client", b"result-header", b"result-payload")]
    supervisor._acknowledge(b"client", header)
    assert not session.tasks


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

    config = WorkerConfig("first", "cpu", "0", 1)
    active = WorkerRecord(config, Process(), None, b"worker:first")
    active.ready = True
    active.task = (("session", 6),)
    active.started_s = time.monotonic() - 2.0
    spare_config = WorkerConfig("second", "cpu", "0", 1)
    spare = WorkerRecord(spare_config, Process(), None, b"worker:second")
    spare.ready = True
    payload = b"scientific-request"
    task = TaskRecord(
        6,
        "direction-state",
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
    worker_config = WorkerConfig("worker", "cpu", "0", 1)
    worker = WorkerRecord(worker_config, None, None, b"worker")
    worker.ready = True
    worker.last_heartbeat_s = time.monotonic() - 0.21
    worker.task = (("session", 4),)
    payload = b"retry-stable"
    task = TaskRecord(
        4,
        "direction-state",
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
    supervisor._expire_workers()

    assert worker.dropped
    assert not worker.ready
    assert list(session.queue) == [4]
    assert task.payload == payload


def test_supervisor_rotates_dispatch_fairly_between_sessions():
    config = WorkerConfig("worker", "cpu", "0", 1)
    worker = WorkerRecord(config, object(), None, b"worker")
    worker.ready = True
    worker.registered.update(("first", "second"))
    first = SessionRecord("first", "one", b"model", b"client-one")
    second = SessionRecord("second", "two", b"model", b"client-two")
    for session, task_id in ((first, 1), (second, 2)):
        task = TaskRecord(task_id, "direction-state", str(task_id), b"payload")
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
        WorkerConfig("worker", "cpu", "0", 1),
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
        WorkerConfig("compatible", "cpu", "0", 2),
        None,
        None,
        b"compatible",
    )
    compatible.ready = True
    compatible.registered.add("session")
    incompatible = WorkerRecord(
        WorkerConfig("incompatible", "gpu", "0", 8),
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


def test_real_pool_runs_scalar_vmap_retries_and_cli_lifecycle(tmp_path):
    config_path = tmp_path / "workers.toml"
    _write_config(config_path)
    _cli(config_path, "up")
    try:
        config = load_runtime_config(config_path)
        manifest = json.loads(config.manifest.read_text(encoding="utf-8"))
        assert manifest["protocol_version"] == PROTOCOL_VERSION
        assert manifest["transports"] == ["ipc"]
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
            config=config_path,
            root_allocation_degree=4,
            delta_K=4,
            max_samples=16,
            initial_capacity=8,
            sampler=sampler,
        )
        checkpoint = distributed.run_until_goal(
            lambda state: int(state.goal_loop_iter) >= 2,
            depth_cond=TerminationCondition(dlogZ=jnp.asarray(0.5)),
            key=jax.random.PRNGKey(21),
        )

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

        # Exercise both worker specializations explicitly. The scheduler is
        # allowed to leave capacity idle once all scientific gaps are already
        # reserved, so pool conformance does not rely on incidental gap width.
        conformance_id = "scalar-vector"
        # The local lambda proves one-time model/session registration supports
        # notebook and closure code rather than only importable module symbols.
        conformance = WorkerSession(
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
        assert any(
            worker["compile_s"] > 0.0
            for worker in workers
            if worker["batch_size"] == 1
        )
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

        # Register a new uncompiled session, kill its active scalar worker,
        # and observe the unchanged task complete on the other scalar worker.
        session_id = "worker-loss"
        loss_sampler = UniDimSliceSampler(model=model, num_slices=50)
        session = WorkerSession(sampler=loss_sampler, args=(), params=None)
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
                    if worker["batch_size"] == 1 and worker["busy"]:
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

        degraded = _status(config_path)
        assert sum(
            worker["ready"] and worker["batch_size"] == 1
            for worker in degraded["workers"]
        ) == 1
        assert any(
            worker["exit_code"] is not None
            for worker in degraded["workers"]
        )

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


def test_authenticated_node_joins_runs_and_drains_over_tcp(tmp_path):
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
                        if worker["worker_id"] == "remote/cpu"
                        and worker["ready"]
                    ),
                    None,
                )
                if remote is not None:
                    break
                time.sleep(0.05)
            assert remote is not None

            model = make_toy_model()
            sampler = UniDimSliceSampler(model=model, num_slices=2)
            distributed = DistributedNestedSampler(
                model=model,
                config=coordinator,
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
                if worker["worker_id"] == "remote/cpu"
            )
            if state == "dropped":
                break
            time.sleep(0.05)
        assert state == "dropped"

        # A certificate that was not installed in the coordinator's allowlist
        # never reaches the pickle-bearing worker protocol.
        unauthorized_public, unauthorized_secret = create_curve_certificate(
            tmp_path / "unauthorized-keys",
            "unauthorized",
        )
        node_config = load_runtime_config(node)
        bad_node = tmp_path / "unauthorized.toml"
        bad_node.write_text(
            node.read_text(encoding="utf-8")
            .replace('node_id = "remote"', 'node_id = "unauthorized"')
            .replace("startup_timeout_s = 30", "startup_timeout_s = 2")
            .replace(
                str(node_config.network.client_public_key),
                str(unauthorized_public),
            )
            .replace(
                str(node_config.network.client_secret_key),
                str(unauthorized_secret),
            ),
            encoding="utf-8",
        )
        rejected = _cli(bad_node, "up", check=False)
        assert rejected.returncode == 1
        assert all(
            worker["node_id"] != "unauthorized"
            for worker in _status(coordinator)["workers"]
        )
    finally:
        stopped = _cli(coordinator, "down", check=False)
        assert stopped.returncode == 0, stopped.stderr
