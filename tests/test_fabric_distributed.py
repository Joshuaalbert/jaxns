from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import pytest

from jaxns.fabric.node import (
    NodeClient,
    RemoteNodeEvaluator,
    build_scheduler_process_manager,
    build_worker_process_manager,
    local_node_evaluator,
    load_service_factory,
)
from jaxns.fabric.process_manager import (
    create_random_ack_address,
    create_random_control_address,
)
from jaxns.fabric.scheduler_launcher import build_argument_parser as build_scheduler_argument_parser
from jaxns.fabric.worker_launcher import build_argument_parser as build_worker_argument_parser
from jaxns.fabric.zmq_p2p import get_free_port
from tests.distributed_support import make_offset_node, make_slow_identity_node


class FabricAddresses(NamedTuple):
    ctl_pub_addr: str
    ack_rep_addr: str
    frontend_addr: str
    backend_addr: str


def make_fabric_addresses() -> FabricAddresses:
    frontend_port = get_free_port()
    backend_port = get_free_port()
    return FabricAddresses(
        ctl_pub_addr=create_random_control_address(),
        ack_rep_addr=create_random_ack_address(),
        frontend_addr=f"tcp://127.0.0.1:{frontend_port}",
        backend_addr=f"tcp://127.0.0.1:{backend_port}",
    )


def test_load_service_factory_supports_import_strings_and_callables():
    imported_factory = load_service_factory("tests.distributed_support:make_offset_node")
    direct_factory = load_service_factory(make_offset_node)

    assert imported_factory().evaluate(2.0) == pytest.approx(3.0)
    assert direct_factory().evaluate(2.0) == pytest.approx(3.0)


def test_launcher_argument_parsers_accept_expected_arguments():
    scheduler_args = build_scheduler_argument_parser().parse_args([
        "--ctl-pub-addr", "ipc://control",
        "--ack-rep-addr", "ipc://ack",
        "--frontend-addr", "tcp://127.0.0.1:6001",
        "--backend-addr", "tcp://127.0.0.1:6002",
    ])
    worker_args = build_worker_argument_parser().parse_args([
        "--ctl-pub-addr", "ipc://control",
        "--ack-rep-addr", "ipc://ack",
        "--backend-addr", "tcp://127.0.0.1:6002",
        "--service-factory", "tests.distributed_support:make_offset_node",
        "--num-workers", "3",
    ])

    assert scheduler_args.frontend_addr == "tcp://127.0.0.1:6001"
    assert scheduler_args.backend_addr == "tcp://127.0.0.1:6002"
    assert worker_args.service_factory == "tests.distributed_support:make_offset_node"
    assert worker_args.num_workers == 3


def test_build_process_managers_create_expected_actor_counts():
    addresses = make_fabric_addresses()

    scheduler_mgr = build_scheduler_process_manager(addresses=addresses, profile=False)
    worker_mgr = build_worker_process_manager(
        addresses=addresses,
        service_factory=make_offset_node,
        num_workers=3,
        profile=False,
    )

    assert len(scheduler_mgr.actors) == 1
    assert len(worker_mgr.actors) == 3


def test_local_node_evaluator_stops_started_managers_when_worker_start_fails(monkeypatch):
    class MockManager:
        def __init__(self, start_error=None):
            self.start_error = start_error
            self.calls = []

        def start_all(self):
            self.calls.append("start_all")
            if self.start_error is not None:
                raise self.start_error

        def stop_all(self):
            self.calls.append("stop_all")

        def print_tracebacks(self):
            self.calls.append("print_tracebacks")

    scheduler_mgr = MockManager()
    worker_mgr = MockManager(start_error=RuntimeError("worker startup failed"))

    monkeypatch.setattr(
        "jaxns.fabric.node.build_scheduler_process_manager",
        lambda **kwargs: scheduler_mgr,
    )
    monkeypatch.setattr(
        "jaxns.fabric.node.build_worker_process_manager",
        lambda **kwargs: worker_mgr,
    )
    monkeypatch.setattr(
        "jaxns.fabric.node.create_local_fabric_addresses",
        make_fabric_addresses,
    )

    with pytest.raises(RuntimeError, match="worker startup failed"):
        with local_node_evaluator(service_factory=make_offset_node):
            raise AssertionError("The evaluator context should not be entered when worker startup fails.")

    assert scheduler_mgr.calls == ["start_all", "stop_all", "print_tracebacks"]
    assert worker_mgr.calls == ["start_all", "stop_all", "print_tracebacks"]


def test_remote_node_evaluator_closes_stale_client_when_thread_id_is_reused(monkeypatch):
    class MockNodeClient:
        next_id = 0

        def __init__(self, ident, frontend_addr, assign_timeout_ms=None, retries=0):
            del frontend_addr, assign_timeout_ms, retries
            self.ident = ident
            self.client_id = MockNodeClient.next_id
            MockNodeClient.next_id += 1
            events.append(("init", self.client_id, self.ident))

        def __enter__(self):
            events.append(("enter", self.client_id, self.ident))
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            del exc_type, exc_val, exc_tb
            events.append(("exit", self.client_id, self.ident))

    events = []
    monkeypatch.setattr("jaxns.fabric.node.NodeClient", MockNodeClient)
    monkeypatch.setattr("jaxns.fabric.node.threading.get_ident", lambda: 17)

    evaluator = RemoteNodeEvaluator(frontend_addr="tcp://127.0.0.1:6553", ident_prefix="reuse")

    first_client = evaluator._get_client()
    evaluator._thread_local = threading.local()
    second_client = evaluator._get_client()

    assert first_client is not second_client
    assert events == [
        ("init", 0, "reuse-17"),
        ("enter", 0, "reuse-17"),
        ("exit", 0, "reuse-17"),
        ("init", 1, "reuse-17"),
        ("enter", 1, "reuse-17"),
    ]

    evaluator.close()

    assert events == [
        ("init", 0, "reuse-17"),
        ("enter", 0, "reuse-17"),
        ("exit", 0, "reuse-17"),
        ("init", 1, "reuse-17"),
        ("enter", 1, "reuse-17"),
        ("exit", 1, "reuse-17"),
    ]


def test_worker_and_scheduler_managers_support_rpc_roundtrip():
    addresses = make_fabric_addresses()
    scheduler_mgr = build_scheduler_process_manager(addresses=addresses, profile=False)
    worker_mgr = build_worker_process_manager(
        addresses=addresses,
        service_factory=make_offset_node,
        num_workers=2,
        profile=False,
    )

    scheduler_mgr.start_all()
    worker_mgr.start_all()
    try:
        with NodeClient(ident="fabric-roundtrip", frontend_addr=addresses.frontend_addr) as client:
            assert client.evaluate(2.5) == pytest.approx(3.5)
            assert client.evaluate(-1.0) == pytest.approx(0.0)
    finally:
        worker_mgr.stop_all()
        worker_mgr.print_tracebacks()
        scheduler_mgr.stop_all()
        scheduler_mgr.print_tracebacks()


def test_remote_node_evaluator_supports_parallel_requests():
    addresses = make_fabric_addresses()
    scheduler_mgr = build_scheduler_process_manager(addresses=addresses, profile=False)
    worker_mgr = build_worker_process_manager(
        addresses=addresses,
        service_factory=make_slow_identity_node,
        num_workers=2,
        profile=False,
    )

    scheduler_mgr.start_all()
    worker_mgr.start_all()
    try:
        with RemoteNodeEvaluator(frontend_addr=addresses.frontend_addr, ident_prefix="parallel-eval") as evaluator:
            start_time = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(evaluator.evaluate, [1.0, 2.0]))
            elapsed_s = time.perf_counter() - start_time

        assert results == pytest.approx([1.0, 2.0])
        assert elapsed_s < 0.9
    finally:
        worker_mgr.stop_all()
        worker_mgr.print_tracebacks()
        scheduler_mgr.stop_all()
        scheduler_mgr.print_tracebacks()


def test_remote_node_evaluator_supports_repeated_sequential_then_parallel_batches():
    addresses = make_fabric_addresses()
    scheduler_mgr = build_scheduler_process_manager(addresses=addresses, profile=False)
    worker_mgr = build_worker_process_manager(
        addresses=addresses,
        service_factory=make_offset_node,
        num_workers=2,
        profile=False,
    )

    scheduler_mgr.start_all()
    worker_mgr.start_all()
    try:
        with RemoteNodeEvaluator(frontend_addr=addresses.frontend_addr, ident_prefix="batched-eval") as evaluator:
            for value in range(4):
                assert evaluator.evaluate(float(value)) == pytest.approx(float(value + 1))

            for batch_idx in range(8):
                inputs = [float(100 + 2 * batch_idx), float(101 + 2 * batch_idx)]
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(executor.map(evaluator.evaluate, inputs))
                assert results == pytest.approx([value + 1.0 for value in inputs])
    finally:
        worker_mgr.stop_all()
        worker_mgr.print_tracebacks()
        scheduler_mgr.stop_all()
        scheduler_mgr.print_tracebacks()


def test_pyproject_declares_fabric_console_scripts():
    pyproject_text = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()

    assert 'jaxns-fabric-scheduler = "jaxns.fabric.scheduler_launcher:main"' in pyproject_text
    assert 'jaxns-fabric-worker = "jaxns.fabric.worker_launcher:main"' in pyproject_text
