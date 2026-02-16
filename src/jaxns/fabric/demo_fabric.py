import contextlib
import os
import time
from abc import ABC, abstractmethod

from jaxns.fabric.process_manager import create_random_control_address, create_random_ack_address, ProcessManager
from jaxns.fabric.zmq_p2p import RPCActor, ZMQRPCClient, RPCClientMeta, LoadBalancer
from jaxns.nested_samplers.logging import jaxns_logger


class AbstractService(ABC):
    @abstractmethod
    def echo(self, x: str | int) -> str:
        ...


class ServiceImpl(AbstractService):

    def echo(self, x: str | int) -> str:
        if x == "crash_once":
            # simulate hard crash
            os._exit(1)
        if isinstance(x, int):
            time.sleep(x / 1000.)
        return f"{x}|pid={os.getpid()}"


class ServiceActor(RPCActor):
    def __init__(self, *a, behavior="normal", **kw):
        super().__init__(*a, **kw)
        self._behavior = behavior

    @contextlib.contextmanager
    def yield_service(self):
        svc = ServiceImpl()
        try:
            yield svc
        finally:
            pass


class Client(ZMQRPCClient, AbstractService, metaclass=RPCClientMeta):
    ...


def _test_rpc():
    ctl_pub_addr = create_random_control_address()
    ack_rep_addr = create_random_ack_address()

    # Use unique IPC endpoints for each run
    frontend_addr = "tcp://127.0.0.1:5555"
    backend_addr = "tcp://127.0.0.1:5556"

    load_balancer = LoadBalancer(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, frontend_addr=frontend_addr,
                                 backend_addr=backend_addr)

    workers = [
        ServiceActor(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, backend_addr=backend_addr),
        ServiceActor(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, backend_addr=backend_addr),
        ServiceActor(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, backend_addr=backend_addr),
    ]

    actors = [load_balancer, *workers]

    mgr = ProcessManager(actors, ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, profile=True)

    mgr.start_all()
    try:
        # Add a finite timeout so tests fail fast on regressions instead of hanging forever
        with (
            Client(ident='client1', frontend_addr=frontend_addr) as client1,
            Client(ident='client2', frontend_addr=frontend_addr) as client2
        ):
            for i in range(15):
                assert client1.echo(f'hello {i} for client 1').startswith(f'hello {i} for client 1|pid=')
                assert client2.echo(f'hello {i} for client 2').startswith(f'hello {i} for client 2|pid=')
    finally:
        mgr.stop_all()
        mgr.print_tracebacks()


def _test_rpc_stress():
    ctl_pub_addr = create_random_control_address()
    ack_rep_addr = create_random_ack_address()

    # Use unique IPC endpoints for each run
    frontend_addr = "tcp://127.0.0.1:5555"
    backend_addr = "tcp://127.0.0.1:5556"

    load_balancer = LoadBalancer(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, frontend_addr=frontend_addr,
                                 backend_addr=backend_addr)

    workers = [
        ServiceActor(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, backend_addr=backend_addr),
        ServiceActor(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, backend_addr=backend_addr),
    ]

    actors = [load_balancer, *workers]

    mgr = ProcessManager(actors, ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, profile=True)

    mgr.start_all()
    try:
        # Add a finite timeout so tests fail fast on regressions instead of hanging forever
        with (
            Client(ident='client1', frontend_addr=frontend_addr) as client1,
            Client(ident='client2', frontend_addr=frontend_addr) as client2
        ):
            for i in range(100):
                assert client1.echo(f'hello {i} for client 1').startswith(f'hello {i} for client 1|pid=')
                assert client2.echo(f'hello {i} for client 2').startswith(f'hello {i} for client 2|pid=')
    finally:
        mgr.stop_all()
        mgr.print_tracebacks()


# ------------------------------
# Test service + client scaffolds
# ------------------------------


# ------------------------------
# Helpers
# ------------------------------

def start_cluster(num_workers=2, profile=False):
    ctl_pub_addr = create_random_control_address()
    ack_rep_addr = create_random_ack_address()
    frontend_addr = "tcp://127.0.0.1:5555"
    backend_addr = "tcp://127.0.0.1:5556"

    lb = LoadBalancer(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, frontend_addr=frontend_addr,
                      backend_addr=backend_addr)

    workers = []
    for b in range(num_workers):
        workers.append(ServiceActor(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr,
                                    backend_addr=backend_addr))

    actors = [lb, *workers]
    mgr = ProcessManager(actors, ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, profile=profile)
    mgr.start_all()
    return mgr, frontend_addr


# ============================================================
# 1) Sticky + Fair assignment
# ============================================================

def _test_sticky_and_fair_assignment():
    mgr, fe = start_cluster(num_workers=2)
    try:
        with Client(ident="A", frontend_addr=fe) as A, \
                Client(ident="B", frontend_addr=fe) as B:
            # First calls should land on different workers, if workers stay busy
            a0 = A.echo(1000)
            b0 = B.echo(1000)
            pidA0 = a0.split("|pid=")[1]
            pidB0 = b0.split("|pid=")[1]
            assert pidA0 != pidB0

            # Subsequent calls should stick
            A_pids = [A.echo(f"a{i}").split("|pid=")[1] for i in range(1, 8)]
            B_pids = [B.echo(f"b{i}").split("|pid=")[1] for i in range(1, 8)]
            assert all(p == pidA0 for p in A_pids)
            assert all(p == pidB0 for p in B_pids)
    finally:
        mgr.stop_all()
        mgr.print_tracebacks()


# ============================================================
# 5) Worker dies mid-flight -> LB reaps and reassigns
# ============================================================
def _test_worker_death_reassigns():
    # Two workers, one will crash on first call; short worker_dead_ms for quick reap
    mgr, fe = start_cluster(num_workers=2)
    try:
        with Client(ident="A", frontend_addr=fe, assign_timeout_ms=10000) as A:
            # A hits the crashing worker: keepalive should fail, and client should be reassigned.
            jaxns_logger.info("Client A: first call should crash a worker")
            try:
                A.echo("crash_once")
                assert False
            except TimeoutError:
                assert True
            jaxns_logger.info("Client A: second call should succeed on other worker.")
            out = A.echo("ok")
            assert out.startswith("ok|pid=")


    finally:
        mgr.stop_all()
        mgr.print_tracebacks()


if __name__ == '__main__':
    # jaxns_logger.info("Running Test 1")
    # _test_rpc()
    # jaxns_logger.info("Running Test 2")
    # _test_rpc_stress()
    # jaxns_logger.info("Running Test 3")
    # _test_sticky_and_fair_assignment()
    jaxns_logger.info("Running Test 4")
    _test_worker_death_reassigns()
