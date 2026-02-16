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


def test_rpc():
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


def test_rpc_stress():
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


def test_sticky_and_fair_assignment():
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


def test_worker_death_reassigns():
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


def test_epoch_change_client_resync():
    """
    Scenario:
      1) Start LB (epoch=E1) and 2 workers; establish a working pool on the client.
      2) Stop LB only, start a fresh LB (epoch=E2) on the same addresses.
      3) Make another RPC. The client should detect the epoch change on the ASSIGN,
         drop its pool, re-request, and complete successfully under E2.

    This asserts the client-side epoch fencing logic:
      - On ASSIGN with new epoch, client drops old sockets/pool and re-REQUESTs.
      - The call finishes successfully (no hang / no resend of old frames to stale sockets).
    """
    # Unique control channels for both managers
    ctl_pub_addr_lb1 = create_random_control_address()
    ack_rep_addr_lb1 = create_random_ack_address()
    ctl_pub_addr_workers = create_random_control_address()
    ack_rep_addr_workers = create_random_ack_address()

    # Fixed TCP for LB <-> client/service so workers survive LB bounce
    frontend_addr = "tcp://127.0.0.1:5555"
    backend_addr = "tcp://127.0.0.1:5556"

    # Start workers in their own manager (so they outlive the first LB)
    workers = [
        ServiceActor(ctl_pub_addr=ctl_pub_addr_workers, ack_rep_addr=ack_rep_addr_workers, backend_addr=backend_addr),
        ServiceActor(ctl_pub_addr=ctl_pub_addr_workers, ack_rep_addr=ack_rep_addr_workers, backend_addr=backend_addr),
    ]
    workers_mgr = ProcessManager(workers, ctl_pub_addr=ctl_pub_addr_workers, ack_rep_addr=ack_rep_addr_workers, profile=False)
    workers_mgr.start_all()

    # Start LB #1 (epoch E1)
    lb1 = LoadBalancer(ctl_pub_addr=ctl_pub_addr_lb1, ack_rep_addr=ack_rep_addr_lb1,
                       frontend_addr=frontend_addr, backend_addr=backend_addr)
    lb_mgr1 = ProcessManager([lb1], ctl_pub_addr=ctl_pub_addr_lb1, ack_rep_addr=ack_rep_addr_lb1, profile=False)
    lb_mgr1.start_all()

    try:
        # Establish client pool and known epoch with an initial successful call
        with Client(ident="C", frontend_addr=frontend_addr, assign_timeout_ms=5_000) as C:
            first = C.echo("warmup")
            assert first.startswith("warmup|pid=")

            # Bounce the LB only — workers stay up
            lb_mgr1.stop_all()

            # Start LB #2 (epoch E2) on the same TCP addresses
            ctl_pub_addr_lb2 = create_random_control_address()
            ack_rep_addr_lb2 = create_random_ack_address()
            lb2 = LoadBalancer(ctl_pub_addr=ctl_pub_addr_lb2, ack_rep_addr=ack_rep_addr_lb2,
                               frontend_addr=frontend_addr, backend_addr=backend_addr)
            lb_mgr2 = ProcessManager([lb2], ctl_pub_addr=ctl_pub_addr_lb2, ack_rep_addr=ack_rep_addr_lb2, profile=False)
            lb_mgr2.start_all()
            try:
                # Now make a new call. The client should detect epoch change on ASSIGN,
                # drop its pool, send a fresh REQUEST, and succeed.
                out = C.echo("after-restart")
                assert out.startswith("after-restart|pid=")

                # Optional: do one more to ensure the resynced pool works fine
                out2 = C.echo("still-good")
                assert out2.startswith("still-good|pid=")
            finally:
                lb_mgr2.stop_all()
    finally:
        workers_mgr.stop_all()
