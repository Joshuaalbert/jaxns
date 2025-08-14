import contextlib
from abc import ABC, abstractmethod

from jaxns.distributed.process_manager import create_random_control_address, create_random_ack_address, ProcessManager
from jaxns.distributed.zmq_loadbalancer import LoadBalancer
from jaxns.distributed.zmq_rpc import RPCActor, ZMQRPCClient, RPCClientMeta


class AbstractService(ABC):

    @abstractmethod
    def echo(self, x):
        ...


class ServiceImpl(AbstractService):

    def echo(self, x):
        return x


class ServiceActor(RPCActor):

    @contextlib.contextmanager
    def yield_service(self):
        service = ServiceImpl()
        try:
            yield service
        finally:
            pass


class Client(ZMQRPCClient, AbstractService, metaclass=RPCClientMeta):
    ...


def test_rpc():
    ctl_pub_addr = create_random_control_address()
    ack_rep_addr = create_random_ack_address()
    backend_addr = "ipc://backend.ipc"
    frontend_addr = "ipc://frontend.ipc"
    load_balancer = LoadBalancer(
        ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr,
        frontend_addr=frontend_addr,
        backend_addr=backend_addr
    )

    workers = [
        ServiceActor(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, backend_addr=backend_addr),
        ServiceActor(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, backend_addr=backend_addr),
        ServiceActor(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, backend_addr=backend_addr),
    ]

    actors = [load_balancer, *workers]

    mgr = ProcessManager(actors, ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr, profile=True)

    mgr.start_all()
    try:
        with (Client(ident='client1', frontend_addr=frontend_addr) as client1,
              Client(ident='client2', frontend_addr=frontend_addr) as client2):

            for i in range(10):
                assert client1.echo(f'hello {i} for client 1') == f'hello {i} for client 1'
                assert client2.echo(f'hello {i} for client 2') == f'hello {i} for client 2'

    finally:
        mgr.stop_all()
        mgr.print_tracebacks()
