import zmq

from jaxns.distributed.process_manager import create_random_control_address, create_random_ack_address, ProcessManager
from jaxns.distributed.zmq_actor import ZMQActor


class MockActorCleanExit(ZMQActor):

    def __init__(
            self,
            ctl_pub_addr: str,
            ack_rep_addr: str
    ):
        ZMQActor.__init__(self, ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr)
        self.ctl_pub_addr = ctl_pub_addr

    def run(self):
        ctl = self.new_socket(zmq.SUB, connect=self.ctl_pub_addr)
        ctl.setsockopt_string(zmq.SUBSCRIBE, "")
        self.ack_startup()

        poller = zmq.Poller()
        poller.register(ctl, zmq.POLLIN)
        try:
            while True:

                socks = dict(poller.poll(3000))
                if ctl in socks:
                    msg = ctl.recv()
                    if msg == b"TERMINATE":
                        break

                break
        finally:
            poller.unregister(ctl)



def test_basic_shutdown_logic():
    ctl_pub_addr = create_random_control_address()
    ack_rep_addr = create_random_ack_address()
    # For each actor spin it up in a single process monitor, then shut it down with wait_all(timeout) and also with stop_all()
    actors = [
        MockActorCleanExit(
            ctl_pub_addr=ctl_pub_addr,
            ack_rep_addr=ack_rep_addr
        )
    ]
    mgr = ProcessManager(
        ctl_pub_addr=ctl_pub_addr,
        ack_rep_addr=ack_rep_addr,
        actors=actors
    )

    mgr.start_all()
    mgr.wait_all()
    for actor_proc in mgr.actor_procs:
        assert not actor_proc.proc.is_alive()
        assert actor_proc.done

class MockActorException(ZMQActor):

    def __init__(
            self,
            ctl_pub_addr: str,
            ack_rep_addr: str
    ):
        ZMQActor.__init__(self, ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr)
        self.ctl_pub_addr = ctl_pub_addr

    def run(self):
        ctl = self.new_socket(zmq.SUB, connect=self.ctl_pub_addr)
        ctl.setsockopt_string(zmq.SUBSCRIBE, "")
        self.ack_startup()

        poller = zmq.Poller()
        poller.register(ctl, zmq.POLLIN)
        try:
            while True:

                socks = dict(poller.poll(2000))
                if ctl in socks:
                    msg = ctl.recv()
                    if msg == b"TERMINATE":
                        break

                raise ValueError("Simulated exception for testing shutdown logic")
        finally:
            poller.unregister(ctl)


def test_exception_shutdown_logic():
    ctl_pub_addr = create_random_control_address()
    ack_rep_addr = create_random_ack_address()
    # For each actor spin it up in a single process monitor, then shut it down with wait_all(timeout) and also with stop_all()
    actors = [
        MockActorException(
            ctl_pub_addr=ctl_pub_addr,
            ack_rep_addr=ack_rep_addr
        ),
        MockActorCTLShutdown(
            ctl_pub_addr=ctl_pub_addr,
            ack_rep_addr=ack_rep_addr,
        ),
        MockActorCleanExit(
            ctl_pub_addr=ctl_pub_addr,
            ack_rep_addr=ack_rep_addr
        )
    ]
    mgr = ProcessManager(
        ctl_pub_addr=ctl_pub_addr,
        ack_rep_addr=ack_rep_addr,
        actors=actors
    )

    mgr.start_all()
    mgr.wait_all(stop_all_on_exception=True)
    mgr.close()
    mgr.print_tracebacks()
    assert isinstance(mgr.actor_procs[0].actor.exception, ValueError)
    for actor_proc in mgr.actor_procs:
        assert not actor_proc.proc.is_alive()
        assert actor_proc.done

def test_exception_shutdown_ctl_logic():
    ctl_pub_addr = create_random_control_address()
    ack_rep_addr = create_random_ack_address()
    # For each actor spin it up in a single process monitor, then shut it down with wait_all(timeout) and also with stop_all()
    actors = [
        MockActorException(
            ctl_pub_addr=ctl_pub_addr,
            ack_rep_addr=ack_rep_addr
        ),
        MockActorCTLShutdown(
            ctl_pub_addr=ctl_pub_addr,
            ack_rep_addr=ack_rep_addr,
        )
    ]
    mgr = ProcessManager(
        ctl_pub_addr=ctl_pub_addr,
        ack_rep_addr=ack_rep_addr,
        actors=actors
    )

    mgr.start_all()
    mgr.wait_all(timeout=5, stop_all_on_exception=False)
    assert mgr.actor_procs[0].done
    assert not mgr.actor_procs[0].proc.is_alive()
    assert isinstance(mgr.actor_procs[0].actor.exception, ValueError)

    assert not mgr.actor_procs[1].done
    assert mgr.actor_procs[1].proc.is_alive()

    mgr.stop_all()
    mgr.close()
    for actor_proc in mgr.actor_procs:
        assert not actor_proc.proc.is_alive()
        assert actor_proc.done

    mgr.print_tracebacks()

class MockActorCTLShutdown(ZMQActor):

    def __init__(
            self,
            ctl_pub_addr: str,
            ack_rep_addr: str,
    ):
        ZMQActor.__init__(self, ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr)
        self.ctl_pub_addr = ctl_pub_addr

    def run(self):
        ctl = self.new_socket(zmq.SUB, connect=self.ctl_pub_addr)
        ctl.setsockopt_string(zmq.SUBSCRIBE, "")
        self.ack_startup()

        poller = zmq.Poller()
        poller.register(ctl, zmq.POLLIN)
        try:
            while True:
                socks = dict(poller.poll(30000))
                if ctl in socks:
                    msg = ctl.recv()
                    if msg == b"TERMINATE":
                        break
        finally:
            poller.unregister(ctl)


def test_ctl_shutdown_immediate_logic():
    ctl_pub_addr = create_random_control_address()
    ack_rep_addr = create_random_ack_address()
    # For each actor spin it up in a single process monitor, then shut it down with wait_all(timeout) and also with stop_all()
    actors = [
        MockActorCTLShutdown(
            ctl_pub_addr=ctl_pub_addr,
            ack_rep_addr=ack_rep_addr,
        )
    ]
    mgr = ProcessManager(
        ctl_pub_addr=ctl_pub_addr,
        ack_rep_addr=ack_rep_addr,
        actors=actors
    )

    mgr.start_all()
    mgr.stop_all()
    mgr.close()
    for actor_proc in mgr.actor_procs:
        assert not actor_proc.proc.is_alive()
        assert actor_proc.done


def test_double_stop_shutdown_logic():
    ctl_pub_addr = create_random_control_address()
    ack_rep_addr = create_random_ack_address()
    # For each actor spin it up in a single process monitor, then shut it down with wait_all(timeout) and also with stop_all()
    actors = [
        MockActorCTLShutdown(
            ctl_pub_addr=ctl_pub_addr,
            ack_rep_addr=ack_rep_addr,
        )
    ]
    mgr = ProcessManager(
        ctl_pub_addr=ctl_pub_addr,
        ack_rep_addr=ack_rep_addr,
        actors=actors
    )

    mgr.start_all()
    mgr.stop_all()
    mgr.stop_all()
    for actor_proc in mgr.actor_procs:
        assert not actor_proc.proc.is_alive()
        assert actor_proc.done