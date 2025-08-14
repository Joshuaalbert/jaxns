from collections import deque

import zmq

from jaxns.distributed.zmq_actor import ZMQActor, CtlTerminate
from jaxns.logging import jaxns_logger


class LoadBalancer(ZMQActor):
    def __init__(self, ctl_pub_addr: str, ack_rep_addr: str, frontend_addr: str, backend_addr: str):
        ZMQActor.__init__(self, ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr)
        self.frontend_addr = frontend_addr
        self.backend_addr = backend_addr

    def run(self):
        ctl = self.new_socket(zmq.SUB, connect=self.ctl_pub_addr)
        ctl.setsockopt_string(zmq.SUBSCRIBE, "")
        self.ack_startup()

        frontend = self.new_socket(zmq.ROUTER, bind=self.frontend_addr)
        backend = self.new_socket(zmq.ROUTER, bind=self.backend_addr)

        backend_ready = False
        workers = deque()
        poller = zmq.Poller()
        # Only poll for requests from backend until workers are available
        poller.register(backend, zmq.POLLIN)
        poller.register(ctl, zmq.POLLIN)
        try:

            while True:

                socks = dict(poller.poll(1000))
                if ctl in socks:
                    msg = ctl.recv()
                    if msg == b"TERMINATE":
                        break

                if backend in socks:
                    # Handle worker activity on the backend
                    request = backend.recv_multipart()
                    # worker, empty, client_or_ack, empty, reply
                    worker, empty, client_or_ack = request[:3]
                    if len(request) == 3:
                        if client_or_ack == b"READY":
                            workers.append(worker)
                        elif client_or_ack == b"DONE":
                            workers.remove(worker)
                        else:
                            raise RuntimeError(f"Unexpected response from backend: {client_or_ack}")
                    else:
                        client = client_or_ack
                        reply = request[4:]
                        frontend.send_multipart([client, b""] + reply)
                        workers.append(worker)
                    if not backend_ready and workers:
                        # Poll for clients now that a worker is available and backend was not ready
                        poller.register(frontend, zmq.POLLIN)
                        backend_ready = True

                if frontend in socks:
                    # Get next client request, route to last-used worker
                    frames = frontend.recv_multipart()
                    client = frames[0]
                    request = frames[2:]
                    worker = workers.popleft()
                    backend.send_multipart([worker, b"", client, b""] + request)
                    if not workers:
                        # Don't poll clients if no workers are available and set backend_ready flag to false
                        poller.unregister(frontend)
                        backend_ready = False
        except CtlTerminate:
            pass
        finally:
            poller.unregister(frontend)
            if backend_ready:
                poller.unregister(backend)
            poller.unregister(ctl)
