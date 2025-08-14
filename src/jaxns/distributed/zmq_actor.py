import cProfile
import multiprocessing
import os
import pickle
import signal
import sys
from abc import abstractmethod, ABC
from typing import List

import zmq

from jaxns.logging import jaxns_logger


class CtlTerminate(Exception):
    ...


class ZMQActor(ABC):
    """
    Manages a ZMQ context and socket list.
    Healthy shutdown expects, ctl socket to be used, which should make run() return, no cleanup necessary.
    If this is not done, then forceful shutdown will occur when the process exits, calling _cleanup().
    """

    def __init__(self, ctl_pub_addr: str, ack_rep_addr: str):
        """
        Initializes the ZMQActor with a context and an empty socket list.
        If ack_rep_addr is provided, it will send an ACK message to indicate readiness.
        """
        self.sockets: List[zmq.Socket]
        self.ctx: zmq.Context
        self.forceful_shutdown: bool = False

        self.ack_rep_addr = ack_rep_addr
        self.ctl_pub_addr = ctl_pub_addr
        self._acked_startup = False
        self._cleaned_up = False
        self._exception: Exception | None = None

    def __repr__(self):
        return f"{self.__class__.__name__}(exception={self._exception})"

    def set_exception(self, e: Exception) -> None:
        """
        Set an exception that occurred in the actor process.
        This is used to propagate exceptions from the actor process to the main process.
        """
        self._exception = e

    @property
    def exception(self) -> Exception | None:
        """
        Get the exception that occurred in the actor process.
        This is used to check if any exceptions were raised during the actor's run.
        """
        return self._exception

    def ack_startup(self) -> None:
        """
        Connects to the ack_rep socket and sends an ACK message to indicate readiness.
        """
        if self._acked_startup:
            raise RuntimeError(f"{self.__class__.__name__} has already acknowledged startup.")
        ack_req = self.ctx.socket(zmq.REQ)
        ack_req.connect(self.ack_rep_addr)
        try:
            ack_req.send(f"{self.__class__.__name__}".encode())
            ack_req.recv()
            jaxns_logger.info(f"{self.__class__.__name__} startup acknowledged.")
        finally:
            ack_req.close(linger=0)
        self._acked_startup = True

    def new_socket(self, sock_type, *, bind=None, connect=None, no_hwm: bool = True, **opts) -> zmq.Socket:
        if bind is None and connect is None:
            raise ValueError("Must specify either bind or connect")
        jaxns_logger.info(
            f"[{self.__class__.__name__}] added socket {sock_type!r}, bind={bind}, connect={connect}, opts={opts}"
        )
        s = self.ctx.socket(sock_type)
        if no_hwm:
            s.setsockopt(zmq.SNDHWM, 0)
            s.setsockopt(zmq.RCVHWM, 0)
        for opt_name, opt_val in opts.items():
            setattr(s, opt_name, opt_val)
        if bind:
            s.bind(bind)
        if connect:
            s.connect(connect)
        self.sockets.append(s)
        return s

    def _install_signal_handlers(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, self._forceful_shutdown)

    @staticmethod
    def cleanup_context(ctx: zmq.Context, sockets: List[zmq.Socket]):
        """
        Clean up the context and sockets.
        This is a static method to be used in the main process.
        """
        for s in sockets:
            try:
                s.close(linger=0)
            except BaseException as e:
                jaxns_logger.error(f"error closing socket: {e}")
        try:
            ctx.term()
        except BaseException as e:
            jaxns_logger.error(f"error terminating context: {e}")

    def _cleanup(self):
        if self._cleaned_up:
            return
        jaxns_logger.info(f"[{self.__class__.__name__}] cleaning up sockets and context...")
        self.cleanup_context(self.ctx, self.sockets)
        self._cleaned_up = True
        try:
            self._extra_shutdown()
        except BaseException as e:
            jaxns_logger.error(f"error in _extra_shutdown: {e}")

    def _forceful_shutdown(self, signum, frame):
        jaxns_logger.info(f"[{self.__class__.__name__}] caught signal {signum}, forcefully shutting down...")
        self.forceful_shutdown = True
        self._cleanup()

    def _graceful_shutdown(self):
        """
        Called when the ctl socket receives a shutdown message.
        This should be overridden by subclasses to do any additional cleanup.
        """
        # Forcefully caught signal, so don't do anymore cleanup.
        if self.forceful_shutdown:
            return
        jaxns_logger.info(f"[{self.__class__.__name__}] gracefully shutting down...")
        self._cleanup()

    def _extra_shutdown(self):
        """Hook for subclasses to do more cleanup if needed."""
        pass

    def start(self, err_pipe: multiprocessing.Pipe, profile: bool = False):
        # create context & socket list
        try:
            self.ctx = zmq.Context()
            self.sockets = []
        except Exception as e:
            jaxns_logger.error(f"Error creating ZMQ context: {e}")
            raise e
        # catch ctrl‐C and terminate
        try:
            self._install_signal_handlers()
        except Exception as e:
            jaxns_logger.error(f"Error installing signal handlers: {e}")
            raise e
        try:
            if profile:
                profile_folder = "./profiles"
                os.makedirs(profile_folder, exist_ok=True)
                profile_fname = f"{profile_folder}/{self.__class__.__name__}-{os.getpid()}.prof"
                profiler = cProfile.Profile()
                profiler.enable()
                try:
                    self.run()
                finally:  # guaranteed to run in the child
                    if profile:
                        profiler.disable()
                        profiler.dump_stats(profile_fname)
            else:
                self.run()
        except Exception as e:
            jaxns_logger.error(str(e))
            import traceback
            tb_str = traceback.format_exc()
            setattr(e, "traceback", tb_str)
            try:
                err_pipe.send(pickle.dumps(e))
            except Exception as e2:
                jaxns_logger.error(f"Error sending exception through error pipe: {str(e2)}.")
            sys.exit(1)  # Raises SystemExit, which is caught by the process manager.
        finally:
            try:
                # ctl socket makes run() return, so we can do cleanup here.
                self._graceful_shutdown()
            except Exception as e:
                jaxns_logger.error(f"Error during graceful shutdown: {e}")
                raise e

    @abstractmethod
    def run(self):
        """
        Blocking work goes here.
        e.g.:

            clt = self.new_socket(zmq.SUB, connect=self.ctl_pub_addr)
            clt.setsockopt_string(zmq.SUBSCRIBE, "")
            poller = zmq.Poller()
            poller.register(clt, zmq.POLLIN)
            while True:
                socks = dict(poller.poll())
                if clt in socks:
                    msg = clt.recv()
                    if msg == b"TERMINATE":
                        break
                # rest of work here
        """
        ...
