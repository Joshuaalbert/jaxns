import dataclasses
import multiprocessing
import os
import pathlib
import pickle
import pstats
import signal
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Connection
from typing import List
from uuid import uuid4

import zmq

from jaxns.distributed.zmq_actor import ZMQActor
from jaxns.logging import jaxns_logger

STARTUP_ACK_TIMEOUT_S = 30


@dataclasses.dataclass
class ActorProc:
    """
    Represents a process that runs a ZMQActor.
    Contains the actor instance and the process object.
    """
    actor: ZMQActor  # the actor instance
    proc: multiprocessing.Process  # the process running the actor
    err_pipe: Connection  # pipe to get exceptions from the actor process
    done: bool


class ProcessManager:
    """
    Starts, supervises, and cleanly shuts down a list of ZMQActor processes.
    """

    def __init__(self, actors: List[ZMQActor], ctl_pub_addr: str, ack_rep_addr: str, shutdown_timeout: float = 1.0,
                 profile: bool = False):
        self.actors = actors
        self.actor_procs: List[ActorProc] = []
        self.shutdown_timeout = shutdown_timeout
        self.ack_rep_addr = ack_rep_addr
        self.ctl_pub_addr = ctl_pub_addr

        self._profiled = False
        self._profile = profile
        self._lock = threading.Lock()

        # Initialize the ZeroMQ context and control socket for process management
        self.ctx = zmq.Context()
        self.ctl = self.ctx.socket(zmq.PUB)
        self.ctl.bind(ctl_pub_addr)

        # Bind signals before starting any processes
        signal.signal(signal.SIGINT, self._cleanup)
        signal.signal(signal.SIGTERM, self._cleanup)

    def print_tracebacks(self):
        """
        Print tracebacks of all actor processes that have raised exceptions.
        """
        for actor_proc in self.actor_procs:
            if actor_proc.done and actor_proc.actor.exception is not None:
                jaxns_logger.error(f"Exception raised in {actor_proc.actor.__class__.__name__}: "
                                   f"{getattr(actor_proc.actor.exception, 'traceback', str(actor_proc.actor.exception))}")

    def start_all(self):
        """Spawn a Process for each actor's run() method."""
        spawn_ctx = multiprocessing.get_context("forkserver")
        ack_rep = self.ctx.socket(zmq.REP)
        ack_rep.bind(self.ack_rep_addr)
        poller = zmq.Poller()
        poller.register(ack_rep, zmq.POLLIN)
        try:
            num_acks = 0
            for actor in self.actors:
                parent_pipe, child_pipe = spawn_ctx.Pipe(duplex=False)
                p = spawn_ctx.Process(
                    target=actor.start,
                    # Pass actor kwargs
                    kwargs=dict(profile=self._profile, err_pipe=child_pipe),
                    daemon=False
                )
                p.start()
                self.actor_procs.append(ActorProc(actor=actor, proc=p, err_pipe=parent_pipe, done=False))
                socks = dict(poller.poll(STARTUP_ACK_TIMEOUT_S * 1000))
                if ack_rep not in socks:
                    jaxns_logger.warning(f"Timeout waiting for actor {actor.__class__.__name__} to ACK startup.")
                else:
                    _ = ack_rep.recv()
                    ack_rep.send(b"")  # Always after CTL SUB so we know the actor is ready to receive CTL messages.
                    num_acks += 1
        finally:
            poller.unregister(ack_rep)
            ack_rep.close(linger=0)
        if num_acks != len(self.actors):
            jaxns_logger.error(f"Only {num_acks} out of {len(self.actors)} processes ACK'd startup.")
            self.stop_all()
            sys.exit(1)
        else:
            jaxns_logger.info(f"All {num_acks} actor processes acknowledged startup successfully.")

    def wait_all(self, timeout: float = None, stop_all_on_exception: bool = True) -> None:
        """
        Wait for all actor processes to finish. After returning actors exception can be gotten via `actor.exception`.
        After timeout, if specified, the method will return regardless of whether all processes have exited.

        Args:
            timeout (float): Maximum time to wait for processes to exit. If None, wait indefinitely.
            stop_all_on_exception (bool): If True, stop all processes if any process exits with a non-zero exit code.
        """
        start_time = time.time()
        remaining: list[int] = list(range(len(self.actor_procs)))
        while remaining:
            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                return
            for p_idx in remaining:
                actor_proc = self.actor_procs[p_idx]
                if actor_proc.done:
                    # Process is already marked as done, skip it
                    remaining.remove(p_idx)
                    continue
                a, p = actor_proc.actor, actor_proc.proc
                p.join(timeout=0)
                if p.exitcode is not None:
                    if p.exitcode == 0:
                        actor_proc.done = True
                        remaining.remove(p_idx)
                    else:
                        # Process has terminated with an exit code != 0
                        err_pipe = self.actor_procs[p_idx].err_pipe
                        if err_pipe.poll(timeout=0.1):  # Check if there is an error message
                            try:
                                e = pickle.loads(err_pipe.recv())
                            except EOFError:
                                # EOFError can occur if the pipe is closed before we read from it
                                pass
                            else:
                                a.set_exception(e)
                                jaxns_logger.error(
                                    f"Process {p.pid} ({a.__class__.__name__}) exited with code {p.exitcode}. "
                                    f"Error message: {e}")
                        actor_proc.done = True
                        remaining.remove(p_idx)
                        if stop_all_on_exception:
                            # If we are configured to stop all on exception, do so, whench the loop will exit once they
                            # are all registered as stopped. This properly sets the exceptions from the actors.
                            self.stop_all()
            time.sleep(0.1)
        # All exited cleanly
        self._maybe_aggregate_profiles()

    def _maybe_aggregate_profiles(self, top_n: int = 256):
        if self._profiled:
            return
        try:
            if self._profile:
                out_file = f'summary-{os.getpid()}.prof'
                profs = pathlib.Path("./profiles").glob("*.prof")
                stats = None
                with open(out_file, "w") as fh:
                    for p in profs:
                        if stats is None:
                            stats = pstats.Stats(str(p), stream=fh)
                        else:
                            stats.add(str(p))  # pstats can merge profiles
                    if stats is None:
                        jaxns_logger.error("No profile files found!")
                        return
                    stats.sort_stats("time").print_stats(top_n)
                jaxns_logger.info(f"Wrote summary proc profiles to {out_file}")
        finally:
            self._profiled = True

    def close(self):
        """Close the control socket and terminate the ZeroMQ context."""
        with self._lock:
            self.ctl.close(linger=0)
            self.ctx.term()

    def stop_all(self):
        """Terminate and join all live child processes."""
        # This is safe to call multiple times, as it will only attempt to terminate processes that are still alive.
        with self._lock:
            jaxns_logger.info("Gracefully terminating all actor processes...")
            self.ctl.send(b"TERMINATE")
            time.sleep(3)  # Give actors time to process the signal

            # Our strategy for graceful shutdown is to send a control socket 'TERMINATE' message, which each actor should
            # handle appropriately. If they do not exit, within 3 seconds, we will attempt to forcefully terminate them by
            # calling terminate() on each process. This sends the SIGTERM signal to the process, which should be caught and
            # a forceful shutdown initiated, allowing the actor to clean up its resources. If they still do not exit, we
            # will call kill() on each process, which sends SIGKILL and does not allow the process to clean up. This is a
            # last resort to ensure all processes are terminated.

            force_term_count = 0
            for actor_proc in filter(lambda ap: not ap.done, self.actor_procs):
                a, p = actor_proc.actor, actor_proc.proc
                p.join(timeout=0)  # Check if process is still alive
                if p.exitcode is not None:
                    if p.exitcode == 0:
                        actor_proc.done = True
                    else:
                        # Process has terminated with an exit code != 0
                        err_pipe = actor_proc.err_pipe
                        try:
                            e = pickle.loads(err_pipe.recv())
                        except EOFError:
                            # EOFError can occur if the pipe is closed before we read from it
                            pass
                        else:
                            a.set_exception(e)
                            jaxns_logger.error(
                                f"Process {p.pid} ({a.__class__.__name__}) exited with code {p.exitcode}. "
                                f"Error message: {e}")
                        actor_proc.done = True
                else:
                    # Process is still alive, attempt to terminate it gracefully
                    jaxns_logger.info(
                        f"Attempting to forcefully terminate actor process {p.pid} ({a.__class__.__name__}).")
                    p.terminate()  # sends SIGTERM
                    force_term_count += 1
            if force_term_count > 0:
                jaxns_logger.info(
                    f"Sent SIGTERM to {force_term_count} of {len(self.actor_procs)} actors which failed to gracefully shutdown...")
            force_kill_count = 0
            for ap in filter(lambda ap: not ap.done, self.actor_procs):
                a, p = ap.actor, ap.proc
                p.join(timeout=self.shutdown_timeout)
                if p.exitcode is not None:
                    if p.exitcode == 0:
                        actor_proc.done = True
                    else:
                        # Process has terminated with an exit code != 0
                        err_pipe = actor_proc.err_pipe
                        e = err_pipe.recv()
                        a.set_exception(e)
                        jaxns_logger.error(f"Process {p.pid} ({a.__class__.__name__}) exited with code {p.exitcode}. "
                                           f"Error message: {e}")
                        actor_proc.done = True
                else:
                    jaxns_logger.warning(f"Process {p.pid} ({a.__class__.__name__}) did not exit cleanly, killing it.")
                    p.kill()
                    p.join()  # Ensure we wait for the process to exit
                    a.set_exception(RuntimeError(
                        f"Process {p.pid} ({a.__class__.__name__}) was forcefully killed after not responding to SIGTERM."))
                    force_kill_count += 1
            if force_kill_count > 0:
                jaxns_logger.info(
                    f"{force_kill_count} of {len(self.actor_procs)} actors failed to forceful termination... Killed them.")
            self._maybe_aggregate_profiles()

    def _cleanup(self, signum, frame):
        """
        Signal handler: shut down children and exit.
        """
        jaxns_logger.info(f"Process Manager received signal {signum}, shutting down...")
        self.stop_all()
        self.close()


def create_random_socket_address(*prefixes: str):
    fd = str(uuid4())  # Random file descriptor-like number
    tmp_dir = tempfile.gettempdir()
    prefix = '-'.join(prefixes) if prefixes else ''
    return f"ipc://{os.path.join(tmp_dir, f'{prefix}_{fd}.ipc')}"


def create_random_control_address() -> str:
    """
    Create a random control address for the process manager.
    This is a placeholder function; in practice, you would generate a unique address.
    """
    fd = str(uuid4())  # Random file descriptor-like number
    tmp_dir = tempfile.gettempdir()
    return f"ipc://{os.path.join(tmp_dir, f'ctl_pub_{fd}.ipc')}"


def create_random_ack_address() -> str:
    """
    Create a random control address for the process manager.
    This is a placeholder function; in practice, you would generate a unique address.
    """
    fd = str(uuid4())  # Random file descriptor-like number
    tmp_dir = tempfile.gettempdir()
    return f"ipc://{os.path.join(tmp_dir, f'ack_rep_{fd}.ipc')}"
