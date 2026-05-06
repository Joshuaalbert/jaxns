"""
Peer-to-Peer RPC over ZeroMQ with a Control-Plane Load Balancer.

- A **control plane**: a single LoadBalancer (LB) that assigns workers via short-lived
  **leases**, handles heartbeats, epoch fencing, and fairness/stickiness.
- A **data plane**: clients talk **directly** to services (peer-to-peer) using ROUTER/DEALER
  sockets.

Messages:

LB <-> Service:
  READY        : [READY, data_endpoint]
  GRANT        : [GRANT, lease_id, client_id, epoch]
  GRANT_ACK    : [GRANT_ACK, lease_id, epoch]
  REVOKE       : [REVOKE, lease_id]
  COMPLETE     : [COMPLETE, lease_id, epoch]
  HEARTBEAT    : [HEARTBEAT]

LB <-> Client:
  REQUEST      : [REQUEST]
  ASSIGN       : [ASSIGN, worker_id, data_endpoint, lease_id, epoch]
  REVOKE       : [REVOKE, lease_id, reason]
  CANCEL       : [CANCEL, lease_id]
   where reason ∈ {b"worker_restart", b"worker_dead", b"cancel", b"wrong_epoch", b"unknown_lease", b"duplicate_grant_ack", b"grant_ack_timeout"}.

Client <-> Service (data plane, direct):
  RPC          : [RPC, lease_id, method_name, main_pickle, *buffers]
  RESULT       : [RESULT, lease_id, status, main_pickle, *buffers]
  RESULT_ACK   : [RESULT_ACK, lease_id]
    where status ∈ {b"ok", b"error", b"invalid_lease"}.

Security & Correctness
----------------------
- Every request is authorized by validating (lease_id, client_id) at the service.
- Epoch fencing ignores stale COMPLETEs after an LB restart.
- `COMPLETE` is sent by the service **only after** receiving `RESULT_ACK` from the client,
  to avoid LB revoking a lease before the client sees the result.
- REVOKE_ACK and COMPLETE act in the same way of freeing up the worker and client state.
"""

import contextlib
import dataclasses
import os
import pickle
import queue
import random
import socket
import threading
import time
from abc import ABC, abstractmethod, ABCMeta
from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from enum import auto, Enum
from typing import Optional, Union

import zmq

from jaxns.fabric.zmq_actor import ZMQActor, CtlTerminate
from jaxns.logging import jaxns_logger

PICKLE_PROTO = pickle.HIGHEST_PROTOCOL

# Control messages (all bytes literals)
MSG_READY = b"READY"  # service -> LB backend:  [READY, data_endpoint]
MSG_COMPLETE = b"COMPLETE"  # service -> LB backend:  [COMPLETE, lease_id, epoch] | client -> LB backend:  [COMPLETE, lease_id, epoch]
MSG_HEARTBEAT = b"HEARTBEAT"  # service -> LB backend:  [HEARTBEAT]

MSG_REQUEST = b"REQUEST"  # client  -> LB frontend: [REQUEST]
MSG_ASSIGN = b"ASSIGN"  # LB -> client:             [ASSIGN, worker_id, data_endpoint, lease_id, epoch]
MSG_GRANT = b"GRANT"  # LB -> service:             [GRANT, lease_id, client_id, epoch]
MSG_GRANT_ACK = b"GRANT_ACK"  # service -> LB backend:  [GRANT_ACK, lease_id, epoch]
MSG_REVOKE = b"REVOKE"  # LB -> service: [REVOKE, lease_id]; LB -> client: [REVOKE, lease_id, reason]
MSG_REVOKE_ACK = b"REVOKE_ACK"  # service -> LB [REVOKE_ACK, lease_id]; client -> LB [REVOKE_ACK, lease_id]
MSG_CANCEL = b"CANCEL"  # client -> LB: [CANCEL, lease_id]

# Data-plane messages
MSG_RPC = b"RPC"  # client -> service: [RPC, lease_id, method_name, main_pickle, *buffers]
MSG_RESULT = b"RESULT"  # service -> client: [RESULT, lease_id, status, main_pickle, *buffers]
MSG_RESULT_ACK = b"RESULT_ACK"  # client -> service: [RESULT_ACK, lease_id]

# Response status codes (data-plane)
STATUS_OK = b"ok"
STATUS_ERR = b"error"
STATUS_INVALID_LEASE = b"invalid_lease"
STATUS_UNKNOWN_WORKER = b"unknown_worker"
STATUS_WRONG_EPOCH = b"wrong_epoch"
STATUS_CANCELLED = b"cancelled"

HB_INTERVAL_MS = 1000
HB_TIMEOUT_MS = 4000
GRANT_ACK_TIMEOUT_MS = 2000
RESULT_ACK_TIMEOUT_MS = 10000


class InvalidTransition(Exception):
    ...


class WorkerLivenessEnum(Enum):
    """
    GAto: live -> suspect
    HBto: live -> dead
    HBto: suspect -> dead
    HB: suspect -> live
    HB: dead -> live
    """
    live = auto()
    suspect = auto()
    dead = auto()


class WorkerStateEnum(Enum):
    """
    G: idle -> granting
    GA: granting -> busy
    R: busy -> revoking
    R: granting -> revoking
    C: busy -> idle
    C: revoking -> idle
    RA: revoking -> idle
    """
    idle = auto()
    granting = auto()
    busy = auto()
    revoking = auto()


class ClientStateEnum(Enum):
    """
    Rq: idle -> waiting
    G: waiting -> granting
    GA: granting -> busy
    R: busy -> revoking
    R: granting -> revoking
    R: waiting -> revoking
    C: busy -> idle
    C: revoking -> idle
    RA: revoking -> idle
    """
    idle = auto()
    waiting = auto()
    granting = auto()
    busy = auto()
    revoking = auto()


@dataclasses.dataclass(slots=True)
class ClientRecord:
    client_id: bytes
    pending_request: bool = False
    lease_id: bytes | None = dataclasses.field(default_factory=lambda: None)
    worker_record: Union['WorkerRecord', None] = dataclasses.field(default_factory=lambda: None)
    state: ClientStateEnum = dataclasses.field(default_factory=lambda: ClientStateEnum.idle)
    mru_cache: OrderedDict[bytes, None] = dataclasses.field(default_factory=OrderedDict)  # worker_id -> None

    def request(self):
        if self.pending_request:
            raise InvalidTransition(f"Client {self.client_id!r} already has a pending request")
        if self.state == ClientStateEnum.busy:
            self.pending_request = True
            return
        if self.state == ClientStateEnum.waiting:
            # Make idempotent
            return
        if self.state != ClientStateEnum.idle:
            raise InvalidTransition(f"Client {self.client_id!r} in state {self.state}, cannot REQUEST")
        self.state = ClientStateEnum.waiting

    def grant(self, lease_id: bytes, worker_record: 'WorkerRecord'):
        if self.state != ClientStateEnum.waiting:
            raise InvalidTransition(f"Client {self.client_id!r} in state {self.state}, cannot GRANT")
        self.lease_id = lease_id
        self.worker_record = worker_record
        # Move to front
        if worker_record.worker_id not in self.mru_cache:
            self.mru_cache[worker_record.worker_id] = None
        self.mru_cache.move_to_end(worker_record.worker_id, last=False)
        self.state = ClientStateEnum.granting

    def grant_ack(self, frontend: zmq.Socket, epoch: bytes):
        if self.state != ClientStateEnum.granting:
            raise InvalidTransition(f"Client {self.client_id!r} in state {self.state}, cannot GRANT_ACK")
        frontend.send_multipart([self.client_id, MSG_ASSIGN, self.worker_record.worker_id,
                                 self.worker_record.endpoint, self.lease_id, epoch], copy=False)
        self.state = ClientStateEnum.busy

    def revoke(self, frontend: zmq.Socket, reason: bytes):
        if self.state not in (ClientStateEnum.busy, ClientStateEnum.granting, ClientStateEnum.waiting):
            raise InvalidTransition(f"Client {self.client_id!r} in state {self.state}, cannot REVOKE")
        jaxns_logger.info(f"Revoking lease {self.lease_id!r} due to {reason!r}")
        frontend.send_multipart([self.client_id, MSG_REVOKE, self.lease_id, reason], copy=False)
        self.state = ClientStateEnum.revoking

    def revoke_ack(self):
        if self.state != ClientStateEnum.revoking:
            raise InvalidTransition(f"Client {self.client_id!r} in state {self.state}, cannot REVOKE_ACK")
        self.lease_id = None
        self.worker_record = None
        if self.pending_request:
            self.state = ClientStateEnum.waiting
            self.pending_request = False
        else:
            self.state = ClientStateEnum.idle

    def complete(self):
        if self.state not in (ClientStateEnum.busy, ClientStateEnum.revoking):
            raise InvalidTransition(f"Client {self.client_id!r} in state {self.state}, cannot COMPLETE")
        self.lease_id = None
        self.worker_record = None
        if self.pending_request:
            self.state = ClientStateEnum.waiting
            self.pending_request = False
        else:
            self.state = ClientStateEnum.idle


@dataclasses.dataclass(slots=True)
class WorkerRecord:
    worker_id: bytes
    endpoint: bytes
    grant_ack_deadline_ms: float | None = None
    heartbeat_deadline_ms: float | None = dataclasses.field(default_factory=lambda: time.monotonic() * 1000.0 + HB_TIMEOUT_MS)
    state: WorkerStateEnum = dataclasses.field(default_factory=lambda: WorkerStateEnum.idle)
    liveness: WorkerLivenessEnum = dataclasses.field(default_factory=lambda: WorkerLivenessEnum.live)
    lease_id: bytes | None = dataclasses.field(default_factory=lambda: None)
    client_record: ClientRecord | None = dataclasses.field(default_factory=lambda: None)

    def grant(self, backend: zmq.Socket, lease_id: bytes, client_record: ClientRecord, epoch: bytes):
        if self.state != WorkerStateEnum.idle:
            raise InvalidTransition(f"Worker {self.worker_id!r} in state {self.state}, cannot GRANT")
        self.lease_id = lease_id
        self.client_record = client_record
        self.grant_ack_deadline_ms = time.monotonic() * 1000 + GRANT_ACK_TIMEOUT_MS
        backend.send_multipart([self.worker_id, MSG_GRANT, lease_id, client_record.client_id, epoch], copy=False)
        self.state = WorkerStateEnum.granting

    def grant_ack(self):
        if self.state != WorkerStateEnum.granting:
            raise InvalidTransition(f"Worker {self.worker_id!r} in state {self.state}, cannot GRANT_ACK")
        self.state = WorkerStateEnum.busy
        self.grant_ack_deadline_ms = None
        if self.liveness in (WorkerLivenessEnum.suspect, WorkerLivenessEnum.dead):
            self.liveness = WorkerLivenessEnum.live
            jaxns_logger.warning(f"Reawakened worker {self.worker_id!r} on grant ack")

    def revoke(self, backend: zmq.Socket):
        if self.state not in (WorkerStateEnum.busy, WorkerStateEnum.granting):
            raise InvalidTransition(f"Worker {self.worker_id!r} in state {self.state}, cannot REVOKE")
        backend.send_multipart([self.worker_id, MSG_REVOKE, self.lease_id], copy=False)
        self.state = WorkerStateEnum.revoking

    def revoke_ack(self):
        if self.state != WorkerStateEnum.revoking:
            raise InvalidTransition(f"Worker {self.worker_id!r} in state {self.state}, cannot REVOKE_ACK")
        self.lease_id = None
        self.client_record = None
        self.state = WorkerStateEnum.idle

    def complete(self):
        if self.state not in (WorkerStateEnum.busy, WorkerStateEnum.revoking):
            raise InvalidTransition(f"Worker {self.worker_id!r} in state {self.state}, cannot COMPLETE")
        self.lease_id = None
        self.client_record = None
        self.state = WorkerStateEnum.idle

    def grant_ack_timeout(self):
        if self.liveness != WorkerLivenessEnum.live:
            raise InvalidTransition(f"Worker {self.worker_id!r} with liveness {self.liveness}, cannot GRANT_ACK_TIMEOUT")
        self.grant_ack_deadline_ms = None
        self.liveness = WorkerLivenessEnum.suspect

    def heartbeat_timeout(self):
        if self.liveness not in (WorkerLivenessEnum.suspect, WorkerLivenessEnum.live):
            raise InvalidTransition(f"Worker {self.worker_id!r} with liveness {self.liveness}, cannot HEARTBEAT_TIMEOUT")
        self.heartbeat_deadline_ms = None
        self.liveness = WorkerLivenessEnum.dead

    def heartbeat(self):
        if self.liveness in (WorkerLivenessEnum.suspect, WorkerLivenessEnum.dead):
            self.liveness = WorkerLivenessEnum.live
            jaxns_logger.warning(f"Reawakened worker {self.worker_id!r} on heartbeat")
        self.heartbeat_deadline_ms = time.monotonic() * 1000.0 + HB_TIMEOUT_MS

    def can_assign(self) -> bool:
        return self.liveness == WorkerLivenessEnum.live and self.state == WorkerStateEnum.idle


def make_lease_id() -> bytes:
    """
    Generate a random 128-bit lease identifier encoded as 32 lowercase hex ASCII bytes.

    Returns
    -------
    bytes
        The generated lease_id, e.g. b'0f2c...'.
    """
    return f"{random.getrandbits(128):032x}".encode("ascii")


# ============================================================
# Load Balancer (control-plane only)
# ============================================================

class LoadBalancer(ZMQActor):

    def __init__(self, ctl_pub_addr: str, ack_rep_addr: str, frontend_addr: str, backend_addr: str) -> None:
        """
        Parameters
        ----------
        ctl_pub_addr : str
            PUB/SUB control channel address for process control (termination, etc.).
        ack_rep_addr : str
            REQ/REP control channel used for startup acknowledgements.
        frontend_addr : str
            TCP address (tcp://host:port) for the LB frontend (clients connect).
        backend_addr : str
            TCP address (tcp://host:port) for the LB backend (services connect).
        """
        super().__init__(ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr)
        if not frontend_addr.startswith('tcp://') or not backend_addr.startswith('tcp://'):
            raise ValueError(f"frontend and backend must use tcp, got {frontend_addr} and {backend_addr}")
        self.frontend_addr = frontend_addr
        self.backend_addr = backend_addr

        # Maintain in LRU order
        self.worker_records: OrderedDict[bytes, WorkerRecord] = OrderedDict()
        self.client_records: OrderedDict[bytes, ClientRecord] = OrderedDict()

        # Epoch fences stale COMPLETEs across LB restarts
        self.epoch: bytes = f"{random.getrandbits(64):016x}".encode("ascii")

    def _now_ms(self) -> float:
        return time.monotonic() * 1000.0

    def mark_worker_least_priority(self, worker_id: bytes) -> None:
        # Move worker MRU, move to end
        if worker_id not in self.worker_records:
            raise RuntimeError(f"Unknown worker {worker_id!r}.")
        self.worker_records.move_to_end(worker_id)

    def mark_worker_highest_priority(self, worker_id: bytes) -> None:
        # Move worker LRu, move to start
        if worker_id not in self.worker_records:
            raise RuntimeError(f"Unknown worker {worker_id!r}.")
        self.worker_records.move_to_end(worker_id, last=False)

    def mark_client_least_priority(self, client_id: bytes) -> None:
        # Move client MRU, move to end
        if client_id not in self.client_records:
            raise RuntimeError(f"Unknown client {client_id!r}.")
        self.client_records.move_to_end(client_id)

    def mark_client_highest_priority(self, client_id: bytes) -> None:
        # Move client LRU, move to start
        if client_id not in self.client_records:
            raise RuntimeError(f"Unknown client {client_id!r}.")
        self.client_records.move_to_end(client_id, last=False)

    def _try_select_worker_for_client(self, client_record: ClientRecord) -> WorkerRecord | None:
        # Try MRU cache
        for wid in client_record.mru_cache:
            wr = self.worker_records.get(wid, None)
            if wr is None:
                raise RuntimeError(f"Unknown worker {wid!r} in client {client_record.client_id!r} MRU cache")
            if wr.can_assign():
                return wr
        # Try any idle worker
        for wid, wr in self.worker_records.items():
            if wr.can_assign():
                return wr
        return None

    def _try_grant_waiting_clients(self, backend: zmq.Socket):
        for cid, cr in list(self.client_records.items()):
            if cr.state == ClientStateEnum.waiting:
                wr = self._try_select_worker_for_client(cr)
                if wr is not None:
                    self._grant(backend, cr, wr)

    def _grant(self, backend: zmq.Socket, client_record: ClientRecord,
               worker_record: WorkerRecord) -> None:
        """
        Grant a lease to (client_id -> worker_id). Performs the GRANT/GRANT_ACK handshake.
        """
        cid = client_record.client_id
        wid = worker_record.worker_id
        lid = make_lease_id()
        jaxns_logger.info(f"Granting lease {lid!r} for client {cid!r} to worker {wid!r}")
        try:
            worker_record.grant(backend, lid, client_record, self.epoch)
            self.mark_worker_least_priority(wid)
            try:
                client_record.grant(lid, worker_record)
                self.mark_client_least_priority(cid)
            except InvalidTransition as e:
                jaxns_logger.exception(e)
        except InvalidTransition as e:
            jaxns_logger.exception(e)

    # ----- run loop -----

    def run(self) -> None:
        """
        Main event loop: polls control sockets, performs scheduling, lease management,
        grant handshakes, heartbeat reaping, and fairness queue draining.
        """
        ctl = self.new_socket(zmq.SUB, connect=self.ctl_pub_addr)
        ctl.setsockopt_string(zmq.SUBSCRIBE, "")
        self.ack_startup()

        frontend = self.new_socket(zmq.ROUTER, bind=self.frontend_addr)
        backend = self.new_socket(zmq.ROUTER, bind=self.backend_addr)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(backend, zmq.POLLIN)
        poller.register(ctl, zmq.POLLIN)

        try:
            while True:
                socks = dict(poller.poll(100))

                # Graceful terminate
                if ctl in socks:
                    if ctl.recv() == b"TERMINATE":
                        break

                if frontend in socks:
                    frames = frontend.recv_multipart(copy=False)
                    if len(frames) < 2:
                        raise RuntimeError(f"Expected >=2 frames [client_id, cmd, *args], got {len(frames)}")
                    client_id = bytes(frames[0])
                    cmd = bytes(frames[1])
                    jaxns_logger.info(f"Received {cmd!r} from frontend {client_id!r}")
                    cr = self.client_records.get(client_id, None)
                    if cr is None:
                        # New client -- goes to end of the line
                        cr = self.client_records[client_id] = ClientRecord(client_id=client_id)
                        self.mark_client_least_priority(client_id)

                    if cmd == MSG_REQUEST:
                        try:
                            cr.request()
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        self._try_grant_waiting_clients(backend)
                        continue
                    elif cmd == MSG_REVOKE_ACK:
                        if len(frames) != 3:
                            raise RuntimeError(
                                f"Expected 3 frames [client_id, REVOKE_ACK, lease_id], got {len(frames)}")
                        lease_id = bytes(frames[2])
                        if cr.lease_id != lease_id:
                            jaxns_logger.warning(f"Client {client_id!r} sent REVOKE_ACK for unknown lease {lease_id!r}")
                            continue
                        try:
                            cr.worker_record.revoke_ack()
                            self.mark_worker_least_priority(cr.worker_record.worker_id)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        try:
                            cr.revoke_ack()
                            self.mark_client_least_priority(cr.client_id)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        self._try_grant_waiting_clients(backend)
                        continue
                    elif cmd == MSG_CANCEL:
                        if len(frames) != 3:
                            raise RuntimeError(f"Expected 3 frames [client_id, CANCEL, lease_id], got {len(frames)}")
                        lease_id = bytes(frames[2])
                        if cr.lease_id != lease_id:
                            jaxns_logger.warning(f"Client {client_id!r} sent CANCEL for unknown lease {lease_id!r}")
                            continue
                        try:
                            cr.revoke(frontend, b"cancel")
                            self.mark_client_least_priority(cr.client_id)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        try:
                            cr.worker_record.revoke(backend)
                            self.mark_worker_least_priority(cr.worker_record.worker_id)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        continue
                    else:
                        jaxns_logger.warning(f"Unknown frontend command {cmd!r} from client: {client_id!r}")

                if backend in socks:
                    frames = backend.recv_multipart(copy=False)
                    if len(frames) < 2:
                        raise RuntimeError(f"Expected >=2 frames [worker_id, cmd, *args], got {len(frames)}")
                    worker_id = bytes(frames[0])
                    cmd = bytes(frames[1])
                    jaxns_logger.info(f"Received {cmd!r} from backend {worker_id!r}")
                    if cmd == MSG_READY:
                        if len(frames) != 3:
                            raise RuntimeError(
                                f"Expected 3 frames [worker_id, READY, data_endpoint], got {len(frames)}")
                        data_ep = bytes(frames[2])
                        wr = self.worker_records.get(worker_id, None)
                        if wr is None:
                            self.worker_records[worker_id] = WorkerRecord(
                                worker_id=worker_id,
                                endpoint=data_ep
                            )
                        else:
                            # Check that endpoint is the same
                            if wr.endpoint != data_ep:
                                wr.endpoint = data_ep
                                if wr.lease_id is not None:
                                    # Lease exists so revoke
                                    try:
                                        wr.revoke(backend)
                                    except InvalidTransition as e:
                                        jaxns_logger.exception(e)
                                    try:
                                        wr.client_record.revoke(frontend, b"worker_restart")
                                    except InvalidTransition as e:
                                        jaxns_logger.exception(e)
                        self.mark_worker_highest_priority(worker_id)
                        # Grant immediately if someone is waiting
                        self._try_grant_waiting_clients(backend)
                        continue
                    else:
                        wr = self.worker_records.get(worker_id, None)
                        if wr is None:
                            jaxns_logger.warning(f"Unknown worker {worker_id!r} sent command {cmd!r}")
                            continue

                    if cmd == MSG_COMPLETE:
                        wr.heartbeat()
                        if len(frames) != 4:
                            raise RuntimeError(
                                f"Expected 4 frames [worker_id, COMPLETE, lease_id, epoch], got {len(frames)}")
                        lease_id = bytes(frames[2])
                        if wr.lease_id != lease_id:
                            jaxns_logger.warning(f"Worker {worker_id!r} sent COMPLETE for unknown lease {lease_id!r}")
                            continue
                        recv_epoch = bytes(frames[3])
                        if recv_epoch != self.epoch:
                            # stale COMPLETE from old epoch; still mark completed locally.
                            pass
                        try:
                            wr.client_record.complete()
                            self.mark_client_least_priority(wr.client_record.client_id)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        try:
                            wr.complete()
                            self.mark_worker_least_priority(wr.worker_id)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        self._try_grant_waiting_clients(backend)
                        continue
                    elif cmd == MSG_REVOKE_ACK:
                        wr.heartbeat()
                        if len(frames) != 3:
                            raise RuntimeError(
                                f"Expected 3 frames [worker_id, REVOKE_ACK, lease_id], got {len(frames)}")
                        lease_id = bytes(frames[2])
                        if wr.lease_id != lease_id:
                            jaxns_logger.warning(f"Worker {worker_id!r} sent REVOKE_ACK for unknown lease {lease_id!r}")
                            continue
                        try:
                            wr.client_record.revoke_ack()
                            self.mark_client_least_priority(wr.client_record.client_id)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        try:
                            wr.revoke_ack()
                            self.mark_worker_least_priority(wr.worker_id)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        self._try_grant_waiting_clients(backend)
                        continue
                    elif cmd == MSG_GRANT_ACK:
                        wr.heartbeat()
                        if len(frames) != 4:
                            jaxns_logger.warning(
                                f"Expected 4 frames [worker_id, GRANT_ACK, lease_id, epoch], got {len(frames)}")
                            continue
                        lease_id = bytes(frames[2])
                        recv_epoch = bytes(frames[3])
                        if wr.lease_id != lease_id:
                            jaxns_logger.warning(f"Worker {worker_id!r} sent GRANT_ACK for unknown lease {lease_id!r}")
                            continue
                        if recv_epoch != self.epoch:
                            # Stale ACK from previous LB epoch; revoke the lease.
                            try:
                                wr.revoke(backend)
                            except InvalidTransition as e:
                                jaxns_logger.exception(e)
                            try:
                                wr.client_record.revoke(frontend, b"wrong_epoch")
                            except InvalidTransition as e:
                                jaxns_logger.exception(e)
                            continue
                        try:
                            wr.grant_ack()
                            self.mark_worker_least_priority(wr.worker_id)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        try:
                            wr.client_record.grant_ack(frontend, self.epoch)
                            self.mark_client_least_priority(wr.client_record.client_id)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        continue
                    elif cmd == MSG_HEARTBEAT:
                        # HB means worker is alive.
                        wr.heartbeat()
                        # If was suspect or dead, now live and idle, so try to assign.
                        self._try_grant_waiting_clients(backend)
                    else:
                        jaxns_logger.warning(f"Unknown backend command {cmd!r} from worker: {worker_id!r}")

                now = self._now_ms()
                # Revoke pending grants that missed GRANT_ACK
                for wid, wr in list(self.worker_records.items()):
                    if wr.grant_ack_deadline_ms is not None and now > wr.grant_ack_deadline_ms:
                        # Missed grant ack; revoke the lease; mark worker suspect until heart beat.
                        try:
                            wr.grant_ack_timeout()
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        try:
                            wr.revoke(backend)
                            self.mark_worker_least_priority(wid)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        try:
                            wr.client_record.revoke(frontend, b"grant_ack_timeout")
                            self.mark_client_least_priority(wr.client_record.client_id)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                    if wr.heartbeat_deadline_ms is not None and now > wr.heartbeat_deadline_ms:
                        # Dead worker; revoke to be safe and mark
                        try:
                            wr.heartbeat_timeout()
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        try:
                            wr.revoke(backend)
                            self.mark_worker_least_priority(wid)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
                        try:
                            wr.client_record.revoke(frontend, b"worker_dead")
                            self.mark_client_least_priority(wr.client_record.client_id)
                        except InvalidTransition as e:
                            jaxns_logger.exception(e)
        except CtlTerminate:
            pass
        finally:
            for s in (frontend, backend, ctl):
                try:
                    poller.unregister(s)
                except Exception:
                    pass


# ============================================================
# Service Actor (data-plane server)
# ============================================================

def get_free_port() -> int:
    """
    Ask the OS for an ephemeral free TCP port and return it.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _parse_tcp_host_port(endpoint: str) -> tuple[str, int]:
    """
    Parse a 'tcp://host:port' endpoint (IPv4 or bracketed IPv6) into (host, port).
    """
    if not endpoint.startswith("tcp://"):
        raise ValueError(f"Only tcp:// supported here, got {endpoint!r}")
    hostport = endpoint[len("tcp://"):]
    if hostport.startswith('['):
        host_end = hostport.index(']')
        host = hostport[1:host_end]
        port = int(hostport[host_end + 2:])
    else:
        host, port_s = hostport.rsplit(":", 1)
        port = int(port_s)
    return host, port


def _format_host_for_zmq(host: str, family: int) -> str:
    """
    Format a host literal for ZeroMQ endpoints, adding brackets for IPv6 when needed.
    """
    if family == socket.AF_INET6 and ':' in host and not host.startswith('['):
        return f'[{host}]'
    return host


def pick_local_ip_for_remote(remote_host: str, remote_port: int) -> tuple[str, socket.AddressFamily]:
    """
    Determine the local source IP and family the kernel would choose to reach (remote_host, remote_port).
    """
    infos = []
    try:
        infos = socket.getaddrinfo(remote_host, remote_port, 0, socket.SOCK_DGRAM)
    except socket.gaierror:
        pass

    for family, _stype, _proto, _canon, sockaddr in infos:
        try:
            s = socket.socket(family, socket.SOCK_DGRAM)
            try:
                s.connect(sockaddr)  # no packets sent for UDP connect
                local_ip = s.getsockname()[0]
                return local_ip, family
            finally:
                s.close()
        except OSError:
            continue

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 53))
            return s.getsockname()[0], socket.AF_INET
        finally:
            s.close()
    except OSError:
        return '127.0.0.1', socket.AF_INET


def compute_advertise_addr(backend_addr: str, port: int, *,
                           override_host: Optional[str] = None) -> str:
    """
    Decide what endpoint to hand to clients for reaching this service.
    """
    if override_host and override_host not in {'*', '0.0.0.0', '::'}:
        host = override_host
        fam = socket.AF_INET6 if ':' in host and not host.replace(':', '').isdigit() else socket.AF_INET
        host_fmt = _format_host_for_zmq(host, fam)
        return f"tcp://{host_fmt}:{port}"

    host, port_dummy = _parse_tcp_host_port(backend_addr)  # we only need host
    local_ip, fam = pick_local_ip_for_remote(host, port_dummy)
    host_fmt = _format_host_for_zmq(local_ip, fam)
    return f"tcp://{host_fmt}:{port}"


class ResultEnum(Enum):
    granted = auto()
    processing = auto()
    awaiting_ack = auto()
    aborted = auto()


@dataclasses.dataclass(slots=True)
class ResultRecord:
    lease_id: bytes
    client_id: bytes
    future: Future | None
    aborted: bool = False
    result_ack_deadline_ms: float | None = None
    state: ResultEnum = dataclasses.field(default_factory=lambda: ResultEnum.granted)

    def mark_processing(self):
        self.state = ResultEnum.processing

    def mark_awaiting_ack(self):
        self.state = ResultEnum.awaiting_ack

    def mark_granted(self):
        self.state = ResultEnum.granted


class RPCActor(ZMQActor, ABC):

    def __init__(self, ctl_pub_addr: str, ack_rep_addr: str, backend_addr: str, data_bind_addr: Optional[str] = None,
                 host_ip: Optional[str] = None, advertise_host: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        ctl_pub_addr : str
            PUB/SUB control channel for process management (termination, etc.).
        ack_rep_addr : str
            REQ/REP channel for startup acknowledgement.
        backend_addr : str
            LB backend address (tcp://host:port) for control-plane DEALER connection.
        data_bind_addr : Optional[str], optional
            Explicit data-plane bind address (tcp://host:port). If None, a free port is chosen.
        host_ip : Optional[str], optional
            Local bind host for the data socket (defaults to '*').
        advertise_host : Optional[str], optional
            Override advertised host (e.g., public/NAT IP); else derived via backend reachability.
        """
        ZMQActor.__init__(self, ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr)
        self.backend_addr = backend_addr

        # Decide port (if not fully provided)
        if data_bind_addr is None:
            port = get_free_port()
            bind_host = host_ip or "*"  # listen on all interfaces by default
            self.data_bind_addr = f"tcp://{bind_host}:{port}"
        else:
            self.data_bind_addr = data_bind_addr
            # Extract port for advertised address if user supplied a bind addr
            _, port = _parse_tcp_host_port(self.data_bind_addr)

        advertise_host = advertise_host or os.getenv("DSA_P2P_HOST")  # None if advertise_host is None else advertise_host
        self.data_advertise_addr = compute_advertise_addr(
            backend_addr=self.backend_addr,
            port=port,
            override_host=advertise_host,
        )
        jaxns_logger.info(f"Resolved host: {self.data_advertise_addr}")

        # Lease and request tracking
        self.lock: threading.Lock | None = None
        self.result_records: dict[bytes, ResultRecord] = dict()  # lease_id -> lease record
        self.lb_epoch: bytes | None = None  # last seen LB epoch (bytes)

    @contextlib.contextmanager
    @abstractmethod
    def yield_service(self):
        """
        Yield a concrete service object exposing RPC methods.
        """
        ...

    def _now_ms(self) -> float:
        return time.monotonic() * 1000.0

    def _send_complete(self, backend: zmq.Socket, lease_id: bytes):
        if self.lb_epoch is None:
            raise RuntimeError("Cannot send COMPLETE before receiving first epoch from LB")
        backend.send_multipart([MSG_COMPLETE, lease_id, self.lb_epoch], copy=False)

    def _send_revoke_ack(self, backend: zmq.Socket, lease_id: bytes):
        backend.send_multipart([MSG_REVOKE_ACK, lease_id], copy=False)

    def _send_result(self, data: zmq.Socket, backend: zmq.Socket, client_id: bytes, lease_id: bytes, result,
                     status: bytes):
        attempt = 0
        while attempt < 2:
            try:
                out_bufs = []
                main_out = pickle.dumps(result, protocol=PICKLE_PROTO,
                                        buffer_callback=out_bufs.append)
                tail = [zmq.Frame(pb.raw()) for pb in out_bufs]
                data.send_multipart([client_id, MSG_RESULT, lease_id, status, zmq.Frame(main_out), *tail], copy=False)
                return
            except zmq.EHOSTUNREACH:
                attempt += 1
                jaxns_logger.warning(f"Client {client_id!r} unreachable when sending result, attempt {attempt} of 2")
                time.sleep(1)
                continue
            except Exception as e:
                msg = RuntimeError(f"Error pickling payload: {str(e)}")
                main_out = pickle.dumps(msg, protocol=PICKLE_PROTO)
                data.send_multipart([client_id, MSG_RESULT, lease_id, STATUS_ERR, zmq.Frame(main_out)], copy=False)
                return
        # Client unreachable; just ACK to LB and drop result on floor.
        # Tells LB to mark both ends idle again.
        self._send_complete(backend, lease_id)
        self.result_records.pop(lease_id, None)

    def _worker_thread(self, stop_event: threading.Event, work_q: queue.Queue):
        """
        Serial worker that executes target methods. Signals completion via inproc PUSH -> main thread PULL.
        """
        done_q = queue.Queue()

        def notifier_thread():
            done_push = self.new_socket(zmq.PUSH, connect="inproc://worker-done")
            try:
                while not stop_event.is_set():
                    try:
                        lease_id = done_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    # single-frame: lease_id
                    done_push.send(lease_id)
                    done_q.task_done()
            finally:
                pass

        fut_to_lease: dict[Future, bytes] = {}

        def done_cb(fut: Future):
            lease_id = fut_to_lease.pop(fut, None)
            if lease_id:
                done_q.put(lease_id)

        notifier = threading.Thread(target=notifier_thread, daemon=True)
        notifier.start()
        try:
            with (
                self.yield_service() as target,
                ThreadPoolExecutor(max_workers=1) as executor
            ):
                while not stop_event.is_set():
                    try:
                        method_name, args, kwargs, client_id, lease_id = work_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    with self.lock:
                        rr = self.result_records.get(lease_id, None)
                        if rr is None:
                            # revoked already; nothing to do
                            work_q.task_done()
                            continue

                        if hasattr(target, method_name):
                            fut = executor.submit(getattr(target, method_name), *args, **kwargs)
                        else:
                            fut = Future()
                            fut.set_exception(AttributeError(f"Method not found: {method_name}"))

                        fut_to_lease[fut] = lease_id

                        rr.future = fut
                        rr.mark_processing()
                        fut.add_done_callback(done_cb)
                        work_q.task_done()
                        del rr, fut, args, kwargs  # make sure no strong reference held between loop iters
        finally:
            notifier.join()

    def run(self) -> None:
        ctl = self.new_socket(zmq.SUB, connect=self.ctl_pub_addr)
        ctl.setsockopt_string(zmq.SUBSCRIBE, "")
        self.ack_startup()
        self.lock = threading.Lock()
        ident = f"{self.__class__.__name__}-{os.getpid()}".encode("ascii")
        backend = self.new_socket(zmq.DEALER, connect=self.backend_addr, identity=ident)
        # ROUTER_MANDATORY=1 -> Drops client with raised EHOSTUNREACH
        data = self.new_socket(
            zmq.ROUTER,
            bind=self.data_bind_addr,
            sockopts={zmq.ROUTER_MANDATORY: 1}
        )
        done_pull = self.new_socket(zmq.PULL, bind="inproc://worker-done")

        work_q = queue.Queue()
        stop_event = threading.Event()
        worker_thread = threading.Thread(target=self._worker_thread, args=(stop_event, work_q), daemon=True)
        worker_thread.start()

        poller = zmq.Poller()
        poller.register(backend, zmq.POLLIN)
        poller.register(ctl, zmq.POLLIN)
        poller.register(data, zmq.POLLIN)
        poller.register(done_pull, zmq.POLLIN)

        try:
            backend.send_multipart([MSG_READY, self.data_advertise_addr.encode("utf-8")], copy=False)
            next_hb = time.monotonic() * 1000.0 + HB_INTERVAL_MS
            while True:
                # HB to LB
                now_ms = self._now_ms()
                if now_ms >= next_hb:
                    backend.send_multipart([MSG_HEARTBEAT], copy=False)
                    next_hb = now_ms + HB_INTERVAL_MS

                # Reap missed result acks
                for lid, rr in list(self.result_records.items()):
                    if rr.state == ResultEnum.awaiting_ack and now_ms > rr.result_ack_deadline_ms:
                        # Missed result ack; send COMPLETE to free worker.
                        jaxns_logger.warning(f"Missed RESULT_ACK for lease {lid!r}, sending COMPLETE.")
                        self._send_complete(backend, lid)
                        self.result_records.pop(lid, None)
                        rr.future = None
                        del rr  # no more strong refs held now

                socks = dict(poller.poll(100))

                if ctl in socks and ctl.recv() == b"TERMINATE":
                    break

                if done_pull in socks:
                    # A worker thread has completed a task
                    lid = done_pull.recv()
                    with self.lock:
                        rr = self.result_records.get(lid, None)
                        if rr is None:
                            # Lease couldn't be revoked, as we only ACK if cancellation worked.
                            raise RuntimeError(f"Unknown lease_id {lid!r} for completed result.")
                        if rr.future is None:
                            raise RuntimeError(f"Lease {lid!r} has no associated future.")
                        cid = rr.client_id
                        if rr.aborted:
                            # Lease was revoked while processing; drop result on floor, but still ACK to LB.
                            self._send_complete(backend, lid)
                            self.result_records.pop(lid, None)
                            rr.future = None
                            del rr  # no more strong refs held now
                            continue
                        else:
                            rr.mark_awaiting_ack()
                            try:
                                self._send_result(data, backend, cid, lid, rr.future.result(), STATUS_OK)
                            except Exception as e:
                                self._send_result(data, backend, cid, lid, e, STATUS_ERR)
                            rr.result_ack_deadline_ms = self._now_ms() + RESULT_ACK_TIMEOUT_MS
                            # can actually remove the future reference now, since we don't need to cache results.
                            rr.future = None
                            del rr  # no more strong refs held now
                            continue

                if data in socks:
                    frames = data.recv_multipart(copy=False)
                    if len(frames) < 2:
                        raise RuntimeError(f"Expected >=2 frames [client_id, cmd, *args], got {len(frames)}")
                    client_id = bytes(frames[0])
                    cmd = bytes(frames[1])

                    if cmd == MSG_RESULT_ACK:
                        if len(frames) != 3:
                            raise RuntimeError(
                                f"Expected 3 frames [client_id, RESULT_ACK, lease_id], got {len(frames)}")
                        lease_id = bytes(frames[2])
                        with self.lock:
                            rr = self.result_records.get(lease_id, None)
                            if rr is None:
                                # lease was revoked already; send COMPLETE to LB to be safe.
                                self._send_complete(backend, lease_id)
                                del frames
                                continue
                            if rr.state == ResultEnum.awaiting_ack:
                                self._send_complete(backend, lease_id)
                                self.result_records.pop(lease_id, None)
                                del rr, frames
                                continue
                            else:
                                raise RuntimeError(
                                    f"Unexpected RESULT_ACK for lease_id {lease_id!r} in state {rr.state}")

                    elif cmd == MSG_RPC:
                        if len(frames) < 5:
                            raise RuntimeError(
                                f"Expected >=5 frames [client_id, RPC, lease_id, method_name, main, *bufs], got {len(frames)}")
                        lease_id = bytes(frames[2])
                        method_name = bytes(frames[3]).decode("utf-8")
                        main = memoryview(frames[4])
                        in_bufs = [memoryview(f) for f in frames[5:]]

                        with self.lock:
                            # Admission: authorize by (lease_id, client_id)
                            rr = self.result_records.get(lease_id, None)

                        if rr is None:
                            # The lease is invalid; either revoked or never granted.
                            self._send_result(data, backend, client_id, lease_id, RuntimeError("Unauthorized."),
                                              STATUS_INVALID_LEASE)
                            del frames
                            continue

                        if rr.client_id != client_id:
                            # Lease doesn't belong to this client
                            self._send_result(data, backend, client_id, lease_id, RuntimeError("Unauthorized."),
                                              STATUS_INVALID_LEASE)
                            # Don't touch record since client was invalid.
                            del rr, frames
                            continue

                        env = pickle.loads(main, buffers=in_bufs)
                        args = env.get("args", ())
                        kwargs = env.get("kwargs", {})
                        work_q.put((method_name, args, kwargs, client_id, lease_id))
                        # will mark processing when the worker thread picks it up
                        del rr, frames, args, kwargs, env, main, in_bufs
                        continue
                    else:
                        jaxns_logger.warning(f"Unknown data command {cmd!r} from client: {client_id!r}")
                        del frames
                        continue

                # Grants / revokes from LB (backend)
                if backend in socks:
                    frames = backend.recv_multipart(copy=False)
                    if len(frames) < 1:
                        raise RuntimeError(f"Expected >=1 frames [cmd, *args], got {len(frames)}")

                    cmd = bytes(frames[0])

                    if cmd == MSG_GRANT:
                        if len(frames) != 4:
                            raise RuntimeError(
                                f"Expected 4 frames [cmd, lease_id, client_id, lb_epoch], got {len(frames)}")
                        lease_id = bytes(frames[1])
                        client_id = bytes(frames[2])
                        lb_epoch = bytes(frames[3])
                        if self.lb_epoch is None:
                            self.lb_epoch = lb_epoch
                        elif lb_epoch != self.lb_epoch:
                            # LB restarted; any mismatch at LB will cause revoke downstream.
                            jaxns_logger.info("Detected LB restart.")
                            self.lb_epoch = lb_epoch

                        with self.lock:
                            self.result_records[lease_id] = ResultRecord(
                                lease_id=lease_id,
                                client_id=client_id,
                                future=None
                            )

                        backend.send_multipart([MSG_GRANT_ACK, lease_id, self.lb_epoch], copy=False)
                        del frames
                        continue

                    elif cmd == MSG_REVOKE:
                        if len(frames) != 2:
                            raise RuntimeError(f"Expected 2 frames [cmd, lease_id], got {len(frames)}")
                        lease_id = bytes(frames[1])
                        with self.lock:
                            rr = self.result_records.get(lease_id, None)
                            if rr is None:
                                #  Lease was already completed, or revoke arrives before grant; send REVOKE_ACK to be safe, as it's harmless and frees worker.
                                self._send_revoke_ack(backend, lease_id)
                                del frames
                                continue
                            if rr.state == ResultEnum.granted:
                                # We can simply drop; and ack
                                self._send_revoke_ack(backend, lease_id)
                                self.result_records.pop(lease_id, None)
                                del rr, frames
                                continue
                            elif rr.state == ResultEnum.processing:
                                # Try to cancel
                                if rr.future.cancel():
                                    # We were able to cancel before it started. Thus, ack, and drop
                                    self._send_revoke_ack(backend, lease_id)
                                    self.result_records.pop(lease_id, None)
                                else:
                                    # Can't stop, but mark aborted; when done, the result will be ignored, and COMPLETE sent to LB.
                                    rr.aborted = True
                                del rr, frames
                                continue
                            elif rr.state == ResultEnum.awaiting_ack:
                                # Already processed, and awaiting result ack; can't revoke, ignore.
                                del rr, frames
                                continue
                            else:
                                raise ValueError(f"Unknown state {rr.state} for lease_id {lease_id!r}")
                    else:
                        raise ValueError(f"Unexpected command from backend: {cmd}")

        except CtlTerminate:
            pass
        finally:
            stop_event.set()
            for s in (data, backend, done_pull, ctl):
                try:
                    poller.unregister(s)
                except Exception:
                    pass
            worker_thread.join()


# ============================================================
# Client (control-plane light LB centralised; data-plane is p2p to services)
# ============================================================
@dataclasses.dataclass(slots=True)
class ClientLeaseRecord:
    lease_id: bytes
    worker_id: bytes
    create_time_ms: float = dataclasses.field(default_factory=lambda: time.monotonic() * 1000.0)


@dataclasses.dataclass(slots=True)
class ClientWorkerRecord:
    worker_id: bytes
    socket: zmq.Socket
    endpoint: bytes


class ZMQRPCClient:

    def __init__(self, ident: str, frontend_addr: str, assign_timeout_ms: int | None = None, retries: int = 0) -> None:
        """
        Parameters
        ----------
        ident : str
            Logical owner tag included in generated request ids.
        frontend_addr : str
            LB frontend address (tcp://host:port).
        """
        self.ident = ident
        self.frontend_addr = frontend_addr
        self.assign_timeout_ms = int(assign_timeout_ms) if assign_timeout_ms is not None else None
        self.retries = int(retries)
        if self.retries < 0:
            raise ValueError("retries must be >= 0")

        self.ctx: zmq.Context | None = None
        self.req_push: zmq.Socket | None = None

        self.lb_epoch: bytes | None = None

        self.stop_event: threading.Event | None = None
        self.control_thread: threading.Thread | None = None
        self.future: Future | None = None

        # Pools and lease
        self.pool: dict[bytes, ClientWorkerRecord] = {}  # worker_id -> client worker record
        self.rpc_frames: list[zmq.Frame] | None = None
        self.lease: ClientLeaseRecord | None = None

    def __enter__(self):
        """
        Initialize ZMQ context and control socket; register for polling.
        """
        if self.ctx is not None:
            raise RuntimeError("ZMQRPCClient context already initialized")
        self.ctx = zmq.Context()
        self.client_id = f"client-{self.ident}-{os.getpid()}".encode("ascii")
        self.stop_event = threading.Event()
        self.control_thread = threading.Thread(
            name=f"zmq-rpc-client-{self.ident}",
            target=self._control_loop,
            args=(self.ctx, self.stop_event),
            daemon=True,
        )
        self.control_thread.start()

        self.req_push = self.ctx.socket(zmq.PUSH)
        self.req_push.connect("inproc://client-ctrl")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Tear down sockets and clear local state (linger=0 for fast shutdown)."""
        # Close worker sockets

        if self.stop_event is not None:
            self.stop_event.set()
        if self.control_thread is not None:
            self.control_thread.join(timeout=1.0)
        control_thread_alive = self.control_thread is not None and self.control_thread.is_alive()

        self.pool.clear()
        self.lease = None
        self.rpc_frames = None
        if self.future and not self.future.done():
            self.future.set_exception(TimeoutError("Client shutting down before result ready."))
        self.future = None
        self.lb_epoch = None

        for s in (self.req_push,):
            try:
                if s:
                    s.close(linger=0)
            except zmq.ZMQError:
                pass

        self.req_push = None

        try:
            if self.ctx:
                if control_thread_alive:
                    self.ctx.destroy(linger=0)
                else:
                    self.ctx.term()
        except zmq.error.ContextTerminated:
            pass
        self.ctx = None
        self.control_thread = None
        self.stop_event = None

    def _drop_worker(self, worker_id, poller):
        # remove the worker
        wr = self.pool.pop(worker_id, None)
        if wr is not None:
            try:
                poller.unregister(wr.socket)
            except Exception:
                pass
            try:
                wr.socket.close(linger=0)
            except Exception:
                pass

    def _drop_pool(self, poller):
        for wid, wr in list(self.pool.items()):
            self._drop_worker(wid, poller)
        self.pool.clear()

    def _control_loop(self, ctx: zmq.Context, stop_event: threading.Event):

        req_pull = ctx.socket(zmq.PULL)
        req_pull.bind("inproc://client-ctrl")
        frontend = self.ctx.socket(zmq.DEALER)
        frontend.setsockopt(zmq.IDENTITY, self.client_id)
        frontend.connect(self.frontend_addr)

        poller = zmq.Poller()
        poller.register(req_pull, zmq.POLLIN)
        poller.register(frontend, zmq.POLLIN)

        IDLE = b'idle'
        AWAITING_ASSIGN = b'awaiting_assign'
        AWAITING_RESULT = b'awaiting_result'

        state = IDLE

        tries = 0  # counts tries

        assign_deadline_ms: float | None = None

        try:
            while not stop_event.is_set():
                if assign_deadline_ms is not None:
                    now = time.monotonic() * 1000
                    if now >= assign_deadline_ms:
                        # Timeout waiting for assign
                        if self.lease is not None:
                            raise RuntimeError("Internal error: lease should be None when waiting for assign")
                        self._drop_pool(poller)
                        self.lb_epoch = None
                        future = self.future
                        self.future = None
                        self.rpc_frames = None
                        state = IDLE
                        assign_deadline_ms = None
                        if future is not None and not future.done():
                            future.set_exception(TimeoutError("Timeout waiting for assignment"))
                        continue
                socks = dict(poller.poll(100))
                if req_pull in socks:
                    self.rpc_frames = req_pull.recv_multipart(copy=False)
                    # method_name = frames[0]
                    # main = frames[1]
                    # tail = frames[2:]

                    # Get assignment
                    if state == IDLE:
                        frontend.send_multipart([MSG_REQUEST], copy=False)
                        state = AWAITING_ASSIGN
                        assign_deadline_ms = time.monotonic() * 1000 + self.assign_timeout_ms if self.assign_timeout_ms is not None else None
                        tries += 1
                        continue
                    else:
                        raise RuntimeError(f"Unexpected new request from {state}")

                for wid, wr in list(self.pool.items()):
                    data_sock = wr.socket
                    if data_sock in socks:
                        resp = data_sock.recv_multipart(copy=False)
                        if self.lease is None:
                            # Stray response from previous (likely canceled) request; drop it.
                            del resp
                            continue
                        if wid != self.lease.worker_id:
                            # Stray response from previous (likely canceled) request; drop it.
                            del resp
                            continue
                        if len(resp) < 1:
                            raise RuntimeError("Malformed response from worker")
                        cmd = bytes(resp[0])
                        if cmd == MSG_RESULT:
                            # RESULT path
                            if len(resp) < 4:
                                raise RuntimeError("Malformed RESULT response")
                            lid = bytes(resp[1])
                            if lid != self.lease.lease_id:
                                # Stray result from a previous (likely canceled) request; drop it.
                                del resp
                                continue
                            status = bytes(resp[2])
                            main = memoryview(resp[3])
                            in_bufs = [memoryview(f) for f in resp[4:]]

                            # Always send ACK RESULT for the matching lease_id
                            # This triggers sending a COMPLETE if service logic says it should.
                            data_sock.send_multipart([MSG_RESULT_ACK, lid], copy=False)
                            self.rpc_frames = None  # No need to send any more, can drop

                            # Handle statuses
                            future = self.future
                            self.future = None
                            state = IDLE
                            if status == STATUS_OK:
                                self.lease = None
                                if future is not None and not future.done():
                                    future.set_result(pickle.loads(main, buffers=in_bufs))
                            elif status == STATUS_ERR:
                                self.lease = None
                                if future is not None and not future.done():
                                    future.set_exception(pickle.loads(main, buffers=in_bufs))
                            elif status == STATUS_INVALID_LEASE:
                                self.lease = None
                                if future is not None and not future.done():
                                    future.set_exception(pickle.loads(main, buffers=in_bufs))
                            else:
                                raise RuntimeError(f"Unknown status '{status}' in RPC response")
                            del resp, main, in_bufs

                if frontend in socks:
                    frames = frontend.recv_multipart(copy=False)
                    if len(frames) < 1:
                        raise RuntimeError("Malformed response from LB")
                    cmd = bytes(frames[0])
                    if cmd == MSG_ASSIGN:
                        if len(frames) != 5:
                            raise RuntimeError("Malformed ASSIGN response from LB")
                        worker_id = bytes(frames[1])
                        endpoint = bytes(frames[2])
                        lease_id = bytes(frames[3])

                        if state != AWAITING_ASSIGN:
                            # Unexpected ASSIGN;
                            # Defensive: free the incorrectly granted lease at the LB
                            frontend.send_multipart([MSG_CANCEL, lease_id], copy=False)
                            continue

                        lb_epoch = bytes(frames[4])
                        if self.lb_epoch is not None and lb_epoch != self.lb_epoch:
                            # Drop pool; send new request.
                            self._drop_pool(poller)
                            self.lb_epoch = lb_epoch
                        elif self.lb_epoch is None:
                            self.lb_epoch = lb_epoch

                        self.lease = ClientLeaseRecord(
                            lease_id=lease_id,
                            worker_id=worker_id
                        )
                        wr = self.pool.get(worker_id, None)
                        if wr is not None:
                            if wr.endpoint != endpoint:
                                # Endpoint changed; drop and recreate
                                self._drop_worker(worker_id, poller)
                                wr = None
                        if wr is None:
                            s = self.ctx.socket(zmq.DEALER)
                            s.setsockopt(zmq.IDENTITY, self.client_id)
                            s.connect(endpoint.decode("utf-8"))
                            wr = self.pool[worker_id] = ClientWorkerRecord(
                                worker_id=worker_id,
                                socket=s,
                                endpoint=endpoint
                            )
                            poller.register(s, zmq.POLLIN)
                        if self.rpc_frames is None:
                            raise RuntimeError("No pending RPC frames after ASSIGN")
                        wr.socket.send_multipart([MSG_RPC, self.lease.lease_id, *self.rpc_frames], copy=False)
                        state = AWAITING_RESULT
                        tries = 0  # reset tries on success
                        assign_deadline_ms = None
                        continue
                    elif cmd == MSG_REVOKE:
                        if len(frames) != 3:
                            raise RuntimeError("Malformed REVOKE response from LB")
                        lease_id = bytes(frames[1])
                        reason = bytes(frames[2])
                        if self.lease is None:
                            # Stale revoke; ignore
                            continue
                        if lease_id != self.lease.lease_id:
                            # Stale revoke; ignore
                            continue

                        frontend.send_multipart([MSG_REVOKE_ACK, lease_id], copy=False)

                        # Handle reasons if successful revoke.
                        if reason == b'worker_dead':
                            # remove the worker
                            wid = self.lease.worker_id
                            self._drop_worker(wid, poller)
                        elif reason == b'worker_restart':
                            # Drop pool; send new request.
                            wid = self.lease.worker_id
                            self._drop_worker(wid, poller)
                        elif reason == b'wrong_epoch':
                            # Drop pool; send new request.
                            self._drop_pool(poller)
                            self.lb_epoch = None
                        self.lease = None

                        if tries < self.retries:
                            # Ask LB again
                            frontend.send_multipart([MSG_REQUEST], copy=False)
                            state = AWAITING_ASSIGN
                            tries += 1
                            assign_deadline_ms = time.monotonic() * 1000 + self.assign_timeout_ms if self.assign_timeout_ms is not None else None
                        else:
                            # Too many tries; give up
                            future = self.future
                            self.future = None
                            self.rpc_frames = None
                            state = IDLE
                            tries = 0
                            assign_deadline_ms = None
                            if future is not None and not future.done():
                                future.set_exception(TimeoutError("Maximum retries exceeded"))

                        continue
                    else:
                        raise RuntimeError(f"Unknown command from LB: {cmd}")

        finally:
            for s in (req_pull, frontend):
                try:
                    poller.unregister(s)
                except Exception:
                    pass
                try:
                    s.close(linger=0)
                except Exception:
                    pass
            self._drop_pool(poller)

    # ----- public RPC -----

    def _call(self, method_name: str, *args, **kwargs) -> Future:
        if self.future is not None and not self.future.done():
            return self.future
        if self.req_push is None:
            raise RuntimeError("Client not initialized; use 'with' context")

        try:
            bufs = []
            main = pickle.dumps({"args": args, "kwargs": kwargs},
                                protocol=PICKLE_PROTO,
                                buffer_callback=bufs.append)
        except Exception as e:
            raise RuntimeError(f"Failed to pickle arguments: {str(e)}")

        frames = [
            method_name.encode("utf-8"),
            zmq.Frame(main),
            *(zmq.Frame(pb.raw()) for pb in bufs),
        ]

        self.future = Future()
        self.req_push.send_multipart(frames, copy=False)

        return self.future

    def __getattr__(self, method_name: str):
        """
        The meta class provides the proxy. If not defined then we reach here.
        """
        raise NotImplementedError()


# ============================================================
# ABC glue (unchanged)
# ============================================================

class RPCClientMeta(ABCMeta):
    """
    Metaclass that auto-generates RPC stubs for abstract methods declared on base classes.

    For each abstract method name `m` found in bases, a proxy is created that forwards to
    `self._call(m, *args, **kwargs)`, and the abstract flag is cleared on the subclass.
    """

    def __new__(mcls, name, bases, namespace):
        """
        Create the subclass, injecting proxies for abstract methods inherited from bases.

        Raises
        ------
        RuntimeError
            If the subclass attempts to override an abstract method directly in `namespace`.
        """
        ns = dict(namespace)
        abstracts = set()
        for base in bases:
            abstracts |= getattr(base, "__abstractmethods__", set())
        for meth in abstracts:
            if meth in ns:
                raise RuntimeError(f"Cannot override abstract method {meth} in {name} as it is already defined.")
            proxy = mcls.make_proxy(meth)
            proxy.__name__ = meth
            proxy.__doc__ = f"auto-RPC stub for {meth}"
            proxy.__isabstractmethod__ = False
            ns[meth] = proxy
        return super().__new__(mcls, name, bases, ns)

    @classmethod
    def make_proxy(mcls, m):
        """Return a proxy function that forwards to `self._call(m, *args, **kwargs)`."""

        def proxy(self, *args, **kwargs):
            return self._call(m, *args, **kwargs).result()

        return proxy
