# dsa2000\_cal – Peer‑to‑Peer RPC over ZeroMQ with a Control‑Plane Load Balancer

A small, fast RPC fabric for distributed services.

* **Data‑plane:** clients talk directly to services (peer‑to‑peer) for request/response RPC.
* **Control‑plane:** a single **LoadBalancer** (LB) assigns workers and manages short‑lived **leases** with **credits**.
* **Stickiness + fairness:** clients stick to their most‑recent usable worker when idle capacity exists; otherwise a FIFO queue preserves fairness.
* **Epoch fencing:** protects the LB from stale `COMPLETE` signals across LB restarts.
* **Service heartbeats:** ensure worker liveness; dead workers are reaped and their leases revoked.
* **Client keepalives:** not implemented yet (see *Limitations*).

This document specifies the architecture, wire protocol, operational semantics, edge‑case handling, tuning and operational guidance.

---

## Contents

* [Quick start](#quick-start)
* [Architecture](#architecture)
* [Message framing & identities](#message-framing--identities)
* [Wire protocol](#wire-protocol)
* [Scheduling & leases](#scheduling--leases)
* [Liveness, epochs & restarts](#liveness-epochs--restarts)
* [Serialization & zero‑copy buffers](#serialization--zero-copy-buffers)
* [Ordering, retries & idempotency](#ordering-retries--idempotency)
* [Flow control & socket options](#flow-control--socket-options)
* [Security considerations](#security-considerations)
* [Edge cases fixed](#edge-cases-fixed)
* [Performance expectations](#performance-expectations)
* [Scaling expectations](#scaling-expectations)
* [Tuning guide](#tuning-guide)
* [Operational tips & troubleshooting](#operational-tips--troubleshooting)
* [Limitations & future work](#limitations--future-work)
* [Appendix: API sketch](#appendix-api-sketch)

---

## Quick start

The tests illustrate starting a cluster; abbreviated example:

```python
from dsa2000_fabric.process_manager import (
    create_random_control_address, create_random_ack_address,
    create_random_socket_address, ProcessManager,
)
from dsa2000_fabric.zmq_p2p import LoadBalancer

# plus your ServiceActor subclass and Client type

ctl_pub = create_random_control_address()
ack_rep = create_random_ack_address()
frontend = create_random_socket_address("frontend")
backend = create_random_socket_address("backend")

lb = LoadBalancer(ctl_pub, ack_rep, frontend, backend, credit_grant=1, K_hint=1)
workers = [ServiceActor(ctl_pub, ack_rep, backend) for _ in range(3)]
mgr = ProcessManager([lb, *workers], ctl_pub_addr=ctl_pub, ack_rep_addr=ack_rep)

mgr.start_all()
try:
    with Client(ident="A", frontend_addr=frontend, timeout_ms=3000) as A:
        print(A.echo("hello"))
finally:
    mgr.stop_all()
    mgr.print_tracebacks()
```

---

## Architecture

### Components

**LoadBalancer (control‑plane only)**

* ROUTER **frontend** (clients)
* ROUTER **backend** (services)
* Assigns services to clients via **leases** with finite **credits**.
* Maintains a per‑client MRU set for sticky assignment, plus an idle queue and a fairness queue.
* Accepts service `READY` announcements, `HEARTBEAT`, `GRANT_ACK`, and `COMPLETE` (credit burn‑down).

**Service actors (`RPCActor`)**

* ROUTER **data** socket bound on a TCP address (ephemeral by default).
* DEALER **control** socket connected to the LB backend.
* Announce `READY`, accept `GRANT`, send `GRANT_ACK`, handle `REVOKE`, send `COMPLETE` after serving a **valid** request, and send `HEARTBEAT` periodically.
* Implements `yield_service()` to expose a concrete RPC API.
* Computes and advertises a routable endpoint (supports IPv4/IPv6 and NAT override via `DSA_P2P_HOST`).

**Clients (`ZMQRPCClient`)**

* DEALER **control** socket to LB frontend.
* For each RPC: reuse existing credits or request a new lease; then speak P2P to the selected service over a DEALER socket keyed by the **client identity**.
* Tracks per‑worker `(lease_id, remaining_credits)` and a pool (≤ `K`) of connected worker sockets (MRU).

**ProcessManager** (test harness helper)

* Boots processes, collects profiles, performs orderly shutdown.

### High‑level flow (with `GRANT_ACK`)

```
Service                       LB                             Client
  | -- READY(endpoint) ------> |                              |
  |                             | <------ REQUEST -------------|
  | <--- GRANT(lease,cid,ep) -- |                              |
  | -- GRANT_ACK(lease,ep) ---> | -- ASSIGN(wid,ep,lease,k) -->|
  |<---------- P2P (lease, req) ------------------------------>|
  |----------- P2P (resp) -----------------------------------> |
  | -- COMPLETE(lease, req, epoch) --> | (credit--)            |
  |                                     revoke when credits==0 |
```

---

## Message framing & identities

* All control sockets are ROUTER/DEALER. The first frame on ROUTER sockets is the peer identity; *payload frames* follow.
* The data plane uses ROUTER (service) ↔ DEALER (client). **No empty delimiter frame** is used; both sides agree on fixed framing.
* **Identities:**

  * The client sets a single ZMQ identity (`client_id`) for *all* sockets it opens; services bind a ROUTER identity (`service_id`).
  * Each **lease** is bound to a tuple `(lease_id, client_id, service_id)`. A forged request with a mismatched `client_id` is rejected.

---

## Wire protocol

Below, frames shown **exclude** the leading ROUTER identity frame seen by receivers.

### Control‑plane (LB ↔ service)

* `READY` (service → LB)
  `[READY, data_endpoint]`

* `GRANT` (LB → service)
  `[GRANT, lease_id, client_id, epoch]`

* `GRANT_ACK` (service → LB)
  `[GRANT_ACK, lease_id, epoch]`
  *LB waits for this before sending `ASSIGN` to the client; it confirms the worker is ready and idle for the lease.*

* `REVOKE` (LB → service)
  `[REVOKE, lease_id]`

* `COMPLETE` (service → LB)
  `[COMPLETE, lease_id, req_id, epoch]`
  *LB ignores `COMPLETE` if `epoch` ≠ current epoch.*

* `HEARTBEAT` (service → LB)
  `[HEARTBEAT]`

### Control‑plane (LB ↔ client)

* `REQUEST` (client → LB)
  `[REQUEST]`

* `ASSIGN` (LB → client)
  `[ASSIGN, worker_id, data_endpoint, lease_id, credit_ascii, epoch]`

* `REVOKE` (LB → client)
  `[REVOKE, lease_id, reason]`

### Data‑plane (client ↔ service; direct P2P)

* `client → service`
  `[lease_id, method_name, req_id, main_pickle, *buffers]`

* `service → client`
  `[req_id, status, main_pickle, *buffers]`, where
  `status ∈ {b"ok", b"error", b"invalid_lease"}` and `main_pickle` is either the result or an exception.

---

## Scheduling & leases

* **Stickiness:** per‑client MRU set (size ≤ `K_hint`). If any MRU worker is idle, LB assigns one of those first.
* **Fairness:** if no sticky worker is idle, the client is enqueued (FIFO) and matched to the next idle worker.
* **Leases:** `lease_id` with `credits = credit_grant`. A **valid** request consumes one credit when the service sends `COMPLETE`.
* **Resend‑ASSIGN:** if a client requests while still holding a valid lease, LB re‑emits `ASSIGN` (amortizes control traffic and avoids stalls).
* **Authorization:** the service validates `(lease_id, client_id)` on every request; invalid or unauthorized requests return `invalid_lease` without burning credits.

---

## Liveness, epochs & restarts

* **Heartbeats:** services send `HEARTBEAT` every `hb_interval_ms` (default 500 ms).
* **Reaping:** LB marks a worker dead after `worker_dead_ms` without heartbeat (default 2000 ms). All leases pointing to that worker are revoked (`reason=b"worker_dead"`).
* **Epoch fencing:** LB generates a random `epoch` on startup/restart and includes it in `GRANT`/`ASSIGN`. Services remember the epoch and echo it on `COMPLETE`. LB ignores `COMPLETE` with a stale epoch. Clients clear local state if the epoch changes.
* **Grant handshake:** LB does not `ASSIGN` until it receives `GRANT_ACK`, ensuring the worker is ready to serve and preventing early client data‑plane sends to a stale or busy socket.

---

## Serialization & zero‑copy buffers

* **Main payload:** Python `pickle` is used for the envelope object (`{"args": ..., "kwargs": ...}` and results/exceptions).
* **Zero‑copy attachments:** large arrays/buffers are sent via `pickle`’s `buffer_callback` and carried as additional frames to avoid extra copies.
* **Compatibility:** sender and receiver must be ABI‑compatible for objects passed by buffer (e.g., NumPy dtype/endian). Prefer simple POD buffers and reconstruct types at the receiver.
* **Security:** never unpickle untrusted data. This system assumes trusted peers within a controlled network.

---

## Ordering, retries & idempotency

* **Per‑socket ordering:** ZeroMQ provides FIFO ordering per connection. When a client talks to multiple workers (pool size `K > 1`), ordering across workers is not defined.
* **Deduplication:** services reject duplicate `req_id` values and do not burn credits for duplicates.
* **Retries:** clients may retry on `invalid_lease` by dropping the local lease and re‑issuing the call. Application methods should be idempotent or tolerate replays where appropriate.

---

## Flow control & socket options

Recommended options and rationale:

* **Teardown:** set `linger=0` on shutdown to avoid stalls.
* **Immediate connect:** consider `ZMQ_IMMEDIATE=1` on DEALER sockets to avoid queueing messages before a TCP connection is fully established.
* **Backpressure:** tune `SNDHWM`/`RCVHWM` (high‑water marks). The control plane messages are small; the data plane should enforce sane HWMs to avoid unbounded memory use.
* **TCP keepalive:** enable keepalives at the OS/ZMQ level in cross‑host deployments.

---

## Security considerations

* **Lease binding:** every request is checked against `(lease_id, client_id)`; mismatches are rejected (`invalid_lease`).
* **Identity collisions:** using multiple sockets with the same ZMQ identity can cause delivery ambiguity on a ROUTER. The client uses a single identity across its sockets; tests include a distinct `rogue` identity for negative cases.
* **Untrusted inputs:** do not use pickle across trust boundaries; prefer explicit schemas if required.
* **Endpoint exposure:** services advertise reachable TCP endpoints. In NAT scenarios, set `DSA_P2P_HOST` to an externally routable address.

---

## Edge cases fixed

1. **Grant/assign race**
   `GRANT_ACK` handshake prevents `ASSIGN` from reaching the client before the worker is ready.

2. **COMPLETE epoch mismatch**
   LB ignores stale `COMPLETE`; workers reset state on LB epoch changes.

3. **Invalid/unauthorized requests**
   Rejected with `invalid_lease`; credits are preserved.

4. **Idle deque drift**
   LB salvage path reconciles registry and idle deque to avoid starvation.

5. **Safe revoke on worker disappearance**
   LB revokes leases and avoids accessing removed worker records.

6. **Identity collision in tests**
   Tests use distinct identities to ensure deterministic routing.

---

## Performance expectations

* **Control‑plane lightness:** `REQUEST`/`ASSIGN`/`GRANT`/`GRANT_ACK`/`COMPLETE` are tiny. The LB is not on the data hot path.
* **Per‑call overhead:**

  * Lease reuse: 0 control hops (data‑plane only).
  * New lease: one `REQUEST`; one `GRANT` + `GRANT_ACK`; one `ASSIGN`.
  * One `COMPLETE` per valid call.
* **Throughput knobs:** increase `credit_grant` to amortize control messages; adjust `K_hint` for cache locality.
* **Latency:** sub‑ms control hops on loopback; data‑plane latency dominated by the service.
* **Complexity:** O(1) fast paths (deques, dicts); salvage O(#workers) in rare reconciliation.

---

## Scaling expectations

* **Workers:** scale linearly; each binds its own data endpoint. Data never transits through LB.
* **Clients:** scale to many; memory scales with active leases and MRU/fairness queues.
* **Single LB:** current design uses a single LB instance. Horizontal sharding would require partitioning or discovery (out of scope).
* **Cross‑host:** works across machines if endpoints are routable. Consider firewalls/NAT and keepalives.

---

## Tuning guide

Key parameters and defaults:

| Component | Parameter        |      Default | Effect                                                    |
| --------- | ---------------- | -----------: | --------------------------------------------------------- |
| LB        | `credit_grant`   |          `1` | Credits per lease; raise to amortize control traffic      |
| LB        | `K_hint`         |          `1` | Max sticky MRU size per client                            |
| LB        | `worker_dead_ms` |       `2000` | Reap threshold for missing heartbeats                     |
| Service   | `hb_interval_ms` |        `500` | Heartbeat cadence                                         |
| Client    | `timeout_ms`     | app‑specific | Per‑call timeout (service latency + network RTT + margin) |
| Client    | `K`              |          `1` | Max connected worker sockets in MRU pool                  |

Additional suggestions:

* **Timeout sizing:** `timeout_ms ≥ p99(service) + 1×RTT + jitter`.
* **HWMs:** set explicit `SNDHWM`/`RCVHWM` for data sockets when large payloads are possible.

---

## Operational tips & troubleshooting

**Timeout waiting for ASSIGN**
No idle workers and no valid lease; or `GRANT_ACK` not received due to a dead worker. Check worker heartbeats and LB logs.

**`invalid_lease` from service**
Stale or forged lease; client identity mismatch; LB epoch changed—client should drop local state and request a new lease.

**Workers never go idle / credits not consumed**
Verify `COMPLETE` framing includes `epoch` and is sent **only** for valid requests.

**LB `KeyError` on revoke (pre‑fix)**
Resolved by checking worker presence before reuse.

**Identity ambiguity**
Avoid multiple sockets with the same identity; use distinct identities for tests and tools.

---

## Limitations & future work

* **No client TTL:** leases do not expire due to client silence alone; only via credit burn‑down or worker death/revoke. Add client heartbeats or idle timers.
* **No auto‑restart:** the LB reaps dead workers but does not respawn them.
* **Single LB / no HA:** epoch fencing protects consistency across restarts, but availability is single‑instance.
* **At‑least‑once semantics:** duplicates are rejected per `req_id` within a worker process; exactly‑once across crashes would require persistence (out of scope).

---

## Appendix: API sketch

**`class LoadBalancer`**

* `credit_grant: int` — credits per lease (≥ 1)
* `K_hint: int` — per‑client MRU size (≥ 1)
* `worker_dead_ms: int` — heartbeat reaping threshold

**`class RPCActor`**

* Override `yield_service()` to yield an object exposing RPC methods.
* Sends `HEARTBEAT` every `hb_interval_ms` (default 500 ms).
* On `GRANT`, records `(lease_id, client_id, epoch)` and replies `GRANT_ACK`.

**`class ZMQRPCClient`**

* `timeout_ms` — per‑call timeout
* `K` — max worker sockets in MRU pool (default 1)
* Call RPC methods directly (generated proxies) or via `_call(name, *args, **kwargs)`.
