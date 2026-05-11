# JAXNS v3 Process-Isolated ZMQ Runtime

Status: v3 runtime target contract.
Source: user runtime architecture proposal.

This note defines the target process-isolated runtime topology for unreleased
v3 runtime work. It is deliberately independent of the current code state. v3
has no backwards-compatibility requirement with earlier unreleased runtime
slices: implementation and tests should target this contract rather than carry
forward direct load-balancer-to-worker paths, threaded worker boundaries, or
remote constrained-sampler worker payloads.

The ordinary v3 local-LB path is the process-isolated topology described here.
There is no legacy local-LB compatibility branch to maintain because v3 has not
been released. Any old in-process worker scheduler, direct
load-balancer-to-worker route, or coarse constrained-sampler worker payload may
exist only as an explicitly named test-only or internal helper, never as the
default or public v3 runtime path.

## Motivation

JAX does not interact well with threaded worker execution as a durable runtime
boundary. Threaded local workers can share interpreter state, JAX runtime state,
compilation caches, and device resources in ways that are hard to reason about
and hard to clean up after failures. The long-term runtime boundary should be
process isolation.

The statistical nested-sampling core should still be single and shared across
local and distributed deployments. Process isolation changes where worker
execution happens, not the race-tree, shrinkage, allocation, retry, or
acceptance semantics.

## Target Architecture

The target runtime has four durable ownership roles:

- Load balancer process: owns public runner creation, node registration,
  scheduling policy, transport/work identity, and client-visible lifecycle.
- Runner: owns statistical parent task identity, local constrained-sampler
  state, accepted child integration, and the acceptance ledger for one submitted
  nested-sampling run.
- Node process manager: owns one node's process tree. It starts exactly one
  node ingress/coordinator process plus many local worker processes, supervises
  them, and tears them down as one node-scoped unit.
- Worker process: owns one isolated execution process. Ordinary v3 worker work
  is deterministic likelihood evaluation only. Each worker process handles at
  most one active likelihood evaluation at a time.

The node ingress/coordinator is a single node-local actor process with two
responsibilities. Its ingress side talks to the load balancer over the
inter-node control/work transport. Its coordinator side fans work out to local
worker processes over node-local IPC endpoints. The load balancer never opens
per-worker connections and never schedules directly against worker sockets; it
schedules against node-advertised capacity.

The load balancer, node ingress/coordinator, and workers should be implemented
as `ZMQActor` processes. One node process manager owns exactly one node
ingress/coordinator process plus that node's worker processes. The owner of a
process tree is responsible for robust `try`/`finally` teardown: start the actor
processes, connect or bind their endpoints, register them with their parent, and
always request shutdown and join or terminate owned processes on context exit or
failure.

The public client should hide these mechanics. A local user should interact
with a context such as:

```python
with LoadBalancerClient(address="local") as lb:
    lb.add_node(...)
    lb.add_workers(...)
    sampler = lb.get_nested_sampler(model, args=args, params=params)
    result = sampler.run_until_goal(...)
```

In this sketch, `add_node(...)` creates or attaches one node process manager,
and `add_workers(...)` starts worker processes under that node. A convenience
implementation may create the default local node lazily on the first
`add_workers(...)` call, but the ownership contract remains the same: workers
belong to a node process manager, and the node ingress/coordinator is the only
node-local process that talks to the load balancer.

The client context owns the local process managers it starts and tears them down
in reverse ownership order. Remote clients that connect to an existing
`tcp://...` load balancer own only their client connection unless they also
explicitly start a node process manager.

## Connection Topology

The load balancer should not maintain a TCP connection to every worker process.
At scale, that creates unnecessary connection fan-out, a larger failure surface,
and more cross-node bookkeeping in the load balancer.

Use one ingress/coordinator process per node:

```text
client process
    |
    | tcp:// or local control endpoint
    v
load-balancer actor
    |
    | tcp:// node control/work endpoint
    v
node ingress/coordinator actor process
    |
    | ipc:// random /tmp paths
    v
worker actor processes
```

The load balancer talks to each node ingress. The node coordinator fans work out
to local workers and returns worker responses through the node ingress to the
load balancer. Local workers on the same node communicate with the node
coordinator over `ipc://` endpoints using random paths under `/tmp`, for example
a node-owned temporary directory with random endpoint filenames. These paths are
owned by the node process manager and removed during teardown.

TCP remains the inter-node transport. IPC is the local transport between a node
coordinator and its worker processes because it avoids unnecessary local TCP
connection overhead while preserving process isolation. The design must not
depend on stable IPC filenames, predictable port numbers, or global worker
socket identities.

## Likelihood-Evaluation Dispatch Runtime

The durable v3 runtime should make likelihood evaluation, not constrained
sampling, the smallest remote work unit. Constrained samplers run in the local
runner process, or in local parallel runner threads/processes when that is the
selected execution mode. They propose prior-space coordinates, own slice or
trajectory state, enforce strict contour logic, retain phantom likelihood
clusters, and decide whether another likelihood probe is needed. Remote workers
only evaluate deterministic likelihood probes.

This split keeps the statistical sampler close to the race-tree coordinator
while still isolating JAX model execution in worker processes. It also avoids
shipping large sampler state, mutable chain state, seed points, direction
snapshots, phantom buffers, or per-step trajectory objects across the transport
for each proposal.

This section supersedes older design text that described ordinary remote work
as serialized constrained-sampler tasks or described load-balancer/runner
creation as ahead-of-time likelihood compilation. Runner creation or an
explicit identity-registration step may make model bytes, args, params, dtype
policy, device class, and expected `U` tree available under
`RuntimeCompileIdentity`. Worker-local JIT compilation happens on the first
matching likelihood request for that identity/device class.

### Work Unit Shape

A likelihood work unit contains exactly the data needed for one deterministic
model evaluation:

```text
LikelihoodEvalRequest
    protocol_version: int
    runner_id: str
    task_id: str
    attempt_id: str
    transport_id: str
    compile_identity_digest: str
    eval_id: str
    U_bytes: bytes
    U_shape_tree: ShapeDtypeTree
    requested_dtype_policy: str
    deadline_ms: int | None
```

The worker response is intentionally small:

```text
LikelihoodEvalResponse
    protocol_version: int
    runner_id: str
    task_id: str
    attempt_id: str
    transport_id: str
    compile_identity_digest: str
    eval_id: str
    status: "ok" | "failed"
    log_L: float | None
    error_type: str | None
    error_message: str | None
    worker_id: str
    cache_event: "compile" | "hit" | "rejected"
    elapsed_seconds: float
```

Only `U` is pickled or otherwise serialized per likelihood evaluation. The
scalar `log_L` is the only successful result value returned to the sampler.
Model bytes, `args`, `params`, sampler configuration, and direction snapshots
are not resent per evaluation. The worker obtains them from the runner's
`RuntimeCompileIdentity` cache, populated during runner creation or a separate
identity-registration step before ordinary evaluation requests.

`U_shape_tree` records the pytree structure, leaf shapes, and dtypes expected by
the compiled likelihood. It is metadata for validation and cache selection, not
a second copy of the payload. A worker rejects a request whose `U` pytree does
not match the identity's declared static tree.

### Runtime Topology

Likelihood dispatch uses the same process-isolated actor hierarchy:

```text
runner and local constrained samplers
    |
    | likelihood eval requests, one U per request
    v
load-balancer actor
    |
    | node-level scheduling
    v
node ingress/coordinator actor process
    |
    | ipc:// local worker routing
    v
likelihood worker process
```

The runner owns parent selection, local constrained-sampler execution, in-flight
statistical task metadata, and the acceptance ledger. The load balancer owns
fair scheduling across runners and nodes. Each node process manager owns one
ingress/coordinator process and many local worker processes. The node
ingress/coordinator owns worker supervision, node-local capacity accounting, and
local routing. Worker processes own only model payload caches, JAX compilation
state, device state, and deterministic evaluation of `U` to `log_L`.

The same runner can issue many independent likelihood requests at once. This is
the mechanism that addresses parallel parent dispatch: multiple parent tasks
can run local constrained samplers concurrently, and every sampler can overlap
its pending likelihood probes through the shared worker pool. Completion order
does not define statistical order. The runner accepts completed child samples
through the existing task ledger, and only accepted children mutate out-degrees,
phantom clusters, supremum state, or allocation diagnostics.

Each likelihood worker process has capacity for at most one active likelihood
evaluation at a time. The node ingress/coordinator advertises aggregate
node-local capacity to the load balancer and schedules against concrete local
worker capacity. Excess requests are queued at the load balancer, node
coordinator, or both according to the scheduling policy. Concurrency comes from
multiple local parent sampler tasks producing demand and from multiple worker
processes consuming it, not from multiplexing concurrent evaluations inside one
worker process.

### Determinism And Failure Semantics

Likelihood workers are deterministic functions of:

- the compile identity's model bytes, args, params, dtype policy, and device
  class;
- the received `U` payload;
- worker-local JAX numerical behavior for the selected device class.

Workers must not draw sampler PRNG keys, choose directions, choose seeds, update
allocation state, or decide acceptance. They must not retain hidden mutable
problem state that changes results for the same `(compile_identity, U)` pair.
If a model has user-visible mutable global state, that state must either be
captured in the compile identity or rejected as unsupported for process
dispatch.

Failures are attempt-local:

- malformed payload, shape mismatch, cache miss for an unknown identity, worker
  exception, timeout, or process death returns or implies a failed likelihood
  evaluation;
- a failed evaluation is reported to the local sampler that requested it;
- the sampler may fail the current parent task, retry a probe, or continue
  according to its bounded trajectory/slice policy, but no statistical state
  mutates until a full child result is accepted;
- duplicate, stale, or late likelihood responses are ignored unless their
  `(task_id, attempt_id, transport_id, eval_id)` still matches live sampler
  demand;
- retrying a parent task reuses the statistical `task_id` contract and creates a
  new `attempt_id`; retrying a likelihood probe within the same sampler attempt
  creates a new `transport_id` or `eval_id` according to the transport layer.

Successful responses may be cached by the runner for idempotence during retry
bookkeeping, but the statistical result remains the constrained sampler's child
sample, not a collection of independent likelihood results.

### JAX Cache And Shape Rules

Worker-side compilation must keep JAX shapes static and predictable:

- `RuntimeCompileIdentity` includes model bytes, args, params, dtype policy,
  device class, `U` pytree structure, and leaf shapes/dtypes.
- Per-evaluation requests may vary values in `U`, but not the pytree structure,
  rank, leaf shapes, or dtypes for a compiled identity.
- Workers JIT the model on first matching work for an identity and device
  class, then reuse that compiled callable for matching `U_shape_tree` values.
- A shape change is either a rejected request or a new compile identity. Silent
  recompilation under the same identity is not allowed.
- Batched likelihood evaluation may be added only with a fixed batch shape in
  the identity. Variable-length batches must be padded and accompanied by a
  static mask, or trimmed in NumPy before calling JAX so compiled functions do
  not see dynamic leading dimensions.
- Candidate lists, parent pools, phantom arrays, and trajectory buffers are
  trimmed, packed, or padded outside JAX. Dynamic JAX array shapes in long runs
  are a correctness and performance hazard because they cause repeated
  compilation.

These rules are deliberately stricter than the transport schema. The transport
can carry any pickleable `U`, but a worker may only execute payloads that match
the static compile contract.

## Lifecycle Boundaries

Ownership should be explicit:

- `LoadBalancerClient(address="local")` owns the load-balancer process it
  starts, any node process managers it starts through public methods, and every
  node ingress/coordinator and worker process owned by those node process
  managers.
- `LoadBalancerClient(address="tcp://...")` owns the client connection by
  default. If it starts a local node under that load balancer, it owns that node
  process manager and its descendants.
- A node process manager owns one node ingress/coordinator process, its local
  worker processes, and local IPC endpoint paths.
- The node ingress/coordinator process owns node registration, heartbeat,
  node-local worker routing, and node-local capacity reporting while it is
  alive.
- A worker owns only its process-local JAX runtime state, model cache, compiled
  likelihood callables, at most one active likelihood evaluation, and actor
  socket resources.

Shutdown order should be child first:

1. Stop accepting new sampler work from public clients.
2. Revoke or drain in-flight work according to the runner failure contract.
3. Ask node ingress/coordinators to stop dispatching new worker work.
4. Ask workers to shut down, then join or terminate them with bounded timeouts.
5. Close node ingress/coordinator sockets and remove IPC paths.
6. Close load-balancer sockets and join or terminate the load-balancer process.

All of these steps should be driven from `try`/`finally` blocks or equivalent
context-manager cleanup so partial startup failures do not leave orphaned JAX
processes or stale IPC endpoints.

## Dynamic Actor Extension

The managed actor set should be extendable while a client context is open.
Examples include adding a GPU node after a CPU-only run has started, adding more
local CPU workers under an existing node process manager, or replacing a failed
node ingress/coordinator process and its worker descendants.

Dynamic extension must preserve ownership:

- Register the new node process manager, or the new worker process under its
  owning node process manager, before advertising it as schedulable.
- If registration with the parent actor fails, immediately tear down the new
  process and remove its local endpoints.
- If advertisement succeeds, include the actor in normal reverse-order context
  cleanup.
- Keep task identity, retry identity, and transport delivery identity
  independent of actor membership changes.
- Do not mutate race-tree state when adding or removing actors. Only accepted
  sampler results mutate statistical state.

The load balancer may update scheduling capacity as nodes and workers join or
leave, but a runner's acceptance ledger remains the source of truth for whether
completed work can update out-degrees or phantom clusters.

## Failure Semantics

Failures should be explicit and local where possible:

- Worker process failure: the node ingress/coordinator marks assigned attempts
  failed, reports failed attempts to the load balancer, and optionally asks the
  node process manager to start replacement workers if policy allows. The
  runner may retry using the same statistical `task_id` and a new
  `attempt_id`.
- Node ingress/coordinator failure: the load balancer marks all in-flight
  attempts on that node failed or unknown according to the retry contract,
  removes the node from schedulable capacity, and lets owner cleanup terminate
  remaining local descendants when possible.
- Load-balancer failure: clients and nodes observe connection closure and clean
  up their owned processes. No remote component should assume a task was
  accepted unless the runner acceptance ledger recorded it before failure.
- Client context failure: the client tears down only the processes it owns. A
  remote shared load balancer remains alive unless the client explicitly owns
  it.
- Partial startup failure: any already-started child processes are shut down in
  reverse order before the exception is re-raised with context.

Dispatch is non-mutating. Pending dispatch, failed attempts, revoked work,
duplicate completions, stale completions, and retried attempts must not update
out-degrees or phantom clusters until the runner accepts exactly one completion
for the statistical task.

## Why This Avoids Threaded-JAX Hazards

Process isolation gives each worker its own Python interpreter, JAX runtime
state, compilation behavior, device initialization, and failure boundary. A
stuck or failed worker can be terminated without relying on thread cancellation
inside JAX or Python. Local IPC keeps same-node communication efficient while
retaining the operational benefits of process boundaries.

The node ingress/coordinator keeps process isolation from turning into a large
TCP mesh. The load balancer schedules against nodes and advertised capacity; the
node coordinator handles local worker fan-out and worker-local failures. This
keeps load-balancer state closer to the statistical runtime responsibilities and
keeps low-level process supervision near the processes being supervised.

## Public Client Surface

The public load-balancer client should expose runtime operations, not process
manager details. Candidate responsibilities:

- create or connect to a load balancer;
- add a node process manager with one ingress/coordinator process;
- add local worker processes under a node process manager;
- add remote-node capacity by connecting a node ingress/coordinator to the load
  balancer;
- create nested-sampler runners;
- run, resume, revoke, and close runners;
- wait for shutdown when acting as a worker-node process.

The client should hide endpoint generation, process start order, ZMQ socket
creation, and `try`/`finally` cleanup. It should still expose enough diagnostics
to explain which nodes and workers exist, which actor owns them, and how failed
attempts were handled.

## Implementation Target

Because v3 is unreleased, this design is a target contract rather than a
compatibility migration. Implementation slices may be incremental, but tests
and review should assert the target behavior directly. Do not add or preserve a
legacy compatibility path for ordinary `LoadBalancerClient(address="local")`
runs; compatibility with earlier unreleased v3 slices is not a requirement.
Old runtime shapes may remain only when explicitly named as test-only/internal
helpers:

1. Every local or remote worker is a separate process.
2. Every node has one node process manager that owns one node
   ingress/coordinator process and all worker processes on that node.
3. The load balancer communicates with node ingress/coordinator processes, not
   individual workers.
4. Node coordinators communicate with local workers over random `ipc://`
   endpoints under `/tmp` and remove those endpoints during teardown.
5. Ordinary remote work units are likelihood-eval requests containing one `U`
   payload, not serialized constrained-sampler tasks.
6. Worker capacity is process count: one active likelihood evaluation per worker
   process.
7. Statistical mutation remains runner-owned and happens only when the runner
   accepts a completed constrained-sampler child.
8. Runtime diagnostics expose node process managers, node
   ingress/coordinators, worker process counts, per-worker capacity, IPC
   endpoint ownership, cache events, dispatch latency, and retry/cancellation
   outcomes.

The implementation should avoid changing statistical acceptance behavior. Any
observable changes should be runtime lifecycle, failure reporting, scheduling
topology, or diagnostics changes.
