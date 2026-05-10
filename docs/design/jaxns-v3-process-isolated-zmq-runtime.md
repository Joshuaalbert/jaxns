# JAXNS v3 Process-Isolated ZMQ Runtime

Status: future-runtime design note.
Source: user runtime architecture proposal.

This note captures a proposed durable runtime architecture for future v3
runtime work. It is not a current Ticket 0014 acceptance requirement and does
not claim that the repository already implements this topology.

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

## Architecture

The proposed runtime has three actor roles:

- Load balancer: owns public runner creation, node registration, scheduling
  policy, task identity, and client-visible lifecycle.
- Node coordinator: owns one machine or process group, starts and supervises
  local worker actors, and fans load-balancer work out to local workers.
- Worker: owns one isolated constrained-sampler or likelihood-evaluation
  execution process.

Each role should be implemented as a `ZMQActor` owned by a `ProcessManager`.
The owner of a process is responsible for robust `try`/`finally` teardown:
start the actor process, connect or bind its endpoints, register it with its
parent, and always request shutdown and join or terminate the process on context
exit or failure.

The public client should hide these mechanics. A local user should interact
with a context such as:

```python
with LoadBalancerClient(address="local") as lb:
    lb.add_node(...)
    lb.add_workers(...)
    sampler = lb.get_nested_sampler(model, args=args, params=params)
    result = sampler.run_until_goal(...)
```

The client context owns the local process managers it starts and tears them down
in reverse ownership order. Remote clients that connect to an existing
`tcp://...` load balancer own only their client connection unless they also
explicitly start node or worker processes.

## Connection Topology

The load balancer should not maintain a TCP connection to every worker process.
At scale, that creates unnecessary connection fan-out, a larger failure surface,
and more cross-node bookkeeping in the load balancer.

Use one node coordinator per node:

```text
client process
    |
    | tcp:// or local control endpoint
    v
load-balancer actor
    |
    | tcp:// node control/work endpoint
    v
node-coordinator actor
    |
    | ipc:// random /tmp paths
    v
worker actor processes
```

The load balancer talks to each node coordinator. The node coordinator fans work
out to local workers and returns worker responses to the load balancer. Local
workers on the same node should communicate with the node coordinator over
`ipc://` endpoints using random paths under `/tmp`, for example a per-context
temporary directory with random endpoint filenames. These paths should be
created with ownership scoped to the process manager and removed during
teardown.

TCP remains the inter-node transport. IPC is the local transport between a node
coordinator and its worker processes because it avoids unnecessary local TCP
connection overhead while preserving process isolation.

## Lifecycle Boundaries

Ownership should be explicit:

- `LoadBalancerClient(address="local")` owns the load-balancer process it
  starts, any node coordinators it starts through public methods, and any worker
  actors started under those nodes.
- `LoadBalancerClient(address="tcp://...")` owns the client connection by
  default. If it starts a remote or local node under that load balancer, it owns
  that node process manager and its descendants.
- A node coordinator owns its local worker process managers and local IPC
  endpoint paths.
- A worker owns only its process-local JAX runtime state, model cache, sampler
  execution state for assigned tasks, and actor socket resources.

Shutdown order should be child first:

1. Stop accepting new sampler work from public clients.
2. Revoke or drain in-flight work according to the runner failure contract.
3. Ask node coordinators to stop dispatching new worker work.
4. Ask workers to shut down, then join or terminate them with bounded timeouts.
5. Close node coordinator sockets and remove IPC paths.
6. Close load-balancer sockets and join or terminate the load-balancer process.

All of these steps should be driven from `try`/`finally` blocks or equivalent
context-manager cleanup so partial startup failures do not leave orphaned JAX
processes or stale IPC endpoints.

## Dynamic Actor Extension

The managed actor set should be extendable while a client context is open.
Examples include adding a GPU node after a CPU-only run has started, adding more
local CPU workers, or replacing a failed node coordinator.

Dynamic extension must preserve ownership:

- Register the new node or worker process manager with the owning context before
  advertising it as schedulable.
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

- Worker process failure: the node coordinator marks assigned attempts failed,
  reports failed attempts to the load balancer, and optionally starts
  replacement workers if policy allows. The runner may retry using the same
  statistical `task_id` and a new `attempt_id`.
- Node coordinator failure: the load balancer marks all in-flight attempts on
  that node failed or unknown according to the retry contract, removes the node
  from schedulable capacity, and lets owner cleanup terminate remaining local
  descendants when possible.
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

The node coordinator keeps process isolation from turning into a large TCP mesh.
The load balancer schedules against nodes and advertised capacity; the node
coordinator handles local worker fan-out and worker-local failures. This keeps
load-balancer state closer to the statistical runtime responsibilities and
keeps low-level process supervision near the processes being supervised.

## Public Client Surface

The public load-balancer client should expose runtime operations, not process
manager details. Candidate responsibilities:

- create or connect to a load balancer;
- add a node coordinator;
- add local workers under a node;
- add remote-node capacity by connecting a node coordinator to the load
  balancer;
- create nested-sampler runners;
- run, resume, revoke, and close runners;
- wait for shutdown when acting as a worker-node process.

The client should hide endpoint generation, process start order, ZMQ socket
creation, and `try`/`finally` cleanup. It should still expose enough diagnostics
to explain which nodes and workers exist, which actor owns them, and how failed
attempts were handled.

## Migration Notes From Current Local LoadBalancerClient

The current local path should migrate in layers:

1. Preserve the public `LoadBalancerClient(address="local")` entry point and
   runner-facing methods.
2. Move local worker execution behind a process-owned worker actor instead of a
   threaded or in-process worker boundary.
3. Insert a node coordinator even for single-node local runs, so local and
   distributed topologies share the same load-balancer-to-node contract.
4. Replace direct load-balancer-to-worker bookkeeping with node-level capacity
   and node-mediated worker dispatch.
5. Keep task identity, attempt identity, acceptance-ledger semantics, and
   diagnostics compatible with the existing v3 runtime contract.
6. Add dynamic actor registration only after fixed local process topology has
   clear ownership and shutdown tests.

The migration should avoid changing statistical acceptance behavior. Any
observable changes should be runtime lifecycle, failure reporting, scheduling
topology, or diagnostics changes.
