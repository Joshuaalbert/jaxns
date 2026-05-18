# Ticket 0019: Process-Isolated Worker Topology

Branch: `feature/v3-process-isolated-worker-topology`
Priority: after Ticket 0018 correctness acceptance, before any further
likelihood-dispatch performance claims
Depends on: Ticket 0005, Ticket 0007, Ticket 0009, Ticket 0011, Ticket 0016,
Ticket 0017, Ticket 0018
Design docs:

- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-process-isolated-zmq-runtime.md`
- `docs/design/jaxns-v3-validation-plan.md`
- `docs/design/interface/run_pattern.py`
- `docs/implementation_orchestration/tickets/0009-distributed-runtime-and-async-identity.md`
- `docs/implementation_orchestration/tickets/0016-standard-problem-speed-benchmarks.md`
- `docs/implementation_orchestration/tickets/0018-likelihood-eval-dispatch-runtime.md`

## Goal

Fix the non-scaling worker wall time from Ticket 0018 by replacing the ordinary
local-LB likelihood worker boundary with the accepted v3 process topology:

- one node process manager owns exactly one node ingress/coordinator process
  plus many local likelihood worker processes;
- the local load balancer itself is a managed actor/process owned by the local
  client context, not only an in-process facade;
- the load balancer talks to node ingress/coordinator processes, not directly to
  worker sockets;
- the node ingress/coordinator fans likelihood work out to local workers over
  random `ipc://` endpoints under `/tmp`;
- each worker process evaluates exactly one `U -> scalar log_L` request at a
  time;
- each worker JITs and caches the registered likelihood on first matching work
  for a stable `RuntimeCompileIdentity`;
- constrained samplers remain local to the runner and may dispatch likelihood
  evaluations concurrently through the shared worker pool.

This ticket is a topology replacement, not another micro-optimization of the
accepted local first-layer scheduler. The intended outcome is that increasing
worker count exposes real process-level likelihood capacity and improves full
8D `basic_mvn` wall time.

## Supersession And No Compatibility Requirement

Ticket 0019 intentionally supersedes the unreleased Ticket 0018 ordinary
local-LB implementation detail where likelihood workers are in-process scheduler
objects. Because v3 is unreleased, do not keep backwards compatibility for that
local in-process worker boundary. The ordinary v3 local-LB path after this
ticket is the process-isolated load-balancer/node/worker topology. There is no
legacy compatibility branch to maintain unless it is explicitly named as a
test-only or internal helper and is not reachable from the public v3 runtime
path.

For ordinary v3 local-LB runs after this ticket:

- `LoadBalancerClient(address="local")` starts or attaches to a local
  managed load-balancer actor/process and node process manager;
- `add_workers(...)` creates worker processes owned by a node process manager,
  not private in-process `LikelihoodEvalWorker` instances;
- runner-local constrained samplers continue to call the likelihood dispatch
  boundary, and the boundary routes through load balancer -> node
  ingress/coordinator -> local worker process;
- tests that assert direct load-balancer-to-worker scheduling, ordinary
  in-process local workers, or remote constrained-sampler payloads are stale
  tests and must be updated or removed unless they exercise an explicitly named
  test-only/internal helper.

The public statistical contract remains unchanged: only completed constrained
sampler child results can mutate the race tree. Raw likelihood responses are
not accepted samples.

## Required Topology

```text
runner process
    |
    | local concurrent constrained samplers
    |
    | likelihood eval requests, one U per request
    v
load-balancer actor/process
    |
    | node-level routing and fair scheduling
    v
node ingress/coordinator process
    |
    | ipc:// random /tmp worker endpoints
    v
likelihood worker processes
```

Responsibilities:

- Runner: owns race-tree state, allocation, parent selection, task/attempt
  identity, local constrained-sampler state, acceptance ledger, result assembly,
  and deterministic integration order.
- Load balancer: owns runner fairness, node selection, node capacity
  accounting, node health, and shutdown orchestration. It must not open or
  schedule against individual worker sockets. For `address="local"` it is still
  a managed actor/process, not a plain in-process facade.
- Node process manager: owns the node ingress/coordinator process and all local
  worker processes for that node. It is responsible for startup, shutdown,
  bounded joins, termination on failure, traceback/diagnostic collection, and
  `/tmp` IPC endpoint cleanup.
- Node ingress/coordinator: owns node-local worker registration, random IPC
  endpoints, per-worker in-flight accounting, local work queueing, worker
  failure detection, retry/failure reporting to the load balancer, and capacity
  snapshots.
- Worker process: owns process-local JAX runtime state, registered problem
  payloads, compiled likelihood callables, cache diagnostics, and one active
  deterministic `U -> scalar log_L` evaluation at a time.

Worker capacity is process count. A single worker process must never execute
two likelihood evaluations concurrently. Concurrency comes from multiple local
parent samplers producing demand and from multiple worker processes consuming
that demand.

## Current Code Notes

Use these observations as starting points, but re-check the code before
implementation because other agents may have changed it:

- `src/jaxns/runtime.py` already defines `ProcessIsolatedLikelihoodEvalWorker`,
  backed by a single-worker `ProcessPoolExecutor` using `spawn`.
- `LikelihoodEvalScheduler._build_workers()` currently constructs
  `LikelihoodEvalWorker` instances for ordinary workers; the process-isolated
  worker class is not the ordinary local-LB scheduler path.
- The current local scheduler has useful request/response, compile identity,
  stale response, timeout, and diagnostics machinery from Ticket 0018. Reuse
  the contract where it still matches the topology, but move capacity and
  lifecycle ownership to the node/LB layer.
- `src/jaxns/fabric/process_manager.py` provides `ProcessManager` and random
  `ipc://` helpers using `/tmp`. Its lifecycle behavior must be made suitable
  for the local-LB runtime path, including bounded shutdown and diagnostic
  collection.
- `src/jaxns/fabric/node.py` contains node/fabric process-manager helpers, but
  the existing node abstraction is generic RPC evaluation. It is not yet the
  v3 likelihood-eval ingress/coordinator with registered compile identities,
  per-worker capacity snapshots, and `LikelihoodEvalRequest` /
  `LikelihoodEvalResponse` routing.
- `LoadBalancerClient.shutdown()` and local-LB state release must shut down
  worker processes and remove node-local IPC endpoints. Leaking processes or
  `/tmp` socket files fails this ticket.

## Spawn, Forkserver, And JAX Environment Pitfalls

Process startup policy must be explicit and tested:

- configure JAX/BLAS CPU threading environment before importing or initializing
  worker-side JAX runtime state;
- avoid forking after JAX has initialized in the parent unless the selected
  start method and environment are proven safe;
- prefer a conservative default start method for local workers and document the
  rationale in code comments or diagnostics;
- keep worker process startup import-safe under `spawn` and `forkserver`;
- ensure registered model/problem payloads are pickleable and independent of
  mutable live parent objects;
- do not rely on hidden global likelihood state in worker processes.

If `forkserver` is used for lower startup overhead, tests must still cover the
case where the parent has already imported JAX. If `spawn` remains the default,
the benchmark gate must account for setup time separately from run time.

## Phases

### Phase 1: Test-First Contract

- Write tests before production code.
- Review the tests against
  `docs/implementation_orchestration/UNIT_TEST_STANDARDS.md` and
  `docs/implementation_orchestration/PERFORMANCE_TEST_STANDARDS.md`.
- Add topology tests that force the future surface rather than proving an
  invented local stand-in.
- Mark direct local in-process worker assertions as old-runtime coverage only if
  they are attached to an explicitly named test-only/internal helper.

### Phase 2: Node Process Topology

- Add or adapt a node ingress/coordinator process for
  `LikelihoodEvalRequest` / `LikelihoodEvalResponse`.
- Add node process-manager ownership of exactly one ingress/coordinator plus
  many worker processes.
- Use random `ipc://` endpoints under `/tmp` between node coordinator and
  workers.
- Route ordinary local-LB likelihood work through the node coordinator.

### Phase 3: Worker Lifecycle And Diagnostics

- Add bounded startup, readiness, shutdown, and failure handling for the node
  coordinator and worker processes.
- Add traceback/error collection for failed node or worker processes.
- Add endpoint cleanup checks for normal shutdown, partial startup failure, and
  worker crash.
- Expose diagnostics for node count, node ingress/coordinator process count,
  worker process count, endpoint ownership, per-worker capacity, active evals,
  queue lengths, worker failures, process start method, and cache events.

### Phase 4: Scheduler Replacement

- Replace the ordinary local-LB in-process likelihood scheduler path with
  load-balancer-to-node routing.
- Preserve Ticket 0018 payload validation, compile identity digest checks,
  stale response rejection, timeout semantics, and raw-response non-acceptance.
- Ensure retries retire stale attempts before later completions can mutate the
  acceptance ledger.
- Keep constrained samplers runner-local and allow multiple parent tasks to
  dispatch likelihood probes concurrently.

### Phase 5: Correctness And Benchmark Hardening

- Run the representative full 8D correctness gates.
- Run the worker-scaling benchmark gates for 1, 2, 4, and 8 workers.
- Inspect diagnostics to confirm no direct LB-to-worker socket path remains.
- Fix only issues directly tied to topology, lifecycle, correctness,
  diagnostics, or benchmark gate failures.

## Test-First Review Against Standards

Unit-test standards review:

- Tests must assert topology contracts, not private implementation accidents:
  LB -> node ingress/coordinator -> worker process, one active eval per worker,
  and no direct LB-to-worker scheduling.
- Use small deterministic fixtures for payload routing, stale responses,
  malformed input, unknown identity, duplicate response, and retry retirement.
- State-transition tests must assert both event outcomes and state snapshots:
  process state, worker capacity, queue state, active evals, endpoint ownership,
  and diagnostics.
- Ordering tests must verify deterministic runner integration order when worker
  responses complete out of order.
- Idempotency tests must call shutdown, retry retirement, and duplicate
  response handling twice and assert the exact allowed outcome.
- The tests must not implement a fake scheduler that duplicates the production
  topology contract.

Performance-test standards review:

- Worker-scaling tests must prove the hot-path bound: adding worker processes
  increases usable process-level likelihood capacity without increasing
  per-worker active evaluation above one.
- Benchmarks must record baseline and final times, not just pass/fail.
- Numeric worker-count defaults and process start-method choices must be
  benchmarked so they can be revisited with evidence.
- Any wall-time threshold must have a recorded rationale and must separate
  setup time from run time.

## Required Unit Tests

- `LoadBalancerClient(address="local").add_workers(["cpu:*:N"])` creates one
  managed local load-balancer actor/process, one node process manager, one node
  ingress/coordinator process, and `N` worker processes for the local node.
- The load balancer's routing table contains node ingress/coordinator entries,
  not per-worker sockets.
- Local load-balancer diagnostics prove the load balancer is a managed
  actor/process with owned shutdown status, not the old in-process
  `LocalLoadBalancerState` as the ordinary runtime boundary.
- Node coordinator diagnostics include random `ipc://` worker endpoints under
  `/tmp`, and endpoint paths are not predictable stable filenames.
- Each worker process handles at most one active likelihood evaluation; a
  second request queues or routes to another worker.
- A multi-worker node reports `max_active_evals_pool <= observed_worker_count`
  and every `max_active_evals_per_worker[worker_id] <= 1`.
- `ProcessIsolatedLikelihoodEvalWorker` or its replacement registers compile
  identity payloads without compiling, then records one first-work compile and
  subsequent cache hits for matching same-shape `U`.
- Shape/dtype/pytree mismatch is rejected without silent recompilation under
  the same identity.
- Unknown identity, malformed payload, worker exception, timeout, process
  death, and transport loss produce structured failed likelihood responses.
- Failed likelihood responses do not mutate out-degrees, accepted samples,
  phantom clusters, allocation plans, or shrinkage inputs.
- Duplicate, stale, and late responses are ignored by
  `(task_id, attempt_id, transport_id, eval_id)`.
- Retry retirement prevents an older live attempt from winning acceptance after
  a newer attempt is issued.
- Shutdown is idempotent and removes live workers and IPC endpoints after
  normal exit.
- Partial startup failure tears down already-started child processes and IPC
  endpoints before surfacing the failure.
- Live node ingress/coordinator process death after successful startup marks
  in-flight work on that node failed or unknown, removes the node from
  schedulable capacity, and leaves cleanup responsible for terminating remaining
  local descendants.
- Worker crash diagnostics include worker id, process id when available,
  endpoint, active eval id when available, failure reason, and traceback or
  structured error message.

## Required Integration Tests

- Ordinary `LoadBalancerClient(address="local")` sampler execution routes the
  sampler's real proposal likelihood calls through the process topology, not
  through the old in-process worker scheduler.
- Multiple local constrained samplers dispatch likelihood evaluations
  concurrently through a shared worker pool, and final race-tree mutation occurs
  on the coordinator/runner integration path in parent-work order.
- Two runners sharing the same local LB do not share compile identities,
  task/attempt identity, likelihood responses, or acceptance ledgers.
- Worker process restart or replacement during an evaluation yields a failed
  attempt and a successful retry without double acceptance.
- Node ingress/coordinator process death during an evaluation yields failed or
  unknown in-flight attempts, removes that node from load-balancer capacity, and
  either retries on surviving capacity or surfaces a structured runtime failure
  without double acceptance.
- A local-LB context manager exits cleanly after a standard-problem smoke run
  with no owned worker processes left alive and no owned `/tmp` IPC endpoints
  left behind.
- Diagnostics expose node count, node ingress/coordinator process count,
  worker process count, requested worker specs, observed worker count,
  completed evals by worker, queue lengths, cache events, failed eval counts,
  start method, and shutdown status.

## Correctness Gates

Run correctness gates after implementation and record the exact commands and
results in this ticket.

- Focused runtime and diagnostics gates:

```text
conda run -n jaxns_py pytest tests/test_runtime.py -q
conda run -n jaxns_py pytest tests/test_likelihood_eval_dispatch_runtime.py -q
conda run -n jaxns_py pytest tests/test_v3_run_pattern.py tests/test_v3_execution_diagnostics.py -q
```

- Representative full 8D standard-problem gate under `MPLBACKEND=Agg`, with
  ordinary local LB, no subprocess isolation wrapper around pytest, and the
  existing `3 * sample_std` evidence criterion. Use the full 8D cases for
  `basic_mvn`, `spike_slab`, and `plateau` across all allocation targets:

```text
tests/test_ns_standard_problems.py::test_nested_sampling_run_results[uniform-basic_mvn]
tests/test_ns_standard_problems.py::test_nested_sampling_run_results[evidence_improving-basic_mvn]
tests/test_ns_standard_problems.py::test_nested_sampling_run_results[posterior_improving-basic_mvn]
tests/test_ns_standard_problems.py::test_nested_sampling_run_results[uniform-spike_slab]
tests/test_ns_standard_problems.py::test_nested_sampling_run_results[evidence_improving-spike_slab]
tests/test_ns_standard_problems.py::test_nested_sampling_run_results[posterior_improving-spike_slab]
tests/test_ns_standard_problems.py::test_nested_sampling_run_results[uniform-plateau]
tests/test_ns_standard_problems.py::test_nested_sampling_run_results[evidence_improving-plateau]
tests/test_ns_standard_problems.py::test_nested_sampling_run_results[posterior_improving-plateau]
```

- The full representative gate must also assert process-topology diagnostics:
  no direct LB-to-worker route, one node ingress/coordinator per local node,
  expected worker process count, max one active eval per worker, and clean
  worker teardown.

## Benchmark Gates

Record a pre-implementation baseline using the accepted Ticket 0016 benchmark
harness before replacing the scheduler. Record final results with the same
problem settings after implementation.

Required worker-scaling benchmark:

- problem: full 8D `basic_mvn`;
- allocation target: at least `uniform`; include other allocation targets if
  the implementation changes scheduling semantics beyond the worker topology;
- worker counts: 1, 2, 4, and 8 local CPU workers;
- report requested worker spec and observed worker process count for every run;
- report setup time separately from run time and total time.

Required timing improvement gate:

- The final full 8D `basic_mvn` run with 2, 4, and 8 workers must each improve
  run wall time over the 1-worker run from the same implementation and benchmark
  command family. The 8-worker run must be the fastest measured run unless the
  ticket records a new strict review acceptance of a hardware saturation limit.
- The final full 8D `basic_mvn` run with 8 workers must improve wall time over
  Ticket 0018's best accepted local-LB result: `218.90s` total / `216.54s` run
  with `cpu:*:1`.
- A multi-worker run where balanced worker completions still increase wall time
  relative to fewer workers fails this ticket; it may be used as diagnostic
  evidence for follow-up work, but not as acceptance.
- If process startup dominates total time, the run-time comparison still must
  improve and setup overhead must be reported separately.

Benchmark output must include:

- total wall time;
- setup time;
- run time;
- result conversion time;
- MC shrinkage time;
- likelihood evaluation count;
- dispatch throughput in likelihood evaluations per second;
- dispatch latency summary;
- requested worker spec;
- observed node count and node ingress/coordinator process count;
- observed worker process count and worker process ids when available;
- observed worker device classes;
- max active likelihood evaluations per worker and across the worker pool;
- completed eval count by worker;
- queue length or queue wait summary at load balancer and node coordinator;
- worker compile count and cache-hit count;
- rejected shape/cache count;
- failed eval count by failure type;
- number of distinct compile identities;
- process start method;
- worker shutdown and IPC cleanup status.

Worker-scaling diagnostics must show real utilization for multi-worker runs:
`max_active_evals_pool >= 2` for `N >= 2`, completed evaluations from at least
two workers for `N >= 2`, and no per-worker active count above one.

## Out Of Scope

- Changing the nested-sampling statistical core, shrinkage law, allocation
  formulas, phantom-conditioning target, GMM direction adaptation, or evidence
  acceptance criteria.
- Sending constrained-sampler state or full sampler tasks to worker processes as
  the ordinary work unit.
- Keeping the unreleased in-process local worker scheduler as the ordinary v3
  local-LB path.
- Adding a backwards-compatibility branch for earlier unreleased v3 local-LB
  worker boundaries.
- Variable-shape JAX batching of likelihood probes.
- Reintroducing the rejected fused compiled constrained-sampler worker path.
- Weakening correctness gates because process startup is expensive.

## Acceptance Criteria

- Test-first draft is accepted against `UNIT_TEST_STANDARDS.md` and
  `PERFORMANCE_TEST_STANDARDS.md` before production implementation starts.
- Ordinary local-LB likelihood work uses LB -> node ingress/coordinator ->
  worker process routing.
- No public or default v3 code path preserves the unreleased in-process
  local-LB worker boundary for backwards compatibility.
- The local load balancer is a managed actor/process owned by the local client
  context, not the old in-process load-balancer facade as the ordinary runtime
  boundary.
- The load balancer never schedules directly against worker sockets.
- A node process manager owns exactly one node ingress/coordinator process plus
  the configured local worker processes.
- Node-local worker communication uses random `ipc://` endpoints under `/tmp`.
- Each worker process evaluates at most one `U -> scalar log_L` request at a
  time and JITs/caches on first matching work.
- Constrained samplers remain local and may dispatch likelihood evaluations
  concurrently through shared worker capacity.
- Worker startup, failure, retry, shutdown, traceback collection, and IPC
  cleanup are tested and diagnostic.
- Correctness gates pass with full 8D `basic_mvn`, `spike_slab`, and `plateau`
  representative cases across allocation targets.
- Benchmark gates record 1, 2, 4, and 8 worker scaling; 2/4/8 worker runs each
  improve over the same-implementation 1-worker run; the accepted 8-worker full
  8D `basic_mvn` run improves over Ticket 0018's best result.
- `ruff check` passes for touched Python files once implementation exists.
