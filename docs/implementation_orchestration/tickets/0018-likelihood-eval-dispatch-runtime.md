# Ticket 0018: Likelihood-Eval Dispatch Runtime

Branch: `feature/v3-likelihood-eval-dispatch-runtime`
Priority: after Ticket 0017 and after the accepted Ticket 0009 identity slices
Depends on: Ticket 0005, Ticket 0007, Ticket 0009, Ticket 0011, Ticket 0016,
Ticket 0017
Design docs:

- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-constrained-sampling.md`
- `docs/design/jaxns-v3-process-isolated-zmq-runtime.md`
- `docs/design/interface/run_pattern.py`
- `docs/implementation_orchestration/tickets/0009-distributed-runtime-and-async-identity.md`
- `docs/implementation_orchestration/tickets/0016-standard-problem-speed-benchmarks.md`
- `docs/implementation_orchestration/tickets/0017-standard-problem-performance-benchmark-driven.md`

## Goal

Implement the next runtime architecture slice: constrained samplers run locally
and may run in parallel, while likelihood evaluations are dispatched to
process-isolated workers as small deterministic work units. Per evaluation, only
the proposed prior-space coordinate `U` crosses the worker boundary and only a
scalar `log_L` returns.

Workers JIT/cache the model for a stable runtime identity on first matching
work. They must not own sampler state, parent selection, allocation, phantom
clusters, or race-tree mutation. The runner remains the source of truth for
statistical task identity and exactly-once acceptance.

## Supersession And Compatibility

Ticket 0018 intentionally supersedes the parts of accepted Ticket 0009 and the
linked execution/run-pattern docs that made ordinary remote work a serialized
constrained-sampler task, or that expected load-balancer/runner creation to
compile likelihoods ahead of time. Those statements are legacy for earlier
local worker-execution slices unless they are explicitly restated here.

For this slice:

- ordinary remote work is likelihood evaluation only: one proposed `U` enters a
  worker request and one scalar `log_L` returns;
- constrained sampler state, seed choice, direction snapshots, phantom buffers,
  parent metadata, allocation decisions, and acceptance remain runner-local;
- model bytes, `args`, `params`, dtype policy, device class, and `U` shape/tree
  are registered under a `RuntimeCompileIdentity` during runner creation or a
  separate identity-registration step before ordinary work;
- identity registration makes the payload available to workers, but it does not
  require ahead-of-time JIT compilation;
- each worker JITs on first matching work for a registered identity/device
  class and then reuses that cache for matching `U` requests;
- older tests or design assertions that require coarse sampler-task payloads
  must be updated for likelihood-eval dispatch or explicitly scoped as legacy
  coverage of pre-0018 behavior.

## Architecture Direction

Use the user's runtime direction as the ticket charter:

- constrained samplers run locally and in parallel where execution policy allows;
- likelihood evaluations are dispatched as small deterministic work units;
- only `U` is pickled/sent per likelihood probe;
- only scalar `log_L` returns on successful evaluation;
- workers are process-isolated and JIT their model on first matching work for a
  registered compile identity;
- each likelihood worker process handles at most one active likelihood
  evaluation at a time;
- dynamic JAX shapes are avoided;
- variable candidate lists, active pools, phantom buffers, and trajectory
  segments are trimmed in NumPy before JAX, or represented with fixed shapes and
  masks where batching is intentional.

## Performance Remediation Follow-Up

The first correctness-passing local likelihood-dispatch implementation
regressed the full 8D `basic_mvn` benchmark from Ticket 0017's `88.99s` total
to `246.27s` total. The strict test review for the remediation work found that
the initial performance gates were not yet strong enough. Follow-up actions:

- parent-concurrency probes must assert that their barrier actually released
  and did not merely continue after a timeout;
- multi-worker benchmark/scaling tests must require multi-worker utilization
  diagnostics such as `max_active_evals_pool >= 2` and completed evals from at
  least two workers for multi-worker specs, without wall-clock pass thresholds;
- benchmark schema validation must reject invalid likelihood-dispatch
  diagnostics, not only missing diagnostic fields;
- the public likelihood-dispatch diagnostics contract must stay aligned with
  benchmark fields, including dispatch eval count, queued eval count, and failed
  eval count.

## Runtime Topology

The topology follows the process-isolated ZMQ runtime design:

```text
runner process
    |
    | local parallel constrained samplers
    |
    | likelihood eval requests, one U per request
    v
load-balancer actor
    |
    | node-level scheduling
    v
node-coordinator actor
    |
    | ipc:// random /tmp worker endpoints
    v
worker process
```

Responsibilities:

- Runner: owns race-tree state, allocation/depth/goal decisions, parent
  selection, local constrained-sampler execution, task/attempt identity,
  acceptance ledger, and result integration.
- Local constrained sampler: owns chain state, seed use, direction snapshot,
  slice or trajectory logic, strict-contour checks, phantom likelihood cluster
  boundaries, and sampler-local bounded retry decisions.
- Load balancer: owns fair scheduling across runners, node capacity, and
  routing of likelihood evaluation work to nodes.
- Node coordinator: owns local worker supervision, IPC routing, failed-worker
  accounting, per-worker in-flight accounting, and worker replacement policy if
  configured.
- Worker process: owns process-local JAX runtime state, model payload cache,
  compiled likelihood callables, device state, and one active deterministic
  evaluation of `U -> log_L` at a time.

This topology addresses parallel parent dispatch by allowing several parent
tasks to advance their local constrained samplers at the same time. Each sampler
can have at most the bounded number of in-flight likelihood probes allowed by
its algorithm, and all probes share the worker pool. Completion order is
transport order only. Race-tree order and mutation are decided by the runner's
acceptance ledger after a full child sample is produced.

Worker capacity is counted in worker processes, not in concurrent evaluations
inside one worker process. A single worker advertises capacity for one active
likelihood evaluation. Additional requests queue at the load balancer or node
coordinator, or are routed to another idle worker. Concurrency comes from
multiple local parent sampler threads/processes producing likelihood demand and
from multiple worker processes consuming that demand.

## Payload Schema

Introduce explicit payload dataclasses rather than tuple protocols. The names
below are contract names; implementation may choose exact module placement.

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

Per-evaluation request payload:

- pickle or serialize only the proposed `U` pytree;
- include metadata needed to validate shape/dtype against the compile identity;
- include task/attempt/transport/eval identity for stale-response rejection.

Per-evaluation response payload:

- return exactly one scalar finite, infinite, or NaN-handled `log_L` according
  to the model likelihood policy when `status == "ok"`;
- return structured failure fields when `status == "failed"`;
- never return transformed coordinates, sampler state, phantom coordinates,
  parent metadata, direction data, or compiled artifacts.

Runner creation or explicit identity registration is responsible for making
model bytes, `args`, `params`, dtype policy, device class, and expected `U` tree
shape available under `RuntimeCompileIdentity`. These identity payloads are
cached and are not resent for every likelihood evaluation. Ordinary evaluation
requests carry only the identity digest needed to select the cached payload.
Registering the identity is not the same as compiling it; compilation is a
worker-local compile event triggered by first matching work for that
identity/device class.

## Determinism And Failure Semantics

Worker likelihood evaluation must be deterministic for
`(compile_identity, U_bytes, U_shape_tree, device_class)`, up to normal device
floating-point behavior. Workers must not draw sampler PRNG keys, choose seeds,
choose directions, adapt direction kernels, mutate allocation state, or decide
child acceptance.

Failure semantics:

- malformed `U`, shape/dtype mismatch, unknown compile identity, worker
  exception, timeout, process death, or transport loss is an evaluation failure;
- failed evaluation does not mutate out-degrees, accepted samples, supremum
  state, phantom clusters, allocation plans, or shrinkage inputs;
- the local sampler that requested the evaluation decides whether the parent
  task fails, retries the probe, or continues within its bounded algorithmic
  policy;
- failed parent attempts keep the same statistical `task_id` if retried and use
  a new `attempt_id`;
- duplicate or stale responses are accepted only if their
  `(task_id, attempt_id, transport_id, eval_id)` still matches live sampler
  demand;
- successful likelihood responses are not themselves statistical samples. Only
  a completed constrained-sampler child result can be accepted into the
  race-tree ledger.

If user code has mutable global likelihood state, the runtime must either
capture that state in `RuntimeCompileIdentity` or reject the problem for
process-isolated dispatch. Silent dependence on worker-local hidden state is
not allowed.

## JAX Cache And Shape Rules

The worker compile/cache contract is strict:

- `RuntimeCompileIdentity` includes model bytes, args, params, dtype policy,
  device class, `U` pytree structure, and all `U` leaf shapes/dtypes.
- A worker JITs the model on first matching work for one compile identity and
  device class, then reuses the compiled callable for matching requests.
- A worker that has a registered identity but no compiled callable records the
  first matching request as a compile event; identity registration alone must
  not be counted as a compile.
- A request may vary `U` values only. It must not vary pytree structure, rank,
  shape, or dtype under the same identity.
- Shape mismatch is a rejected request or a new compile identity. Silent
  recompilation under the same identity fails the ticket.
- Batched worker evaluation is out of scope unless it uses a fixed batch shape
  recorded in the identity. Variable-length batches must be padded with static
  masks or trimmed in NumPy before JAX sees the array.
- Parent candidate arrays, selected active prefixes, phantom buffers, and
  trajectory segment lists must not drive variable-length JAX compilation.

Diagnostics must record compile hits, first compiles, rejected shape mismatches,
and any intentionally new compile identities. A long run should not show
unbounded compile identity growth from ordinary sampler progress.

## Phases

### Phase 1: Contract And Tests

- Confirm current design links and this ticket remain authoritative before
  coding.
- Add test-only coverage for payload dataclasses, identity validation, static
  shape rejection, deterministic result routing, and stale-response rejection.
- Add tests that prove local constrained samplers can request remote likelihood
  probes without shipping sampler state.
- Update tests that previously asserted coarse serialized constrained-sampler
  worker payloads so they are either likelihood-eval dispatch tests or explicit
  legacy-compatibility tests.
- Review tests against `UNIT_TEST_STANDARDS.md` and
  `PERFORMANCE_TEST_STANDARDS.md` before production implementation.

### Phase 2: Local Likelihood Dispatch

- Add the request/response payload boundary.
- Add worker-side model cache lookup and first-work JIT behavior for the
  compile identity.
- Route local constrained-sampler likelihood calls through the dispatch boundary
  while keeping sampler state local.
- Preserve existing public runner behavior and diagnostics.

### Phase 3: Parallel Parent Dispatch

- Allow multiple parent tasks to run local constrained samplers concurrently
  under explicit bounded worker demand.
- Ensure each sampler's pending likelihood probes are routed through the shared
  load balancer.
- Preserve runner-owned task acceptance, out-of-order completion semantics, and
  exactly-once race-tree mutation.

### Phase 4: Process-Isolated Worker Topology

- Move likelihood evaluation from any in-process stand-in to worker processes
  owned by the node coordinator.
- Use local IPC endpoints between node coordinator and workers.
- Add bounded startup, shutdown, failure, and cleanup behavior consistent with
  the process-isolated ZMQ runtime design.

### Phase 5: Benchmark And Hardening

- Measure correctness and speed gates against the accepted benchmark harness.
- Inspect compile/cache diagnostics for shape churn.
- Fix only issues directly related to the dispatch boundary, worker process
  lifecycle, parallel parent dispatch, or cache-shape contract.

## Test Plan

Tests must be written before production implementation.

Required unit tests:

- request/response payload round-trip with nested `U` pytrees and array leaves;
- per-evaluation payload contains `U` but not model bytes, args, params,
  sampler state, phantom buffers, direction snapshots, or parent arrays;
- identity registration can occur before work without recording a JIT compile;
- `RuntimeCompileIdentity` cache hit for repeated matching `U` shapes;
- first-work JIT event recorded once per identity/device class in a deterministic
  small fixture;
- worker process capacity is one active likelihood evaluation: a single worker
  never overlaps two evaluations internally, and queued/rerouted requests do not
  enter that worker until the active evaluation finishes;
- shape/dtype/pytree mismatch is rejected and does not silently recompile under
  the same identity;
- worker response returns only scalar `log_L` on success;
- worker failure, timeout, malformed payload, and unknown identity produce
  structured failed responses;
- duplicate, stale, or late responses are ignored by identity tuple;
- failed evaluation does not mutate out-degree, phantom clusters, accepted
  samples, or allocation diagnostics;
- local sampler state remains local across likelihood dispatch;
- variable candidate lists are trimmed or statically masked before JAX-visible
  calls in touched paths.

Required integration tests:

- local load-balancer run where likelihood evaluation crosses the dispatch
  boundary and produces the same deterministic result as direct local model
  evaluation on a toy problem;
- multiple parent tasks in flight with out-of-order likelihood response
  completion and deterministic final race-tree state;
- multiple worker processes demonstrate capacity-based scheduling: observed
  concurrent active likelihood evaluations never exceed observed worker count,
  and per-worker active evaluation count never exceeds one;
- worker process restart/failure during an evaluation produces a failed attempt
  and a successful retry without double-counting;
- two runners sharing workers do not share compile identity, task identity,
  likelihood responses, or acceptance ledgers;
- process teardown removes worker processes and local IPC endpoints after normal
  exit and after partial startup failure.

Performance/diagnostic tests:

- focused benchmark or smoke proving repeated same-shape evaluations produce
  compile hits rather than repeated compiles;
- no unbounded compile identity growth in a representative local-LB standard
  problem smoke;
- dispatch diagnostics report evaluation latency, worker id, cache event, and
  failure reason when applicable.
- dispatch diagnostics expose enough worker-capacity data to verify observed
  worker count and max active likelihood evaluations per worker.

## Correctness Gates

- Existing focused runtime tests pass:
  `conda run -n jaxns_py pytest tests/test_runtime.py`.
- Existing v3 run-pattern and diagnostics tests pass:
  `conda run -n jaxns_py pytest tests/test_v3_run_pattern.py tests/test_v3_execution_diagnostics.py`.
- Representative standard-problem subset remains accepted under `MPLBACKEND=Agg`
  with the existing `3 * sample_std` evidence acceptance criterion. Use the
  exact Ticket 0017 nodeids below:

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

- Dispatch-enabled toy fixtures match direct local likelihood values exactly or
  within the dtype/device tolerance already used by the model path.
- Out-of-order and retried parent tasks produce the same accepted sample count,
  out-degree totals, finite likelihood count, and phantom-cluster count as a
  serial deterministic fixture.

## Benchmark Gates

- Record a pre-implementation baseline using the Ticket 0016 speed harness.
- Record final benchmark output with the same command and environment.
- Include worker-scaling measurements using the accepted local-LB pattern. Vary
  worker count/spec, for example by changing `--worker-spec`, and report the
  requested worker spec and observed worker count/device classes for each run.
- The dispatch path must not regress full 8D `basic_mvn` wall time by more than
  an explicitly justified tolerance while correctness work is being integrated.
  If process isolation adds startup overhead, report setup time separately from
  run time.
- Report:
  - total wall time;
  - setup time;
  - run time;
  - result conversion time;
  - MC shrinkage time;
  - likelihood evaluation count;
  - dispatch throughput in likelihood evaluations per second;
  - dispatch latency summary;
  - requested worker spec and observed worker count/device classes;
  - max active likelihood evaluations per worker and across the worker pool;
  - worker compile count and cache-hit count;
  - rejected shape/cache count;
  - number of distinct compile identities.
- For worker-scaling runs, report speedup against the single-worker or recorded
  baseline case and efficiency `speedup / observed_worker_count` where the
  comparison is meaningful. If startup or serial runner work dominates a small
  smoke, say so explicitly and keep setup time separate from run time.
- A benchmark run with repeated same-shape likelihood evaluations must show
  cache hits after first matching work and no compile churn tied to parent count,
  candidate-list length, or phantom-buffer length.

## Out Of Scope

- Changing the nested-sampling statistical core, shrinkage law, allocation
  formulas, phantom-conditioning target, or GMM direction adaptation target.
- Shipping full constrained-sampler tasks to remote workers as the primary work
  unit.
- Variable-shape JAX batching of likelihood probes.
- Reintroducing the rejected compiled fused `UniDimSliceSampler` worker path
  before strict seed validation and mutable problem identity are separately
  designed.
- Weakening the standard-problem evidence acceptance criterion.

## Acceptance Criteria

- Design links in this ticket remain correct and are confirmed before
  implementation begins.
- Tests are accepted before implementation work starts.
- Likelihood evaluation has an explicit payload boundary where only `U` is sent
  per probe and only scalar `log_L` returns.
- Workers are process-isolated for the final topology slice and JIT/cache their
  model on first matching work for a stable compile identity.
- Each likelihood worker process handles at most one active likelihood
  evaluation at a time; worker-pool concurrency is obtained by adding worker
  processes and by running multiple local parent sampler tasks.
- Constrained samplers run locally and can run in parallel for multiple parent
  tasks.
- Parallel parent dispatch preserves exactly-once statistical acceptance,
  out-of-order completion semantics, and failure idempotence.
- Dynamic JAX shape churn is rejected by tests and diagnostics.
- Correctness gates and benchmark gates are recorded in this ticket during the
  implementation loop.
- `ruff check` passes for touched Python files once implementation exists.

## Implementation Review Follow-Ups

Strict implementation review rejected the first implementation pass. These
findings must be resolved before the ticket can exit:

- Ordinary local-LB constrained sampling must route the sampler's actual
  proposal likelihood calls through the likelihood-eval boundary. A synthetic
  dispatch of the seed point before calling the old local sampler path is not
  acceptable.
- Worker compile/cache diagnostics must correspond to worker-side compiled
  callables and first-matching-work cache reuse, not only counters around direct
  Python model calls.
- Worker capacity one must be enforced at the worker boundary and in scheduler
  queueing, with diagnostics derived from real active-evaluation accounting.
- Raw `LikelihoodEvalResponse` objects and arbitrary worker payloads must not be
  accepted by the statistical race-tree ledger.
- `ShapeDtypeTree` and request validation must use stable structural metadata
  and validate request metadata, actual `U`, dtype policy, and identity shape.
- Runtime behavior skipped as pre-0018 legacy should be replaced by
  likelihood-eval equivalents where the behavior is still required: retries
  before acceptance, shared-worker runner isolation, and identity/diagnostic
  preservation.

Second strict implementation review confirmed that the sampler proposal
likelihoods now cross the dispatch boundary, but found remaining blockers:

- The local worker pool must be owned by the load balancer/node layer and shared
  across runners. A private scheduler per runner does not exercise fair shared
  worker capacity.
- Retrying a runtime task must retire the previous live attempt/transport so a
  stale completion cannot win the acceptance race after a newer attempt is
  issued.
- Worker compile registration must use immutable serialized problem snapshots,
  or revalidate the live object digest before first compile, so mutable
  args/params or model state cannot compile under a stale identity digest.
- Queued likelihood-eval timeouts must remove or cancel queued work and route a
  structured failed response; abandoned queue entries must not start later with
  no caller.

### Implementation Review Resolution

Final strict review found no remaining implementation blockers for the local
first-layer likelihood-dispatch runtime. Resolved behavior:

- ordinary `UniDimSliceSampler` proposal likelihood calls use the dispatch
  boundary rather than a synthetic seed dispatch;
- the shared local load balancer owns the likelihood scheduler and runners
  register their compile identities against that shared worker pool;
- workers JIT/cache model callables on first matching work and reuse them on
  hits;
- retry supersedes previous live attempts before acceptance-ledger mutation;
- compile identity registration uses serialized problem snapshots so mutable
  live args/params do not compile under stale digests;
- queued likelihood-eval timeout cancellation is atomic with queue start.

Remaining staged risk: the current accepted implementation is still the local
in-process first layer. Process-isolated ZMQ/`ProcessManager` workers remain a
future topology slice.

### Verification And Benchmark Results

Focused verification after final review:

```text
conda run -n jaxns_py pytest tests/test_likelihood_eval_dispatch_runtime.py tests/test_runtime.py -q
70 passed, 15 skipped, 2 warnings in 52.39s

conda run -n jaxns_py pytest tests/test_v3_run_pattern.py tests/test_v3_execution_diagnostics.py -q
54 passed, 2 warnings in 35.33s

conda run -n jaxns_py ruff check src/jaxns/runtime.py src/jaxns/constrained_sampler.py src/jaxns/diagnostics.py tests/test_likelihood_eval_dispatch_runtime.py tests/test_runtime.py
All checks passed

git diff --check
passed
```

Representative standard-problem correctness gate under `MPLBACKEND=Agg`:

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

9 passed, 2 warnings in 1515.58s (0:25:15)
```

Reduced full-8D `basic_mvn` benchmark:

```text
MPLBACKEND=Agg conda run -n jaxns_py python -m benchmarks.v3_performance.standard_problem_speed \
  --allocation-target uniform \
  --target-num-live-points 30 --max-samples 240 --shell-size 15 \
  --num-slices 24 --phantom-burn-in 4 --mc-sample-count 200 \
  --worker-spec cpu:*:2

total_seconds=29.506204472854733
setup_seconds=0.5574116110801697
run_seconds=27.936758944764733
result_conversion_seconds=0.030522150918841362
mc_shrinkage_seconds=0.9815117660909891
worker_sampler_latency_seconds=21.352474225685
likelihood_evaluations=14864
```

Small worker-scaling smoke, full 8D `basic_mvn`:

```text
MPLBACKEND=Agg conda run -n jaxns_py python -m benchmarks.v3_performance.standard_problem_speed \
  --worker-scaling --allocation-target uniform \
  --worker-spec cpu:*:1 --worker-spec cpu:*:2 \
  --target-num-live-points 30 --max-samples 120 --shell-size 15 \
  --num-slices 24 --phantom-burn-in 4 --mc-sample-count 100

cpu:*:1 actual_worker_count=1 total_seconds=15.47643149457872
  setup=0.5672055948525667 run=13.997520795091987
  result=0.025878088548779488 mc=0.8858270160853863
  worker_sampler_latency=9.266149293631315

cpu:*:2 actual_worker_count=2 total_seconds=15.614688359200954
  setup=0.5754122380167246 run=14.028040930628777
  result=0.02517613209784031 mc=0.986059058457613
  worker_sampler_latency=9.264145525172353
```

Full 8D `basic_mvn` benchmark:

```text
MPLBACKEND=Agg conda run -n jaxns_py python -m benchmarks.v3_performance.standard_problem_speed \
  --allocation-target uniform \
  --target-num-live-points 30 --max-samples 1200 --shell-size 15 \
  --num-slices 24 --phantom-burn-in 4 --mc-sample-count 1000 \
  --worker-spec cpu:*:2

total_seconds=246.27120273187757
setup_seconds=0.5555216148495674
run_seconds=243.96522552892566
result_conversion_seconds=0.09984538145363331
mc_shrinkage_seconds=1.6506102066487074
worker_sampler_latency_seconds=214.31621462665498
likelihood_evaluations=301010
```

Performance-remediation update after strict scheduler/identity review and
sampler parity fixes:

```text
MPLBACKEND=Agg conda run -n jaxns_py pytest \
  tests/test_constrained_sampler.py \
  tests/test_likelihood_eval_dispatch_runtime.py \
  tests/test_v3_performance_benchmarks.py -q

87 passed, 2 warnings in 28.03s

MPLBACKEND=Agg conda run -n jaxns_py pytest tests/test_runtime.py -q

48 passed, 15 skipped, 2 warnings in 40.69s

conda run -n jaxns_py ruff check \
  src/jaxns/constrained_sampler.py src/jaxns/runtime.py \
  src/jaxns/diagnostics.py benchmarks/v3_performance/standard_problem_speed.py \
  tests/test_constrained_sampler.py tests/test_likelihood_eval_dispatch_runtime.py \
  tests/test_v3_performance_benchmarks.py tests/test_runtime.py

All checks passed
```

Representative standard-problem correctness gate rerun under `MPLBACKEND=Agg`
with local LB, no subprocess isolation, and full 8D cases:

```text
uniform-plateau: Runtime 38.60s
uniform-basic_mvn: Runtime 293.57s
uniform-spike_slab: Runtime 162.32s
evidence_improving-plateau: Runtime 39.37s
evidence_improving-basic_mvn: Runtime 295.39s
evidence_improving-spike_slab: Runtime 163.24s
posterior_improving-plateau: Runtime 39.71s
posterior_improving-basic_mvn: Runtime 295.73s
posterior_improving-spike_slab: Runtime 163.23s

9 passed, 2 warnings in 1587.17s (0:26:27)
```

Full 8D `basic_mvn` worker-scaling benchmark after remediation:

```text
cpu:*:1 total_seconds=218.898 run_seconds=216.541
  setup=0.561 result_conversion=0.136 mc_shrinkage=1.660
  likelihood_evaluations=301010
  dispatch_eval_count=301010 max_active_pool=1
  completed_by_worker={'worker-000001': 301010}
  compile=1 cache_hit=301009
  worker_sampler_latency_seconds=187.116
  dispatch_latency_seconds_total=21.875

cpu:*:2 total_seconds=265.616 run_seconds=263.186
  setup=0.659 result_conversion=0.103 mc_shrinkage=1.668
  likelihood_evaluations=301010
  dispatch_eval_count=301010 max_active_pool=2
  completed_by_worker={'worker-000001': 150506, 'worker-000002': 150504}
  compile=2 cache_hit=301008
  worker_sampler_latency_seconds=455.460
  dispatch_latency_seconds_total=51.861
```

Benchmark gate status: correctness passes and the best full 8D `basic_mvn`
runtime improved from the first correct likelihood-dispatch reference
(`246.27s` total / `243.97s` run) to `218.90s` total / `216.54s` run by using
one local CPU worker, removing strictly unused direction work, and
host-materializing consumed split keys for forced Python loops. The result is
still much slower than Ticket 0017's pure-local `88.99s` total / `86.81s` run.
Two local CPU workers prove real balanced utilization but remain slower due to
Python/JAX CPU contention, so future performance work should target larger
runtime-topology changes or a parity-proven batched scalar operation path.

Strict-review follow-up resolution:

- Forced Python-loop key materialization no longer materializes the unused final
  next-direction key. Proposal keys still split with `loop_count`, and consumed
  direction keys keep the same prefix schedule as the previous
  `random.split(direction_scan_key, loop_count)[:-1]` path.
- Integrated forced-vs-fused `UniDimSliceSampler.get_sample` parity now covers
  `no_step_out=False` for `num_slices=2` and `num_slices=5`, including final
  sample, `log_L`, likelihood-evaluation count, phantom likelihood logs, and
  phantom valid masks.

Final focused verification after the key-materialization follow-up:

```text
MPLBACKEND=Agg conda run -n jaxns_py pytest \
  tests/test_constrained_sampler.py \
  tests/test_likelihood_eval_dispatch_runtime.py \
  tests/test_v3_performance_benchmarks.py -q

90 passed, 2 warnings in 28.50s

MPLBACKEND=Agg conda run -n jaxns_py pytest tests/test_runtime.py -q

48 passed, 15 skipped, 2 warnings in 40.02s

conda run -n jaxns_py ruff check \
  src/jaxns/constrained_sampler.py src/jaxns/runtime.py \
  src/jaxns/diagnostics.py benchmarks/v3_performance/standard_problem_speed.py \
  tests/test_constrained_sampler.py tests/test_likelihood_eval_dispatch_runtime.py \
  tests/test_v3_performance_benchmarks.py tests/test_runtime.py

All checks passed
```
