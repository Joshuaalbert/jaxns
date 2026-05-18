# Ticket 0009: Load-Balanced Worker Runtime and Runner Interface

Branch: `feature/v3-load-balanced-worker-runtime`
Priority: 9
Depends on: Ticket 0001, Ticket 0003, Ticket 0005, Ticket 0007
Design docs:

- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-constrained-sampling.md`
- `docs/design/jaxns-v3-validation-plan.md`
- `docs/design/interface/run_pattern.py`

## Goal

Implement the v3 load-balancer/worker/runner runtime needed by the target
interface in `docs/design/interface/run_pattern.py`.

Workers establish compute sectors. The load balancer owns worker registration,
fair multi-tenant scheduling, model compilation, and nested-sampler runner
creation. Each runner drives the single v3 core from Ticket 0005 and may overlap
work from different parent contours without changing race-tree semantics.

## Current Code Context

Relevant files likely include:

- `src/jaxns/core_distributed.py`
- `src/jaxns/constrained_sampler_distributed.py`
- `src/jaxns/fabric/node.py`
- `src/jaxns/fabric/zmq_actor.py`
- `src/jaxns/fabric/zmq_p2p.py`
- `src/jaxns/fabric/process_manager.py`
- `src/jaxns/fabric/scheduler_launcher.py`
- `src/jaxns/fabric/worker_launcher.py`
- `pyproject.toml`
- `tests/test_distributed_core.py`
- `tests/test_fabric_distributed.py`
- `tests/test_p2p_rpc.py`
- `tests/test_shutdown_logic.py`

Current code includes a threaded distributed core path and ZMQ fabric/RPC
components. The existing point-to-point/lease-oriented fabric is not the v3
runtime model. Treat it as non-authoritative for this ticket. It may provide
small utilities, tests, or lessons, but v3 should not be implemented by
stretching the current lease model.

The v3 runtime target is:

- one statistical core from Ticket 0005;
- a load balancer that owns worker registration, model compilation, fair
  multi-tenant scheduling, and runner creation;
- workers that join the load balancer and advertise compute sectors such as
  CPU/GPU device pools;
- nested-sampler runners created by the load balancer for submitted models and
  backed by the single v3 core.

Minimal perturbation means isolating or replacing only the non-matching runtime
surface while reusing model, sampler, state, result, and statistical code where
their boundaries match v3. Existing distributed files should be migrated,
deprecated, or bypassed only as needed to expose the new runtime contract and
avoid maintaining two core algorithms.

## Required Behavior

- Add a public `LoadBalancerClient` at `jaxns.runtime.LoadBalancerClient` with
  the interface shape shown in `docs/design/interface/run_pattern.py`.
- `LoadBalancerClient(address="local")` finds or starts a local load balancer,
  connects to it, and tears down only the local resources it owns on context
  exit.
- `LoadBalancerClient(address="tcp://...")` connects to an existing load
  balancer so another host can add workers.
- `add_workers(["cpu:*:5", "gpu:0,1:10"])` parses
  `device_type:device_ids:num_workers_per_device`, starts workers, and registers
  compute sectors with the load balancer.
- Workers advertise compute sectors, not exclusive single-task leases.
- `get_nested_sampler(model, args, params, collect_phantoms=True, ...)` compiles
  or prepares the likelihood on each relevant device type, creates a runner, and
  returns an object with the same public execution methods as `NestedSampler`.
- `args` are the pickleable or pytree positional arguments passed into the prior
  model. `params` are the `CtxParams` produced by `Model.init_params(...)` from
  `.parameter()` declarations. The runner must serialize, ship, compile, and
  cache model code, `args`, and `params` consistently so workers evaluate the
  same likelihood problem as the submitting client.
- A nested-sampler runner owns race-tree state and allocation/depth/goal
  decisions through the single v3 core.
- Workers execute model likelihood and constrained-sampler work assigned by the
  load balancer. Work from different parent contours and different clients may
  overlap.
- Worker payloads include nested literal/array likelihood arguments and the
  attempt/transport metadata needed for retries.
- Runner in-flight task metadata includes `task_id`, `attempt_id`,
  requested parent sample index, effective parent sample index or sentinel,
  effective strict contour, seed identity, and phantom likelihood-cluster
  identity.
- Keep statistical `task_id`, retry `attempt_id`, and transport-level delivery
  identity separate. A runner-owned acceptance ledger keyed by `task_id` decides
  whether a result may mutate out-degrees or phantom clusters.
- Child completion updates only the out-degree of the known parent or sentinel
  fallback.
- Out-of-order child completion is accepted without changing statistical
  ordering.
- Worker arguments are serialized with Python `pickle` and may be nested trees
  of literals and arrays.
- Workers receive serialized likelihood models where the runtime path requires
  model shipment.
- Communication may use ZMQ to match the paper, but not through the old
  lease-first p2p semantics.
- Load balancing and multi-tenant fairness do not change race-tree semantics.
- Dispatching work is non-mutating: pending dispatch, revoke, worker failure, and
  retry leave `K`, shrinkage inputs, out-degree, and phantom clusters unchanged
  until a result is accepted exactly once.
- Failed, revoked, retried, or duplicate worker results cannot double-increment
  out-degree or duplicate phantom likelihood clusters.
- Failed dispatch, revoked work, retried work, replayed completion, and stale
  parent targets after allocation changes all have explicit
  acceptance/idempotence behavior.
- `LoadBalancerClient.wait_until_shutdown()` blocks a worker-node process until
  the connected load balancer requests shutdown or the connection is closed, and
  context exit unregisters workers cleanly.

## Out Of Scope

- Final performance tuning. Covered by Ticket 0010 after correctness.
- New sampler trajectory methods. Covered by Ticket 0008.
- Maintaining the old p2p/lease runtime as a v3 public API.

## Test Plan

Write tests before implementation.

Required unit tests:

- `LoadBalancerClient(address="local")` starts, connects to, and tears down a
  local load balancer without requiring the old fabric API.
- Remote worker-node pattern can connect to `tcp://...`, add workers, block
  until shutdown, and cleanly unregister workers on exit.
- Worker spec parsing covers CPU wildcard, explicit GPU ids, invalid devices,
  zero/negative worker counts, and malformed strings.
- `get_nested_sampler(...)` returns a runner with `run_until_goal` and
  `resume_until_goal` once Ticket 0005 is available.
- Two runners from different clients can share the same worker pool without
  cross-counting tasks, phantom clusters, or model compilation state.
- Out-of-order child completions produce the same final race-tree state as
  in-order completions.
- Duplicate completion for the same task is rejected or idempotent by explicit
  contract and cannot double-count out-degree.
- Failed/retried tasks preserve the in-flight parent target and do not leave
  partial phantom clusters.
- Replayed completion, revoked work, and stale parent target after allocation
  changes have exact state-snapshot tests.
- Stale or duplicate results across multiple `attempt_id` values for the same
  `task_id` are accepted at most once, independent of transport delivery
  identity.
- Serialized nested literal/array arguments round-trip through worker payloads.
- `CtxParams` produced by `Model.init_params(...)` round-trip through worker
  payloads and are included in the compilation/cache identity.
- Serialized likelihood models round-trip to workers and execute with nested
  literal/array arguments.
- Phantom likelihood-cluster identity survives worker result serialization.
- Pending dispatch, revoke, worker failure, and retry have exact state-snapshot
  tests proving no lineage or phantom mutation before acceptance.
- Sentinel fallback is represented explicitly when no seed is available.
- In-flight and diagnostic metadata distinguishes `requested_parent_idx`,
  `effective_parent`, and `effective_log_L_constraint`; persisted state remains
  out-degrees and sample/constraint data only.
- Load-balancer assignment order does not affect final statistical ordering.

Required integration tests:

- Local load-balancer smoke test with multiple workers and overlapping
  constrained-sampling tasks.
- Multi-client load-balancer test that submits tasks from different models and
  verifies isolation, fair scheduling, and parent out-degree updates.
- Shutdown/revoke test ensuring no accepted result is lost and no revoked result
  is counted.

## Implementation Notes

- Keep deterministic runner-owned task ids. They are distinct from retry
  attempts and transport delivery ids.
- Prefer explicit payload dataclasses over tuple protocols for new v3 messages.
- Do not rely on worker-local state to reconstruct the parent target. The
  runner remains the source of truth for accepted out-degree updates.
- Do not preserve a separate distributed core. The runner should call the single
  core execution/acceptance machinery from Ticket 0005.
- Public legacy distributed entry points such as `NestedSamplerDistributed` and
  old fabric tests must be removed, deprecated, or converted into thin wrappers
  over the load-balanced runner. They must not be treated as independently
  validated v3 behavior.
- Do not model worker ownership as one lease at a time. Workers advertise
  compute sectors, and the load balancer schedules tasks across tenants.
- Keep public imports explicit; the target design imports
  `jaxns.runtime.LoadBalancerClient`. Add any top-level re-export only if the
  project decides that is part of the public API.
- Any performance benchmark should report wall-clock timing where meaningful,
  but correctness and identity preservation are the gate for this ticket.

## Acceptance Criteria

- The target `LoadBalancerClient` pattern is backed by tested behavior.
- Load-balanced execution preserves v3 race-tree and phantom-cluster invariants.
- Out-of-order and failure paths are tested.
- Worker-pool sharing supports multiple clients and models without state leaks.
- Legacy distributed APIs are removed, deprecated, or thin wrappers over the new
  runner, with tests reflecting that they are not a second v3 implementation.
- The old p2p/lease public surface is not treated as the v3 runtime contract.

## Current Review Follow-Up

Implementation review found blockers that must be resolved before acceptance:

- `jaxns.runtime.LoadBalancerClient` is currently an in-process facade; it does
  not start/connect to a load balancer, register workers, compile models, or
  create a load-balanced runner.
- `core_distributed.py` still contains an independent distributed nested
  sampling loop rather than a removed/deprecated/thin-wrapper legacy surface.
- Runtime task identity is missing: there is no `task_id`, `attempt_id`,
  transport identity separation, or runner-owned acceptance ledger to prevent
  duplicate/replayed/stale completions from mutating state.
- Model payload identity and round-trip semantics are incomplete for pickled
  model code, nested args, and `CtxParams`.

### Runtime Contract Slice Review

A narrow runtime-contract slice added `jaxns.runtime` identity and payload
dataclasses, local compute-sector registration, serialized model/args/params
helpers, and a runner-owned acceptance ledger. Independent review rejected the
slice as not yet acceptable because the identity foundation is not sound enough
for later worker execution:

- `LoadBalancerClient(address="local")` allocates a fresh local state per
  client, so multiple local clients can collide on `client_id`, `runner_id`,
  and `task_id`. Local clients must share a load-balancer namespace/state.
- `attempt_number` is currently derived from a global attempt counter rather
  than scoped per `task_id`; global uniqueness may stay in `attempt_id`, but the
  ordinal must describe retries for one task.
- `delivery_number` is currently derived from a global transport counter rather
  than scoped per `attempt_id`; global uniqueness may stay in `transport_id`,
  but the ordinal must describe deliveries for one attempt.
- Payload tests must include real `Model.init_params(...)` / `CtxParams`
  round-trip coverage, not only plain dictionaries.

Next implementation slice: fix the shared load-balancer namespace and scoped
identity ordinals, then add multi-client/no-ledger-bleed and `CtxParams`
round-trip regressions before continuing into worker execution.

### Runtime Contract Remediation Accepted

Independent review accepted this remediation slice narrowly. The slice now has:

- shared local load-balancer registry/state for local clients;
- non-colliding client, runner, and task identities within the shared namespace;
- per-`task_id` retry ordinals while preserving globally unique `attempt_id`s;
- per-`attempt_id` delivery ordinals while preserving globally unique
  `transport_id`s;
- per-runner acceptance ledger isolation;
- real `Model.init_params(...)` / `CtxParams` payload round-trip coverage.

Focused acceptance checks passed:

- `conda run -n jaxns_py pytest tests/test_runtime.py tests/test_v3_run_pattern.py`
  (`57 passed`);
- `conda run -n jaxns_py python -m py_compile src/jaxns/runtime.py src/jaxns/model.py`;
- `git diff --check` on the touched runtime/model/test/status files.

The full ticket remains open. The accepted remediation only establishes the
identity and payload foundation needed before implementing local
load-balanced worker execution, worker lifecycle/scheduling, failure/retry
acceptance integration, and legacy distributed API migration.

### Local Worker Execution Test Draft Review

The first test draft for local load-balanced worker execution was rejected by
independent review. The rejected draft required public dispatch records but was
too weak because direct in-process `NestedSampler` execution plus synthetic
records could satisfy it. Required test remediation:

- prove constrained-sampler work executes only after crossing a serialized
  worker payload boundary, rather than through direct client-side sampler calls;
- require a specific public coordinator trace surface instead of accepting many
  possible trace/ledger names;
- strengthen multi-client coverage beyond sequential identity checks so shared
  worker-pool execution is exercised concurrently;
- prove serialized model/args/`CtxParams` payloads are the payload used for
  worker execution, not only independently round-tripped.

The revised tests were independently accepted. They require
`LoadBalancerClient.get_nested_sampler(...)` to return a runner that dispatches
constrained-sampler work through a serialized local worker payload, rejects the
plain direct-client sampler path, exposes public `coordinator_dispatch_records`,
updates the acceptance ledger exactly once per accepted `task_id`, supports
concurrent local clients sharing the same worker pool, and includes serialized
model/args/`CtxParams` payload bytes in each worker dispatch record.

### Local Worker Execution Implementation Review

The first implementation of the local worker-execution slice was rejected by
independent review. Blocking findings:

- the worker task deserialized `SerializedModelProblem` but did not use the
  deserialized model for worker execution, so `model_bytes` was diagnostic
  rather than part of the actual worker payload;
- the core refactor unconditionally set `store_phantom_samples=False` and the
  v3 root/parent sampling path retained phantom coordinates without respecting
  that flag, regressing direct `NestedSampler` behavior.

Next remediation must add focused tests that force the serialized model payload
to be used by worker execution and protect `store_phantom_samples` behavior,
then fix the production code and repeat review.

The second implementation review rejected the remediation because:

- default local runtime execution with the default frozen `UniDimSliceSampler`
  failed when worker-side model injection attempted normal assignment to the
  sampler's frozen `model` field;
- the v3 parent-work path still produced mismatched phantom-coordinate pytrees
  when an existing state with stored phantom coordinates received
  likelihood-only phantom batches.

Next remediation must add regressions for default local runtime execution and
mixed phantom-coordinate append, then repeat implementation review.

### Local Worker Execution Slice Accepted

Independent review accepted the local worker-execution slice narrowly after the
second remediation. The accepted slice now has:

- `LoadBalancerClient.get_nested_sampler(...)` returning a runtime-backed
  runner instead of a plain direct `NestedSampler`;
- local constrained-sampler work executed only after model/args/params and
  sampler payloads cross a pickle serialization boundary;
- worker-side use of the serialized model payload, including frozen sampler
  instances such as the default `UniDimSliceSampler`;
- public `coordinator_dispatch_records` with coordinator-owned
  task/attempt/transport identity, parent metadata, worker/sector identity, and
  serialized problem payload;
- runner-owned acceptance ledger mutation exactly once per accepted `task_id`;
- concurrent local-client sharing of the process-local worker pool without
  runner/task/ledger identity bleed;
- direct v3 `NestedSampler` behavior preserved for `store_phantom_samples` and
  mixed phantom-coordinate/likelihood-only append shapes.

Focused acceptance checks passed:

- `conda run -n jaxns_py pytest tests/test_runtime.py tests/test_v3_run_pattern.py`
  (`64 passed`);
- `conda run -n jaxns_py pytest tests/test_v3_galilean_sampler.py tests/test_distributed_core.py tests/test_v3_sampler_contract.py tests/test_v3_direction_trajectories.py tests/test_constrained_sampler.py`
  (`71 passed`);
- `conda run -n jaxns_py python -m py_compile src/jaxns/core.py src/jaxns/runtime.py src/jaxns/model.py`;
- `conda run -n jaxns_py flake8 src/jaxns/runtime.py`;
- `git diff --check` on touched runtime/core/model/test/status files.

The full ticket remains open for actual load-balancer lifecycle/connectivity,
remote `tcp://...` semantics, worker unregister/shutdown, async retry/revoke/
stale-result acceptance, compile/cache identity, fair scheduling, and legacy
distributed API migration.

### Dispatch Lifecycle Test Draft Review

Independent review rejected the first dispatch-lifecycle test draft. Blocking
findings:

- `complete_runtime_dispatch(...)` was not required to be a non-state-taking
  public lifecycle method, unlike prepare/fail/retry/revoke.
- Failed, revoked, retried, and replayed completions were not covered strongly
  enough to prove no acceptance-ledger, out-degree, or phantom-cluster mutation.
- Later-attempt-first completion and explicit sentinel parent metadata were not
  covered.

The first remediation was also rejected. Remaining blockers:

- retry semantics were internally inconsistent. The contract for this slice is
  first-valid-completion-wins across non-failed/non-revoked attempts; retry
  alone does not make an older attempt stale.
- failure/retry/revoke records must prove the in-flight parent metadata
  (`requested_parent_idx`, `effective_parent_idx`, `accepted_parent_idx`,
  strict contour values, `seed_id`, and `phantom_cluster_id`) is preserved.
- completion must reject a worker result whose task/attempt/transport identity
  does not match the dispatch record, without mutating the acceptance ledger or
  race-tree/phantom state.

Next test remediation must resolve these blockers before production lifecycle
API implementation begins.

The final dispatch-lifecycle test remediation was independently accepted. The
accepted red tests require:

- public lifecycle methods on runtime runners that do not accept `state`;
- explicit pending/fail/retry/revoke/accepted/duplicate/stale/mismatched
  lifecycle records in `coordinator_dispatch_records`;
- first-valid-completion-wins across non-failed/non-revoked attempts, with
  retry alone not staling older attempts;
- terminal failed/revoked attempts rejected even when the late completion uses
  the original issued dispatch record;
- task, attempt, and transport identities validated independently on result
  completion;
- parent metadata, contour values, seed identity, and phantom-cluster identity
  preserved across lifecycle transitions;
- explicit sentinel parent metadata when no parent/seed is available;
- no acceptance-ledger, out-degree, or phantom-state mutation for pending,
  failed, revoked, retried, replayed, stale, or mismatched paths.

Focused red-test check: `conda run -n jaxns_py pytest tests/test_runtime.py`
reports `33 passed, 9 failed`, with all failures at the missing
`prepare_runtime_dispatch(...)` production API. `conda run -n jaxns_py flake8
tests/test_runtime.py` passes.

### Dispatch Lifecycle Slice Accepted

Independent implementation review accepted the dispatch-lifecycle slice
narrowly. The accepted slice now has:

- public non-state-taking lifecycle methods on `RuntimeNestedSampler`;
- explicit lifecycle records in `coordinator_dispatch_records` for pending,
  failed, retried, revoked, accepted, duplicate, stale, stale-parent, and
  mismatched-result paths;
- first-valid-completion-wins semantics across non-failed/non-revoked attempts,
  while retry alone does not stale older attempts;
- terminal failed/revoked tracking by `(task_id, attempt_id, transport_id)` so
  late completions using the original issued dispatch record remain rejected;
- independent task/attempt/transport identity validation before ledger
  mutation;
- stale parent/contour rejection before ledger mutation with current
  parent/contour diagnostics;
- preserved parent metadata, contour values, seed identity,
  phantom-cluster identity, and explicit sentinel parent values across
  lifecycle transitions.

Focused acceptance checks passed:

- `conda run -n jaxns_py pytest tests/test_runtime.py` (`42 passed`);
- `conda run -n jaxns_py flake8 src/jaxns/runtime.py`;
- `conda run -n jaxns_py python -m py_compile src/jaxns/runtime.py`.

The full ticket remains open for real `tcp://` runtime behavior, load-balancer
process lifecycle, worker unregister/shutdown semantics, async scheduling
fairness, compile/cache identity across devices, integration of lifecycle paths
with real worker execution, and legacy distributed API migration.

### Worker Registration And Shutdown Test Draft Accepted

Independent review accepted the worker registration/shutdown test slice. The
accepted red tests require:

- worker compute sectors tracked by owning client and removed by `sector_id` on
  context exit or `shutdown()`, while sectors owned by other same-address
  clients retain their original `sector_id`;
- duplicate same-spec sectors cleaned up by owner in both close orders, not by
  matching worker spec;
- fresh address state after the last client closes, with an empty worker pool
  and sector ids restarting at `sector-000001`;
- identical `tcp://...` addresses sharing the same in-process namespace, while
  distinct `tcp://...` addresses have isolated worker pools and sector id
  counters;
- public shared-address shutdown request API that wakes same-address
  `wait_until_shutdown()` waiters only;
- re-entered local clients and recreated tcp clients using a fresh shutdown
  event rather than stale prior shutdown state.

Focused red-test check: `conda run -n jaxns_py pytest tests/test_runtime.py`
reports `42 passed, 8 failed`, with failures matching owned-sector cleanup and
missing `request_shutdown()` production behavior. `conda run -n jaxns_py flake8
tests/test_runtime.py` passes.

### Worker Registration And Shutdown Slice Accepted

Independent implementation review accepted the worker registration/shutdown
slice narrowly. The accepted slice now has:

- client-owned compute-sector registration and cleanup by exact `sector_id`;
- stable surviving sector ids/order when another same-address client exits;
- duplicate same-spec worker-sector cleanup by owner in both close orders;
- fresh address state after the last client closes, including worker pool,
  counters, dispatch records, and shutdown event;
- same-address in-process `tcp://...` namespace sharing and distinct-address
  isolation;
- public `LoadBalancerClient.request_shutdown()` waking only same-address
  `wait_until_shutdown()` waiters;
- re-entered local clients and recreated tcp clients using fresh shutdown
  events rather than stale prior shutdown state.

Focused acceptance checks passed:

- `conda run -n jaxns_py pytest tests/test_runtime.py` (`50 passed`);
- `conda run -n jaxns_py pytest tests/test_v3_run_pattern.py` (`31 passed`);
- independent review also ran `conda run -n jaxns_py pytest
  tests/test_runtime.py tests/test_v3_run_pattern.py` (`81 passed`);
- `conda run -n jaxns_py flake8 src/jaxns/runtime.py`;
- `conda run -n jaxns_py python -m py_compile src/jaxns/runtime.py`.

The full ticket remains open for real remote `tcp://` transport/process
lifecycle, scheduling fairness, compile/cache identity across devices, full
async worker failure/retry integration with actual worker execution, and legacy
distributed API migration.

### Compile Identity And Fairness Test Draft Accepted

Independent review accepted the compile/cache identity and in-process fairness
test slice. The accepted red tests require:

- deterministic public `runtime_compile_identity` on runtime runners;
- compile identity stability for equivalent model/args/params/sampler/device
  type sets and CPU worker-count changes;
- compile identity changes for model, args, params, `collect_phantoms`, sampler
  type/config, and available worker device-type changes;
- accepted dispatch/lifecycle records exposing `runtime_compile_identity`,
  `client_id`, serialized problem payload, runner/task/attempt/transport
  identity, sector, and worker diagnostics;
- exact deterministic `(sector_id, worker_id)` assignment sequence over two
  full in-process scheduling cycles;
- shared local clients isolating client ids, runner/task ids, compile/cache
  identity, and acceptance ledgers while sharing the same worker pool.

Focused red-test check: `conda run -n jaxns_py pytest tests/test_runtime.py`
reports `51 passed, 3 failed`, with failures matching the missing
`runtime_compile_identity` runner and record surfaces.

### Compile Identity And Fairness Implementation Review

The first implementation of the compile/cache identity and in-process fairness
diagnostics slice was rejected by independent review. Blocking findings:

- lifecycle completion accepted dispatch records owned by another runner and
  could mutate the wrong runner's acceptance ledger. Lifecycle fail/retry/
  revoke/complete paths must validate `runner_id` and `client_id` ownership
  before terminal-status or ledger mutation.
- the implementation globally monkey-patched `Model.init_params` when
  `jaxns.runtime` was imported, adding hidden metadata to all `CtxParams` and
  making compile identity depend on the init PRNG key rather than only the
  submitted public parameter values. This patch is over-broad and must be
  removed. Compile identity should be computed from canonical serialized public
  values; tests should change actual `CtxParams` contents when asserting
  parameter sensitivity.

Next remediation must add a cross-runner lifecycle ownership regression and fix
the params-change fixture/production identity implementation before repeating
review.

### Compile Identity And Fairness Slice Accepted

Independent review accepted the remediated compile/cache identity and
in-process fairness diagnostics slice. The accepted slice now has:

- deterministic public `RuntimeCompileIdentity` built from serialized
  model/args/params, `collect_phantoms`, sampler type/payload, and sorted
  worker device-type set;
- compile identity changes for actual model, args, params, collect flag,
  sampler type/config, and device-type changes, while same-device worker count
  changes do not affect the identity;
- accepted dispatch and lifecycle records carrying `client_id` and
  `runtime_compile_identity` matching the issuing runner;
- exact deterministic sector/worker round-robin assignment diagnostics over
  repeated in-process dispatches;
- multi-client runner/task/client/cache/ledger isolation while sharing one
  worker pool;
- lifecycle fail/retry/revoke/complete ownership validation before terminal
  status or acceptance-ledger mutation;
- no runtime-side/global `Model.init_params` monkey-patch or hidden
  parameter-metadata side effect.

Focused acceptance checks passed:

- `conda run -n jaxns_py pytest tests/test_runtime.py` (`55 passed`);
- `conda run -n jaxns_py pytest tests/test_v3_run_pattern.py` (`31 passed`);
- `conda run -n jaxns_py flake8 tests/test_runtime.py src/jaxns/runtime.py`;
- `conda run -n jaxns_py python -m py_compile src/jaxns/runtime.py`.

`ruff` remains unavailable in the `jaxns_py` environment. The full ticket
remains open for real remote `tcp://` transport/process lifecycle, full async
worker retry/revoke integration with actual worker execution, actual
compile-cache execution across devices, and legacy distributed API migration.

### Legacy Distributed Core Migration Test Draft Accepted

Independent review accepted the legacy distributed core migration test slice.
The accepted red tests require:

- nested-sampler runtime coverage in `tests/test_distributed_core.py` to use
  `jaxns.runtime.LoadBalancerClient(address="local")` instead of validating the
  old independent distributed nested-sampling loop;
- sampler-level `DistributedUniDimSliceSampler` contract coverage to remain;
- public legacy names `NestedSamplerDistributed` and `DistributedNestedSampler`
  to stop silently constructing/running the old independent core. If the names
  remain public, construction or `run(...)` must fail clearly and mention
  `jaxns.runtime.LoadBalancerClient`.

Focused red-test check: `conda run -n jaxns_py pytest
tests/test_distributed_core.py` reports `3 passed, 2 failed`, with the failures
matching the two legacy aliases still executing the old core.

### Legacy Distributed Core Migration Slice Accepted

Independent implementation review accepted the legacy distributed core
migration slice. The accepted slice now has:

- `tests/test_distributed_core.py` nested-sampler runtime coverage using
  `jaxns.runtime.LoadBalancerClient(address="local")`;
- sampler-level `DistributedUniDimSliceSampler` contract coverage retained;
- `NestedSamplerDistributed` and `DistributedNestedSampler` no longer silently
  running the old independent nested-sampling core, instead failing clearly and
  directing callers to `jaxns.runtime.LoadBalancerClient`.

Focused acceptance checks passed:

- `conda run -n jaxns_py pytest tests/test_distributed_core.py` (`5 passed`);
- `conda run -n jaxns_py pytest tests/test_runtime.py
  tests/test_v3_run_pattern.py` (`86 passed`);
- `conda run -n jaxns_py flake8 tests/test_distributed_core.py`;
- `conda run -n jaxns_py python -m py_compile src/jaxns/core_distributed.py`.

`flake8 src/jaxns/core_distributed.py` still fails on pre-existing E501
line-length issues. `ruff` remains unavailable in `jaxns_py`.

The full ticket remains open for physical cleanup/removal of private old-loop
helpers in `core_distributed.py`, real remote `tcp://` transport/process
lifecycle, full async worker retry/revoke integration with actual worker
execution, and actual compile-cache execution across devices.

### Actual Worker Retry Integration Test Draft Accepted

The focused retry-execution test draft was accepted. The red test requires the
real local `LoadBalancerClient(address="local")` / `RuntimeNestedSampler`
worker path to retry an actual serialized worker task after a deterministic
worker-side failure, rather than only exercising synthetic lifecycle APIs. The
test asserts:

- pending, failed, retried, and accepted lifecycle records for the same
  `task_id`;
- a new retry `attempt_id`/`transport_id`, `attempt_number == 2`, and
  preserved parent, contour, seed, and phantom-cluster metadata;
- exactly one acceptance-ledger entry for the statistical task;
- final state sample-count, out-degree, live-point, and finite-likelihood
  invariants after the successful retry;
- no accepted ledger mutation from the failed attempt.

Focused red-test check before implementation:

- `conda run -n jaxns_py pytest
  tests/test_runtime.py::test_local_runtime_retries_failed_serialized_worker_task_before_acceptance
  -q` failed as intended because the runtime raised the deterministic worker
  fixture failure instead of retrying.

### Actual Worker Retry Integration Slice Accepted

Independent review accepted the minimal actual-worker retry integration slice.
The runtime-backed `_sample_constrained(...)` path now prepares a lifecycle
dispatch, executes the serialized worker task, marks a failed worker attempt,
retries through the existing lifecycle machinery, and completes the successful
attempt through the runner-owned acceptance ledger. The accepted slice preserves
the task id and parent/contour/seed/phantom metadata across retry while keeping
failed attempts non-mutating.

Focused acceptance checks passed:

- `conda run -n jaxns_py pytest
  tests/test_runtime.py::test_local_runtime_retries_failed_worker_task_before_acceptance
  -q`;
- `conda run -n jaxns_py pytest tests/test_runtime.py
  tests/test_v3_run_pattern.py` (`87 passed`);
- independent review ran `conda run -n jaxns_py pytest tests/test_runtime.py`
  (`56 passed`) and `conda run -n jaxns_py pytest
  tests/test_v3_run_pattern.py` (`31 passed`);
- `conda run -n jaxns_py flake8 src/jaxns/runtime.py
  tests/test_runtime.py`;
- `conda run -n jaxns_py python -m py_compile src/jaxns/runtime.py
  tests/test_runtime.py`.

The full ticket remains open for physical cleanup/removal of private old-loop
helpers in `core_distributed.py`, real remote `tcp://` transport/process
lifecycle, actual compile-cache execution across devices, and true
out-of-order worker completion over asynchronous worker scheduling.
