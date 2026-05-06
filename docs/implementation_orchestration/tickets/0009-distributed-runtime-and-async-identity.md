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
