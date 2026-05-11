# Repository of learnings from agents

Add relevant things we learned about the project, and how to do certain things. If a preexisting "learning" becomes
contradictory, it must be updated to keep this files consistent. This file shall be read by agents to help them not
repeat mistakes specific to this project. Keep learnings compact.

---

- The updated phantom-conditioning paper target is not the old `rho_g A_g`
  effective-count Dirichlet update. Use gamma-weighted per-cluster counts:
  independent race gammas, independent `v_c ~ Gamma(1, 1)` cluster weights, and
  Kish participation gating.
- For phantom conditioning, the open-interval contribution is
  `R_cg = A_cg - B_cg - E_cg`. The current paper equations print `A_cg` in the
  `M'_<g` update, but the Monte Carlo vector and singleton Dirichlet-recovery
  argument require `R_cg`.
- Use `C_min` as the canonical Kish participating-cluster threshold name for
  phantom conditioning, with default `20` unless a ticket/API explicitly
  overrides it. Blocks with no participating clusters fail the phantom gate.
- In this workspace session, `jaxns_py` was restored after initially being
  absent. Use the required `conda run -n jaxns_py ...` commands for checks.
- Ticket 0013 gamma-weighted phantom sampling must not apply post-hoc
  finite-batch moment correction. Each returned probability draw must equal
  `normalize(race_gamma + sum_c v_c * counts_cg)` for that draw.
- Ticket 0013 standard-problem acceptance must run the full original
  `STANDARD_PROBLEM_CASES` list directly in pytest, with no subprocess
  isolation. Each parametrized case must create and close its own local
  `LoadBalancerClient(address='local')` workers so LB teardown is the intended
  memory cleanup.
- Run standard-problem local-LB cases serially. Parallel JAX compilations of
  these tests can exhaust LLVM section memory and produce misleading failures.
- For the GMM-based non-isotropic direction kernel, the paper target is a GMM
  fit to the posterior from samples collected so far. Treat five-shell refits
  and exact fitting thresholds as v3 policy, not paper-derived statements.
- The accepted Ticket 0008 ellipsoidal direction path only freezes a
  one-component bounding ellipsoid from adaptation history. Full
  posterior-weighted GMM fitting is owned by Ticket 0014.
- The earlier full standard-problem local-LB blocker was
  `uniform-basic_mvn` missing the `3 * sample_std` evidence criterion under
  isotropic directions. Ticket 0014's GMM non-isotropic direction path plus the
  allocation-selector starvation fix resolved the gate; do not resurrect the
  old isotropic miss as a current failure.
- Ticket 0014 GMM direction-kernel cadence is measured since the last
  successful direction update. A failed eligible fit attempt increments
  attempts but does not reset the five-distinct-shell clock, so the next
  distinct shell remains eligible for another attempt.
- Ticket 0014 fitting gates are production gates, not test convenience gates:
  require `N_eff >= max(20, 2 * (D_dim + 1))` and per-component
  responsibility `N_eff,k >= D_dim + 1` without small-fixture bypasses.
- For Ticket 0014 MC-shrinkage fitting weights, normalize posterior weights
  within each shrinkage draw first, then average normalized per-sample weights
  across draws. Averaging raw masses and normalizing once is not equivalent
  when the evidence varies by draw.
- Future runtime design should treat process isolation as the durable JAX
  worker boundary: `ZMQActor` load balancer, node coordinator, and workers owned
  by `ProcessManager`; LB talks to node coordinators, which fan out to local
  workers over random `/tmp` `ipc://` endpoints.
- Runtime tickets that intentionally replace an earlier accepted slice must say
  so explicitly in the ticket and linked design docs. Mark old tests/design
  assertions as legacy or update them to the new contract; otherwise reviewers
  should block the implementation phase.
- In the likelihood-eval dispatch runtime, worker capacity is process count:
  each likelihood worker process handles at most one active `U -> log_L`
  evaluation. Parallelism comes from local parent samplers producing demand and
  multiple worker processes consuming it; benchmarks should report requested
  worker spec, observed worker count, dispatch latency, and cache events.
- For Ticket 0018 tests, simply renaming old serialized constrained-sampler
  worker tests as legacy is not enough if they still exercise the ordinary
  `LoadBalancerClient.get_nested_sampler(...)` path. Legacy coarse-task tests
  must use an explicit legacy compatibility surface or be updated to the
  likelihood-eval contract. Also, raw likelihood responses are not accepted
  statistical samples; only completed constrained-sampler child results may
  mutate the race tree.
- Ticket 0018 tests must include integrated local-LB assertions, not just
  standalone payload/router/scheduler units: ordinary local-LB sampler execution
  should use likelihood-eval dispatch while sampler state stays local, raw
  likelihood responses should leave runner acceptance/out-degree/phantom/
  allocation state unchanged, and worker-capacity diagnostics should be observed
  from the local worker pool.
- Passing Ticket 0018 tests is not enough if the implementation only records a
  synthetic likelihood dispatch before calling the old local sampler path. The
  sampler's actual proposal likelihood callable must consume `U -> log_L`
  results from the dispatch boundary; otherwise worker utilization and pickle
  reduction objectives are not met.
- Ticket 0018 local-LB runtime must share likelihood worker capacity at the load
  balancer/node layer, not create a private worker scheduler per runner. Retry
  must retire prior live attempts before they can mutate the acceptance ledger,
  and worker compile registration should use immutable serialized problem
  snapshots or reject digest drift from mutable live objects.
- Ticket 0018 queued likelihood-eval timeout must be atomic with queue
  cancellation. Avoid separate "wait timed out" then "cancel" windows where a
  worker completion can start the queued request with no caller left to finish
  it; cancellation must remove queued work or safely retire already-started
  work under the scheduler lock.
- The first correct Ticket 0018 local likelihood-dispatch slice preserved the
  representative standard-problem gate but regressed performance: full 8D
  `basic_mvn` benchmark moved from Ticket 0017's `88.99s` total / `86.81s` run
  to `246.27s` total / `243.97s` run, and a 120-sample worker-scaling smoke
  showed no speedup from `cpu:*:1` to `cpu:*:2`. Next performance work should
  focus on reducing per-proposal dispatch overhead and making parent dispatch
  concurrent enough to use shared worker capacity.
- Ticket 0018 performance gates should prove utilization, not just worker
  existence. Multi-worker local-LB tests/benchmarks must expose concurrent
  likelihood demand (`max_active_evals_pool >= 2` for multi-worker specs) and
  completed evals from multiple workers without relying on wall-clock speed
  thresholds; concurrency probes must assert their barrier released and did not
  only pass after a timeout.
- Parallel parent dispatch must keep statistical integration deterministic.
  Parent sampler work may run concurrently to feed likelihood workers, but
  `complete_runtime_dispatch(...)`, acceptance-ledger mutation, and `Samples`
  assembly should happen back on the coordinator thread in parent-work order,
  not in thread completion order.
- Ticket 0018 worker-utilization diagnostics must count only real likelihood
  evaluations requested by constrained sampler work whose result is integrated
  into `ParentWork`/`Samples`. Do not create auxiliary sampler calls or
  discarded likelihood probes solely to occupy workers; that makes utilization
  and benchmark diagnostics untruthful.
- When using barriers to prove parent-dispatch overlap, size the barrier to the
  concurrently started wave, not the full shell, unless the executor can start
  every shell item. A `Barrier(len(parent_work))` with
  `ThreadPoolExecutor(max_workers < len(parent_work))` deadlocks before queued
  parent tasks can start.
- Likelihood scheduler cancellation must distinguish scheduler-visible
  request retirement from physical worker capacity. Cancelling an already
  started eval must not start queued work on that worker until the worker's
  active evaluation actually completes or is otherwise safely stopped, and
  stale cancelled completions must not inflate public completed/dispatch
  diagnostics.
- Allocation target utility should be encoded in `build_allocation_plan()`
  through `target_K`; `select_parent_work()` must service positive deficits
  without multiplying by `unit_peak_utility` again, or baseline target deficits
  in low-utility/high-likelihood blocks can be starved.
- Parent selection runs in Python orchestration and must avoid
  variable-length JAX random draws. Use NumPy weighted-CDF selection with a
  scalar keyed uniform; `jax.random.categorical` over changing candidate
  lengths causes repeated JAX compilation in long standard-problem runs.
- Local runtime dispatch should reuse sampler bytes from
  `RuntimeCompileIdentity` construction. Re-serializing the same sampler once
  per accepted local-LB dispatch is avoidable coordinator overhead and should
  stay bounded by the runner identity setup, not by sample count.
- Do not enable a compiled fused `UniDimSliceSampler` runtime worker path until
  strict seed validation and mutable problem identity are designed explicitly.
  A naive jitted wrapper can bypass Python seed checks and reuse stale compiled
  likelihoods when mutable args/params change in place.
- Standard-problem speed benchmarks must keep wall-clock timing fractions
  separate from aggregate worker sampler latency. Summed worker dispatch
  latency can exceed elapsed run wall time when workers overlap, so report it
  as a diagnostic rather than a wall-clock component.
- Benchmark MC-shrinkage timing must materialize or `block_until_ready()` the
  returned evidence samples before stopping the timer; otherwise JAX async
  dispatch can under-report the completed MC sampling cost.
- For v3 `_sample_parent_work`, strict seed selection should use the sorted
  active likelihood prefix and `searchsorted(..., side="right")` semantics
  rather than rebuilding a dynamic boolean candidate array per parent. This
  preserved plateau strictness and reduced full 8D `basic_mvn` benchmark time
  from about `150.89s` to `88.99s` in Ticket 0017.
- For Ticket 0018 forced-dispatch slice performance, do not use "avoid JAX
  geometry helpers" as an acceptance criterion by itself. A NumPy no-step-out
  geometry rewrite avoided `_slice_bounds`/`_pick_point_in_interval`/
  `_shrink_interval` but was slower and introduced mixed-dtype and shrinkage
  budget parity bugs. Keep semantic parity tests first.
- In `_new_proposal_python`, the returned direction from the proposal is not
  consumed by current callers, and the final forced-Python loop direction draw
  is also unused. Skipping those unused direction draws preserves sampler
  outputs and gives only modest speed gains; larger gains likely require
  reducing forced-loop JAX array indexing/key-splitting overhead or changing
  the process/runtime topology.
- Direct `jax.vmap` direction precomputation is not parity-safe for isotropic
  multi-dimensional directions under strict bitwise determinism. Local probes
  showed tiny normalized-direction differences. Use scalar sampling or prove
  `jax.lax.map`/kernel-specific batching with dedicated parity tests before
  adopting direction precompute.
- Ticket 0018 forced Python-loop dispatch should materialize split PRNG keys
  on the host instead of indexing batched JAX key arrays in every loop
  iteration, but only materialize keys that are actually consumed. Keep
  proposal-key schedules exact and use parity tests for `num_slices` 1, 2, and
  larger loops; also cover integrated `no_step_out=False` forced-vs-fused
  parity with phantom outputs.
- Ticket 0018 full 8D `basic_mvn` after scheduler/identity and sampler
  micro-optimizations measured `218.90s` total / `216.54s` run with
  `cpu:*:1`, improving the first correct local likelihood-dispatch
  `246.27s` total reference. `cpu:*:2` measured `265.62s` total despite
  balanced worker completions, so CPU multi-worker dispatch currently proves
  utilization but not wall-clock speedup.
