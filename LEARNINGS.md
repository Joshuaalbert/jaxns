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
  `R_cg = A_cg - B_cg - E_cg`. Earlier paper drafts printed `A_cg` in the
  `M'_<g` update; the current paper/design should use `R_cg`, matching the
  Monte Carlo vector and singleton Dirichlet-recovery argument.
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
- Difficult-problem benchmark reports with only `30` live points in 10D are
  underpowered and can show misleading single-mode corner plots. Default these
  benchmarks to at least `100 * D` live points, and tie shell/sample budgets to
  that live-point count unless a CLI run intentionally overrides them.
- Ticket 0019 process-topology checks must prove the hot data path, not just
  topology-shaped diagnostics. Local-LB likelihood evals should physically
  route parent -> LB actor -> node ingress actor -> worker actor; started evals
  must not use finite client deadlines that free parent capacity while the
  physical worker is still busy; and failed worker registration must retire the
  bad socket before any later scheduling can reuse it.
- v3 has not been released, so v3 runtime tickets should prefer the target
  design over backwards-compatibility branches. Keep old/in-process scheduler
  paths explicitly test-only or internal when they remain useful for narrow
  unit tests.
- Ticket 0019 node coordinators must not select `sorted(idle_worker_ids)[0]`
  for each request. That makes shallow demand collapse onto worker 1. Use
  node-local fair rotation/deques and verify sequential dispatch touches all
  workers. After this fix, the 8-worker `basic_mvn` speed benchmark balanced
  completions almost exactly across workers, but wall time still stayed near
  `270-281s`.
- The post-round-robin 8-worker `basic_mvn` benchmark showed the remaining
  bottleneck is not physical worker balance: worker likelihood latency totaled
  only about `30s` across `301010` evals, while run wall time was about
  `270-281s`. Fuller shells and bypassing the parent logical scheduler did not
  materially improve time; future speed work should target the forced Python
  slice/proposal loop and per-proposal JAX/host/RPC orchestration.
- Ticket 0019 keeps the default implicit `--worker-scaling` benchmark as a
  strict acceptance gate. It is expected to fail until multi-worker wall time
  improves; pass explicit `--worker-spec` grids only when collecting
  reporting-only diagnostics.
- Runtime diagnostics should keep observed node count distinct from live node
  ingress process count so degraded topology can be reported accurately. Healthy
  benchmark acceptance records should still require one live ingress/coordinator
  per observed node.
- Ticket 0020 pure-core work uses a host public `goal_cond(state)` boundary
  around compiled JAX depth/transition kernels. Do not try to call arbitrary
  Python goal callbacks from inside `jit`, `lax.while_loop`, or `lax.scan`.
- For Ticket 0020, a Python outer loop around `goal_cond(state)` is acceptable
  and intentional because users may compute goals through `state -> result ->
  trim -> metric`. The expensive inner depth/transition loop is the required
  JIT-compiled JAX-control-flow target.
- Ticket 0020 primary pure-core performance should be measured with direct
  `NestedSampler`, not `LoadBalancerClient`: full 8D `basic_mvn`, uniform
  allocation, isotropic directions, standard `30` live points / `1200` max
  samples / `24` slices, MC evidence criterion, and sampler `run_seconds < 60s`.
- Ticket 0020 runtime/subclass execution must honor empty `CoreWorkBatch`
  before calling `_sample_parent_work`. Zero-work allocation iterations should
  advance the allocation target and continue; sampling an empty `ParentWork`
  hits downstream tree-mapping assumptions.
- Ticket 0020 runtime parity should use fixed-shape core work/result buffers:
  pure core owns planning and ordered acceptance; runtime/distributed layers may
  produce proposal or likelihood results, but raw completions must not mutate
  race-tree state directly.
- Ticket 0020 full-standard plateau rows expose the current Python-core gap:
  `run_until_goal(..., goal_cond=lambda state: False, max_samples=1200)` can
  stop around the root/sentinel fallback frontier instead of exhausting the
  requested sample budget. Treat this as an expected-red pure-core acceptance
  gate; the JAX core must keep plateau fallback work advancing to the fixed
  depth budget.
- Ticket 0020 implementation review must reject trace-only progress shims. The
  trace hook needs to cover the same inner transition used by public
  `run_until_goal` / `resume_until_goal`: fixed-shape work planning, seed
  selection, constrained-sampler result production, result-buffer construction,
  and ordered acceptance. A JAXPR that only plans work and increments
  `num_samples` is not enough.
- Ticket 0020 `CoreResultBatch` buffers must be fixed-shape for every field,
  including `U_samples`; partial shells must pad the leading dimension to
  `capacity` with `valid_mask`/`num_results` carrying the dynamic count.
- Ticket 0020 `max_goal_iterations` should cap public host goal-callback
  evaluations, not internal allocation-target increments. Plateau/uniform rows
  may need many allocation target increments to reach a `max_samples` depth
  gate while still checking `goal_cond(state)` only at outer boundaries.
- Ticket 0020 pure-core transition JITs must be module-level/cacheable and
  include sampler configuration in the static cache key. Defining a fresh
  nested `jax.jit` per shell causes severe same-shape cache misses; treating
  sampler fields such as phantom count as dynamic can trigger JAX
  concretization errors.
- Ticket 0020 review should distinguish a compiled shell transition from a
  compiled depth epoch. Public direct `run_until_goal`/`resume_until_goal`
  should call the same JAX-control-flow depth epoch that the trace hook traces;
  invoking a one-shell JIT from a Python allocation loop is still a review
  blocker.
- Ticket 0020 core boundary metadata must be post-fallback metadata. When a
  planned parent has no strict seed, `CoreWorkBatch.effective_parent_idx`,
  `parent_block_idx`, `fallback_to_root`, and `seed_idx` must agree with the
  accepted `ParentWork` and result buffer.
- Ticket 0020 compiled depth epochs should be bounded reusable chunks, not a
  single epoch whose static history length equals all remaining samples. Large
  plateau rows can otherwise ask XLA to compile dozens of shell transitions at
  once and abort during compilation.
- Standard-problem evidence acceptance uses the MC shrinkage ensemble mean,
  `mean(log_Z_samples)`, compared with the reference at `3 * std(log_Z_samples)`.
  Do not replace this with `result.log_Z_mean` or `result.log_Z_uncert`; that is
  a changed acceptance criterion.
- Ticket 0020 direct pure-core Galilean uses a traced-safe streaming trajectory
  sampler inside compiled epochs. Keep the eager Python Galilean trajectory
  builders for standalone trajectory tests, but route traced sampler calls
  through JAX `while_loop` helpers that avoid `bool(np.asarray(...))`.
- Ticket 0020 direct pure-core Galilean root initialization is not traced, so
  `NestedSampler._sample_v3_root_state` must pass a `force_jax_galilean`
  adaptation-context flag. Ordinary standalone local/worker sampler calls
  should still use the eager trajectory path unless they are inside JAX tracing.
- Ticket 0020 traced Galilean support checks must use `jax.lax.cond` around
  the likelihood call. JAX boolean `&` is not short-circuiting, and the
  boundary search intentionally probes outside the unit cube.
- Ticket 0020 traced Galilean reflections must prefer the unit-cube support
  normal when the rejected point leaves `[0, 1]^D`; likelihood gradients can be
  tangent to the cube face and are not a valid support-boundary normal.
- Ticket 0020 pure-core root initialization should call a cached JIT Galilean
  transition wrapper. Calling raw `lax.while_loop` Galilean transitions from
  the Python root loop can trigger repeated XLA compilations and LLVM memory
  failures on the full 8D MVN gate.
- For pure-core Galilean performance, fusing all Galilean slices for one
  constrained sample into a cached chain JIT reduced the 60-sample 8D MVN
  first-run timing from about `49.5s` to about `7.0s`, with steady-state timing
  about `0.33s`. Do not regress back to per-slice transition JIT calls.
- For Galilean root initialization, the sentinel `-inf` contour does not need a
  Galilean trajectory: the model's `sample_U` prior draw is already the exact
  root constrained sample. Skipping root Galilean work reduced the best 1200
  sample 8D MVN timing to about `49s`.
- Current pure-core Galilean is not statistically accepted for full 8D MVN:
  the fast configuration (`num_slices=4`, step `0.01`, boundary limits `4`)
  runs under `60s` for seed `0` but misses the `3 * sample_std` evidence gate,
  and seed `29` fails badly. More work is needed before claiming Galilean
  standard-problem parity.
- A direct 1D Galilean transition check on the interval implied by
  `-(u - 0.5)^2 > -0.04` caught a serious endpoint bias: coarse boundary
  endpoints sampled approximately `[0.4, 0.7]` instead of `[0.3, 0.7]`.
  Fixed-step bisection of inside/outside brackets was still biased because the
  reflection point stayed tied to the inner endpoint. The current target is a
  stochastic bracket shrink: repeatedly sample uniformly between the inside and
  outside bracket endpoints, shrink the outside endpoint on rejections, and use
  the first inside draw as the reflection point.
- In the eager Galilean grow branch, keep `first_outside` synchronized with the
  first candidate that actually leaves the contour. Initializing it from the
  inside proposal makes stochastic bracket refinement sample the wrong segment
  and can reflect deep inside the contour.
- Traced pure-core Galilean should keep a static `max_reflections` and
  materialize fixed-size segment buffers per side. Uniform trajectory sampling
  then chooses a segment proportional to its valid length and samples uniformly
  along that segment. Do not hide the reflection sequence behind an unbounded
  side-level streaming reservoir.
- In traced Galilean, do not create a fake U-turn when a boundary search hits a
  limit. Clip forward proposals to the unit cube and treat that support wall as
  a hard `-inf` likelihood boundary with an explicit support normal.
- On this host JAX reports an NVIDIA GPU may be present, but the installed
  `jaxlib` is CPU-only. Pure-core benchmark reports should record backend and
  devices, and GPU conclusions require a CUDA-enabled `jaxlib`.
- Pure-core efficiency benchmarks should treat loose uncertainty/root-only
  rows as smoke data only. Always report analytic logZ error, MC-shrinkage
  variance, `evals * variance`, `evals * MSE`, target-hit fraction, and the
  standard `3 * sample_std` accuracy gate separately.
- For pure-core `UniDimSliceSampler`, fixed-capacity shells can contain masked
  padded slots when allocation targets do not fill the whole capacity. Those
  slots must not run constrained-sampler likelihood work; otherwise wall time
  is inflated without corresponding accepted likelihood-evaluation counts.
- Pure-core v3 root initialization now draws exact prior samples through
  `model.sample_U` at the sentinel contour and does not call the constrained
  sampler. Contract tests should expect model args/params forwarding at root;
  sampler args/params forwarding must be covered by non-root constrained work.
- Direct pure-core `run_until_goal`/`resume_until_goal` checks the public
  Python `goal_cond` at host boundaries after a compiled depth epoch has made
  sample progress. Tests that observe successive goal callbacks should expect
  the second observation to reflect that progress, not a no-op recheck.
- Pure-core manifest accuracy rows can exhaust CPU XLA code memory when many
  parametrized rows compile in one Python process. Clear JAX caches around
  manifest rows; isolated single rows passing after a full-row OOM usually
  indicates cache pressure, not row semantics.
- For the pure-core 8D `basic_mvn` isotropic row, straight-line step-out
  (`no_step_out=False`) is both faster and more accurate than perfect
  bracketing at the standard 30-live / 1200-sample / 24-slice gate. A seed-0
  probe measured about `176796` evals and a passing `0.70` error-over-3sigma
  ratio versus `431101` evals and failing `1.16` for `no_step_out=True`.
- Pure-core benchmark grids that sweep direction kernels/settings can exhaust
  CPU XLA section memory in one long Python process because compiled variants
  accumulate. Use benchmark row isolation for grid reporting on CPU-only hosts;
  keep algorithm tests in-process unless a test explicitly permits isolation.
- Pure-core direct seed selection should use the sorted active likelihood
  prefix with `searchsorted(..., side="right")` for the strict contour, then a
  bounded uniform integer offset. Rebuilding full `[max_samples]` seed masks
  in every work item adds inner-loop overhead and risks repeated shape work.
- Direct pure-core `run_until_goal` must return to the host goal boundary after
  a compiled depth epoch makes real sample progress. Letting the inner depth
  loop run to `depth_cond.max_samples` makes loose `log_Z_uncert` targets look
  artificially expensive and hides biased early-stop behavior in benchmarks.
- V3 root initialization at the sentinel contour should use exact prior draws,
  not constrained-sampler transitions from `log_L=-inf`. Preserve root phantom
  diagnostics by drawing independent prior phantom likelihoods, and cast root
  U leaves to `mp_policy.measure_dtype` so later ellipsoidal/GMM sampler outputs
  and invalid-slot placeholders have stable JAX dtypes.
- Pure-core efficiency reports on 8D MVN showed `result.log_Z_uncert` can be
  much smaller than analytic error at low depth/live-point budgets. Treat
  `evals * variance` as secondary unless rows also pass analytic
  `3 * sample_std`/RMSE gates; use minimum-sample guards and accuracy-gated
  rollups for ranking.
- For the 8D MVN pure-core benchmark, 40 live points hit a practical uncertainty
  floor near `0.7` even when sample caps are raised, while 80 live points can
  hit `log_Z_uncert < 0.7` too early and remain biased. Precision-target
  benchmarks should sweep live-point count and minimum depth together, not only
  sampler direction kernels.
