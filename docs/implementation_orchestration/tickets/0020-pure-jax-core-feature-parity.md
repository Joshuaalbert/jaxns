# Ticket 0020: Pure-JAX Core Feature Parity

Branch: `feature/v3-pure-jax-core-feature-parity`
Priority: after Ticket 0019 topology review stabilizes, before further v3
core/runtime speed claims
Depends on: Ticket 0001, Ticket 0002, Ticket 0004, Ticket 0005, Ticket 0006,
Ticket 0007, Ticket 0008, Ticket 0010, Ticket 0011, Ticket 0012, Ticket 0013,
Ticket 0014, Ticket 0016, Ticket 0018, Ticket 0019
Design docs:

- `docs/design/jaxns-v3-design-overview.md`
- `docs/design/jaxns-v3-statistical-core.md`
- `docs/design/jaxns-v3-phantom-conditioning.md`
- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-constrained-sampling.md`
- `docs/design/jaxns-v3-process-isolated-zmq-runtime.md`
- `docs/design/jaxns-v3-validation-plan.md`
- `docs/design/interface/run_pattern.py`
- `docs/implementation_orchestration/UNIT_TEST_STANDARDS.md`
- `docs/implementation_orchestration/PERFORMANCE_TEST_STANDARDS.md`
- `docs/implementation_orchestration/tickets/0013-gamma-weighted-phantom-conditioning.md`
- `docs/implementation_orchestration/tickets/0014-gmm-non-isotropic-direction-kernel.md`
- `docs/implementation_orchestration/tickets/0016-standard-problem-speed-benchmarks.md`
- `docs/implementation_orchestration/tickets/0018-likelihood-eval-dispatch-runtime.md`
- `docs/implementation_orchestration/tickets/0019-process-isolated-worker-topology.md`

## Goal

Replace the current Python-orchestrated v3 execution path in
`src/jaxns/core.py` with one pure-JAX statistical core that matches the current
paper/design feature set. The new core must express the nested-sampling
coordinator state transitions with JAX control flow and fixed-shape carries,
while preserving the accepted v3 statistical contracts and public run pattern.

This is a core update, not a runtime-topology ticket. Runtime modules may be
used by tests and benchmarks to prove distributed feature parity, but production
runtime code is out of scope.

## No-v3-backcompat Assumption

v3 has not been released. Do not preserve backwards compatibility with earlier
unreleased v3 core loops, local-LB scheduler behavior, or test expectations
that conflict with the current paper/design target. The implementation may
replace the accepted Python v3 run loop rather than wrap it.

Existing non-v3 compatibility paths such as fixed-live-point `run()` /
`resume()` may remain only as explicitly documented compatibility wrappers or
legacy paths. They must not force a second v3 statistical core or constrain the
paper-aligned `run_until_goal(...)` / `resume_until_goal(...)` behavior.

## Current Code Context

Relevant files likely include:

- `src/jaxns/core.py`
- `src/jaxns/allocation.py`
- `src/jaxns/state.py`
- `src/jaxns/samples.py`
- `src/jaxns/phantom_eval.py`
- `src/jaxns/diagnostics.py`
- `src/jaxns/constrained_sampler.py`
- `src/jaxns/em_gmm.py`
- `tests/test_v3_run_pattern.py`
- `tests/test_v3_execution_diagnostics.py`
- `tests/test_ns_standard_problems.py`
- `tests/test_runtime.py`
- `tests/test_v3_performance_benchmarks.py`
- `benchmarks/v3_performance/standard_problem_speed.py`

At ticket preparation time, `NestedSampler.run_until_goal(...)` and
`resume_until_goal(...)` route through `_resume_until_goal(...)`, which uses
Python `for`/`while` loops, host-side NumPy parent/seed selection, and
per-parent Python sampler orchestration. That path has accumulated the accepted
v3 feature semantics, but it is now the known bottleneck for cheap likelihoods
and distributed worker scaling. The pure-JAX update must migrate those
semantics into JAX-control-flow state transitions rather than deleting feature
coverage.

## Required Core Target

The new `src/jaxns/core.py` v3 path must provide one statistical core used by
local pure-core, batched, process-isolated local-LB, and distributed runs. The
deployment layer may change how likelihood probes are evaluated, but accepted
child integration, out-degree mutation, allocation accounting, phantom
metadata, diagnostics, and result assembly must use the same core semantics.

JAX-control-flow requirements:

- Use explicit fixed-capacity arrays, masks, and scalar counters for core state
  carries.
- Use `jax.lax.while_loop`, `jax.lax.scan`, `jax.lax.cond`, or equivalent JAX
  control flow for data-dependent execution inside the pure core.
- Keep PRNG keys in the JAX state carry with deterministic split schedules.
- Avoid host NumPy selection, Python data-dependent loops, and variable-shape
  JAX arrays in the core transition path.
- Keep dynamic candidate pools, active blocks, phantom buffers, and trajectory
  buffers represented by static shapes plus masks, or trim them before they
  enter the JAX-visible core.
- Ensure the same ordered inputs and PRNG key produce deterministic accepted
  race-tree state and diagnostics for the pure-core path.

## Internal Pure-Core / Runtime Boundary

The test-first branch must not invent this boundary. Ticket 0020 defines an
internal fixed-shape work/result boundary between the statistical core and any
deployment layer:

- The pure-JAX core owns fixed-shape planning: allocation target construction,
  active-block summaries, parent-work selection, seed/sentinel fallback
  identities, direction-kernel snapshot ids, phantom buffer slots, and static
  masks/counters for all planned work.
- For direct pure-core mode, the pure-JAX core also owns the local constrained
  sampler proposal loop, including deterministic proposal-key schedules,
  strict-contour checks, phantom-retention masks, and local likelihood calls
  through JAX-compatible model functions.
- The core owns deterministic acceptance. Completed child results are
  integrated in planned parent-work order, and only this acceptance step mutates
  out-degrees, accepted sample buffers, allocation accounting, phantom
  metadata, diagnostics, and result assembly.
- Runtime or distributed deployment may produce likelihood results or completed
  proposal/child results outside the compiled pure-core sampler loop, but it
  must consume core-defined fixed-shape work buffers and produce core-defined
  fixed-shape result buffers with validity/status masks and parent-work
  identities. Raw likelihood/proposal completions must never mutate the race
  tree directly.
- Distributed and local-LB execution must reuse the same core acceptance
  semantics as pure-core execution. Out-of-order worker completions are allowed
  only before the fixed-shape result buffer is handed back for deterministic
  ordered acceptance.

This is an internal implementation/testing contract, not a request to implement
production runtime changes in this ticket.

## Goal Condition Boundary

The public Python goal loop remains a host boundary around compiled depth and
transition kernels:

- `run_until_goal(...)` / `resume_until_goal(...)` may keep a Python outer loop
  that materializes the current `State` at outer-loop boundaries and calls the
  public `goal_cond(state: State) -> bool`.
- The JAX core must not call arbitrary Python callbacks from inside `jit`,
  `jax.lax.while_loop`, `jax.lax.scan`, or other compiled control flow.
- Built-in depth checks and state-transition kernels are JAX-expressible and
  compiled; the user-supplied Python `goal_cond` is evaluated only between
  those compiled epochs.
- A future API may add an explicitly JAX-expressible goal predicate, but that
  is separate and not required for Ticket 0020.

## Target Feature Set

The pure-JAX core must preserve the current paper/design feature set:

- Strict race-tree accounting with sentinel root lineage count `d_0`, sample
  out-degrees, active lineage counts, no persisted parent requirement, and exact
  in-flight parent preservation until acceptance.
- Plateau-aware block construction with `K_g >= m_g`, equality atom
  separation, and posterior mass split equally across plateau block members.
- Bayesian shrinkage over `(p_{>g}, p_{=g}, p_{<g})` with the accepted epsilon
  policy, Monte Carlo evidence samples, and active v3 posterior weights.
- Gamma-weighted phantom conditioning with parent-contour eligibility,
  per-cluster `A_{cg}`, `B_{cg}`, `E_{cg}`, `R_{cg}` counts, one
  `Gamma(1, 1)` cluster weight per cluster per joint draw, Kish gate
  `C_min = 20` by default, and gate diagnostics. Blocks with no participating
  clusters or insufficient Kish participation use the classic race posterior
  for that block; zero-participant blocks fail the gate even if an override
  sets `C_min <= 0`. Public probability summaries such as `p_gt_mean`,
  `p_eq_mean`, and `p_lt_mean` must describe the active shrinkage target
  returned by MC sampling: phantom-conditioned where the gate is active and
  classic race posterior otherwise. Classic-only summaries must be named
  explicitly as classic-only diagnostics.
- Phantom storage policy: retained post-burn-in phantom likelihoods and cluster
  metadata only; phantom samples never change out-degrees, `K_g`, or `m_g`.
- Allocation targets `"uniform"`, `"evidence_improving"`, and
  `"posterior_improving"` with `K_*^k(L) = d_0 + k DeltaK Ubar(L)`,
  unit-peak utility normalization, explicit integer targets, and positive
  `DeltaK` validation.
- Parent selection from under-allocated contours using the paper's
  parent-block weighting proportional to `(X_{g'} / X_g) 1{X_{g'} < X_g}`,
  with deterministic seed selection and sentinel fallback when no strict seed
  exists.
- Depth and goal handling at the correct loop boundaries: depth inside the
  current allocation target, goal only at outer-loop boundaries, and the public
  host-side `goal_cond(state: State) -> bool` contract above. The
  posterior-mass depth condition uses the active shrinkage volume path
  `X_g = X_{g-1} p_{>g}` for valid blocks only and tests the deepest valid
  remaining strict tail mass against `max_{g'} L_{g'} X_{g'}` with padded blocks
  masked out.
- Accepted constrained-sampler modes from the current design: isotropic
  directions, GMM non-isotropic direction snapshots, straight-line bracketing,
  Galilean trajectories, strict seed validation, and phantom exclusion of
  trajectory-building points.
- Coordinator-owned direction-kernel adaptation from accepted classic samples
  and active v3 posterior weights, including the five-distinct-shell cadence
  and fallback diagnostics. When posterior weights are derived from MC
  shrinkage draws, normalize sample masses within each draw first, then average
  normalized per-sample weights across draws; do not average raw masses and
  normalize once.
- Execution diagnostics for allocation, depth, goal, parent selection, sampler
  behavior, direction adaptation, phantom gates, and worker-runtime observations
  where the deployment layer supplies them.

## Split But Feature-Identical Test And Benchmark Shape

The test-first slice must introduce a shared feature manifest before
implementation. Pure-core and distributed suites must both parameterize from
that manifest, and a focused coverage test must fail if either suite omits a
feature row present in the other.

Each manifest row must specify at least:

- problem fixture and known-reference evidence/posterior metadata;
- seed set;
- allocation target;
- depth condition and goal condition;
- phantom collection on/off and `C_min` when overridden;
- direction kernel mode;
- trajectory mode;
- resume/run pattern;
- expected diagnostic families;
- whether the row is accuracy, benchmark, or both.

The pure-core suite must exercise the direct local core path without requiring
`LoadBalancerClient`. The distributed suite must exercise the public
`LoadBalancerClient(address="local")` path and the accepted process-isolated
node/worker topology once Ticket 0019 is available. The two suites may use
different worker counts and timing metadata, but they must cover the same
statistical feature rows and enforce the same accuracy acceptance criteria.

## Test-First Plan

Write tests before production implementation. Review them against
`UNIT_TEST_STANDARDS.md`, `PERFORMANCE_TEST_STANDARDS.md`, this ticket, and the
linked design docs before implementation starts.

Required pure-core unit tests:

- JAXPR or JIT smoke tests prove the public pure-core run path enters a
  JAX-control-flow core without host-side data-dependent parent loops.
- Core state carries keep static shapes while `num_samples`, active block
  counts, phantom validity, and in-flight parent sets change through masks and
  counters.
- Root initialization draws exactly `d_0` strict sentinel children with the
  deterministic PRNG schedule and no premature goal check.
- Inner depth loop and outer goal loop execute at the paper-defined boundaries.
- Allocation targets construct identical integer target plans to the accepted
  v3 formulas for uniform, evidence-improving, and posterior-improving modes.
- Parent selection preserves requested parent, effective parent, accepted
  parent, target block, parent block, and sentinel fallback identity until
  acceptance.
- Acceptance mutates only the effective parent's out-degree or the sentinel
  root out-degree after fallback.
- Plateau fixtures preserve equality atom mass, reject invalid `K_g < m_g`,
  and keep posterior weights aligned with block membership.
- Phantom-conditioning fixtures cover singleton independent clusters,
  correlated multi-sample clusters, no participating clusters, Kish gate
  pass/fail, `R_cg` open-interval contribution, active probability-summary
  semantics, and the default `C_min = 20` / zero-participant gate behavior.
- Posterior-mass depth-condition fixtures cover masking of padded blocks,
  deepest-valid-tail selection, plateau/equality atom handling through the
  strict endpoint `p_{>g}`, and active phantom-conditioned versus classic
  shrinkage summaries.
- Direction-kernel adaptation tests cover initial isotropic fallback,
  posterior-weighted GMM fitting, five-distinct-shell cadence, failed-fit
  retry semantics, frozen dispatch snapshots, Galilean phantom exclusion, and
  GMM fitting weights aligned with active posterior weights from results.
- Determinism tests compare repeated pure-core runs with identical ordered
  inputs and keys.
- Negative-path tests reject unsupported allocation targets, invalid
  `DeltaK`, malformed phantom metadata, stale or shape-mismatched state, and
  unsupported dynamic-shape proposals.

Required split accuracy tests:

- Define a shared feature manifest and assert exact row coverage equality
  between pure-core and distributed accuracy tests.
- Include analytic evidence, plateau/equality atom, phantom-conditioned,
  dynamic allocation, GMM direction-kernel, Galilean trajectory, and resume
  scenarios.
- For standard problems, keep the accepted direct-pytest policy and
  `3 * sample_std` evidence criterion.
- Run representative standard-problem rows for at least `basic_mvn`,
  `spike_slab`, and `plateau` across all three allocation targets in both
  pure-core and distributed suites.
- Include a focused full original `STANDARD_PROBLEM_CASES` gate for the
  pure-core suite. The distributed full-list gate may be marked slow, but must
  use the same manifest rows and acceptance criteria.

Required distributed parity tests:

- Ordinary `LoadBalancerClient(address="local")` execution routes through the
  process-isolated local-LB topology from Ticket 0019 and still reports the
  same feature row ids as the pure-core suite.
- Distributed accepted state agrees with pure-core invariants for sample count,
  out-degree conservation, block validity, phantom cluster metadata, allocation
  target summaries, and result diagnostic families.
- Raw likelihood responses never mutate the race tree; only completed child
  acceptance through the shared core semantics may update state.
- Distributed work/result adapters consume and produce the same fixed-shape
  work/result buffer schema as the pure-core path, with validity/status masks
  and parent-work identities sufficient for deterministic ordered acceptance.
- Asynchronous or out-of-order worker completion preserves deterministic
  coordinator integration order for accepted parent work.

## Benchmark Gates

Record pre-implementation baselines before changing production code. Record
final benchmark outputs with the same problem settings and include them in this
ticket's implementation log.

Required pure-core benchmark gate:

- Add or extend a pure-core benchmark command that does not require
  `LoadBalancerClient`.
- Use the full 8D `basic_mvn` standard problem, explicit live-point policy,
  `uniform`, `evidence_improving`, and `posterior_improving` allocation
  targets, and the same result/MC-shrinkage synchronization rules as Ticket
  0016.
- Report setup/compile time separately from run time, result conversion time,
  MC shrinkage time, likelihood evaluations, samples, evidence estimate,
  reference evidence, allocation target, direction/trajectory mode, phantom
  settings, and static-shape/JIT-cache diagnostics.
- Final pure-core run time for the primary full 8D `basic_mvn` uniform row must
  use the isotropic direction distribution, pass the MC-shrinkage evidence
  criterion, and complete the sampler body in less than `60s` on the local CPU
  test machine. The report must keep result conversion and MC shrinkage timings
  separate from sampler run time.
- The primary pure-core row should still improve over the pre-0020 Python-core
  baseline on the same machine and command family. If the baseline is noisy,
  require the median of at least three final runs to improve over the median of
  at least three baseline runs.
- The primary pure-core row must not regress more than 10 percent relative to
  Ticket 0017's accepted pure-local reference of `88.99s` total / `86.81s` run
  unless review records a hardware/configuration mismatch and accepts a new
  same-machine reference.
- Repeated same-shape rows must not show per-sample or per-shell recompilation;
  compilation diagnostics must be bounded by static scenario shape, not by
  accepted sample count.

Required distributed benchmark gate:

- Add or extend a distributed benchmark command that uses
  `LoadBalancerClient(address="local")` and the Ticket 0019 process-isolated
  topology.
- Parameterize it from the same feature manifest as the pure-core benchmark.
- Report requested worker spec, observed node count, live node
  ingress/coordinator count, observed worker process count, worker utilization,
  queue diagnostics, compile/cache diagnostics, setup time, run time, result
  time, MC shrinkage time, likelihood evaluations, and feature row id.
- Distributed benchmark rows must pass the same accuracy criteria as pure-core
  rows before timing numbers are accepted.
- Distributed speedup over worker counts remains owned by Ticket 0019 unless
  this ticket changes runtime code, which it must not. This ticket's distributed
  benchmark gate is feature parity, accuracy parity, diagnostics completeness,
  and no regression against the pre-0020 same-worker-count baseline beyond
  10 percent median run time without a strict-review exception.

## Out Of Scope

- Implementing or refactoring `src/jaxns/runtime.py`, fabric process managers,
  node ingress/coordinator code, ZMQ transport, worker process lifecycle, or
  load-balancer scheduling.
- Replacing the Ticket 0019 process-isolated topology.
- Changing the paper statistical target, shrinkage law, phantom-conditioning
  target, allocation formulas, or evidence acceptance criteria.
- Reintroducing the rejected compiled fused constrained-sampler worker path.
- Preserving earlier unreleased v3 Python-loop behavior for compatibility.
- Weakening standard-problem, phantom, plateau, GMM, Galilean, or runtime
  diagnostic coverage to make the pure-JAX transition easier.

## Review Gates

- Test-first review accepts the shared feature manifest, pure-core tests,
  distributed parity tests, and benchmark schemas before implementation starts.
- Reviewers confirm every linked design feature in this ticket has at least one
  pure-core test and one distributed parity row, or an explicit documented
  reason why the feature is pure-core-only or runtime-only.
- Implementation review confirms production runtime modules were not changed
  except for test-only imports or benchmark wiring that does not alter runtime
  behavior.
- Implementation review confirms the v3 public run pattern uses one statistical
  core and that any remaining legacy `run()` / `resume()` path is not a second
  v3 implementation.
- JAX-control-flow review confirms core carries have static shapes and no
  host-side data-dependent parent, seed, allocation, or acceptance loops remain
  in the pure-core transition path.
- Statistical review confirms race-tree, plateau, phantom-conditioning,
  allocation, direction-adaptation, and diagnostics invariants match the
  current paper/design docs.
- Benchmark review confirms pre/post benchmark outputs were recorded with the
  same command family, setup/compile/run timing is separated, MC shrinkage is
  materialized before timing stops, and distributed timing claims are not used
  to assert runtime topology improvements.
- `conda run -n jaxns_py pytest ...` passes for the focused pure-core,
  distributed parity, runtime diagnostics, standard-problem, and benchmark
  tests selected by the test-first review.
- `conda run -n jaxns_py ruff check ...` passes for touched Python files once
  implementation exists.

## Implementation Review Follow-Ups

- Strict review after the first implementation slice blocked acceptance because
  the direct `NestedSampler` public path called a compiled one-shell transition
  from a Python depth/allocation loop. The trace hook used a real
  `jax.lax.while_loop`, but `run_until_goal(...)` / `resume_until_goal(...)`
  did not call the same compiled depth epoch. Resolve by moving the direct
  pure-core depth epoch, allocation-plan construction, work planning, sampling,
  and ordered acceptance into one reusable JAX-control-flow function used by
  both the public direct path and `trace_inner_depth_transition_core(...)`.
- The runtime/distributed fallback still trims `CoreWorkBatch` into variable
  `ParentWork`, samples through `_sample_parent_work(...)`, and accepts through
  `accept_parent_work(...)`. This may remain a follow-up if explicitly
  documented as the runtime adapter slice, but it must not be described as the
  completed fixed-shape runtime boundary.
- Core boundary metadata must be updated after seed/sentinel fallback. If a
  planned parent has no strict seed, `CoreWorkBatch.effective_parent_idx`,
  `fallback_to_root`, `parent_block_idx`, and `seed_idx` must match the
  `ParentWork` accepted by the core result/acceptance path.
- Direct pure-core Galilean trajectories are implemented through a
  traced-safe streaming sampler path. The eager Python trajectory builders
  remain for standalone geometry tests, while compiled pure-core sampler calls
  avoid Python `bool(np.asarray(...))` validation on traced values. The traced
  path must short-circuit unit-cube support checks before model likelihood
  evaluation, must run only the active grow-or-shrink boundary-search branch,
  and must keep ordinary standalone local and worker-backed Galilean sampler
  calls on the eager trajectory path. Unit-cube support hits must reflect on
  the support normal, not on an unrelated likelihood gradient, and the traced
  boundary grow/shrink searches keep the configured Galilean boundary-search
  limits while the outer reflection loop remains U-turn driven.
- Follow-up status: the traced Galilean routing/compilation slice passes
  focused contract tests, but full 8D MVN Galilean standard-problem parity is
  not accepted yet. A fast exploratory configuration can run under 60 seconds,
  but evidence accuracy is seed-unstable and misses the `3 * sample_std` gate.
  Do not mark Galilean standard-problem parity complete until that gate passes.

## Acceptance Criteria

- `src/jaxns/core.py` exposes a paper-aligned v3 pure-JAX core for
  `run_until_goal(...)` and `resume_until_goal(...)`.
- Pure-core and distributed tests are split but feature-identical through a
  shared manifest and explicit coverage equality checks.
- Accuracy gates pass for the shared feature matrix in both pure-core and
  distributed modes.
- Benchmark gates record baseline and final pure-core/distributed rows, with
  pure-core improvement over the Python-core baseline and no unexplained
  distributed same-worker-count regression.
- No runtime production code is implemented in this ticket.
