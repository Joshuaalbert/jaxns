# Implementation Ticket Index

Priority order is based on design risk, hot-path impact, and how much downstream work depends on the service.
Use the dependency DAG below rather than treating the numeric order as a strict
linear blocker. Each ticket should take the minimal perturbation path through
the current code: extend existing `src/jaxns` modules and tests where their
boundaries already match v3, and introduce new helper modules only when they
make the migration smaller and easier to verify.

## Dependency DAG

- `0001-race-tree-state-and-blocks.md`: no dependencies.
- `0002-bayesian-shrinkage-and-plateaus.md`: depends on `0001`.
- `0003-phantom-conditioned-shrinkage.md`: depends on `0001`, `0002`.
- `0004-results-api-and-diagnostics.md`: depends on `0001`, `0002`, `0003`.
- `0007-constrained-sampler-contract.md`: depends on `0001`, `0003`.
- `0005-core-execution-and-depth-first-allocation.md`: depends on `0001`, `0002`, `0007`.
- `0006-allocation-utilities.md`: depends on `0005`; non-blocking for `0008` and `0009`.
- `0008-direction-kernels-and-trajectories.md`: depends on `0005`, `0007`.
- `0009-distributed-runtime-and-async-identity.md`: load-balanced worker runtime and runner interface; depends on `0001`, `0003`, `0005`, `0007`.
- `0010-validation-and-benchmark-suite.md`: deterministic fixtures depend on `0001`, `0002`, `0003`; final roll-up depends on `0004`, `0005`, `0006`, `0008`, `0011`, and applicable feature tickets such as `0009` and `0012`.
- `0011-execution-diagnostics-integration.md`: depends on `0004`, `0005`, `0007`, `0008`, `0009`; include `0006` when evidence/posterior allocation diagnostics are enabled, and include `0012` only if Galilean/gradient-informed diagnostics are enabled.
- `0012-galilean-and-gradient-informed-sampler.md`: depends on `0007`, `0008`.
- `0013-gamma-weighted-phantom-conditioning.md`: supersedes the accepted
  Ticket 0003 `rho_g` phantom-conditioning target with the updated
  gamma-weighted per-cluster target; depends on `0001`, `0002`, `0003`,
  `0004`, `0007`, `0010`, and `0011`.
- `0014-gmm-non-isotropic-direction-kernel.md`: supersedes the accepted Ticket
  0008 one-component history ellipsoid adaptation with the posterior-weighted
  GMM direction-kernel target; depends on `0002`, `0004`, `0005`, `0007`,
  `0008`, `0010`, `0011`, and `0013`.
- `0015-standard-problem-performance.md`: optimizes accepted v3 standard
  problem runtime after statistical acceptance; depends on `0005`, `0006`,
  `0008`, `0009`, `0011`, `0013`, and `0014`.
- `0016-standard-problem-speed-benchmarks.md`: adds repeatable local-LB speed
  reference benchmarks for the accepted standard-problem path; depends on
  `0010` and `0015`.
- `0017-standard-problem-performance-benchmark-driven.md`: uses the accepted
  benchmarks to reduce full 8D `basic_mvn` runtime while preserving
  representative standard-problem correctness; depends on `0015` and `0016`.
- `0018-likelihood-eval-dispatch-runtime.md`: splits local/parallel
  constrained sampling from process-isolated likelihood evaluation dispatch;
  depends on `0005`, `0007`, `0009`, `0011`, `0016`, and `0017`.
- `0019-process-isolated-worker-topology.md`: replaces the unreleased ordinary
  local-LB in-process likelihood worker boundary with the accepted process
  topology of load balancer -> node ingress/coordinator -> local worker
  processes over random `/tmp` IPC endpoints; depends on `0005`, `0007`,
  `0009`, `0011`, `0016`, `0017`, and `0018`.
- `0020-pure-jax-core-feature-parity.md`: replaces the Python-orchestrated v3
  core loop in `src/jaxns/core.py` with a paper-aligned pure-JAX control-flow
  core, and splits pure-core versus distributed accuracy/benchmark suites while
  keeping their feature matrix identical; depends on `0001`, `0002`, `0004`,
  `0005`, `0006`, `0007`, `0008`, `0010`, `0011`, `0012`, `0013`, `0014`,
  `0016`, `0018`, and `0019`.

## Suggested Waves

1. Foundation: `0001`.
2. Statistical core: `0002`, then `0003`, then `0004`; start deterministic validation fixtures from `0010` after `0003`.
3. Sampler/execution: `0007`, then `0005`; `0008` can run after both.
4. Allocation and advanced sampler modes: `0006` and `0012` can run without blocking unrelated work.
5. Load-balanced worker runtime and diagnostics: `0009`, then `0011`.
6. Statistical correction: `0013` updates accepted phantom conditioning before
   final validation claims rely on phantom-conditioned results.
7. Non-isotropic direction adaptation: `0014` updates accepted baseline
   ellipsoidal directions before final sampler-efficiency claims rely on GMM
   direction kernels.
8. Final validation and benchmark roll-up: remaining `0010` work.
9. Performance hardening: `0015` improves the accepted standard-problem gate
   without changing statistical acceptance criteria.
10. Speed reference points: `0016` records phase-level performance benchmarks
    for ongoing optimization work.
11. Benchmark-driven optimization: `0017` applies focused performance fixes
    against those reference points.
12. Likelihood dispatch runtime: `0018` moves the worker boundary to
    deterministic `U -> log_L` evaluations while local constrained samplers
    drive parallel parent tasks.
13. Process-isolated worker topology: `0019` makes that likelihood-eval
    boundary the ordinary local-LB process topology, with node-owned worker
    processes and benchmarked worker scaling.
14. Pure-JAX core feature parity: `0020` moves the accepted v3 statistical
    feature set into one JAX-control-flow core and proves pure-core and
    distributed feature-identical accuracy/benchmark coverage.

## Branch Names

- `feature/v3-race-tree-state-and-blocks`
- `feature/v3-bayesian-shrinkage-and-plateaus`
- `feature/v3-phantom-conditioned-shrinkage`
- `feature/v3-results-api-and-diagnostics`
- `feature/v3-core-execution-depth-first`
- `feature/v3-allocation-utilities`
- `feature/v3-constrained-sampler-contract`
- `feature/v3-baseline-directions-straight-line`
- `feature/v3-load-balanced-worker-runtime`
- `feature/v3-validation-benchmarks`
- `feature/v3-execution-diagnostics-integration`
- `feature/v3-galilean-gradient-sampler`
- `feature/v3-gamma-weighted-phantom-conditioning`
- `feature/v3-gmm-non-isotropic-direction-kernel`
- `feature/v3-standard-problem-performance`
- `feature/v3-standard-problem-speed-benchmarks`
- `feature/v3-benchmark-driven-standard-performance`
- `feature/v3-likelihood-eval-dispatch-runtime`
- `feature/v3-process-isolated-worker-topology`
- `feature/v3-pure-jax-core-feature-parity`

## Ticket Files

- `docs/implementation_orchestration/tickets/0001-race-tree-state-and-blocks.md`
- `docs/implementation_orchestration/tickets/0002-bayesian-shrinkage-and-plateaus.md`
- `docs/implementation_orchestration/tickets/0003-phantom-conditioned-shrinkage.md`
- `docs/implementation_orchestration/tickets/0004-results-api-and-diagnostics.md`
- `docs/implementation_orchestration/tickets/0005-core-execution-and-depth-first-allocation.md`
- `docs/implementation_orchestration/tickets/0006-allocation-utilities.md`
- `docs/implementation_orchestration/tickets/0007-constrained-sampler-contract.md`
- `docs/implementation_orchestration/tickets/0008-direction-kernels-and-trajectories.md`
- `docs/implementation_orchestration/tickets/0009-distributed-runtime-and-async-identity.md`
- `docs/implementation_orchestration/tickets/0010-validation-and-benchmark-suite.md`
- `docs/implementation_orchestration/tickets/0011-execution-diagnostics-integration.md`
- `docs/implementation_orchestration/tickets/0012-galilean-and-gradient-informed-sampler.md`
- `docs/implementation_orchestration/tickets/0013-gamma-weighted-phantom-conditioning.md`
- `docs/implementation_orchestration/tickets/0014-gmm-non-isotropic-direction-kernel.md`
- `docs/implementation_orchestration/tickets/0015-standard-problem-performance.md`
- `docs/implementation_orchestration/tickets/0016-standard-problem-speed-benchmarks.md`
- `docs/implementation_orchestration/tickets/0017-standard-problem-performance-benchmark-driven.md`
- `docs/implementation_orchestration/tickets/0018-likelihood-eval-dispatch-runtime.md`
- `docs/implementation_orchestration/tickets/0019-process-isolated-worker-topology.md`
- `docs/implementation_orchestration/tickets/0020-pure-jax-core-feature-parity.md`
