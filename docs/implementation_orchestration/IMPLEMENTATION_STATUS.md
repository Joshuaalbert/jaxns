# Implementation Status

This file tracks the implementation-loop state after the test-first gate.

Last updated: current orchestration pass.

## Ready For Implementation

None currently.

## Under Review

None.

## Accepted With Follow-Up

- `0001-race-tree-state-and-blocks`: implementation accepted with follow-up for
  a low-level default-validation decision and possible plateau-stable parent
  diagnostics if later diagnostics require reconstructed parent information.

## Accepted

- `0002-bayesian-shrinkage-and-plateaus`: implementation accepted after v3 public
  result evidence, canonical block MC shrinkage, and invalid plateau-capacity
  public-path fixes.
- `0003-phantom-conditioned-shrinkage`: implementation accepted after v3
  Dirichlet phantom-conditioned shrinkage, exact cluster-bootstrap `rho_g`, and
  independent NumPy reference fixes.
- `0004-results-api-and-diagnostics`: implementation accepted after the public
  result surface exposed block metadata, v3 posterior weights, phantom count
  summaries, shrinkage-returned `rho_values`/`rho_fit`, explicit validation,
  and block-safe plotting/resampling behavior.
- `0007-constrained-sampler-contract`: implementation accepted after strict
  contour validation, likelihood-only phantom clusters, and public sampler
  docstrings for the v3 sampler contract.
- `0005-core-execution-depth-first`: implementation accepted after v3
  root-init/outer-goal/inner-depth loop, fixed `d_0`/`k` targets, strict
  weighted parent work, sentinel fallback, v3 `dlogZ` depth handling, and
  legacy `run()`/`resume()` compatibility documentation.
- `0006-allocation-utilities`: implementation accepted after stable
  evidence/posterior utility target construction, conservative posterior option
  plumbing, shared integer targets, multi-deficit scheduling, and zero-utility
  base-target scheduling.
- `0008-baseline-directions-straight-line`: implementation accepted after
  explicit baseline sampler modes, symmetric isotropic directions, frozen
  ellipsoidal direction kernels from adaptation context/sample history,
  bounded straight-line greedy shrinkage with current-point fallback, and clear
  unsupported Galilean/gradient mode failures.
- `0012-galilean-gradient-sampler`: implementation accepted after explicit
  `trajectory="galilean"` local and worker-backed modes, Hausdorff-style
  reflection helpers, reversible unit-cube bounded trajectory construction,
  path-length-uniform sampling, phantom exclusion of internal trajectory
  points, public `NestedSampler.run()` compatibility, and clear rejection of
  `trajectory="gradient_guided"`/legacy `gradient_guided=True`.
- `0010-validation-benchmarks`: implementation accepted after early
  deterministic validation fixtures/producers and the minimal public collector
  skeleton that runs real v3 public toy validations for baseline race-tree,
  phantom-conditioned, dynamic allocation, and Galilean method settings,
  carries result diagnostics, requires real MC shrinkage/rho diagnostics, and
  reports measured smoke-test guardrails without claiming final benchmark
  numbers.
- `0011-execution-diagnostics-integration`: implementation accepted after
  public diagnostics schema/accessors, result-side diagnostics attachment,
  allocation/depth/goal/parent/sampler diagnostics, coordinator-owned
  worker-runtime dispatch records, pre-execution trace publication, explicit
  unavailable timing semantics, true `dlogZ` condition values, non-fabricated
  phantom evaluation counts, and strengthened public diagnostics validation.

## Tests Accepted Provisionally

None currently.

## Completed

None yet. A ticket only moves here after tests pass, implementation review
findings are resolved, and no known design ambiguity remains.

## Integration

- `0009-distributed-runtime-and-async-identity`: in implementation loop. A
  first runtime-contract slice added identity/payload dataclasses, local
  compute-sector registration, serialized model/args/params helpers, and an
  acceptance ledger. The identity remediation has been independently accepted
  after adding shared local load-balancer state, scoped retry/delivery ordinals,
  multi-client ledger-isolation coverage, and real `Model.init_params(...)` /
  `CtxParams` payload coverage. The local worker-execution slice has also been
  independently accepted after serialized worker-boundary execution, default
  sampler support, public coordinator dispatch records, per-task acceptance
  ledger mutation, concurrent local-client isolation, and phantom-coordinate
  append-shape fixes. The dispatch-lifecycle slice has now been independently
  accepted after public non-state-taking lifecycle methods, terminal
  failed/revoked attempt tracking, first-valid-completion-wins retry semantics,
  stale parent rejection, independent task/attempt/transport mismatch
  rejection, and explicit sentinel/parent metadata preservation. The worker
  registration/shutdown slice has also been independently accepted after
  owner-keyed compute-sector cleanup, same-address local/tcp namespace sharing,
  distinct-address isolation, shared-address shutdown requests, and fresh
  re-entry/recreation semantics. The compile/cache identity and in-process
  fairness slice has been independently accepted after deterministic public
  compile identity, record-level client/cache diagnostics, exact sector/worker
  round-robin diagnostics, cross-runner lifecycle ownership validation, and
  removal of an over-broad runtime-side `Model.init_params` patch. Full ticket
  legacy public entry-point migration has been independently accepted after
  moving distributed nested-sampler tests to `LoadBalancerClient` and making
  `NestedSamplerDistributed`/`DistributedNestedSampler` fail clearly toward the
  v3 runtime instead of silently running the old independent core. The actual
  worker failure/retry integration slice has also been independently accepted
  after wiring the real serialized local worker execution path through pending,
  failed, retried, and accepted lifecycle records while preserving task id,
  retry identity, parent/contour/seed/phantom metadata, and exactly-once ledger
  acceptance. Full ticket blockers remain for physical cleanup of private
  old-loop helpers, real remote `tcp://` transport/process lifecycle, actual
  compile-cache execution across devices, and true out-of-order worker
  completion over asynchronous worker scheduling.
- `0010-validation-benchmarks`: minimal production collector accepted. Final
  expensive benchmark numbers remain a release/benchmarking activity using the
  accepted collector and any later runtime transport improvements.

## Current Test Gate

- Passed locally: `tests/test_state.py`, `tests/test_results_plotting.py`,
  `tests/test_runtime.py`, `tests/test_v3_run_pattern.py`,
  `tests/test_v3_phantom_results.py`, `tests/test_v3_sampler_contract.py`,
  `tests/test_v3_race_tree.py`, `tests/test_v3_shrinkage.py`,
  `tests/test_v3_allocation.py`, `tests/test_phantom_eval_jax.py`,
  `tests/test_phantom_eval_ref.py`, `tests/test_constrained_sampler.py`, and
  `tests/test_distributed_core.py` in targeted combinations. The final 0011
  acceptance run passed `tests/test_v3_execution_diagnostics.py`,
  `tests/test_runtime.py`, and `tests/test_v3_run_pattern.py` together. The
  0009 retry-execution acceptance run passed `tests/test_runtime.py` and
  `tests/test_v3_run_pattern.py` in focused review. The 0010 collector
  acceptance run passed `tests/test_runtime.py`, `tests/test_v3_run_pattern.py`,
  `tests/test_v3_execution_diagnostics.py`,
  `tests/test_v3_validation_benchmark_collector.py`,
  `tests/test_v3_validation_benchmark_producers.py`,
  `tests/test_v3_validation_benchmark_schema.py`, and
  `tests/test_v3_validation_deterministic_fixtures.py` together.
- Not rerun in the current pass: `tests/test_ns_standard_problems.py` because
  it is slow/stochastic standard-problem coverage and should be used as part of
  the later validation gate.
- No accepted test-first drafts are currently expected to fail.
