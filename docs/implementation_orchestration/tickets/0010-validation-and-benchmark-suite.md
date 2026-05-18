# Ticket 0010: Validation and Benchmark Suite

Branch: `feature/v3-validation-benchmarks`
Priority: staged validation
Depends on: Ticket 0001, Ticket 0002, Ticket 0003 for deterministic fixtures;
feature-specific benchmark phases depend on the corresponding feature tickets
Design docs:

- `docs/design/jaxns-v3-validation-plan.md`
- `docs/design/jaxns-v3-statistical-core.md`
- `docs/design/jaxns-v3-phantom-conditioning.md`
- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-constrained-sampling.md`

## Goal

Create validation fixtures and benchmark harnesses needed to trust the v3
migration. This ticket has an early deterministic-fixture phase that should land
after Tickets 0001-0003, and a final benchmark-rollup phase after feature
tickets and public diagnostics provide their method settings.

## Current Code Context

Relevant files likely include:

- `tests/test_ns_standard_problems.py`
- `tests/test_phantom_eval_jax.py`
- `tests/test_phantom_eval_ref.py`
- `tests/test_distributed_core.py`
- `tests/test_fabric_distributed.py`
- `benchmarks/`
- `docs/papers/phantom-powered-nested-sampling/`
- `docs/implementation_orchestration/PERFORMANCE_TEST_STANDARDS.md`

Current tests cover some standard problems, phantom evaluator behavior, and
older distributed components. The v3 design needs explicit calibration and
benchmark outputs tied to the paper's diagnostics and the load-balanced worker
runtime.

## Required Behavior

Final validation must compare:

- baseline race-tree nested sampling;
- phantom-conditioned shrinkage;
- dynamic contour allocation;
- gradient-informed constrained sampling through the Galilean method implemented
  in Ticket 0012.

Phase dependencies:

- Early deterministic fixtures depend on Tickets 0001, 0002, and 0003.
- Final public validation roll-up depends on Ticket 0004 for result APIs,
  Ticket 0005 for execution, Ticket 0006 for dynamic utility allocation,
  Tickets 0008 and 0012 for sampler method settings, Ticket 0009 for
  load-balanced worker correctness where distributed timing is reported, and
  Ticket 0011 for runtime diagnostics.

Correctness/calibration diagnostics:

```text
z_logZ = (hat logZ - logZ_ref) / hat sigma_logZ
```

Across independent seeds, expected behavior is mean near zero and standard
deviation near one.

Define and test the estimator used for `hat logZ` and `hat sigma_logZ` before
reporting z-scores. The convention must state whether `hat logZ` is the mean of
`logZ` samples, `logmeanexp` of evidence samples, or another summary, and the
uncertainty estimator must match that choice.

Reports must include:

- bias in `hat logZ`;
- empirical RMSE;
- reported uncertainty versus empirical uncertainty;
- expectation-based estimates versus Monte Carlo shrinkage estimates;
- per-block `rho_g` estimates alongside fitted `rho_g` curves.

Plateau tests must include equality atoms and measure equality-mass recovery in
addition to the diagnostics above.

Evidence-efficiency benchmarks must report:

- likelihood evaluations per effective sample size;
- wall-clock timing where meaningful;
- RMSE versus likelihood evaluations;
- `MSE * likelihood_evals` dominance summaries.

Evidence-efficiency and posterior-quality benchmarks must average over many
independent seeds, not only the calibration table.

Posterior-quality benchmarks must report posterior discrepancy separately from
evidence calibration, using a Monte Carlo estimate of Wasserstein distance
against reference posterior samples.

Load-balanced worker validation should report wall-clock timing where
meaningful. When distributed scaling is evaluated, include worker count,
throughput, scheduler overhead, serialization/communication issues, multi-tenant
fairness, and async identity preservation as useful diagnostics rather than
paper-mandated pass/fail thresholds.

## Out Of Scope

- Implementing core v3 features. Covered by Tickets 0001-0009.
- Claiming final benchmark numbers before the implementation is complete.
- Public result and runtime diagnostics APIs. Covered by Tickets 0004 and 0011.

## Test Plan

Write tests and benchmark harnesses before using them for claims.

Required deterministic tests:

- Analytic evidence/survival-curve toy problems.
- Plateau/equality atom toy problems.
- Phantom-conditioned toy problems with known count effects.
- Race-tree accounting assumptions: strict constraints, sentinel/sample
  out-degrees, in-flight parent target handling, and active lineage counts.
- Posterior weighting on plateau and non-plateau samples.

Required benchmark harnesses:

- Early deterministic validation fixtures for analytic evidence, plateaus,
  phantom count effects, race-tree accounting, and posterior weights.
- Multi-seed evidence calibration table producer.
- Pareto plot data producer for RMSE versus likelihood evaluations.
- Posterior Wasserstein-distance data producer.
- Optional load-balanced worker wall-clock timing harness where likelihood cost
  is high enough to make timing meaningful.
- Performance guardrails with measured thresholds and rationale for hot paths:
  block construction, Bayesian shrinkage sampling, phantom counting,
  `rho_g` bootstrap estimation, constrained-sampler trajectories, serialization,
  and worker task latency.
- A release-version timing history or append-only benchmark record sufficient to
  catch regressions, following `PERFORMANCE_TEST_STANDARDS.md`.

## Implementation Notes

- Keep unit tests deterministic and fast. Put slow benchmarks under benchmark
  scripts or marked tests.
- Extend existing tests and benchmark directories first:
  `tests/test_ns_standard_problems.py`, `tests/test_phantom_eval_jax.py`,
  `tests/test_phantom_eval_ref.py`, relevant worker-runtime tests, and
  `benchmarks/`. Add new harness modules only when existing locations cannot
  express the validation cleanly.
- Do not mix posterior-quality metrics into evidence calibration summaries.
- Store benchmark metadata: method setting, seed, model/problem, likelihood
  evaluations, wall-clock time, worker count where applicable, and software
  version or commit.
- Use current placeholder paper tables as output schema, not as proof that the
  implementation is correct.

## Acceptance Criteria

- The suite can produce the paper-style evidence calibration table.
- Plateau tests measure equality mass recovery and log-evidence calibration.
- Phantom diagnostics include `rho_g` information.
- Evidence and posterior quality are reported separately.
- Benchmarks are reproducible from recorded seeds and metadata.

## Review Status

Accepted for the early deterministic validation/benchmark harness phase. The
current harness covers analytic evidence, plateau equality, phantom count
effects, race-tree accounting, posterior weighting, multi-seed calibration
records, RMSE/Pareto producers, posterior quality/Wasserstein records,
performance guardrails, timing history, and method-conditional `rho_g`/`rho_fit`
diagnostics. Final benchmark production remains blocked on the feature tickets
that provide the completed v3 execution, allocation, sampler, runtime, and
diagnostics surfaces.

### Minimal Public Collector Test Draft Accepted

The final benchmark-production test draft was accepted for a minimal collector
slice. The accepted red test requires
`benchmarks.v3_validation.collector.collect_minimal_v3_validation_records(...)`
to run cheap real public v3 sampler/result/diagnostics paths and emit
schema-checked records for:

- baseline race-tree, phantom-conditioned, dynamic allocation, and Galilean
  method settings over deterministic seeds;
- evidence calibration and calibration rollups with `result_diagnostics`
  metadata from public results/diagnostics;
- phantom `rho_g`/`rho_fit` diagnostics for phantom-conditioned rows;
- plateau equality recovery derived from actual result block output;
- separate posterior quality, posterior Wasserstein, RMSE/Pareto,
  performance guardrail, timing-row, and append-only timing-history sections.

Focused red-test check before implementation:

- `conda run -n jaxns_py pytest
  tests/test_v3_validation_benchmark_collector.py` failed as intended because
  `benchmarks.v3_validation.collector` did not exist.

### Minimal Public Collector Slice Accepted

Independent review accepted the minimal public collector implementation after
remediation. The collector now:

- runs real public `NestedSampler.run_until_goal(...)` toy validations for the
  required four method settings;
- builds evidence, plateau equality, posterior quality, posterior
  Wasserstein, RMSE/Pareto, performance guardrail, timing-row, and timing
  history records through the existing schema/producer helpers;
- carries public result diagnostics into benchmark metadata;
- fails loudly for non-finite, too-short, non-1D, or degenerate MC shrinkage
  log-Z samples;
- requires finite/nonempty `rho_g` and `rho_fit` diagnostics for
  phantom-conditioned rows instead of using synthetic fallback curves;
- derives guardrail observations from measured collector-stage timings and
  labels thresholds as coarse smoke-test ceilings, not final benchmark claims.

Focused acceptance checks passed:

- `conda run -n jaxns_py pytest
  tests/test_v3_validation_benchmark_collector.py -q`;
- `conda run -n jaxns_py pytest
  tests/test_v3_validation_benchmark_collector.py
  tests/test_v3_validation_benchmark_producers.py
  tests/test_v3_validation_benchmark_schema.py
  tests/test_v3_validation_deterministic_fixtures.py -q` (`29 passed`);
- combined focused gate:
  `conda run -n jaxns_py pytest tests/test_runtime.py
  tests/test_v3_run_pattern.py tests/test_v3_execution_diagnostics.py
  tests/test_v3_validation_benchmark_collector.py
  tests/test_v3_validation_benchmark_producers.py
  tests/test_v3_validation_benchmark_schema.py
  tests/test_v3_validation_deterministic_fixtures.py -q` (`127 passed`);
- `conda run -n jaxns_py flake8
  benchmarks/v3_validation/collector.py
  tests/test_v3_validation_benchmark_collector.py`;
- `conda run -n jaxns_py python -m py_compile
  benchmarks/v3_validation/collector.py
  tests/test_v3_validation_benchmark_collector.py`.

This accepts the minimal production collector/harness skeleton. It does not
claim final expensive benchmark numbers; those remain a release/benchmarking
activity using this collector and any later runtime transport improvements.
