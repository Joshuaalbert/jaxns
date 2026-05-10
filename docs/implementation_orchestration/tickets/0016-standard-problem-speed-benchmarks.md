# Ticket 0016: Standard-Problem Speed Benchmarks

Branch: `feature/v3-standard-problem-speed-benchmarks`
Priority: after Ticket 0015 standard-problem performance
Depends on: Ticket 0010, Ticket 0015
Design docs:

- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-constrained-sampling.md`
- `docs/design/interface/run_pattern.py`
- `docs/implementation_orchestration/tickets/0015-standard-problem-performance.md`

## Goal

Add solid, repeatable speed reference points for the accepted v3 standard
problem path. These benchmarks must be cheap enough to run while optimizing,
but must still exercise the same full-dimensional local-load-balancer runtime
shape as the strict standard-problem gate.

The primary reference is the `basic_mvn` standard problem because earlier pure
JAX implementations completed that case in seconds even with about `50 * D`
live points. The benchmark must make live-point policy explicit so the current
runtime can be compared against both the accepted strict-test defaults and
historical `50 * D` reference settings.

## Required Behavior

- Run through the public local runtime pattern:
  `LoadBalancerClient(address="local")`, `lb.add_workers(...)`, and
  `lb.get_nested_sampler(...)`.
- Tear down the local load balancer between allocation-target runs.
- Cover the allocation targets `uniform`, `evidence_improving`, and
  `posterior_improving` by default.
- Provide a worker-scaling benchmark mode that varies local worker specs while
  holding the problem and sampler configuration fixed.
- Use a full-dimensional 8D `basic_mvn` problem definition, not a low-dimensional
  substitute.
- Keep benchmark configuration explicit: seed, worker specs, target live
  points, optional live-points-per-dimension label, max samples, shell size,
  number of slices, phantom burn-in, direction kernel, and MC sample count.
- Emit structured records with enough fields to compare:
  - elapsed run wall time;
  - aggregate constrained-sampler worker latency as a diagnostic, not a
    wall-clock timing component;
  - result conversion time;
  - MC shrinkage time;
  - setup/runner construction time;
  - actual worker count observed from load-balancer compute sectors;
  - likelihood evaluations, total samples, evidence estimate, and reference
    evidence.
- Make timing fractions machine-readable and derived from measured seconds.
- Do not change standard-problem statistical acceptance criteria or production
  sampler behavior.

## Out Of Scope

- Adding a hard CI wall-clock threshold for the full benchmark.
- Reintroducing the rejected Ticket 0015 compiled-fused runtime path.
- Moving all standard-problem test cases into benchmark modules.
- Running subprocess-isolated benchmarks.

## Test Plan

Tests must be written before implementation and reviewed against
`UNIT_TEST_STANDARDS.md` and `PERFORMANCE_TEST_STANDARDS.md`.

Required tests:

- Schema validation accepts complete speed records and rejects missing timing,
  non-finite, or negative timing fields.
- Relative timing fractions are computed from non-overlapping measured wall
  phases: setup, run, result conversion, and MC shrinkage.
- The collector covers all three allocation targets by default.
- Worker-scaling records vary the worker specification, tear down the local LB
  between scaling runs, and report the observed worker count.
- Each allocation-target benchmark run creates and exits its own
  `LoadBalancerClient(address="local")` context.
- The default problem specification is the full-dimensional 8D `basic_mvn`
  reference and exposes the live-point policy in metadata.
- The JSON/CLI-facing API returns deterministic, serializable records.
- MC shrinkage timing materializes/synchronizes the returned evidence samples
  before stopping the timer.

## Acceptance Criteria

- Test-first review accepts the tests as design-driven and not merely mirroring
  implementation internals.
- Implementation review finds no changes to production statistical semantics.
- Focused benchmark tests pass.
- `ruff check` passes for touched benchmark/test/doc files.
- The benchmark command can be invoked manually under
  `MPLBACKEND=Agg conda run -n jaxns_py python -m
  benchmarks.v3_performance.standard_problem_speed ...`.
- The ticket records at least one local benchmark measurement or an explicit
  reason why a measurement was deferred.

## Implementation Review Log

### Test-First Review

Initial test-first coverage was rejected as incomplete. Required fixes were:

- enforce live-point policy metadata, including explicit `50_per_dimension`
  support;
- assert that declared benchmark configuration is actually passed to the local
  runtime, not only echoed in metadata;
- cover the CLI-facing JSON output path;
- add schema negative tests for required metadata/result fields, not only
  timing fields.

The test suite was strengthened accordingly and remains red until the public
`benchmarks.v3_performance.standard_problem_speed` module is implemented:

```text
tests/test_v3_performance_benchmarks.py
15 failed because benchmarks.v3_performance does not exist yet
```

### Implementation Review Pass 1

The initial benchmark implementation passed focused tests, but strict review
rejected the timing decomposition:

- **High:** summed worker sampler latency was treated as a wall-clock timing
  component. With multiple workers, aggregate worker latency can exceed elapsed
  run wall time, so timing fractions must be derived from non-overlapping wall
  phases while worker sampler latency is reported separately as a diagnostic.
- **Medium:** fake-runtime tests did not prove the real problem builder is the
  full 8D `basic_mvn` reference.
- **Low:** schema validation accepted a bare string as `worker_specs`.

These findings require another implementation/review pass before acceptance.

### Implementation Review Pass 2

Strict-review findings were addressed:

- worker sampler latency is reported in a separate `diagnostics` field and no
  longer contributes to wall-clock timing totals or timing fractions;
- focused tests now call the real `basic_mvn` builder and check the 8D
  reference evidence against the canonical formula;
- schema validation rejects bare-string `worker_specs` and non-string or empty
  worker spec entries.

### Scope Extension: Worker Scaling

User requested a benchmark that measures scaling when adding more workers. The
benchmark remains high-level and runtime-facing: it varies worker specs over a
small explicit grid, records actual worker count from compute sectors, and
emits ordinary speed records so run wall time and aggregate worker sampler
latency can be compared across worker counts.

Strict re-review also found that `mc_shrinkage_seconds` must synchronize the
returned MC samples; otherwise JAX async dispatch could under-report completed
MC work. This must be fixed with focused coverage.

Implementation added:

- public `collect_worker_scaling_speed_records(...)`;
- CLI `--worker-scaling`, where each repeated `--worker-spec` is one scaling
  grid point;
- `diagnostics.actual_worker_count`, computed from `lb.compute_sectors` after
  workers are added;
- MC shrinkage sample materialization/synchronization before stopping
  `mc_shrinkage_seconds`.
- fast config validation for `phantom_burn_in <= num_slices - 1`.

Final strict review found no blockers.

Verification:

```text
MPLBACKEND=Agg conda run -n jaxns_py pytest tests/test_v3_performance_benchmarks.py
24 passed, 2 warnings in 1.63s

conda run -n jaxns_py ruff check \
  benchmarks/v3_performance/standard_problem_speed.py \
  tests/test_v3_performance_benchmarks.py \
  docs/implementation_orchestration/tickets/0016-standard-problem-speed-benchmarks.md
All checks passed.
```

Small real worker-scaling smoke, full 8D `basic_mvn`, local LB, direct CLI:

```text
MPLBACKEND=Agg conda run -n jaxns_py python -m \
  benchmarks.v3_performance.standard_problem_speed \
  --worker-scaling --allocation-target uniform \
  --worker-spec cpu:*:1 --worker-spec cpu:*:2 \
  --target-num-live-points 8 --max-samples 12 --shell-size 4 \
  --num-slices 5 --phantom-burn-in 1 --mc-sample-count 2
```

Observed smoke records:

```text
cpu:*:1 actual_worker_count=1 total_seconds=6.336957657709718
  setup=0.419297294691205 run=5.1465608309954405
  result=0.02235371246933937 mc=0.7487458195537329
  worker_sampler_latency=1.609335944056511

cpu:*:2 actual_worker_count=2 total_seconds=6.470547446981072
  setup=0.4195526894181967 run=5.2760567702353
  result=0.022344207391142845 mc=0.7525937799364328
  worker_sampler_latency=1.7002852372825146
```

The tiny smoke is a schema/runtime exercise, not a speedup claim; the sample
count is too small for worker scaling to dominate compile/setup overhead.
