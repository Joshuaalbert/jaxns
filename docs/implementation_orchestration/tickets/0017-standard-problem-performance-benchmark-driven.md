# Ticket 0017: Benchmark-Driven Standard-Problem Performance

Branch: `feature/v3-benchmark-driven-standard-performance`
Priority: after Ticket 0016 speed benchmarks
Depends on: Ticket 0015, Ticket 0016
Design docs:

- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-constrained-sampling.md`
- `docs/design/interface/run_pattern.py`
- `docs/implementation_orchestration/tickets/0015-standard-problem-performance.md`
- `docs/implementation_orchestration/tickets/0016-standard-problem-speed-benchmarks.md`

## Goal

Use the accepted speed benchmarks to reduce wall-clock time for the full 8D
`basic_mvn` standard-problem path while preserving correctness on representative
standard problems.

Correctness gates for this ticket are the standard-problem `basic_mvn`,
`spike_slab`, and `plateau` cases under the local-LB setup and the existing
`3 * sample_std` evidence acceptance criterion. The ticket must not weaken
statistical tolerances, dimensions, allocation targets, phantom conditioning,
GMM direction adaptation, or local-LB teardown semantics.

## Required Behavior

- Measure a baseline with the Ticket 0016 benchmark before implementation.
- Improve full 8D `basic_mvn` benchmark time, not only a toy low-dimensional
  or tiny-sample smoke.
- Keep benchmark timing semantics intact:
  - wall-clock fractions from non-overlapping phases;
  - aggregate worker sampler latency diagnostic only;
  - MC shrinkage synchronized before timing stops.
- Preserve correctness on `basic_mvn`, `spike_slab`, and `plateau`.
- Keep changes narrow and evidence-driven.

## Initial Focus Areas

- Run-loop and runtime orchestration overhead around the local-LB path.
- Constrained-sampler hot-path costs that affect full 8D `basic_mvn`.
- Direction-adaptation and GMM fitting cadence only if profiling shows they
  dominate.
- Worker scaling only if profiling shows the run loop can exploit additional
  local workers.

## Out Of Scope

- Reintroducing the rejected Ticket 0015 compiled-fused slice-sampler runtime
  path without first resolving strict seed validation and mutable problem
  identity.
- Weakening correctness criteria or reducing standard problem dimensionality.
- Replacing the future process-isolated runtime design in this ticket.

## Test Plan

Tests must be written before production implementation and reviewed against
`UNIT_TEST_STANDARDS.md` and `PERFORMANCE_TEST_STANDARDS.md`.

Required tests depend on the selected implementation, but must include:

- focused regression coverage for any changed runtime/sampler/result behavior;
- a correctness run over the representative standard-problem subset:
  `basic_mvn`, `spike_slab`, and `plateau`;
- the Ticket 0016 benchmark tests if benchmark code is touched;
- a benchmark before/after record showing full 8D `basic_mvn` time decreases.

## Acceptance Criteria

- Baseline benchmark and final benchmark are both recorded in this ticket.
- Implementation review finds no correctness or statistical-semantics
  regression.
- Focused tests pass.
- Representative correctness subset passes under `MPLBACKEND=Agg`.
- Full 8D `basic_mvn` benchmark time decreases relative to the recorded
  baseline.
- `ruff check` passes for touched files.

## Implementation Review Log

### Baseline Benchmark

Full 8D `basic_mvn`, local LB, uniform allocation, accepted standard-problem
benchmark settings:

```text
MPLBACKEND=Agg conda run -n jaxns_py python -m \
  benchmarks.v3_performance.standard_problem_speed \
  --allocation-target uniform \
  --target-num-live-points 30 --max-samples 1200 --shell-size 15 \
  --num-slices 24 --phantom-burn-in 4 --mc-sample-count 1000 \
  --worker-spec cpu:*:2
```

Observed baseline:

```text
total_seconds=150.8852138966322
setup_seconds=0.4161775317043066
run_seconds=148.67072185501456
result_conversion_seconds=0.10252952389419079
mc_shrinkage_seconds=1.6957849860191345
worker_sampler_latency_seconds=63.725022381171584
likelihood_evaluations=274320
```

The run loop dominates. Aggregate worker sampler latency is a large diagnostic,
but it is still less than half of elapsed run wall time, so the first target is
run-loop orchestration/parent-work overhead rather than result conversion or MC
shrinkage.

Reduced iteration benchmark for faster comparison:

```text
max_samples=240 mc_sample_count=200
total_seconds=34.652317302301526
run_seconds=33.23636349849403
worker_sampler_latency_seconds=13.740194922313094
mc_shrinkage_seconds=0.9700238239020109
```

### Correctness Gate

Use exact standard-problem nodeids to avoid accidentally broadening or
weakening the gate:

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

Collect-only confirmed these are exactly 9 tests.

### Test-First Pass 1

Focused behavioral tests were added around `_sample_parent_work` seed selection:

- equality plateaus must use strict `>` contour behavior;
- no-strict-seed cases fall back to root with `-inf` constraint;
- selected seed indices must be active and strict-valid;
- padded inactive rows must not be selected.

A source-inspection test for a private helper was rejected during review as too
implementation-driven and removed. The accepted focused tests passed:

```text
conda run -n jaxns_py pytest tests/test_v3_run_pattern.py -k sample_parent_work
4 passed, 39 deselected
```

### Implementation Pass 1

Implemented a bounded `_sample_parent_work` seed-selection optimization in
`src/jaxns/core.py`:

- build a host active log-likelihood view once per parent-work batch;
- use `np.searchsorted(..., side="right")` for strict `log_L > constraint`;
- exclude padded inactive rows through `state.num_samples`;
- preserve root fallback and parent-work diagnostics.

Focused verification:

```text
conda run -n jaxns_py pytest tests/test_v3_run_pattern.py -k sample_parent_work
4 passed, 39 deselected

conda run -n jaxns_py ruff check src/jaxns/core.py tests/test_v3_run_pattern.py
All checks passed.
```

Strict review found no finite-state correctness blocker. It noted that an
externally supplied state with no finite active seed remains outside the useful
sampling contract and should be handled deliberately if resumed invalid states
become a supported surface.

Reduced benchmark after implementation:

```text
max_samples=240 mc_sample_count=200
total_seconds=20.85467102751136
run_seconds=19.457116089761257
worker_sampler_latency_seconds=12.618375968188047
mc_shrinkage_seconds=0.9544597901403904
```

This reduced run improved total time from `34.65s` to `20.85s` and run time
from `33.24s` to `19.46s`.

Full 8D `basic_mvn` benchmark after implementation:

```text
total_seconds=88.98535172082484
setup_seconds=0.4121498893946409
run_seconds=86.81302922964096
result_conversion_seconds=0.09604676440358162
mc_shrinkage_seconds=1.6641258373856544
worker_sampler_latency_seconds=56.96391236037016
likelihood_evaluations=274320
```

This full benchmark improved total time from `150.89s` to `88.99s` and run
time from `148.67s` to `86.81s`.

Representative correctness subset after implementation:

```text
MPLBACKEND=Agg conda run -n jaxns_py pytest \
  tests/test_ns_standard_problems.py::test_nested_sampling_run_results[uniform-basic_mvn] \
  tests/test_ns_standard_problems.py::test_nested_sampling_run_results[evidence_improving-basic_mvn] \
  tests/test_ns_standard_problems.py::test_nested_sampling_run_results[posterior_improving-basic_mvn] \
  tests/test_ns_standard_problems.py::test_nested_sampling_run_results[uniform-spike_slab] \
  tests/test_ns_standard_problems.py::test_nested_sampling_run_results[evidence_improving-spike_slab] \
  tests/test_ns_standard_problems.py::test_nested_sampling_run_results[posterior_improving-spike_slab] \
  tests/test_ns_standard_problems.py::test_nested_sampling_run_results[uniform-plateau] \
  tests/test_ns_standard_problems.py::test_nested_sampling_run_results[evidence_improving-plateau] \
  tests/test_ns_standard_problems.py::test_nested_sampling_run_results[posterior_improving-plateau]

9 passed, 2 warnings in 746.84s (0:12:26)
```
