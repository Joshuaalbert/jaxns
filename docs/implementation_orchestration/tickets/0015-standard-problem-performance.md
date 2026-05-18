# Ticket 0015: Standard-Problem Performance

Branch: `feature/v3-standard-problem-performance`
Priority: after Ticket 0014 standard-problem acceptance
Depends on: Ticket 0005, Ticket 0006, Ticket 0008, Ticket 0009, Ticket 0011,
Ticket 0013, Ticket 0014
Design docs:

- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-constrained-sampling.md`
- `docs/design/jaxns-v3-process-isolated-zmq-runtime.md`
- `docs/design/interface/run_pattern.py`

## Goal

Reduce wall-clock time for `tests/test_ns_standard_problems.py` without
weakening its statistical acceptance contract.

The performance target has three separate cost centers:

- inner-loop constrained sampling time;
- outer-loop allocation, direction adaptation, state conversion, and dispatch
  orchestration time;
- Monte Carlo shrinkage/result-sampling time after the run.

The work must preserve the full standard-problem gate: full-dimensional
problems, all allocation targets (`uniform`, `evidence_improving`,
`posterior_improving`), direct pytest execution with no subprocess isolation,
`MPLBACKEND=Agg`, `3 * sample_std`, and per-case
`LoadBalancerClient(address="local")` teardown.

## Current Observations

The accepted Ticket 0014 gate passed but was slow:

```text
tests/test_ns_standard_problems.py
31 passed, 2 warnings in 9167.84s (2:32:47)
```

Most full standard-problem cases reported about 320 seconds from runner
construction through `state.to_result().trim()`. The local runtime path also
currently wraps `UniDimSliceSampler` calls with `force_python_loop=True`.
Initial direct micro-profiling showed that simply using the current eager
`lax.scan` branch is not automatically faster (`50` basic-model samples:
Python-loop path about `0.73s`, current scan branch about `9.15s`), so fused
execution is a valid target only if it is introduced as a genuinely compiled
path with tests and measurements.

A reduced 90-sample local-LB cProfile run also showed about `8.8s` in
`allocation.select_parent_work()`, mostly through `jax.random.categorical` on
variable-length candidate arrays. That is outer-loop overhead from repeated
JAX compilation in Python orchestration code and is a high-priority target for
a deterministic Python/NumPy weighted-CDF replacement.

## Required Behavior

Performance improvements must not change the statistical model, shrinkage law,
phantom-conditioning target, direction-kernel fitting target, allocation target
semantics, or acceptance criteria.

Inner-loop work:

- keep the current fast path until profiling proves a better one, and allow the
  runtime worker path to use compiled/fused slice-step execution only when the
  sampler and adaptation context support a measured faster path;
- keep `UniDimSliceSampler` direction adaptation snapshots fresh per dispatch;
- keep exact phantom likelihood retention semantics and phantom-coordinate
  discard policy;
- preserve strict seed and contour validation.

Outer-loop work:

- avoid unnecessary repeated result conversion, block-state construction, or
  posterior-weight construction when a cheaper state-derived helper can provide
  the required data;
- avoid JAX compilation churn in Python orchestration code, especially parent
  selection over variable-length candidate arrays;
- keep direction-kernel cadence measured since the last successful update;
- keep allocation target diagnostics accurate;
- preserve parent-selection and acceptance-ledger semantics.

MC shrinkage/result work:

- keep gamma-weighted phantom conditioning exactly as designed;
- preserve Kish gate diagnostics and returned evidence-sample fields;
- use structured APIs or JAX transforms for speedups rather than ad hoc array
  manipulation;
- any batching or caching must not reuse stale state.

Runtime diagnostics:

- expose enough worker latency statistics to separate dispatch overhead,
  payload/cache overhead, and sampler execution time;
- diagnostics must be per dispatch and aggregateable without hidden global
  mutable state;
- latency fields must be optional for existing tests but validated when
  present.

## Out Of Scope

- Changing `tests/test_ns_standard_problems.py` acceptance tolerance,
  dimensions, allocation targets, or local-LB teardown pattern.
- Implementing the full future process-isolated ZMQ topology from
  `docs/design/jaxns-v3-process-isolated-zmq-runtime.md`.
- Parallel JAX worker execution in Python threads unless a design review proves
  it does not reintroduce the threaded-JAX hazard already captured in
  `LEARNINGS.md`.
- Removing diagnostic fields needed by accepted tickets.

## Test Plan

Tests must be written before implementation and reviewed against
`UNIT_TEST_STANDARDS.md` and `PERFORMANCE_TEST_STANDARDS.md`.

Required tests:

- A runtime/worker test proving per-dispatch direction-adaptation context
  freshness under payload caching, regardless of which loop mode is selected.
- A sampler-loop equivalence test proving any newly introduced compiled/fused
  slice execution produces the same sample, likelihood, likelihood-evaluation
  count, and phantom likelihood diagnostics as the current Python-loop path for
  a fixed key, seed, constraint, and frozen direction context.
- A focused performance/observability guard proving worker dispatch records
  expose which loop mode was used. Do not hard-code the current eager scan
  branch as faster without measurement.
- A direction-fitting/posterior-weight helper test proving any cheaper
  state-derived path matches `state.to_result().trim().v3_log_posterior_weights`
  on a small fixture, including plateau/equality mass behavior if that helper
  is changed.
- Parent-selection tests proving weighted candidate selection remains
  proportional for positive finite weights, preserves single-candidate exact
  behavior, and preserves zero/invalid-weight fallback behavior without relying
  on `jax.random.categorical` over variable-length arrays.
- A runtime diagnostics test proving accepted dispatch records expose finite
  non-negative latency fields and that worker-runtime diagnostics preserve
  those records.
- If MC shrinkage is optimized, a focused equivalence test against the accepted
  JAX/reference shrinkage fixtures and a hot-path bound test proving the
  optimized path avoids repeated block-state reconstruction when a `BlockState`
  is supplied.

Required profiling/measurement:

- Record before/after timings for at least one representative standard-problem
  case or reduced deterministic benchmark that uses the same local-LB runtime
  path.
- Separate measured or diagnostic timing into inner sampler execution,
  orchestration/dispatch, and MC shrinkage/result-sampling where possible.
- Do not claim full standard-problem speedup until the full gate passes.

## Acceptance Criteria

- Test-first review accepts the tests as design-driven and sufficient.
- Implementation review finds no statistical-semantics regressions.
- Focused tests for sampler loop equivalence, runtime adaptation freshness,
  diagnostics, and any changed posterior-weight/MC shrinkage helper pass.
- `tests/test_ns_standard_problems.py` still passes under the strict accepted
  setup.
- `ruff check` passes for touched files.
- Performance measurements show a defensible wall-clock improvement or expose
  a documented blocker with latency diagnostics strong enough to guide the next
  ticket.

## Implementation Review Log

### Pass 1

Focused implementation added:

- NumPy weighted-CDF parent selection for `_choose_index_from_weights()`, avoiding
  variable-shape `jax.random.categorical` compilation in Python orchestration.
- Runtime accepted dispatch observability for `sampler_loop_mode`,
  `dispatch_latency_seconds`, `payload_cache_latency_seconds`, and
  `sampler_execution_latency_seconds`.
- Worker payload cache reuse while keeping per-dispatch adaptation context fresh.

Strict review findings and actions:

- **Resolved:** target-block selection still used `unit_peak_utility` through a
  hash perturbation. Removed the utility hash so allocation utility influences
  target choice only through `target_K`, and added regression coverage that
  identical targets/current counts produce identical target choices under
  different utility vectors.
- **Resolved:** stale-parent and terminal late-completion lifecycle records could
  drop worker timing metadata. Worker execution stats are now applied before
  stale/terminal lifecycle status decisions, with focused stale-parent and
  failed-late-completion coverage.
- **Resolved:** weighted-CDF selection could choose a zero-weight prefix entry
  for an exact zero quantile. Selection now uses `searchsorted(..., side="right")`
  and has explicit regression coverage.

Performance-review follow-ups retained for profiling:

- Coordinator-side `serialize_sampler(self.sampler)` still happens on every
  dispatch before worker latency timers start.
- Low-overhead sampler timing does not force JAX outputs ready, so blocking
  profiling should separately measure true device completion time.

Reduced local-LB profiling after the first patch:

```text
basic_mvn/uniform max_samples=90 mc_samples=1000
run_seconds=19.927193 result_seconds=0.804226 mc_seconds=4.166290
dispatch_latency_sum=9.571255 payload_latency_sum=0.000383
sampler_latency_sum=9.513398 accepted=90
```

`select_parent_work()` was no longer in the top 25 cumulative `cProfile`
entries; remaining dominant costs were sampler execution and JAX
compile/cache misses inside the slice sampler.

### Pass 2

Focused implementation tried a runtime-only compiled fused path for eligible
`UniDimSliceSampler` worker dispatches:

- `LocalLoadBalancerState` caches a worker payload containing a compiled fused
  sampler callable alongside the deserialized sampler/problem.
- Runtime `_sample_constrained()` opts into the compiled fused path for local-LB
  dispatches; direct `_execute_serialized_worker_task()` calls keep the explicit
  Python-loop path for compatibility and adaptation-context observability.
- Accepted dispatch records report `sampler_loop_mode="compiled_fused"` only
  when the compiled path ran.

Reduced local-LB profiling after the compiled fused path:

```text
basic_mvn/uniform max_samples=90 mc_samples=1000
run_seconds=12.062797 result_seconds=0.783942 mc_seconds=4.125693
dispatch_latency_sum=1.623042 payload_latency_sum=0.000574
sampler_latency_sum=1.566906 accepted=90

basic_mvn/evidence_improving max_samples=90 mc_samples=1000
run_seconds=12.368638 result_seconds=0.812921 mc_seconds=4.141941
dispatch_latency_sum=1.636891 payload_latency_sum=0.000544
sampler_latency_sum=1.580489 accepted=90
```

The inner sampler latency dropped by about `8s` on the reduced 90-sample
`basic_mvn` probe. MC shrinkage remains about `4.1s` for the first call and is
the next result-sampling target.

Strict review rejected this pass, and it was removed before continuing:

- **Blocked:** the jitted wrapper bypassed strict seed validation because
  `UniDimSliceSampler.get_sample()` treats tracer conversion failures as
  inconclusive during tracing.
- **Blocked:** `_StaticLogLikelihoodFn` uses object identity for static JAX
  cache keys, so mutable in-place args/params/model changes can reuse a stale
  compiled likelihood.
- **Blocked:** eligibility did not exclude non-JIT-safe sampler modes such as
  Galilean trajectories.

Compiled-fused UniDim runtime dispatch is therefore a future design task, not
an accepted Ticket 0015 optimization.

### Pass 3

Focused implementation added a block-state MC shrinkage helper for public
`NestedSamplerResults.sample_mc_shrinkage()`:

- The public method still runs Python validation first, including stale
  `block_state` alignment and strict-contour checks.
- When result block arrays are available, the method routes through
  `_sample_mc_shrinkage_with_block_state()` and a jitted array-level helper
  rather than passing the full results object and diagnostics through JAX.
- Equivalence coverage compares the public result call against the explicit
  `jaxns.phantom_eval.sample_mc_shrinkage(..., block_state=...)` call with the
  same key.

Reduced local-LB profiling after the block-state MC helper:

```text
basic_mvn/uniform max_samples=90 mc_samples=1000
run_seconds=19.348621 result_seconds=0.790193 mc_seconds=0.933625
dispatch_latency_sum=9.386357 payload_latency_sum=0.000345
sampler_latency_sum=9.330360 accepted=90
```

The accepted runtime remains on the Python-loop `UniDimSliceSampler` path after
the compiled-fused rollback, but first-call MC shrinkage dropped from about
`4.17s` to about `0.93s` on the reduced probe.

### Pass 4

Focused implementation bounded coordinator-side sampler serialization:

- `RuntimeNestedSampler.__post_init__()` now serializes the sampler once while
  constructing `runtime_compile_identity`.
- `_sample_constrained()` reuses those bytes for every local-LB dispatch under
  that runner.
- `_build_runtime_compile_identity()` still supports direct helper callers that
  do not provide precomputed sampler bytes.

Review found no production blockers. The remaining test whitelist for rejected
`compiled_fused` runtime records was removed, so accepted runtime diagnostics
now reject that mode unless a future design reintroduces it deliberately.

Reduced local-LB profiling after sampler-byte reuse:

```text
basic_mvn/uniform max_samples=90 mc_samples=1000
run_seconds=19.541291 result_seconds=0.814251 mc_seconds=0.946640
dispatch_latency_sum=9.369168 payload_latency_sum=0.000387
sampler_latency_sum=9.311442 accepted=90
```

This pass removes avoidable coordinator work and adds regression coverage, but
the reduced probe remains dominated by the accepted Python-loop slice sampler
and JAX compile/cache overhead inside it.

## Verification

Final focused and broad checks:

```text
tests/test_runtime.py tests/test_v3_run_pattern.py tests/test_constrained_sampler.py
tests/test_v3_phantom_results.py tests/test_v3_shrinkage.py tests/test_phantom_eval_jax.py
143 passed, 2 warnings in 117.52s
```

Strict standard-problem gate:

```text
MPLBACKEND=Agg conda run -n jaxns_py pytest tests/test_ns_standard_problems.py
31 passed, 2 warnings in 4754.84s (1:19:14)
```

This preserved the full accepted setup: direct pytest execution, full standard
problems, all allocation targets, local `LoadBalancerClient(address="local")`
worker setup and teardown, no subprocess isolation, and the `3 * sample_std`
acceptance criterion.
