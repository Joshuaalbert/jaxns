# Develop audit performance and intent review

Compared the audit changes with `255b33d` on CPU (Intel Core i7-8750H),
Python 3.12.9, JAX 0.11.1, with x64 enabled and no extra XLA options.
The sampling workload has 10 dimensions, 100 concurrent chains, 100 slice
transitions, and all 99 intermediate states retained. Both versions use the
same keys, valid interior seeds, model, and strict contour.

All seven sampling output leaves match exactly. The lowered StableHLO is
byte-identical (183,323 bytes), with SHA-256
`1ce151ec633b58e01e9effbcaf3c479eeb96c5280d9de4dcb9495cab27ac1929`.
Compiler-reported memory is unchanged:

| Memory | Before | After |
| --- | ---: | ---: |
| Arguments | 6,600 B | 6,600 B |
| Outputs | 86,556 B | 86,556 B |
| Temporaries | 904,208 B | 904,208 B |
| Aliases | 0 B | 0 B |

The end-to-end Gaussian run uses 300 roots, width 100, evidence-improving
allocation with delta_K=300, the default depth condition, initial capacity
2,048, and maximum 65,536 samples. All 20 state leaves and all 42 result/evidence
snapshot leaves match exactly, including 128 classic and 128 phantom evidence
draws. Both runs produce 3,678 classics from 645,662 likelihood evaluations,
with expected log evidence -10.157836190208506 and uncertainty
0.12097271981586784. The analytic reference is -10.077421137436366.

`before.json` and `after.json` separate lowering, compilation, and eleven
synchronised warm executions. Other tests were running during these measurements,
so the timing difference is not a speedup claim. The identical compiled program,
memory plan, and scientific outputs provide the controlled comparison here.

Enabling phantom collection now intentionally stores every intermediate state.
For the default 5D-transition sampler this changes storage from D to 5D-1
likelihoods per chain. That requested increase is separate from the matched
comparison above. Direct sampler calls still accept a validated retained count,
and final inference can use a shorter prefix.

The intent review checked configuration ownership, static validation, JAX
specialisation, and all changed consumers. Count and merge validation run at
host boundaries. The shared local orchestration retains one depth/interrupt
lifecycle, and shared schedule-storage growth belongs to `algorithm/`.
No likelihood, random-stream, allocation, or compiled depth kernel was changed.
Signed cumulative sums carry `LogSpace` through the existing cumulative operator,
while unsigned sums keep their original implementation. JSON restores NumPy array
leaves and leaves subsequent device placement to the caller.

To reproduce, use the same script with each checkout on PYTHONPATH:

```bash
PYTHONPATH=/path/to/parent/src conda run --no-capture-output -n jaxns_py \
  python benchmarks/issue_307/review.py --baseline --output /tmp/review-before
PYTHONPATH=src conda run --no-capture-output -n jaxns_py \
  python benchmarks/issue_307/review.py --output /tmp/review-after
```

Compare every array in the generated `sampling.npz`, `state.npz`, and
`inference.npz` files with `numpy.testing.assert_array_equal`. Raw snapshots and
compiler dumps are generated artifacts, not repository inputs.

## Fixed-prefix accuracy gate and full-retention calibration

Changing default retention from D to 5D-1 also changes the data used by the
default phantom evidence call. The standard-problem regression therefore
explicitly selects its original D-observation prefix. Its reference evidence,
seed, and 2-sigma tolerance are unchanged. Full retention remains exercised by
the end-to-end matched comparison above and the system test.

The Jones case (seed 1001, four dimensions, 20 slice transitions) demonstrates
why retention and uncertainty calibration must be distinguished. Running the
parent with 19 phantoms explicitly configured reproduces all 26 candidate
state leaves exactly. For 1,000 evidence draws with key 20260823, both versions
give the following identical values:

| Inference | Mean log evidence | Reported MC standard deviation |
| --- | ---: | ---: |
| Classic | 35.142736005275445 | 0.228260531809239 |
| Original four-phantom prefix | 35.080828460746720 | 0.149780387568636 |
| All 19 phantoms | 35.111009103357560 | 0.119889924976638 |

The reference is 34.803948945405. Full conditioning misses its own 2-sigma
interval on both versions. This is an existing calibration limitation exposed
by the requested default change, not a change to the inference or samples.
The classic expectation, sample count (1,390), and likelihood count (154,494)
also agree exactly.

To reproduce, build `STANDARD_PROBLEM_CASES_BY_NAME["jones_scalar"]` from
`cicd.tests.test_ns_standard_problems`, run with its `run_seed`, and request
1,000 evidence draws at prefixes 0 (classic), 4, and 19. On the parent, supply
`UniDimSliceSampler(num_slices=20, collect_phantom_samples=True,
max_phantom_samples=19)` to the runner. On the candidate, the default sampler
retains all 19 when `collect_phantom_samples=True`.

## Posterior integration and gradients

The padding regression also covers derivatives with respect to integrand
parameters. Before the final boundary fix, a function containing `log(theta*x)`
or `1/(theta*x)` produced a NaN gradient at padded `x=0`, despite the zero
posterior weight. Padded and trimmed values and gradients now agree for both
integration modes, with and without batching.

A matched moments calculation uses 16,384 ten-dimensional normal samples
(key 307), equal normalised weights on the first 12,288 rows, zero weight on
the rest, batch size 128, and integrand `x**2 + 1`. The ten returned values
match the parent exactly. Compiler-reported temporary memory is 20,896 B before
and 21,032 B after (+136 B), with unchanged arguments (1,441,792 B) and output
(80 B). The fix selects one supported observation for evaluating ignored rows
and retains their zero weight, without copying the posterior or adding state.

In 51 alternating, synchronised warm calls in the same process, medians were
8.06 ms before and 8.26 ms after; ranges were 5.70–15.62 ms and 5.87–15.05 ms.
Other tests were running, so these timings do not establish a speed difference.
The additional work is confined to posterior integration; the sampling program
and the exact end-to-end comparisons above are unchanged.
