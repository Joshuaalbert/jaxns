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
