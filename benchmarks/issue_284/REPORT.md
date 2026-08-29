# Issue 284 MC prefix benchmark

## Question

Does selecting a shorter phantom prefix through the public evidence API keep
the unused retained suffix out of the compiled MC hot path?

## Method

`run_mc_prefix.py` completed an 8D Gaussian nested-sampling run with 512
classic samples, 512 likelihood blocks, and eight retained phantom slots per
cluster. It then evaluated 256 evidence draws in batches of 64 with prefixes
of one, four, and eight. Each steady-state result is the median of seven warm
runs. XLA's CPU memory analysis supplies the compiler-plan peak; it is the sum
of argument, output, and temporary bytes less aliases. The benchmark also
asserts draw-for-draw equality between public prefix selection and a physically
sliced `log_L_phantom` result.

Environment: CPU (`cpu:0`), JAX/JAXLIB 0.10.0, Python 3.12.9, x64 enabled,
Linux 6.8.0-87-generic. Run on 2026-08-29 with:

```bash
PYTHONPATH=/tmp/jaxns-issue-284/src conda run -n jaxns_py python \
  benchmarks/issue_284/run_mc_prefix.py \
  --max-samples 512 --draws 256 --batch-size 64 --repeats 7 \
  --output /tmp/issue284-final.json
```

## Results

| Used prefix | Lower (s) | Compile (s) | Steady median (s) | Range (s) | Compiler peak |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.680 | 2.646 | 0.1333 | 0.1257–0.1501 | 10,406,152 B |
| 4 | 0.373 | 2.323 | 0.1335 | 0.1256–0.1492 | 10,609,736 B |
| 8 | 0.462 | 2.525 | 0.1413 | 0.1329–0.1589 | 13,591,176 B |

The one-phantom executable used 23.4% less compiler-planned peak memory than
the eight-phantom executable and had a 5.7% lower median runtime in this CPU
run. The runtime ranges overlap, so the robust conclusion is the physical-plan
result: the selected prefix changes the compiled phantom axis and the unused
suffix is absent from its arguments and temporary plan. Prefix-one phantom
arguments were 29,720 bytes versus 58,392 bytes for prefix eight. Exact output
equality with the explicitly sliced reference confirms that this plan change
does not alter the selected-prefix calculation.
