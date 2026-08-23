# Issue 247 final accuracy and performance report

## Scope and provenance

Measurements were taken on an Intel Core i7-8750H CPU (6 physical cores,
12 hardware threads) with the JAX CPU backend, float64 enabled, Python 3.12.9,
JAX/JAXLIB 0.10.0, and Linux 6.8. The comparison uses all ten standard
problems, phantoms off and on, and seeds 11, 23, and 37: 60 records per
implementation. Each final evidence calculation uses 1,000 Monte Carlo draws.

`n` counts independent nested-sampling runs, not MC draws. Thus each
aggregate conditioning row has `n=30` (10 problems times 3 sampler seeds),
while each per-problem version/mode entry below has `n=3` (seeds 11, 23, and
37). Every one of those runs uses 1,000 MC evidence draws.

The baseline is `jaxns==2.6.9`, the latest PyPI release on 2026-08-23.
`origin/main` at `2f356d6d497ce3ac471fb9a06f9d22587487aaaa`
and the installed wheel have byte-identical `jaxns` package trees (excluding
`__pycache__`), so the one baseline matrix represents both sources. Every raw
record identifies the imported version and absolute module path.

Common sampling settings are 30 root chains and five isotropic perfect slice
transitions per dimension, float64, and `dlogZ=log1p(1e-3)`. V3 uses the
paper scheduler's full vmap replacement width of ten chains per dimension.
Phantom mode retains `D` generated states from the start of each chain; it does
not change the v3 race tree.

The 30D chains and 5D slices are the actual released v2 defaults and are set
explicitly in both runners. V2 uses `c=30D` directly because passing
`num_live_points` would divide the independent chain count by `k+1` when
phantoms are enabled. The active statistical termination threshold is
identical. V2's storage cap is its released `100*c*(k+1)` because it stores
classic and phantom outputs together; v3 uses `100*c` classic rows and stores
phantoms separately. No matrix run reached either cap. The implementations
necessarily estimate the common dlogZ condition differently: released v2 uses
its existing register, while v3 uses the paper-required classic block
expectation estimator.

## Required release gate

`tests/test_ns_standard_problems.py` passes all 20 problem/mode combinations
without weakening the v2 tolerances:

- deterministic expectation estimate within 3 reported standard deviations;
- mean of 1,000 MC evidence draws within 2 empirical standard deviations;
- finite results and non-empty phantom output when enabled.

Result: **20 passed in 319.87 s**. The focused v3 implementation suite also
passes **118 tests in 110.40 s**.

## Three-seed accuracy matrix

| implementation | conditioning | n | expectation bias | expectation RMSE | 3σ failures | MC bias | MC RMSE | 2σ coverage (95% Wilson CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v2.6.9/main | classic | 30 | +0.0595 | 0.1828 | 0 | +0.0655 | 0.1821 | 30/30 = 100% (88.6%, 100%) |
| v2.6.9/main | phantom-recording run | 30 | +0.0410 | 0.1524 | 0 | +0.0465 | 0.1523 | 27/30 = 90.0% (74.4%, 96.5%) |
| v3 | classic | 30 | +0.0344 | 0.1967 | 0 | +0.0358 | 0.1960 | 28/30 = 93.3% (78.7%, 98.2%) |
| v3 | phantom-conditioned | 30 | +0.0344 | 0.1967 | 0 | +0.0334 | 0.1956 | 26/30 = 86.7% (70.3%, 94.7%) |

V2's phantom-labelled row records phantoms but its evidence sampler remains the
released classic shrinkage calculation. V3's phantom row uses the new
Kish-gated, root-lineage-grouped phantom conditioning. Recording phantoms does
not alter the v3 classic samples, hence its identical expectation columns.

Across all 60 records, v3 reduces absolute bias (expectation +0.0344 versus
+0.0503; MC +0.0346 versus +0.0560), while its RMSE is higher (expectation
0.1967 versus 0.1683; MC 0.1958 versus 0.1679). All deterministic estimates
remain calibrated at 3σ. The descriptive MC matrix has 6/60 v3 2σ misses
versus 3/60 in v2; the mode-specific Wilson intervals overlap. These misses
are retained in the raw data and are not hidden by seed selection. The fixed
release matrix above passes every MC gate.

### Per-standard-problem accuracy (v2 → v3)

Each arrow is the released v2.6.9/main value followed by v3. In phantom-on
rows, v2 records phantoms but still reports classic MC evidence; v3 reports
phantom-conditioned MC evidence.

| problem | mode | n each | expectation bias v2 → v3 | expectation RMSE v2 → v3 | 3σ failures v2 → v3 | MC bias v2 → v3 | MC RMSE v2 → v3 | 2σ coverage v2 → v3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| basic | off: classic | 3 | -0.0104 → -0.0001 | 0.0549 → 0.0409 | 0 → 0 | +0.0062 → +0.0003 | 0.0544 → 0.0408 | 100.0% → 100.0% |
| basic | on: recorded → conditioned | 3 | -0.0115 → -0.0001 | 0.0525 → 0.0409 | 0 → 0 | +0.0043 → +0.0003 | 0.0518 → 0.0417 | 100.0% → 100.0% |
| basic2 | off: classic | 3 | -0.0054 → +0.0062 | 0.0886 → 0.0676 | 0 → 0 | +0.0115 → +0.0063 | 0.0894 → 0.0672 | 100.0% → 100.0% |
| basic2 | on: recorded → conditioned | 3 | -0.0062 → +0.0062 | 0.0850 → 0.0676 | 0 → 0 | +0.0096 → +0.0052 | 0.0859 → 0.0702 | 100.0% → 100.0% |
| basic3 | off: classic | 3 | -0.0114 → +0.0086 | 0.1967 → 0.1150 | 0 → 0 | -0.0024 → +0.0130 | 0.2022 → 0.1117 | 100.0% → 100.0% |
| basic3 | on: recorded → conditioned | 3 | -0.0438 → +0.0086 | 0.1846 → 0.1150 | 0 → 0 | -0.0420 → +0.0028 | 0.1872 → 0.1731 | 100.0% → 100.0% |
| plateau | off: classic | 3 | -0.0503 → -0.0322 | 0.0503 → 0.0322 | 0 → 0 | -0.0332 → -0.0324 | 0.0332 → 0.0324 | 100.0% → 100.0% |
| plateau | on: recorded → conditioned | 3 | -0.0503 → -0.0322 | 0.0503 → 0.0322 | 0 → 0 | -0.0332 → -0.0328 | 0.0332 → 0.0328 | 100.0% → 100.0% |
| basic_mvn | off: classic | 3 | +0.0402 → +0.0559 | 0.1717 → 0.2067 | 0 → 0 | +0.0424 → +0.0562 | 0.1727 → 0.2070 | 100.0% → 100.0% |
| basic_mvn | on: recorded → conditioned | 3 | -0.0407 → +0.0559 | 0.0492 → 0.2067 | 0 → 0 | -0.0385 → +0.1516 | 0.0460 → 0.1984 | 100.0% → 66.7% |
| spike_slab | off: classic | 3 | +0.0080 → +0.2178 | 0.2137 → 0.3720 | 0 → 0 | +0.0074 → +0.2174 | 0.2165 → 0.3703 | 100.0% → 66.7% |
| spike_slab | on: recorded → conditioned | 3 | +0.1402 → +0.2178 | 0.3115 → 0.3720 | 0 → 0 | +0.1416 → +0.1622 | 0.3129 → 0.3484 | 66.7% → 66.7% |
| spike_slab10 | off: classic | 3 | +0.1293 → +0.0803 | 0.2313 → 0.3807 | 0 → 0 | +0.1325 → +0.0833 | 0.2257 → 0.3797 | 100.0% → 66.7% |
| spike_slab10 | on: recorded → conditioned | 3 | +0.0820 → +0.0803 | 0.1252 → 0.3807 | 0 → 0 | +0.0820 → +0.0527 | 0.1250 → 0.3615 | 66.7% → 33.3% |
| weak_curved_mvn8 | off: classic | 3 | +0.0737 → +0.0109 | 0.1627 → 0.0578 | 0 → 0 | +0.0722 → +0.0126 | 0.1596 → 0.0579 | 100.0% → 100.0% |
| weak_curved_mvn8 | on: recorded → conditioned | 3 | +0.1267 → +0.0109 | 0.1279 → 0.0578 | 0 → 0 | +0.1267 → -0.0358 | 0.1283 → 0.0796 | 100.0% → 100.0% |
| weak_curved_spike_slab8 | off: classic | 3 | +0.1784 → +0.0394 | 0.1911 → 0.1712 | 0 → 0 | +0.1763 → +0.0421 | 0.1885 → 0.1730 | 100.0% → 100.0% |
| weak_curved_spike_slab8 | on: recorded → conditioned | 3 | +0.0454 → +0.0394 | 0.0576 → 0.1712 | 0 → 0 | +0.0463 → +0.0190 | 0.0578 → 0.1331 | 100.0% → 100.0% |
| weak_curved_spike_slab10 | off: classic | 3 | +0.2431 → -0.0430 | 0.3009 → 0.0872 | 0 → 0 | +0.2418 → -0.0406 | 0.2999 → 0.0842 | 100.0% → 100.0% |
| weak_curved_spike_slab10 | on: recorded → conditioned | 3 | +0.1684 → -0.0430 | 0.2256 → 0.0872 | 0 → 0 | +0.1683 → +0.0085 | 0.2245 → 0.1713 | 66.7% → 100.0% |

## Performance matrix

| implementation | conditioning | median core run | warmed median core run | median end-to-end | median likelihood evals | total likelihood evals | median ESS/eval | max process RSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v2.6.9/main | classic | 1.407 s | 1.407 s | 4.025 s | 398,594 | 16,060,538 | 1.331e-3 | 1,132 MiB |
| v2.6.9/main | phantom-recording run | 1.445 s | 1.445 s | 5.388 s | 398,594 | 16,060,538 | 1.465e-3 | 2,908 MiB |
| v3 | classic | 4.910 s | 3.326 s | 7.434 s | 162,191 | 9,689,231 | 2.966e-3 | 1,324 MiB |
| v3 | phantom-conditioned | 5.274 s | 3.338 s | 10.754 s | 162,191 | 9,689,231 | 2.966e-3 | 3,659 MiB |

The v3 core uses **39.7%** of the v2 likelihood evaluations (60.3% fewer) and
has **2.13x** the median ESS per likelihood evaluation. On these deliberately
cheap CPU likelihoods, that scientific-efficiency gain does not offset the
compiled scheduler and vmap slowest-lane overhead: the paired warmed core-time
ratio has median 2.43x (IQR 2.03x–2.84x). This is a measured residual wall-time
regression, not presented as a speedup. More expensive user likelihoods should
shift the trade-off toward the evaluation reduction, but that is an inference,
not measured here.

Phantom evidence is also more expensive because v3 evaluates the paper's
cluster-conditioned gamma model, whereas v2's evidence calculation is classic.
The raw records separate core runtime, result construction, and MC time.

### Per-standard-problem performance (v2 → v3)

The first run for each problem/mode includes call-site compilation; the warmed
column uses seeds 23 and 37. End-to-end includes core execution, result
construction, and the 1,000-draw evidence calculation.

| problem | mode | n each | core s median [IQR] v2 → v3 | warmed core s v2 → v3 | end-to-end s median [IQR] v2 → v3 | evals median v2 → v3 | ESS/eval v2 → v3 | max RSS MiB v2 → v3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| basic | off | 3 | 0.013 [0.013, 0.016] → 0.026 [0.025, 2.172] | 0.016 → 0.025 | 0.426 [0.421, 2.215] → 2.196 [1.147, 4.840] | 6,283 → 1,391 | 4.765e-03 → 2.955e-02 | 613 → 627 |
| basic | on | 3 | 0.012 [0.010, 0.013] → 0.027 [0.025, 2.204] | 0.013 → 0.025 | 0.447 [0.438, 1.820] → 3.134 [1.666, 5.938] | 6,283 → 1,391 | 3.085e-03 → 2.955e-02 | 619 → 714 |
| basic2 | off | 3 | 0.011 [0.010, 0.015] → 0.027 [0.024, 2.142] | 0.010 → 0.024 | 0.391 [0.387, 2.008] → 2.318 [2.146, 4.851] | 6,283 → 1,418 | 4.771e-03 → 2.931e-02 | 604 → 679 |
| basic2 | on | 3 | 0.014 [0.013, 0.015] → 0.029 [0.026, 2.395] | 0.013 → 0.026 | 0.434 [0.422, 1.967] → 3.382 [3.336, 6.250] | 6,283 → 1,418 | 3.208e-03 → 2.931e-02 | 610 → 786 |
| basic3 | off | 3 | 0.070 [0.062, 0.078] → 0.128 [0.127, 2.815] | 0.070 → 0.127 | 3.850 [2.215, 4.196] → 2.157 [1.195, 5.512] | 44,053 → 10,527 | 2.243e-03 → 1.272e-02 | 809 → 689 |
| basic3 | on | 3 | 0.070 [0.060, 0.072] → 0.142 [0.136, 2.964] | 0.062 → 0.136 | 3.785 [2.235, 4.215] → 3.616 [2.114, 7.214] | 44,053 → 10,527 | 2.431e-03 → 1.272e-02 | 896 → 811 |
| plateau | off | 3 | 0.000 [0.000, 0.000] → 0.003 [0.002, 2.069] | 0.000 → 0.002 | 0.351 [0.350, 1.805] → 0.162 [0.162, 3.689] | 30 → 30 | 5.119e-01 → 3.330e-02 | 593 → 611 |
| plateau | on | 3 | 0.000 [0.000, 0.000] → 0.002 [0.002, 2.071] | 0.000 → 0.002 | 0.373 [0.369, 1.835] → 0.214 [0.206, 4.207] | 30 → 30 | 5.119e-01 → 3.330e-02 | 590 → 610 |
| basic_mvn | off | 3 | 4.008 [3.884, 4.059] → 8.161 [7.911, 11.310] | 3.935 → 7.911 | 4.873 [4.694, 6.811] → 10.460 [10.083, 14.525] | 1,952,545 → 1,205,779 | 2.523e-04 → 3.682e-04 | 831 → 1106 |
| basic_mvn | on | 3 | 4.263 [4.085, 4.272] → 7.967 [7.762, 11.134] | 4.085 → 7.762 | 7.855 [7.720, 9.992] → 16.028 [15.776, 19.368] | 1,952,545 → 1,205,779 | 2.905e-04 → 3.682e-04 | 1818 → 2556 |
| spike_slab | off | 3 | 2.341 [2.232, 2.353] → 6.995 [6.969, 9.976] | 2.243 → 6.969 | 6.019 [4.576, 6.694] → 9.134 [8.244, 12.992] | 587,711 → 358,262 | 1.021e-03 → 1.927e-03 | 1046 → 1086 |
| spike_slab | on | 3 | 2.451 [2.293, 2.521] → 6.997 [6.959, 9.922] | 2.293 → 6.959 | 9.026 [7.579, 9.783] → 15.080 [12.888, 18.687] | 587,711 → 358,262 | 1.131e-03 → 1.927e-03 | 2009 → 2465 |
| spike_slab10 | off | 3 | 5.011 [4.957, 5.012] → 13.004 [12.808, 16.341] | 4.959 → 12.808 | 9.293 [7.564, 9.884] → 15.321 [15.161, 19.543] | 1,214,668 → 879,950 | 6.584e-04 → 1.151e-03 | 1099 → 1324 |
| spike_slab10 | on | 3 | 4.852 [4.785, 4.917] → 12.935 [12.690, 16.256] | 4.785 → 12.690 | 14.032 [12.275, 14.724] → 23.149 [22.867, 27.002] | 1,214,668 → 879,950 | 7.232e-04 → 1.151e-03 | 2908 → 3659 |
| weak_curved_mvn8 | off | 3 | 1.076 [1.043, 1.092] → 3.470 [3.251, 6.435] | 1.092 → 3.251 | 1.802 [1.792, 3.923] → 5.506 [5.295, 9.413] | 368,159 → 158,343 | 1.453e-03 → 3.618e-03 | 953 → 903 |
| weak_curved_mvn8 | on | 3 | 1.182 [1.152, 1.190] → 3.520 [3.338, 6.486] | 1.159 → 3.338 | 3.806 [3.748, 6.130] → 8.762 [8.432, 13.110] | 368,159 → 158,343 | 1.567e-03 → 3.618e-03 | 1561 → 1653 |
| weak_curved_spike_slab8 | off | 3 | 1.713 [1.709, 1.758] → 4.165 [3.674, 7.119] | 1.709 → 3.674 | 6.091 [4.243, 6.638] → 6.206 [5.747, 9.950] | 440,820 → 189,169 | 1.199e-03 → 2.767e-03 | 1132 → 937 |
| weak_curved_spike_slab8 | on | 3 | 1.835 [1.764, 1.858] → 4.092 [3.588, 7.104] | 1.764 → 3.588 | 7.778 [6.192, 8.417] → 10.944 [9.591, 14.099] | 440,820 → 189,169 | 1.333e-03 → 2.767e-03 | 1800 → 1803 |
| weak_curved_spike_slab10 | off | 3 | 3.218 [3.209, 3.285] → 8.249 [7.962, 11.741] | 3.285 → 7.962 | 7.800 [5.923, 8.294] → 10.442 [9.247, 14.780] | 773,245 → 423,829 | 9.808e-04 → 1.901e-03 | 1101 → 1092 |
| weak_curved_spike_slab10 | on | 3 | 3.384 [3.380, 3.447] → 8.213 [7.949, 11.739] | 3.380 → 7.949 | 11.938 [9.993, 12.387] → 15.467 [13.825, 19.949] | 773,245 → 423,829 | 1.075e-03 → 1.901e-03 | 2494 → 2665 |

## Compiler and batching observations

For the representative 8D spike/slab case at width 80, the v3 depth program:

- lowers in 1.28 s and compiles in 3.56 s;
- has 499,924 bytes of StableHLO text;
- reports 557,564 argument bytes, 557,716 output bytes, and 1,097,856 temporary
  bytes from XLA memory analysis;
- executes in 8.07 s on the second isolated warmed depth call;
- contains 2 scalar-identity sort operations, 24 while operations (the depth,
  scheduling, and data-dependent sampler loops), and 30 scatters in the
  StableHLO text.

The corresponding v2 whole-run executable lowers in 0.83 s, compiles in
1.77 s, and has 304,907 bytes of StableHLO text. These are different
specialisation boundaries, so they explain compile-plan size but are not a
like-for-like device-runtime comparison.

The implementation contains one full `jax.vmap` for replacement sampling and
no `lax.map`. It sorts the root likelihood identities once, incrementally
merges only the `S` new scalar likelihood identities, and never sorts the full
scientific payload inside the depth loop. A width sweep on the same 8D case
measured widths 16, 40, and 80 at 18.3 s, 15.7 s, and 11.4 s cold end-to-end;
width 80 is the best measured choice. A separate 10D sweep rejected the former
half-root width because 150 lanes took 215.6 s versus 20.9 s for 100 lanes,
showing the expected vmap slowest-lane tail.

## Review findings addressed

The performance-and-intent review found and fixed the following before the
final matrix:

- nested-mask seed de-duplication biased shallow stationary seed ranks; seed
  uniqueness is now enforced only among lanes sharing the exact contour;
- singleton blocks now use the exact `Beta(K, 1)` inverse-CDF path, avoiding
  unnecessary gamma fields while preserving the paper distribution;
- `Gamma(1, 1)` fields use exact exponential draws, substantially reducing
  final-MC cost and memory;
- phantom independence is grouped by root race lineage rather than treating
  finite Markov descendants as independent clusters;
- result resampling clears topology metadata instead of exposing stale parent
  and seed identities;
- a superseded Python parent scheduler and an unused alternative phantom model
  were removed so there is one readable implementation of each algorithm;
- comments now state the stationary interval, in-flight allocation estimate,
  early-chain phantom retention, batching, and reparenting invariants.

Remaining performance risks are explicit: vmapped rejection sampling waits for
the slowest lane, phantom MC materialises arrays proportional to draws times
independence groups, and this CPU matrix does not measure expensive or
accelerator-resident likelihoods. Likelihood-call parallelism, gradient-guided
sampling, and ellipsoidal direction selection remain intentionally deferred to
issues 244, 245, and 246.

## Reproduction

The commands, isolation setup, and interpretation rules are in
`benchmarks/issue_247/README.md`. Final raw records are under
`results/final_v2/` and `results/final_v3_final/`; `summarise.py` emits the
per-problem table without dropping failed gates.
