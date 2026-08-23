# Issue 247 final accuracy and performance report

## Scope and provenance

Measurements were taken on an Intel Core i7-8750H CPU (6 physical cores,
12 hardware threads) with the JAX CPU backend, float64 enabled, Python 3.12.9,
JAX/JAXLIB 0.10.0, and Linux 6.8. The comparison uses all ten standard
problems, phantoms off and on, and seeds 11, 23, and 37: 60 records per
implementation. Each final evidence calculation uses 1,000 Monte Carlo draws.

The baseline is `jaxns==2.6.9`, the latest PyPI release on 2026-08-23.
`origin/main` at `2f356d6d497ce3ac471fb9a06f9d22587487aaaa`
and the installed wheel have byte-identical `jaxns` package trees (excluding
`__pycache__`), so the one baseline matrix represents both sources. Every raw
record identifies the imported version and absolute module path.

Common sampling settings are 30 root chains and five isotropic perfect slice
transitions per dimension, float64, and `dZ/Z = log1p(1e-3)`. V3 uses the
paper scheduler's full vmap replacement width of ten chains per dimension.
Phantom mode retains `D` generated states from the start of each chain; it does
not change the v3 race tree.

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
