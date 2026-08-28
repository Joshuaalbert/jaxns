# Issue 275: periodic random-chart evidence

## Method

All accuracy summaries use 30 independent nested-sampling seeds per row. The
Jones evidence uses 256 final MC shrinkage draws per run; the analytic circular
problems use the deterministic race expectation. Cold wall time is the first
end-to-end run after `jax.clear_caches()`. Warm wall time is the median of the
remaining 29 runs and excludes final MC evidence sampling. Likelihood calls are
logical scientific evaluations reported by JAXNS.

Measurements ran on the JAX 0.10.0 CPU backend with one Intel i7-8750H
(6 cores/12 threads). JAXNS enabled x64 for evidence arithmetic; JAXCTX's
homogeneous U coordinates used their normal float32 base dtype. The direct
hotpath shape was 8 concurrent chains, 32 slice transitions, and one U
coordinate. Compiler memory values are XLA `memory_analysis()` reports, not
process RSS.

The Jones reference is independent of JAXNS. Circular constant phase is
integrated through its modified-Bessel form, unknown HalfNormal measurement
noise is integrated in log scale, and DTEC/clock use tensor Gauss-Legendre
quadrature. Orders 32, 64, 128, 256, and 512 all gave
`log Z = 34.803948945405`; noise quadrature orders 256 and 512 agreed at the
displayed precision.

## Ordinary all-false path versus develop

Paired scientific outputs and evaluation counts identical: **True**.

| Problem/method | n | log Z bias | log Z RMS | circular moment RMS | median likelihood evals | cold wall s | median warm wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| develop | 30 | +0.0128 | 0.0527 | -- | 5105 | 5.022 | 0.023 |
| all-false random-chart branch | 30 | +0.0128 | 0.0527 | -- | 5105 | 5.105 | 0.024 |

The constrained-sampler StableHLO SHA-256 is
`d4bb889d1efa6c9067cf182a9e2b7baac3167608550abe9afe32a9df14d4ebee` on develop and
`d4bb889d1efa6c9067cf182a9e2b7baac3167608550abe9afe32a9df14d4ebee` on the all-false branch. Both have
1152 lines, zero remainder operations,
8136 temporary bytes, and identical argument/output
memory. Direct compile time was 0.498 s versus
0.526 s; median execution was
0.413 ms versus
0.423 ms.

With periodic charts enabled, the same hotpath used
2 remainder operations,
8872 temporary bytes, compiled in
0.658 s, and executed in a median
0.806 ms.
This deliberately exposes the modulo cost on an unrealistically cheap toy
likelihood. In the end-to-end circular benchmark below, random charts add only
a few percent to warm wall time because likelihood work dominates.

## Known circular posterior at three chart placements

| Problem/method | n | log Z bias | log Z RMS | circular moment RMS | median likelihood evals | cold wall s | median warm wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| von_mises_-3.11159 | 30 | -0.0514 | 0.1488 | 0.0128 | 38227 | 6.641 | 0.157 |
| von_mises_+0.00000 | 30 | +0.0179 | 0.1304 | 0.0153 | 37848 | 6.594 | 0.155 |
| von_mises_+1.20000 | 30 | +0.0287 | 0.1719 | 0.0161 | 38154 | 6.535 | 0.154 |

## Sharp seam stress: hard cube versus random chart

This uses concentration 100 and 120 root lineages to make split modes at
the canonical seam consequential.

| Problem/method | n | log Z bias | log Z RMS | circular moment RMS | median likelihood evals | cold wall s | median warm wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard_seam | 30 | -0.0383 | 0.1700 | 0.0086 | 31820 | 5.794 | 0.153 |
| hard_interior | 30 | +0.0180 | 0.1437 | 0.0055 | 44932 | 5.691 | 0.157 |
| random_chart_seam | 30 | -0.0524 | 0.1803 | 0.0052 | 42891 | 6.231 | 0.160 |
| random_chart_interior | 30 | +0.0052 | 0.1453 | 0.0051 | 42004 | 6.029 | 0.159 |

Moving the physical mode from the seam to the interior changes the hard-cube
median likelihood count by 41% and its circular-moment RMS by 56%. Random
charts reduce those placement effects to 2% for both metrics, while their warm
end-to-end wall times agree within 1%. The log-evidence biases remain small
relative to the per-run RMS errors for every placement.

## Full Jones calibration problem

Twenty channels span 700--702.6 MHz. The inferred variables are DTEC in mTECU,
clock in ns, periodic constant phase, and unknown measurement uncertainty; the
truth noise is 0.1.

| Problem/method | n | log Z bias | log Z RMS | circular moment RMS | median likelihood evals | cold wall s | median warm wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| jones_classic | 30 | +0.0036 | 0.2106 | -- | 267182 | 7.075 | 0.741 |
| jones_phantom | 30 | -0.0013 | 0.2272 | -- | 267182 | 7.651 | 0.762 |
