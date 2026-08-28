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
likelihood. In the end-to-end default-allocation controls below, random charts
add 3--12% to warm wall time because likelihood work dominates.

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

## Marginal geometry control: coupled periodic ridge

The initial seam stress only showed placement invariance. This stronger
control holds the physical target, stopping condition, slice count, lineage
count, and random keys fixed while changing only whether the two angular
coordinates have hard cube boundaries or independent random periodic charts.
The target is a coupled toroidal ridge; at the canonical seam its posterior is
split among four cube corners. Root lineage counts 4, 8, 16, and 32 deliberately
test under-resolved mode recovery, while 60 is the normal JAXNS default for
this two-dimensional model. The interior placement is a negative control.

| Problem/method | n | log Z bias | log Z RMS | circular moment RMS | median likelihood evals | cold wall s | median warm wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| ridge_hard_seam_K4 | 30 | +0.1531 | 0.9340 | 0.1026 | 1920 | 4.589 | 0.065 |
| ridge_hard_seam_K8 | 30 | +0.1235 | 0.7540 | 0.0908 | 3704 | 4.589 | 0.063 |
| ridge_hard_seam_K16 | 30 | +0.0653 | 0.6347 | 0.0686 | 7443 | 4.908 | 0.077 |
| ridge_hard_seam_K32 | 30 | -0.0613 | 0.3810 | 0.0417 | 14565 | 4.923 | 0.126 |
| ridge_hard_seam_K60 | 30 | +0.0719 | 0.2560 | 0.0275 | 26400 | 4.826 | 0.133 |
| ridge_hard_interior_K4 | 30 | +0.3313 | 0.8441 | 0.0402 | 3671 | 4.588 | 0.013 |
| ridge_hard_interior_K8 | 30 | +0.2685 | 0.7376 | 0.0246 | 7156 | 4.502 | 0.071 |
| ridge_hard_interior_K16 | 30 | +0.0067 | 0.5943 | 0.0201 | 14369 | 4.979 | 0.105 |
| ridge_hard_interior_K32 | 30 | -0.0213 | 0.4174 | 0.0129 | 27770 | 4.756 | 0.124 |
| ridge_hard_interior_K60 | 30 | -0.0370 | 0.3079 | 0.0089 | 51386 | 4.857 | 0.140 |
| ridge_random_chart_seam_K4 | 30 | +0.0885 | 0.9920 | 0.0356 | 3276 | 4.590 | 0.042 |
| ridge_random_chart_seam_K8 | 30 | +0.1081 | 0.8238 | 0.0278 | 6666 | 4.696 | 0.076 |
| ridge_random_chart_seam_K16 | 30 | +0.0744 | 0.5906 | 0.0203 | 12808 | 4.846 | 0.088 |
| ridge_random_chart_seam_K32 | 30 | +0.0766 | 0.4337 | 0.0138 | 24784 | 4.781 | 0.133 |
| ridge_random_chart_seam_K60 | 30 | +0.1530 | 0.3538 | 0.0119 | 45576 | 4.894 | 0.149 |
| ridge_random_chart_interior_K4 | 30 | +0.4496 | 0.9903 | 0.0337 | 3250 | 4.662 | 0.042 |
| ridge_random_chart_interior_K8 | 30 | +0.2253 | 0.8449 | 0.0254 | 6558 | 4.705 | 0.100 |
| ridge_random_chart_interior_K16 | 30 | +0.0305 | 0.5163 | 0.0181 | 12976 | 4.685 | 0.110 |
| ridge_random_chart_interior_K32 | 30 | -0.0860 | 0.3474 | 0.0139 | 25022 | 4.942 | 0.129 |
| ridge_random_chart_interior_K60 | 30 | -0.0443 | 0.2773 | 0.0108 | 46016 | 4.868 | 0.150 |

The following comparisons use a 10,000-resample paired seed bootstrap;
brackets are 95% intervals. Bias is chart minus hard, while the other columns
are chart/hard ratios for which values below one favour random charts.

| Placement | root degree | log Z bias difference | log Z RMS ratio | moment RMS ratio | likelihood-eval ratio | warm-time ratio |
|---|---:|---:|---:|---:|---:|---:|
| seam | 4 | -0.065 [-0.527, +0.373] | 1.06 [0.73, 1.55] | 0.35 [0.29, 0.41] | 1.71 [1.62, 1.75] | 0.64 [0.15, 5.29] |
| seam | 8 | -0.015 [-0.396, +0.339] | 1.09 [0.83, 1.52] | 0.31 [0.24, 0.38] | 1.80 [1.71, 1.85] | 1.21 [0.79, 4.04] |
| seam | 16 | +0.009 [-0.233, +0.255] | 0.93 [0.62, 1.26] | 0.30 [0.23, 0.38] | 1.72 [1.69, 1.76] | 1.14 [0.82, 1.51] |
| seam | 32 | +0.138 [-0.058, +0.332] | 1.14 [0.74, 1.65] | 0.33 [0.25, 0.44] | 1.70 [1.68, 1.75] | 1.06 [1.03, 1.35] |
| seam | 60 | +0.081 [-0.044, +0.207] | 1.38 [1.00, 1.85] | 0.43 [0.32, 0.58] | 1.73 [1.70, 1.75] | 1.12 [1.09, 1.37] |
| interior | 4 | +0.118 [-0.260, +0.507] | 1.17 [0.79, 1.64] | 0.84 [0.61, 1.15] | 0.89 [0.85, 0.92] | 3.19 [0.35, 6.78] |
| interior | 8 | -0.043 [-0.394, +0.275] | 1.15 [0.82, 1.44] | 1.03 [0.80, 1.32] | 0.92 [0.89, 0.94] | 1.41 [0.73, 1.97] |
| interior | 16 | +0.024 [-0.253, +0.303] | 0.87 [0.64, 1.23] | 0.90 [0.70, 1.17] | 0.90 [0.88, 0.93] | 1.05 [0.81, 1.39] |
| interior | 32 | -0.065 [-0.247, +0.118] | 0.83 [0.59, 1.20] | 1.08 [0.80, 1.47] | 0.90 [0.89, 0.91] | 1.04 [0.81, 1.35] |
| interior | 60 | -0.007 [-0.166, +0.149] | 0.90 [0.68, 1.15] | 1.20 [0.93, 1.56] | 0.90 [0.88, 0.90] | 1.07 [0.88, 1.10] |

## Marginal geometry control: Jones calibration

This is the full Jones model with the same data, priors, stopping condition,
slice count, random keys, and MC evidence calculation in each pair. Only the
constant phase coordinate changes from hard bounded to periodic. Root degree
120 is the normal four-dimensional default; lower values test whether topology
helps when the run begins with less mode coverage.

| Problem/method | n | log Z bias | log Z RMS | circular moment RMS | median likelihood evals | cold wall s | median warm wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| jones_hard_K16 | 30 | +0.1720 | 0.8280 | 0.6312 | 36373 | 5.483 | 0.186 |
| jones_hard_K32 | 30 | -0.2245 | 0.6755 | 0.5608 | 75618 | 5.721 | 0.316 |
| jones_hard_K64 | 30 | +0.0388 | 0.4703 | 0.3453 | 142447 | 6.063 | 0.432 |
| jones_hard_K120 | 30 | -0.0277 | 0.2279 | 0.2374 | 267236 | 6.925 | 0.602 |
| jones_random_chart_K16 | 30 | -0.0820 | 0.7400 | 0.7074 | 37793 | 6.001 | 0.192 |
| jones_random_chart_K32 | 30 | -0.0163 | 0.5524 | 0.5641 | 74866 | 6.434 | 0.292 |
| jones_random_chart_K64 | 30 | +0.0792 | 0.4701 | 0.3431 | 142475 | 6.607 | 0.400 |
| jones_random_chart_K120 | 30 | +0.0036 | 0.2106 | 0.1917 | 267182 | 7.357 | 0.621 |

| Placement | root degree | log Z bias difference | log Z RMS ratio | moment RMS ratio | likelihood-eval ratio | warm-time ratio |
|---|---:|---:|---:|---:|---:|---:|
| same model | 16 | -0.254 [-0.613, +0.134] | 0.89 [0.64, 1.26] | 1.12 [0.96, 1.31] | 1.04 [0.98, 1.07] | 1.03 [0.96, 1.09] |
| same model | 32 | +0.208 [-0.052, +0.477] | 0.82 [0.64, 1.08] | 1.01 [0.85, 1.20] | 0.99 [0.96, 1.04] | 0.93 [0.82, 0.95] |
| same model | 64 | +0.040 [-0.175, +0.259] | 1.00 [0.70, 1.46] | 0.99 [0.78, 1.26] | 1.00 [0.97, 1.02] | 0.93 [0.87, 1.01] |
| same model | 120 | +0.031 [-0.087, +0.159] | 0.92 [0.70, 1.18] | 0.81 [0.66, 0.99] | 1.00 [0.98, 1.02] | 1.03 [1.00, 1.06] |

## Interpretation

The marginal value is specific rather than universal. On the coupled ridge at
the seam, random charts reduce circular posterior-moment RMS by 57--70% at
every tested lineage allocation. At the normal 60-root allocation the ratio is
0.43 [0.32, 0.58], whereas the interior negative control is 1.20
[0.93, 1.56]. This is strong evidence that charts repair seam-fragmented
posterior exploration, rather than generally improving an easy interior
target.

There is no evidence that charts change log-evidence bias: every paired bias
difference interval contains zero. Nor is there a consistent evidence-RMS
improvement. The normal ridge setting has a higher RMS point estimate, 1.38
[1.00, 1.85], while the normal Jones setting is statistically unresolved at
0.92 [0.70, 1.18]. The feature should therefore not be presented as an
evidence-accuracy optimisation.

On Jones, the normal 120-root setting has unchanged likelihood work, 1.00
[0.98, 1.02], 3% more warm wall time, and a smaller periodic-phase moment RMS,
0.81 [0.66, 0.99]. Together with the ridge result, this supports keeping random
charts as the topology-correct sampler for explicitly periodic coordinates.
The cost is isolated: when no coordinate is periodic, scientific results,
evaluation counts, StableHLO, and compiler memory are identical to develop.

## Full Jones calibration problem

Twenty channels span 700--702.6 MHz. The inferred variables are DTEC in mTECU,
clock in ns, periodic constant phase, and unknown measurement uncertainty; the
truth noise is 0.1.

| Problem/method | n | log Z bias | log Z RMS | circular moment RMS | median likelihood evals | cold wall s | median warm wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| jones_classic | 30 | +0.0036 | 0.2106 | -- | 267182 | 7.075 | 0.741 |
| jones_phantom | 30 | -0.0013 | 0.2272 | -- | 267182 | 7.651 | 0.762 |
