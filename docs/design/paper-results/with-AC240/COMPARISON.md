# R/A/C/C′/A+C at 240 roots

Thirty seeds per problem and variant. A+C combines all-phantom seeds with C’s evidence-improving allocation after a first uniform goal identical to A’s. R/A/C/C′ retain their previous measurements. SS8 uses covariance multiplier 0.43 and correlation 0.031; CSS8 is its component-centered β=0.4 shear. All runs use CPU float64, isotropic directions, width 80, 80 slice transitions and classic expected log-evidence uncertainty<0.05.

Evidence analysis uses 2048 paired shrinkage draws and prefixes 0, 8, ..., 72. A and A+C retain all 79 phantoms for seed selection. Mode RMSE is measured once per classic posterior and does not depend on phantom prefix. The paper tables remain R240.

| Problem | Variant | Classic RMSE | Ph72 RMSE | Mode RMSE | Mean calls (M) | Mean goals |
|---|---|---:|---:|---:|---:|---:|
| G8 | R240 | 0.0537 ± 0.0069 | 0.0303 ± 0.0040 | — | 63.832 | 28.00 |
| G8 | A240 | 0.0534 ± 0.0066 | 0.0271 ± 0.0030 | — | 64.029 | 28.00 |
| G8 | C240 | 0.0396 ± 0.0040 | 0.0311 ± 0.0044 | — | 32.350 | 29.00 |
| G8 | C′240 | 0.0499 ± 0.0062 | 0.0227 ± 0.0046 | — | 43.325 | 54.03 |
| G8 | A+C240 | 0.0478 ± 0.0057 | 0.0237 ± 0.0044 | — | 32.271 | 29.00 |
| CG8 | R240 | 0.0582 ± 0.0058 | 0.0281 ± 0.0034 | — | 65.254 | 28.30 |
| CG8 | A240 | 0.0480 ± 0.0055 | 0.0335 ± 0.0044 | — | 64.596 | 28.20 |
| CG8 | C240 | 0.0454 ± 0.0045 | 0.0299 ± 0.0037 | — | 32.899 | 29.33 |
| CG8 | C′240 | 0.0439 ± 0.0047 | 0.0301 ± 0.0031 | — | 43.776 | 54.17 |
| CG8 | A+C240 | 0.0468 ± 0.0060 | 0.0304 ± 0.0033 | — | 32.808 | 29.23 |
| SS8 | R240 | 0.0786 ± 0.0077 | 0.0649 ± 0.0074 | 0.0479 | 4.295 | 5.00 |
| SS8 | A240 | 0.0597 ± 0.0066 | 0.0453 ± 0.0068 | 0.0315 | 4.292 | 5.00 |
| SS8 | C240 | 0.0942 ± 0.0145 | 0.0634 ± 0.0074 | 0.0512 | 1.292 | 6.00 |
| SS8 | C′240 | 0.0888 ± 0.0091 | 0.0659 ± 0.0067 | 0.0503 | 1.761 | 6.73 |
| SS8 | A+C240 | 0.0616 ± 0.0101 | 0.0376 ± 0.0037 | 0.0295 | 1.277 | 6.00 |
| CSS8 | R240 | 0.0710 ± 0.0104 | 0.0570 ± 0.0076 | 0.0405 | 5.118 | 5.80 |
| CSS8 | A240 | 0.0609 ± 0.0073 | 0.0389 ± 0.0052 | 0.0282 | 5.061 | 5.77 |
| CSS8 | C240 | 0.0701 ± 0.0105 | 0.0581 ± 0.0072 | 0.0386 | 1.509 | 6.90 |
| CSS8 | C′240 | 0.0891 ± 0.0118 | 0.0607 ± 0.0096 | 0.0392 | 2.123 | 8.07 |
| CSS8 | A+C240 | 0.0531 ± 0.0056 | 0.0332 ± 0.0037 | 0.0262 | 1.483 | 6.87 |

Evidence errors are log-Z RMSE ± bootstrap SE (10,000 seed resamples). Mode reference is the component evidence fraction; hard component assignment can introduce a small overlap discrepancy.

## A+C policy contrasts

Differences are A+C minus the reference; negative RMSE differences favor A+C. Intervals are paired-seed bootstrap 95% intervals, unadjusted for multiple comparisons. An interval crossing zero is unresolved, not evidence of equivalence. Calls and core-time ratios compare means. Core times include compilation and different cohort host loads.

| Problem | Reference | Classic RMSE difference [95% CI] | Ph72 RMSE difference [95% CI] | Mode RMSE difference [95% CI] | Calls ratio [95% CI] |
|---|---|---|---|---|---|
| G8 | R240 | -0.0059 [-0.0213, +0.0094] | -0.0066 [-0.0183, +0.0059] | — | 0.506 [0.501, 0.510] |
| G8 | A240 | -0.0056 [-0.0210, +0.0101] | -0.0034 [-0.0131, +0.0055] | — | 0.504 [0.503, 0.505] |
| G8 | C240 | +0.0082 [-0.0057, +0.0211] | -0.0074 [-0.0178, +0.0048] | — | 0.998 [0.990, 1.005] |
| G8 | C′240 | -0.0022 [-0.0196, +0.0136] | +0.0010 [-0.0119, +0.0135] | — | 0.745 [0.739, 0.751] |
| CG8 | R240 | -0.0114 [-0.0281, +0.0060] | +0.0023 [-0.0069, +0.0110] | — | 0.503 [0.497, 0.508] |
| CG8 | A240 | -0.0012 [-0.0174, +0.0131] | -0.0031 [-0.0135, +0.0081] | — | 0.508 [0.504, 0.512] |
| CG8 | C240 | +0.0014 [-0.0138, +0.0167] | +0.0005 [-0.0075, +0.0087] | — | 0.997 [0.987, 1.007] |
| CG8 | C′240 | +0.0029 [-0.0118, +0.0174] | +0.0003 [-0.0086, +0.0096] | — | 0.749 [0.743, 0.756] |
| SS8 | R240 | -0.0170 [-0.0421, +0.0101] | -0.0273 [-0.0435, -0.0113] | -0.0184 [-0.0321, -0.0055] | 0.297 [0.287, 0.309] |
| SS8 | A240 | +0.0019 [-0.0130, +0.0170] | -0.0077 [-0.0159, +0.0020] | -0.0020 [-0.0079, +0.0042] | 0.298 [0.287, 0.309] |
| SS8 | C240 | -0.0326 [-0.0666, +0.0016] | -0.0259 [-0.0430, -0.0089] | -0.0216 [-0.0389, -0.0055] | 0.989 [0.936, 1.048] |
| SS8 | C′240 | -0.0272 [-0.0552, +0.0021] | -0.0284 [-0.0424, -0.0133] | -0.0207 [-0.0351, -0.0069] | 0.725 [0.688, 0.765] |
| CSS8 | R240 | -0.0179 [-0.0378, +0.0013] | -0.0239 [-0.0391, -0.0080] | -0.0143 [-0.0266, -0.0004] | 0.290 [0.277, 0.304] |
| CSS8 | A240 | -0.0079 [-0.0222, +0.0072] | -0.0057 [-0.0122, +0.0017] | -0.0020 [-0.0052, +0.0018] | 0.293 [0.282, 0.304] |
| CSS8 | C240 | -0.0171 [-0.0351, +0.0026] | -0.0250 [-0.0403, -0.0096] | -0.0124 [-0.0229, -0.0002] | 0.983 [0.939, 1.027] |
| CSS8 | C′240 | -0.0360 [-0.0570, -0.0134] | -0.0275 [-0.0468, -0.0086] | -0.0130 [-0.0243, -0.0007] | 0.698 [0.674, 0.724] |

## Runtime and memory

Times include compilation. RSS is peak resident process memory; saved-state size is the serialized trimmed State. Cohorts ran under different host loads, so likelihood counts are the more controlled measure of work.

| Problem | Variant | Median core (s) | Median analysis (s) | Median / max core RSS (GiB) | Median classics | Median state (MiB) |
|---|---|---:|---:|---:|---:|---:|
| G8 | R240 | 269.1 | 582.7 | 3.672 / 3.708 | 186456 | 128.0 |
| G8 | A240 | 1652.5 | 621.3 | 6.249 / 6.314 | 186546 | 701.5 |
| G8 | C240 | 201.4 | 421.9 | 2.907 / 2.934 | 128673 | 88.4 |
| G8 | C′240 | 249.6 | 412.9 | 2.887 / 2.921 | 150392 | 103.3 |
| G8 | A+C240 | 975.7 | 359.8 | 4.317 / 4.413 | 128488 | 483.2 |
| CG8 | R240 | 272.5 | 594.7 | 3.645 / 3.707 | 187244 | 128.6 |
| CG8 | A240 | 1682.2 | 631.1 | 6.230 / 6.292 | 187077 | 703.5 |
| CG8 | C240 | 202.5 | 426.4 | 2.891 / 2.943 | 129683 | 89.0 |
| CG8 | C′240 | 250.3 | 416.9 | 2.866 / 2.917 | 151301 | 103.9 |
| CG8 | A+C240 | 1001.7 | 356.6 | 4.291 / 4.416 | 129717 | 487.8 |
| SS8 | R240 | 73.4 | 40.5 | 1.945 / 1.976 | 17921 | 12.3 |
| SS8 | A240 | 95.7 | 38.0 | 2.310 / 2.334 | 17950 | 67.5 |
| SS8 | C240 | 57.0 | 18.5 | 1.714 / 1.758 | 8226 | 5.7 |
| SS8 | C′240 | 56.7 | 19.9 | 1.740 / 1.942 | 10305 | 7.1 |
| SS8 | A+C240 | 60.0 | 34.0 | 1.949 / 1.997 | 8108 | 30.5 |
| CSS8 | R240 | 94.5 | 48.5 | 2.105 / 2.122 | 21720 | 14.9 |
| CSS8 | A240 | 121.2 | 40.3 | 2.567 / 2.632 | 21722 | 81.7 |
| CSS8 | C240 | 59.2 | 20.8 | 1.705 / 1.837 | 9572 | 6.6 |
| CSS8 | C′240 | 74.4 | 21.6 | 1.897 / 1.951 | 11681 | 8.0 |
| CSS8 | A+C240 | 61.6 | 38.4 | 1.935 / 2.006 | 9630 | 36.2 |
