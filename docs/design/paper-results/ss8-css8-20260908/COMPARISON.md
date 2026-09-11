# R/A/C/C′ at 240 roots: revised SS8 and CSS8

SS8 uses covariance scale 0.43 and correlation 0.031. CSS8 applies β=0.4 separately around each component mean and latent variance. Thirty seeds per row; isotropic kernel, one pinned core per worker, at most 60 concurrent workers, classic expected log-evidence uncertainty <0.05. G8 and CG8 reuse their unchanged earlier 240-root measurements.

A includes all phantom U points in seed selection. C uses evidence-improving allocation after the first uniform goal. C′ uses an equal weighted sum of evidence and posterior allocation gaps after the same first goal. All evidence comparisons retain the same leading 72 phantoms. Posterior mode RMSE is computed once from classic weights and hard component labels; the reference is the component evidence fraction.

| Problem | Variant | Classic log Z RMSE | Ph72 log Z RMSE | Mode RMSE | Mean calls (M) | Mean goals | Median core (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| G8 | R240 | 0.0537 ± 0.0069 | 0.0303 ± 0.0040 | — | 63.832 | 28.00 | 269.1 |
| G8 | A240 | 0.0534 ± 0.0066 | 0.0271 ± 0.0030 | — | 64.029 | 28.00 | 1652.5 |
| G8 | C240 | 0.0396 ± 0.0040 | 0.0311 ± 0.0044 | — | 32.350 | 29.00 | 201.4 |
| G8 | C′240 | 0.0499 ± 0.0062 | 0.0227 ± 0.0046 | — | 43.325 | 54.03 | 249.6 |
| CG8 | R240 | 0.0582 ± 0.0058 | 0.0281 ± 0.0034 | — | 65.254 | 28.30 | 272.5 |
| CG8 | A240 | 0.0480 ± 0.0055 | 0.0335 ± 0.0044 | — | 64.596 | 28.20 | 1682.2 |
| CG8 | C240 | 0.0454 ± 0.0045 | 0.0299 ± 0.0037 | — | 32.899 | 29.33 | 202.5 |
| CG8 | C′240 | 0.0439 ± 0.0047 | 0.0301 ± 0.0031 | — | 43.776 | 54.17 | 250.3 |
| SS8 | R240 | 0.0786 ± 0.0077 | 0.0649 ± 0.0074 | 0.0479 | 4.295 | 5.00 | 73.4 |
| SS8 | A240 | 0.0597 ± 0.0066 | 0.0453 ± 0.0068 | 0.0315 | 4.292 | 5.00 | 95.7 |
| SS8 | C240 | 0.0942 ± 0.0145 | 0.0634 ± 0.0074 | 0.0512 | 1.292 | 6.00 | 57.0 |
| SS8 | C′240 | 0.0888 ± 0.0091 | 0.0659 ± 0.0067 | 0.0503 | 1.761 | 6.73 | 56.7 |
| CSS8 | R240 | 0.0710 ± 0.0104 | 0.0570 ± 0.0076 | 0.0405 | 5.118 | 5.80 | 94.5 |
| CSS8 | A240 | 0.0609 ± 0.0073 | 0.0389 ± 0.0052 | 0.0282 | 5.061 | 5.77 | 121.2 |
| CSS8 | C240 | 0.0701 ± 0.0105 | 0.0581 ± 0.0072 | 0.0386 | 1.509 | 6.90 | 59.2 |
| CSS8 | C′240 | 0.0891 ± 0.0118 | 0.0607 ± 0.0096 | 0.0392 | 2.123 | 8.07 | 74.4 |

RMSE ± bootstrap SE uses 10,000 seed resamples. Core times include compilation and concurrent host load; likelihood counts are the primary work comparison. The mode reference is a component fraction; hard-label overlap can contribute a small systematic discrepancy.

## Paired evidence improvement from conditioning

Positive classic-minus-Ph72 RMSE means improvement. Intervals are unadjusted paired-bootstrap 95% intervals, not corrected for multiple comparisons.

| Problem | Variant | Classic − Ph72 RMSE [95% CI] |
|---|---|---|
| G8 | R240 | +0.0234 [+0.0062, +0.0401] |
| G8 | A240 | +0.0263 [+0.0133, +0.0398] |
| G8 | C240 | +0.0085 [-0.0034, +0.0196] |
| G8 | Cprime240 | +0.0272 [+0.0138, +0.0429] |
| CG8 | R240 | +0.0301 [+0.0176, +0.0426] |
| CG8 | A240 | +0.0145 [+0.0011, +0.0294] |
| CG8 | C240 | +0.0155 [+0.0041, +0.0264] |
| CG8 | Cprime240 | +0.0139 [+0.0030, +0.0243] |
| SS8 | R240 | +0.0137 [+0.0033, +0.0239] |
| SS8 | A240 | +0.0144 [-0.0029, +0.0328] |
| SS8 | C240 | +0.0307 [+0.0114, +0.0485] |
| SS8 | Cprime240 | +0.0229 [+0.0100, +0.0362] |
| CSS8 | R240 | +0.0139 [+0.0004, +0.0274] |
| CSS8 | A240 | +0.0220 [+0.0101, +0.0348] |
| CSS8 | C240 | +0.0120 [-0.0038, +0.0260] |
| CSS8 | Cprime240 | +0.0284 [+0.0109, +0.0459] |

## Matched policy contrasts

Differences are intervention minus R; negative RMSE differences favor the intervention.

| Problem | Variant − R | Classic RMSE difference [95% CI] | Ph72 RMSE difference [95% CI] | Calls ratio |
|---|---|---|---|---:|
| G8 | A240 − R240 | -0.0003 [-0.0148, +0.0133] | -0.0033 [-0.0140, +0.0076] | 1.003 |
| G8 | C240 − R240 | -0.0141 [-0.0282, +0.0010] | +0.0008 [-0.0089, +0.0108] | 0.507 |
| G8 | Cprime240 − R240 | -0.0038 [-0.0192, +0.0143] | -0.0076 [-0.0168, +0.0002] | 0.679 |
| CG8 | A240 − R240 | -0.0101 [-0.0268, +0.0063] | +0.0054 [-0.0050, +0.0161] | 0.990 |
| CG8 | C240 − R240 | -0.0128 [-0.0262, +0.0010] | +0.0018 [-0.0083, +0.0124] | 0.504 |
| CG8 | Cprime240 − R240 | -0.0142 [-0.0280, +0.0002] | +0.0020 [-0.0062, +0.0108] | 0.671 |
| SS8 | A240 − R240 | -0.0189 [-0.0390, +0.0024] | -0.0196 [-0.0402, -0.0000] | 0.999 |
| SS8 | C240 − R240 | +0.0156 [-0.0063, +0.0370] | -0.0014 [-0.0103, +0.0078] | 0.301 |
| SS8 | Cprime240 − R240 | +0.0102 [-0.0044, +0.0251] | +0.0010 [-0.0080, +0.0111] | 0.410 |
| CSS8 | A240 − R240 | -0.0100 [-0.0316, +0.0132] | -0.0181 [-0.0337, -0.0017] | 0.989 |
| CSS8 | C240 − R240 | -0.0008 [-0.0119, +0.0119] | +0.0011 [-0.0057, +0.0084] | 0.295 |
| CSS8 | Cprime240 − R240 | +0.0181 [-0.0010, +0.0358] | +0.0037 [-0.0056, +0.0123] | 0.415 |
