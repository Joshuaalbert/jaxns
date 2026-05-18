# Pure-Core Efficiency Grid

Generated: `2026-05-12_113958 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn, spike_slab`
- Allocation targets: `uniform, evidence_improving, posterior_improving`
- LogZ uncertainty targets: `0.85, 0.7`
- Seeds: `0, 17`
- Target live points: `40`
- Max samples: `240`
- Shell size: `10`
- delta_K: `10`
- MC shrinkage samples: `128`
- JAX backend: `cpu`
- JAX devices: `cpu:0`

## Best By Problem And Target

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.7 | evidence_improving | isotropic_6 | 2681.5 | 8.19965 | 180289 | 3536.38 | 0 | 0.5 |
| basic_mvn | 0.85 | evidence_improving | gmm_6 | 3190.5 | 8.62348 | 237260 | 3474.78 | 0 | 0 |
| spike_slab | 0.7 | uniform | isotropic_6 | 2139.5 | 5.58168 | 66656.4 | 1327.31 | 0 | 0 |
| spike_slab | 0.85 | evidence_improving | isotropic_6 | 1177.5 | 30.6591 | 1.10683e+06 | 1179.12 | 0.5 | 0 |

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.7 | evidence_improving | gmm_6 | 2 | 3190.5 | 21.6957 | 23.1563 | 8.69949 | -7.57381 | 1.64351 | 5214.94 | 241461 | 0 | 0.5 | 1 |
| basic_mvn | 0.7 | evidence_improving | isotropic_6 | 2 | 2681.5 | 20.0343 | 21.3748 | 8.19965 | -7.00378 | 1.30459 | 3536.38 | 180289 | 0 | 0.5 | 1 |
| basic_mvn | 0.7 | posterior_improving | gmm_6 | 2 | 3190.5 | 21.6831 | 23.1127 | 8.60609 | -7.46424 | 1.61571 | 5134.22 | 236304 | 0 | 0.5 | 1 |
| basic_mvn | 0.7 | posterior_improving | isotropic_6 | 2 | 2681.5 | 20.3419 | 21.6918 | 8.25489 | -6.99409 | 0.973197 | 2615.52 | 182726 | 0 | 0.5 | 1 |
| basic_mvn | 0.7 | uniform | gmm_6 | 2 | 3190.5 | 21.1505 | 22.5028 | 8.83407 | -7.63325 | 1.5708 | 4979.42 | 248989 | 0 | 0.5 | 1 |
| basic_mvn | 0.7 | uniform | isotropic_6 | 2 | 2681.5 | 19.8456 | 21.2351 | 8.24569 | -6.99627 | 0.984532 | 2663.29 | 182319 | 0 | 0.5 | 1 |
| basic_mvn | 0.85 | evidence_improving | gmm_6 | 2 | 3190.5 | 21.5548 | 23 | 8.62348 | -7.52108 | 1.08807 | 3474.78 | 237260 | 0 | 0 | 1 |
| basic_mvn | 0.85 | evidence_improving | isotropic_6 | 2 | 240 | 16.9594 | 18.2634 | 59.3771 | -57.9206 | 0.670468 | 160.912 | 846153 | 1 | 0 | 0 |
| basic_mvn | 0.85 | posterior_improving | gmm_6 | 2 | 3190.5 | 21.8352 | 23.274 | 8.8937 | -7.7764 | 1.51592 | 4852.53 | 252362 | 0 | 0.5 | 1 |
| basic_mvn | 0.85 | posterior_improving | isotropic_6 | 2 | 240 | 17.0176 | 18.3359 | 59.3897 | -57.9387 | 0.636659 | 152.798 | 846511 | 1 | 0 | 0 |
| basic_mvn | 0.85 | uniform | gmm_6 | 2 | 3190.5 | 21.0097 | 22.377 | 8.85477 | -7.81287 | 1.62961 | 5126.92 | 250157 | 0 | 0 | 1 |
| basic_mvn | 0.85 | uniform | isotropic_6 | 2 | 240 | 16.9524 | 18.2659 | 59.3835 | -57.9337 | 0.745383 | 178.892 | 846336 | 1 | 0 | 0 |
| spike_slab | 0.7 | evidence_improving | gmm_6 | 2 | 2106.5 | 23.1472 | 24.7035 | 6.98786 | -6.96994 | 0.762512 | 1605.61 | 102861 | 0 | 0 | 1 |
| spike_slab | 0.7 | evidence_improving | isotropic_6 | 2 | 2139.5 | 21.4947 | 23.0504 | 5.58186 | -5.36543 | 0.548877 | 1170.07 | 66660.8 | 0 | 0 | 1 |
| spike_slab | 0.7 | posterior_improving | gmm_6 | 2 | 2106.5 | 22.9487 | 24.4981 | 7.01017 | -6.99776 | 0.851021 | 1794.98 | 103519 | 0 | 0 | 1 |
| spike_slab | 0.7 | posterior_improving | isotropic_6 | 2 | 2139.5 | 21.3756 | 22.965 | 5.63465 | -5.39196 | 0.58578 | 1248.3 | 67927.7 | 0 | 0 | 1 |
| spike_slab | 0.7 | uniform | gmm_6 | 2 | 2106.5 | 22.5536 | 24.1307 | 7.05208 | -7.03677 | 0.785258 | 1651.5 | 104760 | 0 | 0 | 1 |
| spike_slab | 0.7 | uniform | isotropic_6 | 2 | 2139.5 | 20.8969 | 22.5057 | 5.58168 | -5.36005 | 0.624029 | 1327.31 | 66656.4 | 0 | 0 | 1 |
| spike_slab | 0.85 | evidence_improving | gmm_6 | 2 | 1183.5 | 21.2096 | 22.8265 | 34.5072 | -27.4403 | 0.567821 | 811.294 | 1.40925e+06 | 0.5 | 0 | 0.5 |
| spike_slab | 0.85 | evidence_improving | isotropic_6 | 2 | 1177.5 | 19.9515 | 21.4941 | 30.6591 | -24.8527 | 0.944498 | 1179.12 | 1.10683e+06 | 0.5 | 0 | 0.5 |
| spike_slab | 0.85 | posterior_improving | gmm_6 | 2 | 1183.5 | 21.1912 | 22.7848 | 34.4622 | -27.3858 | 0.59415 | 805.231 | 1.40558e+06 | 0.5 | 0 | 0.5 |
| spike_slab | 0.85 | posterior_improving | isotropic_6 | 2 | 1177.5 | 19.8651 | 21.4457 | 30.7953 | -24.9183 | 0.835504 | 1015.78 | 1.11668e+06 | 0.5 | 0 | 0.5 |
| spike_slab | 0.85 | uniform | gmm_6 | 2 | 1183.5 | 20.7717 | 22.3933 | 34.57 | -27.5274 | 0.623629 | 839.447 | 1.41438e+06 | 0.5 | 0 | 0.5 |
| spike_slab | 0.85 | uniform | isotropic_6 | 2 | 1177.5 | 19.4259 | 20.9927 | 30.7763 | -24.9553 | 0.830524 | 992.786 | 1.11531e+06 | 0.5 | 0 | 0.5 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.85 | uniform | isotropic_6 | 0 | 240 | 16.9291 | 18.2496 | -24.6065 | -95.5817 | 0.714261 | 0.840511 | -70.9752 | 169.55 | 1.209e+06 | 84.443 | false | true |
| basic_mvn | 0.85 | uniform | isotropic_6 | 17 | 240 | 16.9757 | 18.2822 | -24.6065 | -69.4987 | 0.756731 | 0.885611 | -44.8923 | 188.234 | 483675 | 50.6907 | false | true |
| basic_mvn | 0.85 | uniform | gmm_6 | 0 | 2895 | 20.9904 | 22.3548 | -24.6065 | -36.5866 | 1.15402 | 1.36909 | -11.9801 | 5426.4 | 415500 | 8.75044 | false | false |
| basic_mvn | 0.85 | uniform | gmm_6 | 17 | 3486 | 21.0291 | 22.3992 | -24.6065 | -28.2521 | 1.19372 | 1.17678 | -3.64561 | 4827.43 | 46330.5 | 3.09796 | false | false |
| basic_mvn | 0.85 | evidence_improving | isotropic_6 | 0 | 240 | 17.0484 | 18.352 | -24.6065 | -95.5976 | 0.714261 | 0.737998 | -70.9912 | 130.714 | 1.20954e+06 | 96.1943 | false | true |
| basic_mvn | 0.85 | evidence_improving | isotropic_6 | 17 | 240 | 16.8704 | 18.1749 | -24.6065 | -69.4565 | 0.756731 | 0.892354 | -44.85 | 191.111 | 482766 | 50.2604 | false | true |
| basic_mvn | 0.85 | evidence_improving | gmm_6 | 0 | 2895 | 21.3889 | 22.8198 | -24.6065 | -36.3463 | 1.15402 | 1.03776 | -11.7398 | 3117.75 | 398999 | 11.3127 | false | false |
| basic_mvn | 0.85 | evidence_improving | gmm_6 | 17 | 3486 | 21.7207 | 23.1802 | -24.6065 | -27.9088 | 1.19372 | 1.04843 | -3.30234 | 3831.81 | 38016.5 | 3.14981 | false | false |
| basic_mvn | 0.85 | posterior_improving | isotropic_6 | 0 | 240 | 16.8249 | 18.15 | -24.6065 | -95.5925 | 0.714261 | 0.73127 | -70.9861 | 128.341 | 1.20936e+06 | 97.0723 | false | true |
| basic_mvn | 0.85 | posterior_improving | isotropic_6 | 17 | 240 | 17.2104 | 18.5219 | -24.6065 | -69.4979 | 0.756731 | 0.859397 | -44.8914 | 177.255 | 483658 | 52.236 | false | true |
| basic_mvn | 0.85 | posterior_improving | gmm_6 | 0 | 2895 | 21.9195 | 23.3611 | -24.6065 | -36.6986 | 1.15402 | 1.20908 | -12.0921 | 4232.11 | 423305 | 10.0011 | false | false |
| basic_mvn | 0.85 | posterior_improving | gmm_6 | 17 | 3486 | 21.7509 | 23.187 | -24.6065 | -28.0671 | 1.19372 | 1.25299 | -3.46069 | 5472.94 | 41749.6 | 2.76195 | true | false |
| basic_mvn | 0.7 | uniform | isotropic_6 | 0 | 2600 | 19.8339 | 21.2201 | -24.6065 | -35.9666 | 1.01743 | 0.83607 | -11.3602 | 1817.43 | 335539 | 13.5876 | false | false |
| basic_mvn | 0.7 | uniform | isotropic_6 | 17 | 2763 | 19.8573 | 21.2501 | -24.6065 | -27.2388 | 1.0561 | 1.12697 | -2.63238 | 3509.15 | 19146 | 2.33581 | true | false |
| basic_mvn | 0.7 | uniform | gmm_6 | 0 | 2895 | 21.0864 | 22.4467 | -24.6065 | -36.6866 | 1.15402 | 1.29607 | -12.0801 | 4862.98 | 422463 | 9.32059 | false | false |
| basic_mvn | 0.7 | uniform | gmm_6 | 17 | 3486 | 21.2145 | 22.5589 | -24.6065 | -27.7929 | 1.19372 | 1.20905 | -3.18641 | 5095.85 | 35394 | 2.63546 | true | false |
| basic_mvn | 0.7 | evidence_improving | isotropic_6 | 0 | 2600 | 20.1592 | 21.4903 | -24.6065 | -35.8742 | 1.01743 | 0.914875 | -11.2677 | 2176.19 | 330101 | 12.3161 | false | false |
| basic_mvn | 0.7 | evidence_improving | isotropic_6 | 17 | 2763 | 19.9095 | 21.2593 | -24.6065 | -27.3463 | 1.0561 | 1.33124 | -2.73983 | 4896.56 | 20741 | 2.05811 | true | false |
| basic_mvn | 0.7 | evidence_improving | gmm_6 | 0 | 2895 | 21.7203 | 23.1752 | -24.6065 | -36.4603 | 1.15402 | 1.31929 | -11.8538 | 5038.79 | 406785 | 8.98503 | false | false |
| basic_mvn | 0.7 | evidence_improving | gmm_6 | 17 | 3486 | 21.6711 | 23.1373 | -24.6065 | -27.9003 | 1.19372 | 1.24358 | -3.2938 | 5391.1 | 37820 | 2.64864 | true | false |
| basic_mvn | 0.7 | posterior_improving | isotropic_6 | 0 | 2600 | 20.2616 | 21.6087 | -24.6065 | -35.9853 | 1.01743 | 0.949176 | -11.3788 | 2342.43 | 336641 | 11.9881 | false | false |
| basic_mvn | 0.7 | posterior_improving | isotropic_6 | 17 | 2763 | 20.4223 | 21.775 | -24.6065 | -27.2158 | 1.0561 | 1.02248 | -2.60937 | 2888.61 | 18812.7 | 2.55201 | true | false |
| basic_mvn | 0.7 | posterior_improving | gmm_6 | 0 | 2895 | 21.8198 | 23.2338 | -24.6065 | -36.3544 | 1.15402 | 1.29839 | -11.7479 | 4880.45 | 399549 | 9.04806 | false | false |
| basic_mvn | 0.7 | posterior_improving | gmm_6 | 17 | 3486 | 21.5464 | 22.9916 | -24.6065 | -27.787 | 1.19372 | 1.24323 | -3.18055 | 5388 | 35264.1 | 2.55831 | true | false |
| spike_slab | 0.85 | uniform | isotropic_6 | 0 | 240 | 17.8668 | 19.4004 | -18.3959 | -61.3627 | 0.806009 | 0.902602 | -42.9668 | 195.526 | 443076 | 47.6033 | false | true |
| spike_slab | 0.85 | uniform | isotropic_6 | 17 | 2115 | 20.985 | 22.5849 | -18.3959 | -25.3397 | 0.981648 | 0.919977 | -6.94382 | 1790.05 | 101978 | 7.54782 | false | false |
| spike_slab | 0.85 | uniform | gmm_6 | 0 | 2127 | 22.612 | 24.1976 | -18.3959 | -25.0109 | 1.03238 | 0.855033 | -6.61497 | 1555.01 | 93073 | 7.73651 | false | false |
| spike_slab | 0.85 | uniform | gmm_6 | 17 | 240 | 18.9314 | 20.589 | -18.3959 | -66.8356 | 0.703491 | 0.718454 | -48.4398 | 123.882 | 563138 | 67.4222 | false | true |
| spike_slab | 0.85 | evidence_improving | isotropic_6 | 0 | 240 | 18.1944 | 19.7169 | -18.3959 | -61.202 | 0.806009 | 0.934378 | -42.8061 | 209.535 | 439768 | 45.8124 | false | true |
| spike_slab | 0.85 | evidence_improving | isotropic_6 | 17 | 2115 | 21.7087 | 23.2714 | -18.3959 | -25.2951 | 0.981648 | 1.00794 | -6.89921 | 2148.7 | 100672 | 6.84489 | false | false |
| spike_slab | 0.85 | evidence_improving | gmm_6 | 0 | 2127 | 22.8779 | 24.4314 | -18.3959 | -24.913 | 1.03238 | 0.845837 | -6.51708 | 1521.74 | 90338.7 | 7.70489 | false | false |
| spike_slab | 0.85 | evidence_improving | gmm_6 | 17 | 240 | 19.5412 | 21.2216 | -18.3959 | -66.7593 | 0.703491 | 0.648229 | -48.3635 | 100.848 | 561366 | 74.6086 | false | true |
| spike_slab | 0.85 | posterior_improving | isotropic_6 | 0 | 240 | 18.3614 | 19.8792 | -18.3959 | -61.4092 | 0.806009 | 0.89521 | -43.0133 | 192.336 | 444034 | 48.0483 | false | true |
| spike_slab | 0.85 | posterior_improving | isotropic_6 | 17 | 2115 | 21.3688 | 23.0123 | -18.3959 | -25.2193 | 0.981648 | 0.932527 | -6.82337 | 1839.22 | 98470.8 | 7.31707 | false | false |
| spike_slab | 0.85 | posterior_improving | gmm_6 | 0 | 2127 | 23.3985 | 24.9465 | -18.3959 | -24.8613 | 1.03238 | 0.838043 | -6.46542 | 1493.83 | 88912.2 | 7.71491 | false | false |
| spike_slab | 0.85 | posterior_improving | gmm_6 | 17 | 240 | 18.9838 | 20.6231 | -18.3959 | -66.7021 | 0.703491 | 0.697126 | -48.3062 | 116.636 | 560038 | 69.2934 | false | true |
| spike_slab | 0.7 | uniform | isotropic_6 | 0 | 2164 | 20.5721 | 22.2255 | -18.3959 | -22.1987 | 0.85508 | 0.552761 | -3.80283 | 661.199 | 31294.6 | 6.87969 | false | false |
| spike_slab | 0.7 | uniform | isotropic_6 | 17 | 2115 | 21.2217 | 22.7858 | -18.3959 | -25.3132 | 0.981648 | 0.970831 | -6.91728 | 1993.42 | 101200 | 7.12511 | false | false |
| spike_slab | 0.7 | uniform | gmm_6 | 0 | 2127 | 22.5431 | 24.1241 | -18.3959 | -24.9682 | 1.03238 | 0.809917 | -6.5723 | 1395.24 | 91875.9 | 8.11477 | false | false |
| spike_slab | 0.7 | uniform | gmm_6 | 17 | 2086 | 22.5642 | 24.1373 | -18.3959 | -25.8971 | 0.96584 | 0.956321 | -7.50125 | 1907.75 | 117377 | 7.84386 | false | false |
| spike_slab | 0.7 | evidence_improving | isotropic_6 | 0 | 2164 | 21.524 | 23.0765 | -18.3959 | -22.222 | 0.85508 | 0.612701 | -3.82615 | 812.372 | 31679.6 | 6.24472 | false | false |
| spike_slab | 0.7 | evidence_improving | isotropic_6 | 17 | 2115 | 21.4654 | 23.0244 | -18.3959 | -25.3006 | 0.981648 | 0.849913 | -6.90471 | 1527.77 | 100833 | 8.12402 | false | false |
| spike_slab | 0.7 | evidence_improving | gmm_6 | 0 | 2127 | 22.9532 | 24.5144 | -18.3959 | -24.8658 | 1.03238 | 0.85573 | -6.46994 | 1557.55 | 89036.4 | 7.56072 | false | false |
| spike_slab | 0.7 | evidence_improving | gmm_6 | 17 | 2086 | 23.3412 | 24.8927 | -18.3959 | -25.8658 | 0.96584 | 0.890365 | -7.46995 | 1653.68 | 116399 | 8.38976 | false | false |
| spike_slab | 0.7 | posterior_improving | isotropic_6 | 0 | 2164 | 21.4131 | 23.0498 | -18.3959 | -22.152 | 0.85508 | 0.61842 | -3.75608 | 827.607 | 30530.1 | 6.07368 | false | false |
| spike_slab | 0.7 | posterior_improving | isotropic_6 | 17 | 2115 | 21.3382 | 22.8802 | -18.3959 | -25.4237 | 0.981648 | 0.888323 | -7.02784 | 1668.98 | 104461 | 7.91135 | false | false |
| spike_slab | 0.7 | posterior_improving | gmm_6 | 0 | 2127 | 22.9361 | 24.4967 | -18.3959 | -24.9768 | 1.03238 | 0.98162 | -6.58088 | 2049.53 | 92116.2 | 6.7041 | false | false |
| spike_slab | 0.7 | posterior_improving | gmm_6 | 17 | 2086 | 22.9613 | 24.4995 | -18.3959 | -25.8105 | 0.96584 | 0.859339 | -7.41464 | 1540.43 | 114682 | 8.62831 | false | false |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: isotropic_6/evidence_improving, gmm_6/evidence_improving.
- `spike_slab` best rows by target were: isotropic_6/uniform, isotropic_6/evidence_improving.
- Some rows did not reach the requested uncertainty. Check `sample_cap_fraction` and `max_goal_iterations` before drawing strong efficiency conclusions.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.
- GMM settings won 1 problem/target groups; compare against ellipsoidal rows to decide whether GMM fitting cost is buying lower variance.