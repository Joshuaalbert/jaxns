# Pure-Core Efficiency Grid

Generated: `2026-05-12_120410 UTC`

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
- Row isolation: `True`
- JAX backend: `cpu`
- JAX devices: `cpu:0`

## Best By Problem And Target

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.7 | posterior_improving | gmm_6 | 3112.5 | 5.3623 | 89497.5 | 4331.91 | 0 | 0.5 |
| basic_mvn | 0.85 | evidence_improving | gmm_6 | 3112.5 | 5.31217 | 87832.2 | 3083.78 | 0 | 0.5 |
| spike_slab | 0.7 | posterior_improving | gmm_6 | 2107.5 | 4.9124 | 50857.5 | 2570.55 | 0 | 0 |
| spike_slab | 0.85 | evidence_improving | isotropic_6 | 1155.5 | 31.4598 | 1.14362e+06 | 981.002 | 0.5 | 0 |

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.7 | evidence_improving | gmm_6 | 2 | 3112.5 | 21.6974 | 23.1391 | 5.44858 | -4.66398 | 1.40798 | 4369.06 | 92400.7 | 0 | 0.5 | 1 |
| basic_mvn | 0.7 | evidence_improving | isotropic_6 | 2 | 2703.5 | 20.104 | 21.4541 | 10.8718 | -10.8415 | 1.49977 | 4063.51 | 319541 | 0 | 0 | 1 |
| basic_mvn | 0.7 | posterior_improving | gmm_6 | 2 | 3112.5 | 21.576 | 23.0117 | 5.3623 | -4.55392 | 1.39624 | 4331.91 | 89497.5 | 0 | 0.5 | 1 |
| basic_mvn | 0.7 | posterior_improving | isotropic_6 | 2 | 2703.5 | 20.0649 | 21.3877 | 10.8537 | -10.8153 | 1.18151 | 3194.94 | 318479 | 0 | 0 | 1 |
| basic_mvn | 0.7 | uniform | gmm_6 | 2 | 3112.5 | 21.3563 | 22.7886 | 5.6089 | -4.80172 | 1.42408 | 4412.78 | 97918.5 | 0 | 0.5 | 1 |
| basic_mvn | 0.7 | uniform | isotropic_6 | 2 | 2703.5 | 19.8153 | 21.2377 | 10.9305 | -10.8882 | 1.14484 | 3100.23 | 323003 | 0 | 0 | 1 |
| basic_mvn | 0.85 | evidence_improving | gmm_6 | 2 | 3112.5 | 21.5905 | 23.0924 | 5.31217 | -4.5292 | 0.991557 | 3083.78 | 87832.2 | 0 | 0.5 | 1 |
| basic_mvn | 0.85 | evidence_improving | isotropic_6 | 2 | 240 | 17.0391 | 18.3446 | 59.3771 | -57.9206 | 0.670468 | 160.912 | 846153 | 1 | 0 | 0 |
| basic_mvn | 0.85 | posterior_improving | gmm_6 | 2 | 3112.5 | 21.8847 | 23.301 | 5.61508 | -4.82545 | 1.41504 | 4402.1 | 98134.4 | 0 | 0.5 | 1 |
| basic_mvn | 0.85 | posterior_improving | isotropic_6 | 2 | 240 | 17.0199 | 18.3168 | 59.3897 | -57.9387 | 0.636659 | 152.798 | 846511 | 1 | 0 | 0 |
| basic_mvn | 0.85 | uniform | gmm_6 | 2 | 3112.5 | 21.25 | 22.6154 | 5.60374 | -4.88634 | 1.50529 | 4671.8 | 97738.4 | 0 | 0.5 | 1 |
| basic_mvn | 0.85 | uniform | isotropic_6 | 2 | 240 | 16.9625 | 18.2683 | 59.3835 | -57.9337 | 0.745383 | 178.892 | 846336 | 1 | 0 | 0 |
| spike_slab | 0.7 | evidence_improving | gmm_6 | 2 | 2107.5 | 23.0657 | 24.6091 | 4.92385 | -4.65352 | 1.18405 | 2495.8 | 51094.9 | 0 | 0 | 1 |
| spike_slab | 0.7 | evidence_improving | isotropic_6 | 2 | 2065.5 | 21.3622 | 22.9247 | 8.90663 | -7.68062 | 0.628006 | 1297.08 | 163852 | 0 | 0 | 1 |
| spike_slab | 0.7 | posterior_improving | gmm_6 | 2 | 2107.5 | 23.1146 | 24.6612 | 4.9124 | -4.67472 | 1.21956 | 2570.55 | 50857.5 | 0 | 0 | 1 |
| spike_slab | 0.7 | posterior_improving | isotropic_6 | 2 | 2065.5 | 21.5938 | 23.1431 | 8.92964 | -7.68938 | 0.578303 | 1194.59 | 164700 | 0 | 0 | 1 |
| spike_slab | 0.7 | uniform | gmm_6 | 2 | 2107.5 | 22.9151 | 24.5201 | 5.00632 | -4.7287 | 1.11046 | 2340.82 | 52820.7 | 0 | 0 | 1 |
| spike_slab | 0.7 | uniform | isotropic_6 | 2 | 2065.5 | 21.1856 | 22.7466 | 8.87527 | -7.63139 | 0.767472 | 1585.53 | 162700 | 0 | 0 | 1 |
| spike_slab | 0.85 | evidence_improving | gmm_6 | 2 | 1173 | 21.3156 | 22.9087 | 34.2666 | -25.7127 | 0.63108 | 937.007 | 1.37734e+06 | 0.5 | 0 | 0.5 |
| spike_slab | 0.85 | evidence_improving | isotropic_6 | 2 | 1155.5 | 19.9302 | 21.5749 | 31.4598 | -27.4667 | 0.859629 | 981.002 | 1.14362e+06 | 0.5 | 0 | 0.5 |
| spike_slab | 0.85 | posterior_improving | gmm_6 | 2 | 1173 | 21.2627 | 22.8694 | 34.2215 | -25.6305 | 0.611688 | 834.79 | 1.37371e+06 | 0.5 | 0 | 0.5 |
| spike_slab | 0.85 | posterior_improving | isotropic_6 | 2 | 1155.5 | 19.8279 | 21.38 | 31.6053 | -27.5821 | 0.822955 | 970.657 | 1.15422e+06 | 0.5 | 0 | 0.5 |
| spike_slab | 0.85 | uniform | gmm_6 | 2 | 1173 | 21.245 | 22.849 | 34.3276 | -25.8289 | 0.670718 | 930.941 | 1.38224e+06 | 0.5 | 0 | 0.5 |
| spike_slab | 0.85 | uniform | isotropic_6 | 2 | 1155.5 | 19.855 | 21.4065 | 31.5721 | -27.5547 | 0.746727 | 800.622 | 1.1518e+06 | 0.5 | 0 | 0.5 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.85 | uniform | isotropic_6 | 0 | 240 | 17.0649 | 18.383 | -24.6065 | -95.5817 | 0.714261 | 0.840511 | -70.9752 | 169.55 | 1.209e+06 | 84.443 | false | true |
| basic_mvn | 0.85 | uniform | isotropic_6 | 17 | 240 | 16.8602 | 18.1535 | -24.6065 | -69.4987 | 0.756731 | 0.885611 | -44.8923 | 188.234 | 483675 | 50.6907 | false | true |
| basic_mvn | 0.85 | uniform | gmm_6 | 0 | 3050 | 21.1773 | 22.533 | -24.6065 | -32.2361 | 1.13388 | 1.31138 | -7.62961 | 5245.12 | 177544 | 5.81802 | false | false |
| basic_mvn | 0.85 | uniform | gmm_6 | 17 | 3175 | 21.3227 | 22.6977 | -24.6065 | -26.7495 | 1.14079 | 1.13616 | -2.14307 | 4098.48 | 14582 | 1.88624 | true | false |
| basic_mvn | 0.85 | evidence_improving | isotropic_6 | 0 | 240 | 16.9537 | 18.2471 | -24.6065 | -95.5976 | 0.714261 | 0.737998 | -70.9912 | 130.714 | 1.20954e+06 | 96.1943 | false | true |
| basic_mvn | 0.85 | evidence_improving | isotropic_6 | 17 | 240 | 17.1244 | 18.4421 | -24.6065 | -69.4565 | 0.756731 | 0.892354 | -44.85 | 191.111 | 482766 | 50.2604 | false | true |
| basic_mvn | 0.85 | evidence_improving | gmm_6 | 0 | 3050 | 21.6967 | 23.1228 | -24.6065 | -31.9115 | 1.13388 | 1.01518 | -7.30508 | 3143.33 | 162761 | 7.19582 | false | false |
| basic_mvn | 0.85 | evidence_improving | gmm_6 | 17 | 3175 | 21.4844 | 23.0619 | -24.6065 | -26.3598 | 1.14079 | 0.975968 | -1.75331 | 3024.23 | 9760.28 | 1.79649 | true | false |
| basic_mvn | 0.85 | posterior_improving | isotropic_6 | 0 | 240 | 16.9157 | 18.2224 | -24.6065 | -95.5925 | 0.714261 | 0.73127 | -70.9861 | 128.341 | 1.20936e+06 | 97.0723 | false | true |
| basic_mvn | 0.85 | posterior_improving | isotropic_6 | 17 | 240 | 17.1241 | 18.4111 | -24.6065 | -69.4979 | 0.756731 | 0.859397 | -44.8914 | 177.255 | 483658 | 52.236 | false | true |
| basic_mvn | 0.85 | posterior_improving | gmm_6 | 0 | 3050 | 22.0873 | 23.5075 | -24.6065 | -32.3032 | 1.13388 | 1.20434 | -7.69672 | 4423.82 | 180680 | 6.39082 | false | false |
| basic_mvn | 0.85 | posterior_improving | gmm_6 | 17 | 3175 | 21.6821 | 23.0945 | -24.6065 | -26.5606 | 1.14079 | 1.17458 | -1.95418 | 4380.38 | 12124.7 | 1.66372 | true | false |
| basic_mvn | 0.7 | uniform | isotropic_6 | 0 | 2657 | 19.8205 | 21.3051 | -24.6065 | -36.4553 | 1.08968 | 1.01672 | -11.8489 | 2746.6 | 373031 | 11.654 | false | false |
| basic_mvn | 0.7 | uniform | isotropic_6 | 17 | 2750 | 19.8102 | 21.1703 | -24.6065 | -34.534 | 1.09644 | 1.12069 | -9.92755 | 3453.87 | 271029 | 8.8584 | false | false |
| basic_mvn | 0.7 | uniform | gmm_6 | 0 | 3050 | 21.3864 | 22.7684 | -24.6065 | -32.307 | 1.13388 | 1.31854 | -7.70056 | 5302.57 | 180861 | 5.84021 | false | false |
| basic_mvn | 0.7 | uniform | gmm_6 | 17 | 3175 | 21.3262 | 22.8088 | -24.6065 | -26.5093 | 1.14079 | 1.05338 | -1.90288 | 3522.99 | 11496.5 | 1.80646 | true | false |
| basic_mvn | 0.7 | evidence_improving | isotropic_6 | 0 | 2657 | 20.2391 | 21.5984 | -24.6065 | -36.2588 | 1.08968 | 1.14394 | -11.6523 | 3476.96 | 360759 | 10.1861 | false | false |
| basic_mvn | 0.7 | evidence_improving | isotropic_6 | 17 | 2750 | 19.9688 | 21.3099 | -24.6065 | -34.6371 | 1.09644 | 1.30036 | -10.0307 | 4650.06 | 276689 | 7.71376 | false | false |
| basic_mvn | 0.7 | evidence_improving | gmm_6 | 0 | 3050 | 21.602 | 23.0187 | -24.6065 | -32.0872 | 1.13388 | 1.27297 | -7.48076 | 4942.4 | 170683 | 5.87661 | false | false |
| basic_mvn | 0.7 | evidence_improving | gmm_6 | 17 | 3175 | 21.7928 | 23.2595 | -24.6065 | -26.4537 | 1.14079 | 1.09339 | -1.8472 | 3795.72 | 10833.6 | 1.68943 | true | false |
| basic_mvn | 0.7 | posterior_improving | isotropic_6 | 0 | 2657 | 20.0544 | 21.3786 | -24.6065 | -36.3335 | 1.08968 | 1.07978 | -11.7271 | 3097.88 | 365402 | 10.8606 | false | false |
| basic_mvn | 0.7 | posterior_improving | isotropic_6 | 17 | 2750 | 20.0753 | 21.3969 | -24.6065 | -34.51 | 1.09644 | 1.09412 | -9.90357 | 3292 | 269722 | 9.05167 | false | false |
| basic_mvn | 0.7 | posterior_improving | gmm_6 | 0 | 3050 | 21.576 | 23.0184 | -24.6065 | -31.9916 | 1.13388 | 1.27219 | -7.38518 | 4936.35 | 166350 | 5.80508 | false | false |
| basic_mvn | 0.7 | posterior_improving | gmm_6 | 17 | 3175 | 21.576 | 23.005 | -24.6065 | -26.3291 | 1.14079 | 1.08351 | -1.72265 | 3727.46 | 9421.93 | 1.58988 | true | false |
| spike_slab | 0.85 | uniform | isotropic_6 | 0 | 240 | 18.3401 | 19.8736 | -18.3959 | -61.3627 | 0.806009 | 0.902602 | -42.9668 | 195.526 | 443076 | 47.6033 | false | true |
| spike_slab | 0.85 | uniform | isotropic_6 | 17 | 2071 | 21.3699 | 22.9393 | -18.3959 | -30.5385 | 1.01495 | 0.823871 | -12.1426 | 1405.72 | 305355 | 14.7385 | false | false |
| spike_slab | 0.85 | uniform | gmm_6 | 0 | 2106 | 23.1275 | 24.6975 | -18.3959 | -21.6139 | 1.04946 | 0.908439 | -3.21799 | 1738 | 21808.6 | 3.54233 | false | false |
| spike_slab | 0.85 | uniform | gmm_6 | 17 | 240 | 19.3625 | 21.0004 | -18.3959 | -66.8356 | 0.703491 | 0.718454 | -48.4398 | 123.882 | 563138 | 67.4222 | false | true |
| spike_slab | 0.85 | evidence_improving | isotropic_6 | 0 | 240 | 18.258 | 19.7864 | -18.3959 | -61.202 | 0.806009 | 0.934378 | -42.8061 | 209.535 | 439768 | 45.8124 | false | true |
| spike_slab | 0.85 | evidence_improving | isotropic_6 | 17 | 2071 | 21.6023 | 23.3634 | -18.3959 | -30.5232 | 1.01495 | 0.919888 | -12.1273 | 1752.47 | 304586 | 13.1835 | false | false |
| spike_slab | 0.85 | evidence_improving | gmm_6 | 0 | 2106 | 23.1557 | 24.6883 | -18.3959 | -21.4578 | 1.04946 | 0.917583 | -3.06195 | 1773.17 | 19744.9 | 3.33698 | false | false |
| spike_slab | 0.85 | evidence_improving | gmm_6 | 17 | 240 | 19.4755 | 21.129 | -18.3959 | -66.7593 | 0.703491 | 0.648229 | -48.3635 | 100.848 | 561366 | 74.6086 | false | true |
| spike_slab | 0.85 | posterior_improving | isotropic_6 | 0 | 240 | 17.9147 | 19.4498 | -18.3959 | -61.4092 | 0.806009 | 0.89521 | -43.0133 | 192.336 | 444034 | 48.0483 | false | true |
| spike_slab | 0.85 | posterior_improving | isotropic_6 | 17 | 2071 | 21.7412 | 23.3101 | -18.3959 | -30.5469 | 1.01495 | 0.918972 | -12.151 | 1748.98 | 305776 | 13.2224 | false | false |
| spike_slab | 0.85 | posterior_improving | gmm_6 | 0 | 2106 | 23.4932 | 25.0333 | -18.3959 | -21.3506 | 1.04946 | 0.858714 | -2.95473 | 1552.94 | 18386.2 | 3.44087 | false | false |
| spike_slab | 0.85 | posterior_improving | gmm_6 | 17 | 240 | 19.0321 | 20.7055 | -18.3959 | -66.7021 | 0.703491 | 0.697126 | -48.3062 | 116.636 | 560038 | 69.2934 | false | true |
| spike_slab | 0.7 | uniform | isotropic_6 | 0 | 2060 | 21.2568 | 22.821 | -18.3959 | -21.496 | 0.955617 | 0.842526 | -3.10012 | 1462.29 | 19798.2 | 3.67956 | false | false |
| spike_slab | 0.7 | uniform | isotropic_6 | 17 | 2071 | 21.1144 | 22.6721 | -18.3959 | -30.5585 | 1.01495 | 0.908346 | -12.1626 | 1708.77 | 306363 | 13.3899 | false | false |
| spike_slab | 0.7 | uniform | gmm_6 | 0 | 2106 | 22.8332 | 24.4169 | -18.3959 | -21.4806 | 1.04946 | 0.872869 | -3.08472 | 1604.56 | 20039.6 | 3.534 | false | false |
| spike_slab | 0.7 | uniform | gmm_6 | 17 | 2109 | 22.9969 | 24.6232 | -18.3959 | -24.7686 | 1.18925 | 1.2079 | -6.37267 | 3077.08 | 85648.5 | 5.27583 | false | false |
| spike_slab | 0.7 | evidence_improving | isotropic_6 | 0 | 2060 | 21.2045 | 22.7818 | -18.3959 | -21.5669 | 0.955617 | 0.799965 | -3.17106 | 1318.28 | 20714.5 | 3.96399 | false | false |
| spike_slab | 0.7 | evidence_improving | isotropic_6 | 17 | 2071 | 21.5199 | 23.0676 | -18.3959 | -30.5861 | 1.01495 | 0.7849 | -12.1902 | 1275.88 | 307751 | 15.5309 | false | false |
| spike_slab | 0.7 | evidence_improving | gmm_6 | 0 | 2106 | 23.275 | 24.8167 | -18.3959 | -21.4403 | 1.04946 | 0.950478 | -3.04446 | 1902.58 | 19519.9 | 3.20308 | false | false |
| spike_slab | 0.7 | evidence_improving | gmm_6 | 17 | 2109 | 22.8564 | 24.4016 | -18.3959 | -24.6585 | 1.18925 | 1.21024 | -6.26258 | 3089.02 | 82714.9 | 5.17466 | false | false |
| spike_slab | 0.7 | posterior_improving | isotropic_6 | 0 | 2060 | 21.4397 | 22.9621 | -18.3959 | -21.5452 | 0.955617 | 0.747456 | -3.14934 | 1150.9 | 20431.8 | 4.21342 | false | false |
| spike_slab | 0.7 | posterior_improving | isotropic_6 | 17 | 2071 | 21.748 | 23.324 | -18.3959 | -30.6253 | 1.01495 | 0.77325 | -12.2294 | 1238.28 | 309736 | 15.8156 | false | false |
| spike_slab | 0.7 | posterior_improving | gmm_6 | 0 | 2106 | 23.2071 | 24.7382 | -18.3959 | -21.5611 | 1.04946 | 0.99902 | -3.16521 | 2101.87 | 21099.1 | 3.16832 | false | false |
| spike_slab | 0.7 | posterior_improving | gmm_6 | 17 | 2109 | 23.0221 | 24.5841 | -18.3959 | -24.5801 | 1.18925 | 1.20045 | -6.18423 | 3039.22 | 80658.2 | 5.15161 | false | false |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: gmm_6/posterior_improving, gmm_6/evidence_improving.
- `spike_slab` best rows by target were: gmm_6/posterior_improving, isotropic_6/evidence_improving.
- Some rows did not reach the requested uncertainty. Check `sample_cap_fraction` and `max_goal_iterations` before drawing strong efficiency conclusions.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.
- GMM settings won 3 problem/target groups; compare against ellipsoidal rows to decide whether GMM fitting cost is buying lower variance.