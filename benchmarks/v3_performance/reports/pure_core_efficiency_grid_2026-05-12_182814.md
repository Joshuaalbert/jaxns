# Pure-Core Efficiency Grid

Generated: `2026-05-12_182814 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn, spike_slab`
- Allocation targets: `uniform, evidence_improving, posterior_improving`
- LogZ uncertainty targets: `0.9, 0.7, 0.5`
- Seeds: `0, 17`
- Target live points: `40`
- Max samples: `1200`
- Shell size: `10`
- Minimum samples before uncertainty stopping: `480`
- delta_K: `10`
- MC shrinkage samples: `128`
- Row isolation: `True`
- JAX backend: `cpu`
- JAX devices: `cpu:0`

## Best By Problem And Target

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 136110 | 0.660757 | 59425.4 | 13176.5 | 0 | 1 |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 136110 | 0.659854 | 59263 | 11167.4 | 0 | 1 |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 25958 | 1.47554 | 56516 | 2615.04 | 1 | 0.5 |
| spike_slab | 0.5 | uniform | ellipsoidal_12 | 40602 | 0.461248 | 8638.07 | 5134.57 | 0 | 1 |
| spike_slab | 0.7 | evidence_improving | isotropic_24_stepout | 43691 | 0.122024 | 650.556 | 3755.5 | 1 | 1 |
| spike_slab | 0.9 | uniform | isotropic_24_stepout | 43691 | 0.0939861 | 385.94 | 3532.09 | 1 | 1 |

## Best Usable By Problem And Target

Rows here require `accuracy_success_fraction >= 0.5`; this prevents biased early stops from winning solely because the MC variance estimate is small.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 136110 | 0.660757 | 59425.4 | 13176.5 | 0 | 1 |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 136110 | 0.659854 | 59263 | 11167.4 | 0 | 1 |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 25958 | 1.47554 | 56516 | 2615.04 | 1 | 0.5 |
| spike_slab | 0.5 | uniform | ellipsoidal_12 | 40602 | 0.461248 | 8638.07 | 5134.57 | 0 | 1 |
| spike_slab | 0.7 | evidence_improving | isotropic_24_stepout | 43691 | 0.122024 | 650.556 | 3755.5 | 1 | 1 |
| spike_slab | 0.9 | uniform | isotropic_24_stepout | 43691 | 0.0939861 | 385.94 | 3532.09 | 1 | 1 |

## Best Strict By Problem And Target

Rows here require both `accuracy_success_fraction == 1` and `target_success_fraction == 1`.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spike_slab | 0.7 | evidence_improving | isotropic_24_stepout | 43691 | 0.122024 | 650.556 | 3755.5 | 1 | 1 |
| spike_slab | 0.9 | uniform | isotropic_24_stepout | 43691 | 0.0939861 | 385.94 | 3532.09 | 1 | 1 |

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | min_sample_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 2 | 136110 | 22.36 | 23.8942 | 0.660757 | -0.558255 | 0.0965918 | 13176.5 | 59425.4 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.5 | evidence_improving | gmm_12 | 2 | 136110 | 22.1195 | 23.6376 | 0.694959 | -0.57304 | 0.0827878 | 11323.5 | 65736.5 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.5 | evidence_improving | isotropic_24_stepout | 2 | 163928 | 12.9595 | 14.5014 | 0.757244 | -0.739944 | 0.0372198 | 6098.33 | 93999.5 | 0 | 0 | 1 | 1 |
| basic_mvn | 0.5 | posterior_improving | ellipsoidal_12 | 2 | 136110 | 22.1801 | 23.7134 | 0.669998 | -0.541016 | 0.102636 | 13967.5 | 61099.2 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.5 | posterior_improving | gmm_12 | 2 | 136110 | 22.4224 | 23.9971 | 0.676184 | -0.530719 | 0.0904006 | 12327 | 62232.7 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.5 | posterior_improving | isotropic_24_stepout | 2 | 163928 | 12.9491 | 14.508 | 0.730464 | -0.716491 | 0.0349794 | 5733.11 | 87468.5 | 0 | 0 | 1 | 1 |
| basic_mvn | 0.5 | uniform | ellipsoidal_12 | 2 | 136110 | 21.4931 | 23.0225 | 0.703008 | -0.547399 | 0.104338 | 14186.2 | 67268 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.5 | uniform | gmm_12 | 2 | 136110 | 21.656 | 23.1945 | 0.723352 | -0.602354 | 0.0942158 | 12902 | 71217.7 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | 2 | 163928 | 12.4197 | 14.0172 | 0.754988 | -0.737956 | 0.0392665 | 6434.83 | 93440.4 | 0 | 0 | 1 | 1 |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 2 | 136110 | 22.1581 | 23.7207 | 0.697244 | -0.546805 | 0.0930116 | 12678.3 | 66169.4 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.7 | evidence_improving | gmm_12 | 2 | 136110 | 22.19 | 23.6965 | 0.712615 | -0.56634 | 0.102093 | 13989.8 | 69119.1 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.7 | evidence_improving | isotropic_24_stepout | 2 | 163928 | 12.9405 | 14.4727 | 0.754804 | -0.742076 | 0.0384356 | 6298.79 | 93394.8 | 0 | 0 | 1 | 1 |
| basic_mvn | 0.7 | posterior_improving | ellipsoidal_12 | 2 | 136110 | 22.7185 | 24.3337 | 0.706397 | -0.555104 | 0.102891 | 14054.4 | 67918.1 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 2 | 136110 | 22.3541 | 23.9118 | 0.689833 | -0.56811 | 0.0953098 | 13007.6 | 64770.4 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.7 | posterior_improving | isotropic_24_stepout | 2 | 163928 | 12.8895 | 14.4264 | 0.741832 | -0.727661 | 0.0343732 | 5633.73 | 90212.3 | 0 | 0 | 1 | 1 |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 2 | 136110 | 21.6341 | 23.1953 | 0.659854 | -0.541804 | 0.0815063 | 11167.4 | 59263 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.7 | uniform | gmm_12 | 2 | 136110 | 21.4218 | 22.9484 | 0.686665 | -0.560632 | 0.109674 | 14940.1 | 64176.8 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.7 | uniform | isotropic_24_stepout | 2 | 163928 | 12.4929 | 14.0802 | 0.798664 | -0.780153 | 0.0339247 | 5559.9 | 104564 | 0 | 0 | 1 | 1 |
| basic_mvn | 0.9 | evidence_improving | ellipsoidal_12 | 2 | 25958 | 8.12641 | 9.69059 | 1.49863 | -1.30113 | 0.104789 | 2722.43 | 58299.2 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | evidence_improving | gmm_12 | 2 | 25958 | 8.1036 | 9.65529 | 1.50047 | -1.27469 | 0.10377 | 2656.51 | 58441.8 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | evidence_improving | isotropic_24_stepout | 2 | 54922.5 | 8.37562 | 9.86967 | 1.25109 | -1.23943 | 0.0570393 | 3136.54 | 85966.8 | 1 | 0 | 1 | 0 |
| basic_mvn | 0.9 | posterior_improving | ellipsoidal_12 | 2 | 25958 | 8.24093 | 9.79344 | 1.5064 | -1.28478 | 0.121439 | 3150.75 | 58904.6 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 2 | 25958 | 8.2448 | 9.77826 | 1.52883 | -1.32087 | 0.108114 | 2765.92 | 60672.4 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | posterior_improving | isotropic_24_stepout | 2 | 54922.5 | 8.4896 | 9.98439 | 1.26666 | -1.25547 | 0.0605834 | 3334.81 | 88119.7 | 1 | 0 | 1 | 0 |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 2 | 25958 | 7.67285 | 9.23783 | 1.47554 | -1.25593 | 0.101362 | 2615.04 | 56516 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | uniform | gmm_12 | 2 | 25958 | 7.64729 | 9.20151 | 1.47754 | -1.26582 | 0.0983136 | 2539.59 | 56669.7 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | uniform | isotropic_24_stepout | 2 | 54922.5 | 7.94733 | 9.48578 | 1.23241 | -1.22333 | 0.0612776 | 3367.81 | 83417.8 | 1 | 0 | 1 | 0 |
| spike_slab | 0.5 | evidence_improving | ellipsoidal_12 | 2 | 40602 | 22.1431 | 23.8471 | 0.494425 | 0.447334 | 0.119883 | 4846.07 | 9925.4 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | evidence_improving | gmm_12 | 2 | 40602 | 22.279 | 24.016 | 0.493258 | 0.455301 | 0.0987136 | 4011.84 | 9878.61 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | evidence_improving | isotropic_24_stepout | 2 | 129480 | 13.1107 | 14.8769 | 0.424609 | 0.405642 | 0.0604552 | 7827.42 | 23344.4 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | posterior_improving | ellipsoidal_12 | 2 | 40602 | 22.495 | 24.2296 | 0.475396 | 0.410948 | 0.129015 | 5249.23 | 9176.09 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | posterior_improving | gmm_12 | 2 | 40602 | 22.3198 | 24.0803 | 0.508921 | 0.460242 | 0.108656 | 4404.06 | 10515.9 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | posterior_improving | isotropic_24_stepout | 2 | 129480 | 13.1942 | 14.9533 | 0.416787 | 0.398676 | 0.0698555 | 9043.9 | 22492.2 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | uniform | ellipsoidal_12 | 2 | 40602 | 21.5982 | 23.3584 | 0.461248 | 0.423474 | 0.127105 | 5134.57 | 8638.07 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | uniform | gmm_12 | 2 | 40602 | 21.7517 | 23.5798 | 0.494339 | 0.437728 | 0.11147 | 4481.48 | 9921.96 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | uniform | isotropic_24_stepout | 2 | 129480 | 12.5879 | 14.3961 | 0.436771 | 0.418987 | 0.0700804 | 9073.26 | 24700.9 | 0 | 1 | 1 | 1 |
| spike_slab | 0.7 | evidence_improving | ellipsoidal_12 | 2 | 10015 | 9.46811 | 11.15 | 0.455427 | -0.396012 | 0.131569 | 1317.53 | 2077.24 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | evidence_improving | gmm_12 | 2 | 10015 | 9.38248 | 11.0595 | 0.450144 | -0.394157 | 0.131416 | 1316.19 | 2029.34 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | evidence_improving | isotropic_24_stepout | 2 | 43691 | 8.60488 | 10.3248 | 0.122024 | -0.0122333 | 0.0859186 | 3755.5 | 650.556 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | posterior_improving | ellipsoidal_12 | 2 | 10015 | 9.52667 | 11.2368 | 0.440806 | -0.390464 | 0.135093 | 1352.6 | 1946.02 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | posterior_improving | gmm_12 | 2 | 10015 | 9.53882 | 11.213 | 0.414806 | -0.356535 | 0.108447 | 1085.7 | 1723.22 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | posterior_improving | isotropic_24_stepout | 2 | 43691 | 8.6535 | 10.3672 | 0.136033 | 0.0173127 | 0.088676 | 3874.22 | 808.502 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | uniform | ellipsoidal_12 | 2 | 10015 | 9.0117 | 10.6879 | 0.447489 | -0.386358 | 0.146326 | 1465.13 | 2005.47 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | uniform | gmm_12 | 2 | 10015 | 9.01334 | 10.6966 | 0.431671 | -0.369666 | 0.139451 | 1396.54 | 1866.2 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | uniform | isotropic_24_stepout | 2 | 43691 | 8.08719 | 9.83234 | 0.125519 | 0.000468352 | 0.0755527 | 3302.34 | 688.349 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | evidence_improving | ellipsoidal_12 | 2 | 10015 | 9.52551 | 11.2378 | 0.475129 | -0.411693 | 0.129646 | 1298.09 | 2260.87 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 2 | 10015 | 9.53816 | 11.2196 | 0.448404 | -0.379947 | 0.128544 | 1287.3 | 2013.68 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | evidence_improving | isotropic_24_stepout | 2 | 43691 | 8.53223 | 10.2606 | 0.14561 | 0.00364281 | 0.0792688 | 3464.46 | 926.347 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | posterior_improving | ellipsoidal_12 | 2 | 10015 | 9.60895 | 11.2854 | 0.429332 | -0.376969 | 0.140015 | 1402.25 | 1846.02 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | posterior_improving | gmm_12 | 2 | 10015 | 9.56835 | 11.244 | 0.413737 | -0.359498 | 0.135084 | 1352.36 | 1714.35 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | posterior_improving | isotropic_24_stepout | 2 | 43691 | 8.64092 | 10.3687 | 0.1617 | -0.00347935 | 0.0919519 | 4019.71 | 1142.39 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | uniform | ellipsoidal_12 | 2 | 10015 | 9.03134 | 10.7105 | 0.454186 | -0.396272 | 0.125687 | 1258.75 | 2065.94 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | uniform | gmm_12 | 2 | 10015 | 9.04662 | 10.7143 | 0.407644 | -0.344457 | 0.13986 | 1400.6 | 1664.23 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | uniform | isotropic_24_stepout | 2 | 43691 | 8.15955 | 9.86833 | 0.0939861 | 0.0298257 | 0.0808392 | 3532.09 | 385.94 | 1 | 1 | 1 | 0 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok | min_samples_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | uniform | isotropic_24_stepout | 0 | 55440 | 7.89078 | 9.45142 | -24.6065 | -25.6805 | 0.872158 | 0.256309 | -1.07403 | 3642.08 | 63952.1 | 4.19037 | false | true | true |
| basic_mvn | 0.9 | uniform | isotropic_24_stepout | 17 | 54405 | 8.00388 | 9.52015 | -24.6065 | -25.9791 | 0.820844 | 0.238456 | -1.37263 | 3093.53 | 102506 | 5.75634 | false | true | true |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 0 | 27555 | 7.56786 | 9.12889 | -24.6065 | -26.6369 | 0.85652 | 0.302104 | -2.03043 | 2514.85 | 113599 | 6.72097 | false | true | true |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 17 | 24361 | 7.77784 | 9.34678 | -24.6065 | -25.0879 | 0.78093 | 0.333853 | -0.481429 | 2715.23 | 5646.24 | 1.44204 | true | true | true |
| basic_mvn | 0.9 | uniform | gmm_12 | 0 | 27555 | 7.54324 | 9.09855 | -24.6065 | -26.6344 | 0.85652 | 0.300882 | -2.02794 | 2494.55 | 113321 | 6.74 | false | true | true |
| basic_mvn | 0.9 | uniform | gmm_12 | 17 | 24361 | 7.75134 | 9.30447 | -24.6065 | -25.1102 | 0.78093 | 0.325726 | -0.503699 | 2584.64 | 6180.7 | 1.54639 | true | true | true |
| basic_mvn | 0.9 | evidence_improving | isotropic_24_stepout | 0 | 55440 | 8.39385 | 9.86113 | -24.6065 | -25.6755 | 0.872158 | 0.253743 | -1.06902 | 3569.53 | 63356.9 | 4.213 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | isotropic_24_stepout | 17 | 54405 | 8.35738 | 9.87822 | -24.6065 | -26.0163 | 0.820844 | 0.22292 | -1.40985 | 2703.56 | 108139 | 6.32447 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | ellipsoidal_12 | 0 | 27555 | 7.96798 | 9.5086 | -24.6065 | -26.6512 | 0.85652 | 0.325932 | -2.04475 | 2927.21 | 115207 | 6.27354 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | ellipsoidal_12 | 17 | 24361 | 8.28484 | 9.87258 | -24.6065 | -25.164 | 0.78093 | 0.321477 | -0.557515 | 2517.64 | 7571.95 | 1.73423 | true | true | true |
| basic_mvn | 0.9 | evidence_improving | gmm_12 | 0 | 27555 | 8.00002 | 9.53261 | -24.6065 | -26.6727 | 0.85652 | 0.283725 | -2.06625 | 2218.18 | 117643 | 7.28257 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | gmm_12 | 17 | 24361 | 8.20717 | 9.77797 | -24.6065 | -25.0896 | 0.78093 | 0.356428 | -0.483125 | 3094.84 | 5686.09 | 1.35546 | true | true | true |
| basic_mvn | 0.9 | posterior_improving | isotropic_24_stepout | 0 | 55440 | 8.49947 | 10.0178 | -24.6065 | -25.6939 | 0.872158 | 0.273709 | -1.08744 | 4153.38 | 65559.1 | 3.97297 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | isotropic_24_stepout | 17 | 54405 | 8.47972 | 9.95095 | -24.6065 | -26.03 | 0.820844 | 0.215058 | -1.4235 | 2516.23 | 110244 | 6.61914 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | ellipsoidal_12 | 0 | 27555 | 8.09881 | 9.639 | -24.6065 | -26.6777 | 0.85652 | 0.347073 | -2.07127 | 3319.27 | 118216 | 5.96783 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | ellipsoidal_12 | 17 | 24361 | 8.38304 | 9.94789 | -24.6065 | -25.1047 | 0.78093 | 0.349883 | -0.498284 | 2982.23 | 6048.51 | 1.42414 | true | true | true |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 0 | 27555 | 8.11778 | 9.65151 | -24.6065 | -26.6972 | 0.85652 | 0.287674 | -2.0907 | 2280.35 | 120444 | 7.2676 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 17 | 24361 | 8.37183 | 9.90501 | -24.6065 | -25.1575 | 0.78093 | 0.365337 | -0.551031 | 3251.49 | 7396.86 | 1.50828 | true | true | true |
| basic_mvn | 0.7 | uniform | isotropic_24_stepout | 0 | 163529 | 12.5598 | 14.1644 | -24.6065 | -25.5576 | 0.757298 | 0.192997 | -0.951107 | 6091.12 | 147929 | 4.92808 | false | false | true |
| basic_mvn | 0.7 | uniform | isotropic_24_stepout | 17 | 164328 | 12.426 | 13.9959 | -24.6065 | -25.2157 | 0.732402 | 0.174933 | -0.609199 | 5028.68 | 60986.1 | 3.48248 | false | false | true |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 0 | 141166 | 21.612 | 23.1715 | -24.6065 | -25.5249 | 0.70999 | 0.309958 | -0.918441 | 13562.4 | 119078 | 2.96311 | true | false | true |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 17 | 131053 | 21.6562 | 23.219 | -24.6065 | -24.7716 | 0.720743 | 0.258725 | -0.165167 | 8772.51 | 3575.15 | 0.638388 | true | false | true |
| basic_mvn | 0.7 | uniform | gmm_12 | 0 | 141166 | 21.3666 | 22.8864 | -24.6065 | -25.5636 | 0.70999 | 0.334861 | -0.957117 | 15829.2 | 129318 | 2.85825 | true | false | true |
| basic_mvn | 0.7 | uniform | gmm_12 | 17 | 131053 | 21.477 | 23.0104 | -24.6065 | -24.7706 | 0.720743 | 0.32744 | -0.164147 | 14051.1 | 3531.12 | 0.501304 | true | false | true |
| basic_mvn | 0.7 | evidence_improving | isotropic_24_stepout | 0 | 163529 | 12.9294 | 14.472 | -24.6065 | -25.4866 | 0.757298 | 0.207848 | -0.880104 | 7064.6 | 126667 | 4.23436 | false | false | true |
| basic_mvn | 0.7 | evidence_improving | isotropic_24_stepout | 17 | 164328 | 12.9516 | 14.4734 | -24.6065 | -25.2105 | 0.732402 | 0.183495 | -0.604048 | 5532.98 | 59959 | 3.29191 | false | false | true |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 0 | 141166 | 22.2318 | 23.704 | -24.6065 | -25.5859 | 0.70999 | 0.310926 | -0.979417 | 13647.3 | 135414 | 3.15 | false | false | true |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 17 | 131053 | 22.0844 | 23.7374 | -24.6065 | -24.7207 | 0.720743 | 0.298911 | -0.114194 | 11709.3 | 1708.97 | 0.382033 | true | false | true |
| basic_mvn | 0.7 | evidence_improving | gmm_12 | 0 | 141166 | 22.0549 | 23.5585 | -24.6065 | -25.6053 | 0.70999 | 0.347402 | -0.998866 | 17037.1 | 140846 | 2.87525 | true | false | true |
| basic_mvn | 0.7 | evidence_improving | gmm_12 | 17 | 131053 | 22.325 | 23.8345 | -24.6065 | -24.7403 | 0.720743 | 0.288959 | -0.133814 | 10942.6 | 2346.66 | 0.46309 | true | false | true |
| basic_mvn | 0.7 | posterior_improving | isotropic_24_stepout | 0 | 163529 | 12.8966 | 14.4269 | -24.6065 | -25.4784 | 0.757298 | 0.192176 | -0.871968 | 6039.38 | 124336 | 4.53734 | false | false | true |
| basic_mvn | 0.7 | posterior_improving | isotropic_24_stepout | 17 | 164328 | 12.8824 | 14.4259 | -24.6065 | -25.1898 | 0.732402 | 0.178367 | -0.583355 | 5228.08 | 55921.3 | 3.27052 | false | false | true |
| basic_mvn | 0.7 | posterior_improving | ellipsoidal_12 | 0 | 141166 | 22.5071 | 24.0208 | -24.6065 | -25.5984 | 0.70999 | 0.335809 | -0.991974 | 15919 | 138909 | 2.95398 | true | false | true |
| basic_mvn | 0.7 | posterior_improving | ellipsoidal_12 | 17 | 131053 | 22.9299 | 24.6466 | -24.6065 | -24.7247 | 0.720743 | 0.304983 | -0.118233 | 12189.9 | 1832 | 0.387672 | true | false | true |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 0 | 141166 | 22.1869 | 23.717 | -24.6065 | -25.5659 | 0.70999 | 0.319759 | -0.959416 | 14433.6 | 129940 | 3.00044 | false | false | true |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 17 | 131053 | 22.5213 | 24.1065 | -24.6065 | -24.7833 | 0.720743 | 0.297278 | -0.176805 | 11581.7 | 4096.71 | 0.594746 | true | false | true |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | 0 | 163529 | 12.3608 | 13.9881 | -24.6065 | -25.5039 | 0.757298 | 0.210819 | -0.897416 | 7267.99 | 131699 | 4.25681 | false | false | true |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | 17 | 164328 | 12.4786 | 14.0462 | -24.6065 | -25.185 | 0.732402 | 0.18463 | -0.578496 | 5601.67 | 54993.6 | 3.13327 | false | false | true |
| basic_mvn | 0.5 | uniform | ellipsoidal_12 | 0 | 141166 | 21.4908 | 23.0159 | -24.6065 | -25.595 | 0.70999 | 0.318325 | -0.988504 | 14304.5 | 137939 | 3.10533 | false | false | true |
| basic_mvn | 0.5 | uniform | ellipsoidal_12 | 17 | 131053 | 21.4953 | 23.0291 | -24.6065 | -24.7128 | 0.720743 | 0.327635 | -0.106293 | 14067.8 | 1480.67 | 0.324426 | true | false | true |
| basic_mvn | 0.5 | uniform | gmm_12 | 0 | 141166 | 21.5178 | 23.0702 | -24.6065 | -25.6093 | 0.70999 | 0.331221 | -1.00286 | 15487 | 141976 | 3.02778 | false | false | true |
| basic_mvn | 0.5 | uniform | gmm_12 | 17 | 131053 | 21.7942 | 23.3187 | -24.6065 | -24.8083 | 0.720743 | 0.280578 | -0.201843 | 10317 | 5339.19 | 0.719383 | true | false | true |
| basic_mvn | 0.5 | evidence_improving | isotropic_24_stepout | 0 | 163529 | 12.9782 | 14.5151 | -24.6065 | -25.5073 | 0.757298 | 0.211796 | -0.900883 | 7335.49 | 132718 | 4.25355 | false | false | true |
| basic_mvn | 0.5 | evidence_improving | isotropic_24_stepout | 17 | 164328 | 12.9409 | 14.4876 | -24.6065 | -25.1855 | 0.732402 | 0.171995 | -0.579004 | 4861.18 | 55090.3 | 3.36641 | false | false | true |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 0 | 141166 | 22.2528 | 23.7802 | -24.6065 | -25.5182 | 0.70999 | 0.320036 | -0.911739 | 14458.7 | 117347 | 2.84886 | true | false | true |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 17 | 131053 | 22.4672 | 24.0081 | -24.6065 | -24.8112 | 0.720743 | 0.301265 | -0.204771 | 11894.4 | 5495.18 | 0.679703 | true | false | true |
| basic_mvn | 0.5 | evidence_improving | gmm_12 | 0 | 141166 | 21.9874 | 23.4944 | -24.6065 | -25.5727 | 0.70999 | 0.306158 | -0.966223 | 13231.9 | 131791 | 3.15596 | false | false | true |
| basic_mvn | 0.5 | evidence_improving | gmm_12 | 17 | 131053 | 22.2516 | 23.7808 | -24.6065 | -24.7863 | 0.720743 | 0.268035 | -0.179856 | 9415.23 | 4239.34 | 0.671017 | true | false | true |
| basic_mvn | 0.5 | posterior_improving | isotropic_24_stepout | 0 | 163529 | 12.8247 | 14.3715 | -24.6065 | -25.4651 | 0.757298 | 0.193672 | -0.858683 | 6133.76 | 120576 | 4.4337 | false | false | true |
| basic_mvn | 0.5 | posterior_improving | isotropic_24_stepout | 17 | 164328 | 13.0735 | 14.6444 | -24.6065 | -25.1808 | 0.732402 | 0.180139 | -0.574298 | 5332.47 | 54198.4 | 3.18808 | false | false | true |
| basic_mvn | 0.5 | posterior_improving | ellipsoidal_12 | 0 | 141166 | 22.0843 | 23.554 | -24.6065 | -25.5427 | 0.70999 | 0.319687 | -0.936236 | 14427.2 | 123737 | 2.9286 | true | false | true |
| basic_mvn | 0.5 | posterior_improving | ellipsoidal_12 | 17 | 131053 | 22.276 | 23.8728 | -24.6065 | -24.7523 | 0.720743 | 0.321048 | -0.145797 | 13507.9 | 2785.75 | 0.454127 | true | false | true |
| basic_mvn | 0.5 | posterior_improving | gmm_12 | 0 | 141166 | 22.1446 | 23.6819 | -24.6065 | -25.5562 | 0.70999 | 0.308022 | -0.94972 | 13393.5 | 127327 | 3.08328 | false | false | true |
| basic_mvn | 0.5 | posterior_improving | gmm_12 | 17 | 131053 | 22.7001 | 24.3123 | -24.6065 | -24.7182 | 0.720743 | 0.293127 | -0.111718 | 11260.5 | 1635.67 | 0.381126 | true | false | true |
| spike_slab | 0.9 | uniform | isotropic_24_stepout | 0 | 43875 | 8.13969 | 9.86098 | -18.3959 | -18.2769 | 0.677254 | 0.28574 | 0.118954 | 3582.27 | 620.832 | 0.416301 | true | true | true |
| spike_slab | 0.9 | uniform | isotropic_24_stepout | 17 | 43507 | 8.17942 | 9.87568 | -18.3959 | -18.4552 | 0.690241 | 0.282898 | -0.0593023 | 3481.92 | 153.004 | 0.209625 | true | true | true |
| spike_slab | 0.9 | uniform | ellipsoidal_12 | 0 | 9995 | 9.02306 | 10.7082 | -18.3959 | -19.0141 | 0.691539 | 0.355291 | -0.618202 | 1261.68 | 3819.83 | 1.73999 | true | true | true |
| spike_slab | 0.9 | uniform | ellipsoidal_12 | 17 | 10035 | 9.03961 | 10.7128 | -18.3959 | -18.5702 | 0.68635 | 0.353756 | -0.174342 | 1255.81 | 305.017 | 0.492832 | true | true | true |
| spike_slab | 0.9 | uniform | gmm_12 | 0 | 9995 | 9.10049 | 10.7784 | -18.3959 | -18.9583 | 0.691539 | 0.380709 | -0.562455 | 1448.67 | 3161.98 | 1.47739 | true | true | true |
| spike_slab | 0.9 | uniform | gmm_12 | 17 | 10035 | 8.99276 | 10.6502 | -18.3959 | -18.5223 | 0.68635 | 0.367126 | -0.126459 | 1352.53 | 160.479 | 0.344457 | true | true | true |
| spike_slab | 0.9 | evidence_improving | isotropic_24_stepout | 0 | 43875 | 8.61162 | 10.3507 | -18.3959 | -18.2467 | 0.677254 | 0.292209 | 0.149207 | 3746.32 | 976.779 | 0.510618 | true | true | true |
| spike_slab | 0.9 | evidence_improving | isotropic_24_stepout | 17 | 43507 | 8.45285 | 10.1705 | -18.3959 | -18.5378 | 0.690241 | 0.270465 | -0.141922 | 3182.6 | 876.306 | 0.524731 | true | true | true |
| spike_slab | 0.9 | evidence_improving | ellipsoidal_12 | 0 | 9995 | 9.50471 | 11.2243 | -18.3959 | -19.0448 | 0.691539 | 0.381727 | -0.648878 | 1456.42 | 4208.32 | 1.69985 | true | true | true |
| spike_slab | 0.9 | evidence_improving | ellipsoidal_12 | 17 | 10035 | 9.54632 | 11.2514 | -18.3959 | -18.5704 | 0.68635 | 0.337013 | -0.174508 | 1139.75 | 305.597 | 0.517809 | true | true | true |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 0 | 9995 | 9.44127 | 11.1357 | -18.3959 | -19.014 | 0.691539 | 0.363461 | -0.618078 | 1320.38 | 3818.3 | 1.70054 | true | true | true |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 17 | 10035 | 9.63506 | 11.3034 | -18.3959 | -18.5377 | 0.68635 | 0.353532 | -0.141816 | 1254.22 | 201.822 | 0.401141 | true | true | true |
| spike_slab | 0.9 | posterior_improving | isotropic_24_stepout | 0 | 43875 | 8.64398 | 10.3675 | -18.3959 | -18.2377 | 0.677254 | 0.322715 | 0.158184 | 4569.37 | 1097.84 | 0.490165 | true | true | true |
| spike_slab | 0.9 | posterior_improving | isotropic_24_stepout | 17 | 43507 | 8.63786 | 10.3699 | -18.3959 | -18.561 | 0.690241 | 0.282416 | -0.165142 | 3470.06 | 1186.52 | 0.584749 | true | true | true |
| spike_slab | 0.9 | posterior_improving | ellipsoidal_12 | 0 | 9995 | 9.64974 | 11.3215 | -18.3959 | -18.9783 | 0.691539 | 0.374409 | -0.582444 | 1401.12 | 3390.71 | 1.55563 | true | true | true |
| spike_slab | 0.9 | posterior_improving | ellipsoidal_12 | 17 | 10035 | 9.56816 | 11.2493 | -18.3959 | -18.5674 | 0.68635 | 0.373963 | -0.171495 | 1403.38 | 295.133 | 0.458587 | true | true | true |
| spike_slab | 0.9 | posterior_improving | gmm_12 | 0 | 9995 | 9.58615 | 11.2686 | -18.3959 | -18.9602 | 0.691539 | 0.400451 | -0.56429 | 1602.81 | 3182.64 | 1.40913 | true | true | true |
| spike_slab | 0.9 | posterior_improving | gmm_12 | 17 | 10035 | 9.55055 | 11.2194 | -18.3959 | -18.5506 | 0.68635 | 0.331371 | -0.154707 | 1101.91 | 240.181 | 0.46687 | true | true | true |
| spike_slab | 0.7 | uniform | isotropic_24_stepout | 0 | 43875 | 8.04599 | 9.81416 | -18.3959 | -18.2699 | 0.677254 | 0.288068 | 0.125986 | 3640.89 | 696.406 | 0.437348 | true | true | true |
| spike_slab | 0.7 | uniform | isotropic_24_stepout | 17 | 43507 | 8.12839 | 9.85051 | -18.3959 | -18.5209 | 0.690241 | 0.261002 | -0.125049 | 2963.78 | 680.335 | 0.479113 | true | true | true |
| spike_slab | 0.7 | uniform | ellipsoidal_12 | 0 | 9995 | 8.96124 | 10.6289 | -18.3959 | -19.008 | 0.691539 | 0.402827 | -0.612132 | 1621.89 | 3745.18 | 1.51959 | true | true | true |
| spike_slab | 0.7 | uniform | ellipsoidal_12 | 17 | 10035 | 9.06215 | 10.7469 | -18.3959 | -18.5565 | 0.68635 | 0.361084 | -0.160585 | 1308.38 | 258.777 | 0.44473 | true | true | true |
| spike_slab | 0.7 | uniform | gmm_12 | 0 | 9995 | 9.08235 | 10.7787 | -18.3959 | -18.9885 | 0.691539 | 0.377349 | -0.592572 | 1423.21 | 3509.66 | 1.57036 | true | true | true |
| spike_slab | 0.7 | uniform | gmm_12 | 17 | 10035 | 8.94432 | 10.6145 | -18.3959 | -18.5427 | 0.68635 | 0.369472 | -0.14676 | 1369.87 | 216.138 | 0.397215 | true | true | true |
| spike_slab | 0.7 | evidence_improving | isotropic_24_stepout | 0 | 43875 | 8.57054 | 10.2738 | -18.3959 | -18.2867 | 0.677254 | 0.307834 | 0.109176 | 4157.69 | 522.966 | 0.354659 | true | true | true |
| spike_slab | 0.7 | evidence_improving | isotropic_24_stepout | 17 | 43507 | 8.63923 | 10.3758 | -18.3959 | -18.5295 | 0.690241 | 0.277624 | -0.133643 | 3353.31 | 777.053 | 0.481381 | true | true | true |
| spike_slab | 0.7 | evidence_improving | ellipsoidal_12 | 0 | 9995 | 9.47257 | 11.1514 | -18.3959 | -19.0168 | 0.691539 | 0.372021 | -0.620929 | 1383.31 | 3853.6 | 1.66907 | true | true | true |
| spike_slab | 0.7 | evidence_improving | ellipsoidal_12 | 17 | 10035 | 9.46365 | 11.1487 | -18.3959 | -18.567 | 0.68635 | 0.353183 | -0.171096 | 1251.75 | 293.761 | 0.484439 | true | true | true |
| spike_slab | 0.7 | evidence_improving | gmm_12 | 0 | 9995 | 9.39416 | 11.0721 | -18.3959 | -19.0075 | 0.691539 | 0.358555 | -0.611574 | 1284.97 | 3738.36 | 1.70566 | true | true | true |
| spike_slab | 0.7 | evidence_improving | gmm_12 | 17 | 10035 | 9.37081 | 11.0469 | -18.3959 | -18.5726 | 0.68635 | 0.366429 | -0.17674 | 1347.4 | 313.465 | 0.482331 | true | true | true |
| spike_slab | 0.7 | posterior_improving | isotropic_24_stepout | 0 | 43875 | 8.63673 | 10.3058 | -18.3959 | -18.2437 | 0.677254 | 0.296655 | 0.15224 | 3861.19 | 1016.89 | 0.513187 | true | true | true |
| spike_slab | 0.7 | posterior_improving | isotropic_24_stepout | 17 | 43507 | 8.67027 | 10.4285 | -18.3959 | -18.5135 | 0.690241 | 0.298911 | -0.117614 | 3887.26 | 601.836 | 0.393476 | true | true | true |
| spike_slab | 0.7 | posterior_improving | ellipsoidal_12 | 0 | 9995 | 9.53976 | 11.2982 | -18.3959 | -18.9909 | 0.691539 | 0.390967 | -0.595031 | 1527.79 | 3538.85 | 1.52195 | true | true | true |
| spike_slab | 0.7 | posterior_improving | ellipsoidal_12 | 17 | 10035 | 9.51357 | 11.1754 | -18.3959 | -18.5818 | 0.68635 | 0.342535 | -0.185897 | 1177.41 | 346.788 | 0.54271 | true | true | true |
| spike_slab | 0.7 | posterior_improving | gmm_12 | 0 | 9995 | 9.5588 | 11.2303 | -18.3959 | -18.9644 | 0.691539 | 0.358339 | -0.568541 | 1283.43 | 3230.77 | 1.5866 | true | true | true |
| spike_slab | 0.7 | posterior_improving | gmm_12 | 17 | 10035 | 9.51883 | 11.1957 | -18.3959 | -18.5404 | 0.68635 | 0.297467 | -0.14453 | 887.966 | 209.62 | 0.485867 | true | true | true |
| spike_slab | 0.5 | uniform | isotropic_24_stepout | 0 | 129824 | 12.6338 | 14.4335 | -18.3959 | -18.1003 | 0.636561 | 0.260351 | 0.295623 | 8799.78 | 11345.7 | 1.13548 | true | false | true |
| spike_slab | 0.5 | uniform | isotropic_24_stepout | 17 | 129137 | 12.542 | 14.3587 | -18.3959 | -17.8535 | 0.65122 | 0.269032 | 0.542352 | 9346.74 | 37985 | 2.01593 | true | false | true |
| spike_slab | 0.5 | uniform | ellipsoidal_12 | 0 | 39428 | 21.5905 | 23.3481 | -18.3959 | -18.1552 | 0.64948 | 0.386519 | 0.240663 | 5890.43 | 2283.61 | 0.622641 | true | false | true |
| spike_slab | 0.5 | uniform | ellipsoidal_12 | 17 | 41776 | 21.6059 | 23.3687 | -18.3959 | -17.7896 | 0.655171 | 0.32375 | 0.606285 | 4378.7 | 15356.1 | 1.8727 | true | false | true |
| spike_slab | 0.5 | uniform | gmm_12 | 0 | 39428 | 21.63 | 23.5171 | -18.3959 | -18.1879 | 0.64948 | 0.386418 | 0.208022 | 5887.35 | 1706.18 | 0.538335 | true | false | true |
| spike_slab | 0.5 | uniform | gmm_12 | 17 | 41776 | 21.8734 | 23.6424 | -18.3959 | -17.7285 | 0.655171 | 0.271333 | 0.667435 | 3075.61 | 18609.9 | 2.45984 | true | false | true |
| spike_slab | 0.5 | evidence_improving | isotropic_24_stepout | 0 | 129824 | 13.1377 | 14.9298 | -18.3959 | -18.1157 | 0.636561 | 0.24374 | 0.280152 | 7712.72 | 10189.2 | 1.14939 | true | false | true |
| spike_slab | 0.5 | evidence_improving | isotropic_24_stepout | 17 | 129137 | 13.0836 | 14.824 | -18.3959 | -17.8648 | 0.65122 | 0.247995 | 0.531131 | 7942.11 | 36429.6 | 2.1417 | true | false | true |
| spike_slab | 0.5 | evidence_improving | ellipsoidal_12 | 0 | 39428 | 22.1216 | 23.8306 | -18.3959 | -18.1591 | 0.64948 | 0.371652 | 0.236745 | 5445.99 | 2209.88 | 0.637009 | true | false | true |
| spike_slab | 0.5 | evidence_improving | ellipsoidal_12 | 17 | 41776 | 22.1647 | 23.8636 | -18.3959 | -17.738 | 0.655171 | 0.318811 | 0.657923 | 4246.14 | 18083.3 | 2.06368 | true | false | true |
| spike_slab | 0.5 | evidence_improving | gmm_12 | 0 | 39428 | 22.2453 | 23.996 | -18.3959 | -18.1303 | 0.64948 | 0.308894 | 0.265551 | 3762.04 | 2780.35 | 0.859682 | true | false | true |
| spike_slab | 0.5 | evidence_improving | gmm_12 | 17 | 41776 | 22.3126 | 24.0359 | -18.3959 | -17.7508 | 0.655171 | 0.319393 | 0.64505 | 4261.64 | 17382.6 | 2.01962 | true | false | true |
| spike_slab | 0.5 | posterior_improving | isotropic_24_stepout | 0 | 129824 | 13.2079 | 14.9742 | -18.3959 | -18.1187 | 0.636561 | 0.258571 | 0.277148 | 8679.88 | 9971.95 | 1.07185 | true | false | true |
| spike_slab | 0.5 | posterior_improving | isotropic_24_stepout | 17 | 129137 | 13.1805 | 14.9324 | -18.3959 | -17.8757 | 0.65122 | 0.269912 | 0.520203 | 9407.92 | 34945.9 | 1.92731 | true | false | true |
| spike_slab | 0.5 | posterior_improving | ellipsoidal_12 | 0 | 39428 | 22.5775 | 24.312 | -18.3959 | -18.2239 | 0.64948 | 0.345946 | 0.171944 | 4718.69 | 1165.67 | 0.497024 | true | false | true |
| spike_slab | 0.5 | posterior_improving | ellipsoidal_12 | 17 | 41776 | 22.4125 | 24.1471 | -18.3959 | -17.7459 | 0.655171 | 0.371957 | 0.649952 | 5779.78 | 17647.7 | 1.74739 | true | false | true |
| spike_slab | 0.5 | posterior_improving | gmm_12 | 0 | 39428 | 22.1034 | 23.8298 | -18.3959 | -18.1529 | 0.64948 | 0.339284 | 0.243036 | 4538.7 | 2328.88 | 0.716322 | true | false | true |
| spike_slab | 0.5 | posterior_improving | gmm_12 | 17 | 41776 | 22.5362 | 24.3308 | -18.3959 | -17.7184 | 0.655171 | 0.319684 | 0.677447 | 4269.42 | 19172.4 | 2.11911 | true | false | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: ellipsoidal_12/evidence_improving, ellipsoidal_12/uniform, ellipsoidal_12/uniform.
- `spike_slab` best rows by target were: ellipsoidal_12/uniform, isotropic_24_stepout/evidence_improving, isotropic_24_stepout/uniform.
- Some rows did not reach the requested uncertainty. Check `sample_cap_fraction` and `max_goal_iterations` before drawing strong efficiency conclusions.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.
- At least one row reached the uncertainty target while missing the analytic accuracy gate for every seed. Use the minimum-sample guard or stricter accuracy-gated tables when comparing `evals * variance`.
- GMM settings won 0 problem/target groups; compare against ellipsoidal rows to decide whether GMM fitting cost is buying lower variance.