# Pure-Core Efficiency Grid

Generated: `2026-05-12_160750 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn, spike_slab`
- Allocation targets: `uniform, evidence_improving, posterior_improving`
- LogZ uncertainty targets: `0.9, 0.7, 0.5, 0.35`
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
| basic_mvn | 0.35 | uniform | gmm_12 | 136110 | 0.663201 | 59865.8 | 12641.2 | 0 | 1 |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 136110 | 0.660757 | 59425.4 | 13176.5 | 0 | 1 |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 136110 | 0.659854 | 59263 | 11167.4 | 0 | 1 |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 25958 | 1.47554 | 56516 | 2615.04 | 1 | 0.5 |
| spike_slab | 0.35 | uniform | gmm_12 | 40602 | 0.46494 | 8776.9 | 5176.65 | 0 | 1 |
| spike_slab | 0.5 | uniform | ellipsoidal_12 | 40602 | 0.461248 | 8638.07 | 5134.57 | 0 | 1 |
| spike_slab | 0.7 | posterior_improving | gmm_12 | 10015 | 0.414806 | 1723.22 | 1085.7 | 1 | 1 |
| spike_slab | 0.9 | uniform | gmm_12 | 10015 | 0.407644 | 1664.23 | 1400.6 | 1 | 1 |

## Best Usable By Problem And Target

Rows here require `accuracy_success_fraction >= 0.5`; this prevents biased early stops from winning solely because the MC variance estimate is small.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.35 | uniform | gmm_12 | 136110 | 0.663201 | 59865.8 | 12641.2 | 0 | 1 |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 136110 | 0.660757 | 59425.4 | 13176.5 | 0 | 1 |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 136110 | 0.659854 | 59263 | 11167.4 | 0 | 1 |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 25958 | 1.47554 | 56516 | 2615.04 | 1 | 0.5 |
| spike_slab | 0.35 | uniform | gmm_12 | 40602 | 0.46494 | 8776.9 | 5176.65 | 0 | 1 |
| spike_slab | 0.5 | uniform | ellipsoidal_12 | 40602 | 0.461248 | 8638.07 | 5134.57 | 0 | 1 |
| spike_slab | 0.7 | posterior_improving | gmm_12 | 10015 | 0.414806 | 1723.22 | 1085.7 | 1 | 1 |
| spike_slab | 0.9 | uniform | gmm_12 | 10015 | 0.407644 | 1664.23 | 1400.6 | 1 | 1 |

## Best Strict By Problem And Target

Rows here require both `accuracy_success_fraction == 1` and `target_success_fraction == 1`.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spike_slab | 0.7 | posterior_improving | gmm_12 | 10015 | 0.414806 | 1723.22 | 1085.7 | 1 | 1 |
| spike_slab | 0.9 | uniform | gmm_12 | 10015 | 0.407644 | 1664.23 | 1400.6 | 1 | 1 |

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | min_sample_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.35 | evidence_improving | ellipsoidal_12 | 2 | 136110 | 22.1618 | 23.6697 | 0.686498 | -0.533949 | 0.0884598 | 12042 | 64145.6 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.35 | evidence_improving | gmm_12 | 2 | 136110 | 22.3143 | 23.8735 | 0.668944 | -0.516973 | 0.0908109 | 12415.2 | 60907.2 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.35 | posterior_improving | ellipsoidal_12 | 2 | 136110 | 22.3275 | 23.8971 | 0.684505 | -0.534458 | 0.105769 | 14464.9 | 63773.7 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.35 | posterior_improving | gmm_12 | 2 | 136110 | 22.3405 | 23.9156 | 0.663325 | -0.515627 | 0.112995 | 15441.6 | 59888.1 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.35 | uniform | ellipsoidal_12 | 2 | 136110 | 21.521 | 23.0569 | 0.694351 | -0.562003 | 0.0891208 | 12198 | 65621.5 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.35 | uniform | gmm_12 | 2 | 136110 | 21.5347 | 23.077 | 0.663201 | -0.546665 | 0.0926172 | 12641.2 | 59865.8 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 2 | 136110 | 22.3118 | 23.8932 | 0.660757 | -0.558255 | 0.0965918 | 13176.5 | 59425.4 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.5 | evidence_improving | gmm_12 | 2 | 136110 | 22.1803 | 23.7312 | 0.694959 | -0.57304 | 0.0827878 | 11323.5 | 65736.5 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.5 | posterior_improving | ellipsoidal_12 | 2 | 136110 | 22.3578 | 23.9181 | 0.669998 | -0.541016 | 0.102636 | 13967.5 | 61099.2 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.5 | posterior_improving | gmm_12 | 2 | 136110 | 22.2418 | 23.8261 | 0.676184 | -0.530719 | 0.0904006 | 12327 | 62232.7 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.5 | uniform | ellipsoidal_12 | 2 | 136110 | 21.5012 | 23.008 | 0.703008 | -0.547399 | 0.104338 | 14186.2 | 67268 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.5 | uniform | gmm_12 | 2 | 136110 | 21.4993 | 23.0444 | 0.723352 | -0.602354 | 0.0942158 | 12902 | 71217.7 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 2 | 136110 | 22.2616 | 23.7811 | 0.697244 | -0.546805 | 0.0930116 | 12678.3 | 66169.4 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.7 | evidence_improving | gmm_12 | 2 | 136110 | 22.218 | 23.7734 | 0.712615 | -0.56634 | 0.102093 | 13989.8 | 69119.1 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.7 | posterior_improving | ellipsoidal_12 | 2 | 136110 | 22.3868 | 23.9604 | 0.706397 | -0.555104 | 0.102891 | 14054.4 | 67918.1 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 2 | 136110 | 22.3703 | 23.9329 | 0.689833 | -0.56811 | 0.0953098 | 13007.6 | 64770.4 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 2 | 136110 | 21.4184 | 22.96 | 0.659854 | -0.541804 | 0.0815063 | 11167.4 | 59263 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.7 | uniform | gmm_12 | 2 | 136110 | 21.566 | 23.093 | 0.686665 | -0.560632 | 0.109674 | 14940.1 | 64176.8 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.9 | evidence_improving | ellipsoidal_12 | 2 | 25958 | 8.14541 | 9.67806 | 1.49863 | -1.30113 | 0.104789 | 2722.43 | 58299.2 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | evidence_improving | gmm_12 | 2 | 25958 | 8.10532 | 9.64665 | 1.50047 | -1.27469 | 0.10377 | 2656.51 | 58441.8 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | posterior_improving | ellipsoidal_12 | 2 | 25958 | 8.23783 | 9.78564 | 1.5064 | -1.28478 | 0.121439 | 3150.75 | 58904.6 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 2 | 25958 | 8.25015 | 9.79742 | 1.52883 | -1.32087 | 0.108114 | 2765.92 | 60672.4 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 2 | 25958 | 7.67184 | 9.21964 | 1.47554 | -1.25593 | 0.101362 | 2615.04 | 56516 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | uniform | gmm_12 | 2 | 25958 | 7.6929 | 9.2296 | 1.47754 | -1.26582 | 0.0983136 | 2539.59 | 56669.7 | 1 | 0.5 | 1 | 0 |
| spike_slab | 0.35 | evidence_improving | ellipsoidal_12 | 2 | 40602 | 22.0847 | 23.7934 | 0.505407 | 0.464827 | 0.107525 | 4356.58 | 10371.2 | 0 | 1 | 1 | 1 |
| spike_slab | 0.35 | evidence_improving | gmm_12 | 2 | 40602 | 22.3056 | 24.0469 | 0.507258 | 0.468903 | 0.119785 | 4854.93 | 10447.3 | 0 | 1 | 1 | 1 |
| spike_slab | 0.35 | posterior_improving | ellipsoidal_12 | 2 | 40602 | 22.3672 | 24.1078 | 0.513278 | 0.471629 | 0.13895 | 5658.81 | 10696.8 | 0 | 1 | 1 | 1 |
| spike_slab | 0.35 | posterior_improving | gmm_12 | 2 | 40602 | 22.233 | 23.9599 | 0.536741 | 0.504444 | 0.119167 | 4837.14 | 11697 | 0 | 1 | 1 | 1 |
| spike_slab | 0.35 | uniform | ellipsoidal_12 | 2 | 40602 | 21.599 | 23.3369 | 0.490049 | 0.439483 | 0.128585 | 5197.41 | 9750.48 | 0 | 1 | 1 | 1 |
| spike_slab | 0.35 | uniform | gmm_12 | 2 | 40602 | 21.6166 | 23.3507 | 0.46494 | 0.416607 | 0.127697 | 5176.65 | 8776.9 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | evidence_improving | ellipsoidal_12 | 2 | 40602 | 22.3902 | 24.1088 | 0.494425 | 0.447334 | 0.119883 | 4846.07 | 9925.4 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | evidence_improving | gmm_12 | 2 | 40602 | 22.3503 | 24.098 | 0.493258 | 0.455301 | 0.0987136 | 4011.84 | 9878.61 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | posterior_improving | ellipsoidal_12 | 2 | 40602 | 22.3424 | 24.0636 | 0.475396 | 0.410948 | 0.129015 | 5249.23 | 9176.09 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | posterior_improving | gmm_12 | 2 | 40602 | 22.2873 | 24.0151 | 0.508921 | 0.460242 | 0.108656 | 4404.06 | 10515.9 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | uniform | ellipsoidal_12 | 2 | 40602 | 21.6492 | 23.4131 | 0.461248 | 0.423474 | 0.127105 | 5134.57 | 8638.07 | 0 | 1 | 1 | 1 |
| spike_slab | 0.5 | uniform | gmm_12 | 2 | 40602 | 21.6797 | 23.4382 | 0.494339 | 0.437728 | 0.11147 | 4481.48 | 9921.96 | 0 | 1 | 1 | 1 |
| spike_slab | 0.7 | evidence_improving | ellipsoidal_12 | 2 | 10015 | 9.43062 | 11.1204 | 0.455427 | -0.396012 | 0.131569 | 1317.53 | 2077.24 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | evidence_improving | gmm_12 | 2 | 10015 | 9.47527 | 11.1743 | 0.450144 | -0.394157 | 0.131416 | 1316.19 | 2029.34 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | posterior_improving | ellipsoidal_12 | 2 | 10015 | 9.56308 | 11.2698 | 0.440806 | -0.390464 | 0.135093 | 1352.6 | 1946.02 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | posterior_improving | gmm_12 | 2 | 10015 | 9.59296 | 11.3408 | 0.414806 | -0.356535 | 0.108447 | 1085.7 | 1723.22 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | uniform | ellipsoidal_12 | 2 | 10015 | 9.00728 | 10.6906 | 0.447489 | -0.386358 | 0.146326 | 1465.13 | 2005.47 | 1 | 1 | 1 | 0 |
| spike_slab | 0.7 | uniform | gmm_12 | 2 | 10015 | 9.10671 | 10.7989 | 0.431671 | -0.369666 | 0.139451 | 1396.54 | 1866.2 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | evidence_improving | ellipsoidal_12 | 2 | 10015 | 9.43327 | 11.1101 | 0.475129 | -0.411693 | 0.129646 | 1298.09 | 2260.87 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 2 | 10015 | 9.44672 | 11.2291 | 0.448404 | -0.379947 | 0.128544 | 1287.3 | 2013.68 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | posterior_improving | ellipsoidal_12 | 2 | 10015 | 9.55975 | 11.2221 | 0.429332 | -0.376969 | 0.140015 | 1402.25 | 1846.02 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | posterior_improving | gmm_12 | 2 | 10015 | 9.56631 | 11.2596 | 0.413737 | -0.359498 | 0.135084 | 1352.36 | 1714.35 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | uniform | ellipsoidal_12 | 2 | 10015 | 9.01571 | 10.702 | 0.454186 | -0.396272 | 0.125687 | 1258.75 | 2065.94 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | uniform | gmm_12 | 2 | 10015 | 8.97678 | 10.6795 | 0.407644 | -0.344457 | 0.13986 | 1400.6 | 1664.23 | 1 | 1 | 1 | 0 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok | min_samples_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 0 | 27555 | 7.55422 | 9.09535 | -24.6065 | -26.6369 | 0.85652 | 0.302104 | -2.03043 | 2514.85 | 113599 | 6.72097 | false | true | true |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 17 | 24361 | 7.78946 | 9.34393 | -24.6065 | -25.0879 | 0.78093 | 0.333853 | -0.481429 | 2715.23 | 5646.24 | 1.44204 | true | true | true |
| basic_mvn | 0.9 | uniform | gmm_12 | 0 | 27555 | 7.52157 | 9.02712 | -24.6065 | -26.6344 | 0.85652 | 0.300882 | -2.02794 | 2494.55 | 113321 | 6.74 | false | true | true |
| basic_mvn | 0.9 | uniform | gmm_12 | 17 | 24361 | 7.86424 | 9.43208 | -24.6065 | -25.1102 | 0.78093 | 0.325726 | -0.503699 | 2584.64 | 6180.7 | 1.54639 | true | true | true |
| basic_mvn | 0.9 | evidence_improving | ellipsoidal_12 | 0 | 27555 | 8.07377 | 9.61475 | -24.6065 | -26.6512 | 0.85652 | 0.325932 | -2.04475 | 2927.21 | 115207 | 6.27354 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | ellipsoidal_12 | 17 | 24361 | 8.21705 | 9.74137 | -24.6065 | -25.164 | 0.78093 | 0.321477 | -0.557515 | 2517.64 | 7571.95 | 1.73423 | true | true | true |
| basic_mvn | 0.9 | evidence_improving | gmm_12 | 0 | 27555 | 7.98895 | 9.52338 | -24.6065 | -26.6727 | 0.85652 | 0.283725 | -2.06625 | 2218.18 | 117643 | 7.28257 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | gmm_12 | 17 | 24361 | 8.22169 | 9.76992 | -24.6065 | -25.0896 | 0.78093 | 0.356428 | -0.483125 | 3094.84 | 5686.09 | 1.35546 | true | true | true |
| basic_mvn | 0.9 | posterior_improving | ellipsoidal_12 | 0 | 27555 | 8.09219 | 9.62866 | -24.6065 | -26.6777 | 0.85652 | 0.347073 | -2.07127 | 3319.27 | 118216 | 5.96783 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | ellipsoidal_12 | 17 | 24361 | 8.38346 | 9.94262 | -24.6065 | -25.1047 | 0.78093 | 0.349883 | -0.498284 | 2982.23 | 6048.51 | 1.42414 | true | true | true |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 0 | 27555 | 8.1097 | 9.64899 | -24.6065 | -26.6972 | 0.85652 | 0.287674 | -2.0907 | 2280.35 | 120444 | 7.2676 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 17 | 24361 | 8.3906 | 9.94585 | -24.6065 | -25.1575 | 0.78093 | 0.365337 | -0.551031 | 3251.49 | 7396.86 | 1.50828 | true | true | true |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 0 | 141166 | 21.3172 | 22.8526 | -24.6065 | -25.5249 | 0.70999 | 0.309958 | -0.918441 | 13562.4 | 119078 | 2.96311 | true | false | true |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 17 | 131053 | 21.5196 | 23.0675 | -24.6065 | -24.7716 | 0.720743 | 0.258725 | -0.165167 | 8772.51 | 3575.15 | 0.638388 | true | false | true |
| basic_mvn | 0.7 | uniform | gmm_12 | 0 | 141166 | 21.4181 | 22.912 | -24.6065 | -25.5636 | 0.70999 | 0.334861 | -0.957117 | 15829.2 | 129318 | 2.85825 | true | false | true |
| basic_mvn | 0.7 | uniform | gmm_12 | 17 | 131053 | 21.7139 | 23.274 | -24.6065 | -24.7706 | 0.720743 | 0.32744 | -0.164147 | 14051.1 | 3531.12 | 0.501304 | true | false | true |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 0 | 141166 | 22.3615 | 23.8586 | -24.6065 | -25.5859 | 0.70999 | 0.310926 | -0.979417 | 13647.3 | 135414 | 3.15 | false | false | true |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 17 | 131053 | 22.1617 | 23.7036 | -24.6065 | -24.7207 | 0.720743 | 0.298911 | -0.114194 | 11709.3 | 1708.97 | 0.382033 | true | false | true |
| basic_mvn | 0.7 | evidence_improving | gmm_12 | 0 | 141166 | 22.1866 | 23.6823 | -24.6065 | -25.6053 | 0.70999 | 0.347402 | -0.998866 | 17037.1 | 140846 | 2.87525 | true | false | true |
| basic_mvn | 0.7 | evidence_improving | gmm_12 | 17 | 131053 | 22.2494 | 23.8645 | -24.6065 | -24.7403 | 0.720743 | 0.288959 | -0.133814 | 10942.6 | 2346.66 | 0.46309 | true | false | true |
| basic_mvn | 0.7 | posterior_improving | ellipsoidal_12 | 0 | 141166 | 22.3587 | 23.8856 | -24.6065 | -25.5984 | 0.70999 | 0.335809 | -0.991974 | 15919 | 138909 | 2.95398 | true | false | true |
| basic_mvn | 0.7 | posterior_improving | ellipsoidal_12 | 17 | 131053 | 22.415 | 24.0353 | -24.6065 | -24.7247 | 0.720743 | 0.304983 | -0.118233 | 12189.9 | 1832 | 0.387672 | true | false | true |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 0 | 141166 | 22.2589 | 23.7883 | -24.6065 | -25.5659 | 0.70999 | 0.319759 | -0.959416 | 14433.6 | 129940 | 3.00044 | false | false | true |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 17 | 131053 | 22.4816 | 24.0774 | -24.6065 | -24.7833 | 0.720743 | 0.297278 | -0.176805 | 11581.7 | 4096.71 | 0.594746 | true | false | true |
| basic_mvn | 0.5 | uniform | ellipsoidal_12 | 0 | 141166 | 21.5975 | 23.1262 | -24.6065 | -25.595 | 0.70999 | 0.318325 | -0.988504 | 14304.5 | 137939 | 3.10533 | false | false | true |
| basic_mvn | 0.5 | uniform | ellipsoidal_12 | 17 | 131053 | 21.4049 | 22.8898 | -24.6065 | -24.7128 | 0.720743 | 0.327635 | -0.106293 | 14067.8 | 1480.67 | 0.324426 | true | false | true |
| basic_mvn | 0.5 | uniform | gmm_12 | 0 | 141166 | 21.4129 | 22.9635 | -24.6065 | -25.6093 | 0.70999 | 0.331221 | -1.00286 | 15487 | 141976 | 3.02778 | false | false | true |
| basic_mvn | 0.5 | uniform | gmm_12 | 17 | 131053 | 21.5856 | 23.1253 | -24.6065 | -24.8083 | 0.720743 | 0.280578 | -0.201843 | 10317 | 5339.19 | 0.719383 | true | false | true |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 0 | 141166 | 22.1757 | 23.6706 | -24.6065 | -25.5182 | 0.70999 | 0.320036 | -0.911739 | 14458.7 | 117347 | 2.84886 | true | false | true |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 17 | 131053 | 22.4479 | 24.1159 | -24.6065 | -24.8112 | 0.720743 | 0.301265 | -0.204771 | 11894.4 | 5495.18 | 0.679703 | true | false | true |
| basic_mvn | 0.5 | evidence_improving | gmm_12 | 0 | 141166 | 22.1668 | 23.7508 | -24.6065 | -25.5727 | 0.70999 | 0.306158 | -0.966223 | 13231.9 | 131791 | 3.15596 | false | false | true |
| basic_mvn | 0.5 | evidence_improving | gmm_12 | 17 | 131053 | 22.1937 | 23.7116 | -24.6065 | -24.7863 | 0.720743 | 0.268035 | -0.179856 | 9415.23 | 4239.34 | 0.671017 | true | false | true |
| basic_mvn | 0.5 | posterior_improving | ellipsoidal_12 | 0 | 141166 | 22.2705 | 23.7916 | -24.6065 | -25.5427 | 0.70999 | 0.319687 | -0.936236 | 14427.2 | 123737 | 2.9286 | true | false | true |
| basic_mvn | 0.5 | posterior_improving | ellipsoidal_12 | 17 | 131053 | 22.4452 | 24.0447 | -24.6065 | -24.7523 | 0.720743 | 0.321048 | -0.145797 | 13507.9 | 2785.75 | 0.454127 | true | false | true |
| basic_mvn | 0.5 | posterior_improving | gmm_12 | 0 | 141166 | 22.1836 | 23.7403 | -24.6065 | -25.5562 | 0.70999 | 0.308022 | -0.94972 | 13393.5 | 127327 | 3.08328 | false | false | true |
| basic_mvn | 0.5 | posterior_improving | gmm_12 | 17 | 131053 | 22.3001 | 23.912 | -24.6065 | -24.7182 | 0.720743 | 0.293127 | -0.111718 | 11260.5 | 1635.67 | 0.381126 | true | false | true |
| basic_mvn | 0.35 | uniform | ellipsoidal_12 | 0 | 141166 | 21.4256 | 22.9698 | -24.6065 | -25.5762 | 0.70999 | 0.320215 | -0.969772 | 14474.8 | 132761 | 3.0285 | false | false | true |
| basic_mvn | 0.35 | uniform | ellipsoidal_12 | 17 | 131053 | 21.6163 | 23.1439 | -24.6065 | -24.7607 | 0.720743 | 0.275143 | -0.154233 | 9921.22 | 3117.47 | 0.560556 | true | false | true |
| basic_mvn | 0.35 | uniform | gmm_12 | 0 | 141166 | 21.5048 | 23.0669 | -24.6065 | -25.5286 | 0.70999 | 0.315525 | -0.922156 | 14053.9 | 120044 | 2.92261 | true | false | true |
| basic_mvn | 0.35 | uniform | gmm_12 | 17 | 131053 | 21.5646 | 23.0871 | -24.6065 | -24.7776 | 0.720743 | 0.292709 | -0.171173 | 11228.4 | 3839.9 | 0.584792 | true | false | true |
| basic_mvn | 0.35 | evidence_improving | ellipsoidal_12 | 0 | 141166 | 22.0754 | 23.5808 | -24.6065 | -25.5719 | 0.70999 | 0.298022 | -0.965432 | 12538 | 131575 | 3.23946 | false | false | true |
| basic_mvn | 0.35 | evidence_improving | ellipsoidal_12 | 17 | 131053 | 22.2482 | 23.7587 | -24.6065 | -24.7089 | 0.720743 | 0.29682 | -0.102465 | 11546 | 1375.93 | 0.345209 | true | false | true |
| basic_mvn | 0.35 | evidence_improving | gmm_12 | 0 | 141166 | 22.2742 | 23.8557 | -24.6065 | -25.548 | 0.70999 | 0.318893 | -0.941502 | 14355.6 | 125133 | 2.95241 | true | false | true |
| basic_mvn | 0.35 | evidence_improving | gmm_12 | 17 | 131053 | 22.3544 | 23.8913 | -24.6065 | -24.6989 | 0.720743 | 0.282717 | -0.0924444 | 10474.9 | 1119.97 | 0.326985 | true | false | true |
| basic_mvn | 0.35 | posterior_improving | ellipsoidal_12 | 0 | 141166 | 22.2142 | 23.7195 | -24.6065 | -25.5686 | 0.70999 | 0.345502 | -0.962128 | 16851.2 | 130676 | 2.78473 | true | false | true |
| basic_mvn | 0.35 | posterior_improving | ellipsoidal_12 | 17 | 131053 | 22.4407 | 24.0747 | -24.6065 | -24.7133 | 0.720743 | 0.303588 | -0.106788 | 12078.6 | 1494.49 | 0.351753 | true | false | true |
| basic_mvn | 0.35 | posterior_improving | gmm_12 | 0 | 141166 | 22.2197 | 23.7396 | -24.6065 | -25.5394 | 0.70999 | 0.353878 | -0.932914 | 17678.2 | 122861 | 2.63626 | true | false | true |
| basic_mvn | 0.35 | posterior_improving | gmm_12 | 17 | 131053 | 22.4613 | 24.0916 | -24.6065 | -24.7048 | 0.720743 | 0.317427 | -0.0983391 | 13204.9 | 1267.36 | 0.309801 | true | false | true |
| spike_slab | 0.9 | uniform | ellipsoidal_12 | 0 | 9995 | 9.02036 | 10.7222 | -18.3959 | -19.0141 | 0.691539 | 0.355291 | -0.618202 | 1261.68 | 3819.83 | 1.73999 | true | true | true |
| spike_slab | 0.9 | uniform | ellipsoidal_12 | 17 | 10035 | 9.01106 | 10.6818 | -18.3959 | -18.5702 | 0.68635 | 0.353756 | -0.174342 | 1255.81 | 305.017 | 0.492832 | true | true | true |
| spike_slab | 0.9 | uniform | gmm_12 | 0 | 9995 | 8.97764 | 10.7213 | -18.3959 | -18.9583 | 0.691539 | 0.380709 | -0.562455 | 1448.67 | 3161.98 | 1.47739 | true | true | true |
| spike_slab | 0.9 | uniform | gmm_12 | 17 | 10035 | 8.97592 | 10.6378 | -18.3959 | -18.5223 | 0.68635 | 0.367126 | -0.126459 | 1352.53 | 160.479 | 0.344457 | true | true | true |
| spike_slab | 0.9 | evidence_improving | ellipsoidal_12 | 0 | 9995 | 9.45914 | 11.1202 | -18.3959 | -19.0448 | 0.691539 | 0.381727 | -0.648878 | 1456.42 | 4208.32 | 1.69985 | true | true | true |
| spike_slab | 0.9 | evidence_improving | ellipsoidal_12 | 17 | 10035 | 9.4074 | 11.1 | -18.3959 | -18.5704 | 0.68635 | 0.337013 | -0.174508 | 1139.75 | 305.597 | 0.517809 | true | true | true |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 0 | 9995 | 9.44421 | 11.1881 | -18.3959 | -19.014 | 0.691539 | 0.363461 | -0.618078 | 1320.38 | 3818.3 | 1.70054 | true | true | true |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 17 | 10035 | 9.44923 | 11.27 | -18.3959 | -18.5377 | 0.68635 | 0.353532 | -0.141816 | 1254.22 | 201.822 | 0.401141 | true | true | true |
| spike_slab | 0.9 | posterior_improving | ellipsoidal_12 | 0 | 9995 | 9.56605 | 11.21 | -18.3959 | -18.9783 | 0.691539 | 0.374409 | -0.582444 | 1401.12 | 3390.71 | 1.55563 | true | true | true |
| spike_slab | 0.9 | posterior_improving | ellipsoidal_12 | 17 | 10035 | 9.55346 | 11.2341 | -18.3959 | -18.5674 | 0.68635 | 0.373963 | -0.171495 | 1403.38 | 295.133 | 0.458587 | true | true | true |
| spike_slab | 0.9 | posterior_improving | gmm_12 | 0 | 9995 | 9.56688 | 11.2629 | -18.3959 | -18.9602 | 0.691539 | 0.400451 | -0.56429 | 1602.81 | 3182.64 | 1.40913 | true | true | true |
| spike_slab | 0.9 | posterior_improving | gmm_12 | 17 | 10035 | 9.56574 | 11.2563 | -18.3959 | -18.5506 | 0.68635 | 0.331371 | -0.154707 | 1101.91 | 240.181 | 0.46687 | true | true | true |
| spike_slab | 0.7 | uniform | ellipsoidal_12 | 0 | 9995 | 9.0405 | 10.7227 | -18.3959 | -19.008 | 0.691539 | 0.402827 | -0.612132 | 1621.89 | 3745.18 | 1.51959 | true | true | true |
| spike_slab | 0.7 | uniform | ellipsoidal_12 | 17 | 10035 | 8.97406 | 10.6585 | -18.3959 | -18.5565 | 0.68635 | 0.361084 | -0.160585 | 1308.38 | 258.777 | 0.44473 | true | true | true |
| spike_slab | 0.7 | uniform | gmm_12 | 0 | 9995 | 9.15814 | 10.8819 | -18.3959 | -18.9885 | 0.691539 | 0.377349 | -0.592572 | 1423.21 | 3509.66 | 1.57036 | true | true | true |
| spike_slab | 0.7 | uniform | gmm_12 | 17 | 10035 | 9.05527 | 10.7159 | -18.3959 | -18.5427 | 0.68635 | 0.369472 | -0.14676 | 1369.87 | 216.138 | 0.397215 | true | true | true |
| spike_slab | 0.7 | evidence_improving | ellipsoidal_12 | 0 | 9995 | 9.43663 | 11.1381 | -18.3959 | -19.0168 | 0.691539 | 0.372021 | -0.620929 | 1383.31 | 3853.6 | 1.66907 | true | true | true |
| spike_slab | 0.7 | evidence_improving | ellipsoidal_12 | 17 | 10035 | 9.42461 | 11.1028 | -18.3959 | -18.567 | 0.68635 | 0.353183 | -0.171096 | 1251.75 | 293.761 | 0.484439 | true | true | true |
| spike_slab | 0.7 | evidence_improving | gmm_12 | 0 | 9995 | 9.45897 | 11.1525 | -18.3959 | -19.0075 | 0.691539 | 0.358555 | -0.611574 | 1284.97 | 3738.36 | 1.70566 | true | true | true |
| spike_slab | 0.7 | evidence_improving | gmm_12 | 17 | 10035 | 9.49157 | 11.196 | -18.3959 | -18.5726 | 0.68635 | 0.366429 | -0.17674 | 1347.4 | 313.465 | 0.482331 | true | true | true |
| spike_slab | 0.7 | posterior_improving | ellipsoidal_12 | 0 | 9995 | 9.51045 | 11.1616 | -18.3959 | -18.9909 | 0.691539 | 0.390967 | -0.595031 | 1527.79 | 3538.85 | 1.52195 | true | true | true |
| spike_slab | 0.7 | posterior_improving | ellipsoidal_12 | 17 | 10035 | 9.61571 | 11.378 | -18.3959 | -18.5818 | 0.68635 | 0.342535 | -0.185897 | 1177.41 | 346.788 | 0.54271 | true | true | true |
| spike_slab | 0.7 | posterior_improving | gmm_12 | 0 | 9995 | 9.57332 | 11.3769 | -18.3959 | -18.9644 | 0.691539 | 0.358339 | -0.568541 | 1283.43 | 3230.77 | 1.5866 | true | true | true |
| spike_slab | 0.7 | posterior_improving | gmm_12 | 17 | 10035 | 9.6126 | 11.3047 | -18.3959 | -18.5404 | 0.68635 | 0.297467 | -0.14453 | 887.966 | 209.62 | 0.485867 | true | true | true |
| spike_slab | 0.5 | uniform | ellipsoidal_12 | 0 | 39428 | 21.6724 | 23.4217 | -18.3959 | -18.1552 | 0.64948 | 0.386519 | 0.240663 | 5890.43 | 2283.61 | 0.622641 | true | false | true |
| spike_slab | 0.5 | uniform | ellipsoidal_12 | 17 | 41776 | 21.626 | 23.4046 | -18.3959 | -17.7896 | 0.655171 | 0.32375 | 0.606285 | 4378.7 | 15356.1 | 1.8727 | true | false | true |
| spike_slab | 0.5 | uniform | gmm_12 | 0 | 39428 | 21.6855 | 23.4598 | -18.3959 | -18.1879 | 0.64948 | 0.386418 | 0.208022 | 5887.35 | 1706.18 | 0.538335 | true | false | true |
| spike_slab | 0.5 | uniform | gmm_12 | 17 | 41776 | 21.6739 | 23.4166 | -18.3959 | -17.7285 | 0.655171 | 0.271333 | 0.667435 | 3075.61 | 18609.9 | 2.45984 | true | false | true |
| spike_slab | 0.5 | evidence_improving | ellipsoidal_12 | 0 | 39428 | 22.376 | 24.115 | -18.3959 | -18.1591 | 0.64948 | 0.371652 | 0.236745 | 5445.99 | 2209.88 | 0.637009 | true | false | true |
| spike_slab | 0.5 | evidence_improving | ellipsoidal_12 | 17 | 41776 | 22.4044 | 24.1026 | -18.3959 | -17.738 | 0.655171 | 0.318811 | 0.657923 | 4246.14 | 18083.3 | 2.06368 | true | false | true |
| spike_slab | 0.5 | evidence_improving | gmm_12 | 0 | 39428 | 22.3323 | 24.0756 | -18.3959 | -18.1303 | 0.64948 | 0.308894 | 0.265551 | 3762.04 | 2780.35 | 0.859682 | true | false | true |
| spike_slab | 0.5 | evidence_improving | gmm_12 | 17 | 41776 | 22.3684 | 24.1203 | -18.3959 | -17.7508 | 0.655171 | 0.319393 | 0.64505 | 4261.64 | 17382.6 | 2.01962 | true | false | true |
| spike_slab | 0.5 | posterior_improving | ellipsoidal_12 | 0 | 39428 | 22.4585 | 24.1974 | -18.3959 | -18.2239 | 0.64948 | 0.345946 | 0.171944 | 4718.69 | 1165.67 | 0.497024 | true | false | true |
| spike_slab | 0.5 | posterior_improving | ellipsoidal_12 | 17 | 41776 | 22.2263 | 23.9299 | -18.3959 | -17.7459 | 0.655171 | 0.371957 | 0.649952 | 5779.78 | 17647.7 | 1.74739 | true | false | true |
| spike_slab | 0.5 | posterior_improving | gmm_12 | 0 | 39428 | 22.2908 | 24.0345 | -18.3959 | -18.1529 | 0.64948 | 0.339284 | 0.243036 | 4538.7 | 2328.88 | 0.716322 | true | false | true |
| spike_slab | 0.5 | posterior_improving | gmm_12 | 17 | 41776 | 22.2838 | 23.9958 | -18.3959 | -17.7184 | 0.655171 | 0.319684 | 0.677447 | 4269.42 | 19172.4 | 2.11911 | true | false | true |
| spike_slab | 0.35 | uniform | ellipsoidal_12 | 0 | 39428 | 21.5405 | 23.2753 | -18.3959 | -18.1732 | 0.64948 | 0.385362 | 0.222682 | 5855.2 | 1955.13 | 0.577853 | true | false | true |
| spike_slab | 0.35 | uniform | ellipsoidal_12 | 17 | 41776 | 21.6575 | 23.3984 | -18.3959 | -17.7396 | 0.655171 | 0.329645 | 0.656284 | 4539.63 | 17993.3 | 1.99088 | true | false | true |
| spike_slab | 0.35 | uniform | gmm_12 | 0 | 39428 | 21.7112 | 23.4294 | -18.3959 | -18.1857 | 0.64948 | 0.36687 | 0.210191 | 5306.77 | 1741.94 | 0.572929 | true | false | true |
| spike_slab | 0.35 | uniform | gmm_12 | 17 | 41776 | 21.5219 | 23.2721 | -18.3959 | -17.7729 | 0.655171 | 0.347562 | 0.623024 | 5046.53 | 16215.7 | 1.79255 | true | false | true |
| spike_slab | 0.35 | evidence_improving | ellipsoidal_12 | 0 | 39428 | 22.1155 | 23.8236 | -18.3959 | -18.1295 | 0.64948 | 0.33958 | 0.266403 | 4546.61 | 2798.23 | 0.784509 | true | false | true |
| spike_slab | 0.35 | evidence_improving | ellipsoidal_12 | 17 | 41776 | 22.054 | 23.7631 | -18.3959 | -17.7326 | 0.655171 | 0.315809 | 0.663251 | 4166.54 | 18377.3 | 2.10016 | true | false | true |
| spike_slab | 0.35 | evidence_improving | gmm_12 | 0 | 39428 | 22.2579 | 24.0031 | -18.3959 | -18.1205 | 0.64948 | 0.356508 | 0.275406 | 5011.23 | 2990.56 | 0.77251 | true | false | true |
| spike_slab | 0.35 | evidence_improving | gmm_12 | 17 | 41776 | 22.3533 | 24.0908 | -18.3959 | -17.7335 | 0.655171 | 0.335369 | 0.6624 | 4698.64 | 18330.2 | 1.97514 | true | false | true |
| spike_slab | 0.35 | posterior_improving | ellipsoidal_12 | 0 | 39428 | 22.3415 | 24.0763 | -18.3959 | -18.1268 | 0.64948 | 0.352609 | 0.269096 | 4902.2 | 2855.09 | 0.763157 | true | false | true |
| spike_slab | 0.35 | posterior_improving | ellipsoidal_12 | 17 | 41776 | 22.3929 | 24.1393 | -18.3959 | -17.7217 | 0.655171 | 0.391876 | 0.674163 | 6415.41 | 18987 | 1.72035 | true | false | true |
| spike_slab | 0.35 | posterior_improving | gmm_12 | 0 | 39428 | 22.1128 | 23.8312 | -18.3959 | -18.0748 | 0.64948 | 0.346754 | 0.321067 | 4740.75 | 4064.39 | 0.925922 | true | false | true |
| spike_slab | 0.35 | posterior_improving | gmm_12 | 17 | 41776 | 22.3532 | 24.0887 | -18.3959 | -17.7081 | 0.655171 | 0.343649 | 0.687821 | 4933.54 | 19764.1 | 2.00152 | true | false | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: gmm_12/uniform, ellipsoidal_12/evidence_improving, ellipsoidal_12/uniform, ellipsoidal_12/uniform.
- `spike_slab` best rows by target were: gmm_12/uniform, ellipsoidal_12/uniform, gmm_12/posterior_improving, gmm_12/uniform.
- Some rows did not reach the requested uncertainty. Check `sample_cap_fraction` and `max_goal_iterations` before drawing strong efficiency conclusions.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.
- GMM settings won 4 problem/target groups; compare against ellipsoidal rows to decide whether GMM fitting cost is buying lower variance.