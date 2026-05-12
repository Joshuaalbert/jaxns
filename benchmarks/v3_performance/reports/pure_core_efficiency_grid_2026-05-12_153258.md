# Pure-Core Efficiency Grid

Generated: `2026-05-12_153258 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn, spike_slab`
- Allocation targets: `uniform, evidence_improving, posterior_improving`
- LogZ uncertainty targets: `0.9`
- Seeds: `0, 17`
- Target live points: `40`
- Max samples: `960`
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
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 25958 | 1.47554 | 56516 | 2615.04 | 1 | 0.5 |
| spike_slab | 0.9 | uniform | gmm_12 | 10015 | 0.407644 | 1664.23 | 1400.6 | 1 | 1 |

## Best Usable By Problem And Target

Rows here require `accuracy_success_fraction >= 0.5`; this prevents biased early stops from winning solely because the MC variance estimate is small.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 25958 | 1.47554 | 56516 | 2615.04 | 1 | 0.5 |
| spike_slab | 0.9 | uniform | gmm_12 | 10015 | 0.407644 | 1664.23 | 1400.6 | 1 | 1 |

## Best Strict By Problem And Target

Rows here require both `accuracy_success_fraction == 1` and `target_success_fraction == 1`.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spike_slab | 0.9 | uniform | gmm_12 | 10015 | 0.407644 | 1664.23 | 1400.6 | 1 | 1 |

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | min_sample_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | evidence_improving | ellipsoidal_12 | 2 | 25958 | 7.79147 | 9.33276 | 1.49863 | -1.30113 | 0.104789 | 2722.43 | 58299.2 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | evidence_improving | gmm_12 | 2 | 25958 | 7.80754 | 9.34945 | 1.50047 | -1.27469 | 0.10377 | 2656.51 | 58441.8 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | evidence_improving | isotropic_12 | 2 | 25402.5 | 7.5093 | 9.04393 | 1.74357 | -1.72326 | 0.0971976 | 2484.66 | 77224.1 | 1 | 0 | 1 | 0 |
| basic_mvn | 0.9 | posterior_improving | ellipsoidal_12 | 2 | 25958 | 7.90438 | 9.45789 | 1.5064 | -1.28478 | 0.121439 | 3150.75 | 58904.6 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 2 | 25958 | 7.94706 | 9.47786 | 1.52883 | -1.32087 | 0.108114 | 2765.92 | 60672.4 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | posterior_improving | isotropic_12 | 2 | 25402.5 | 7.56651 | 9.11936 | 1.76233 | -1.74428 | 0.10635 | 2706.88 | 78895.5 | 1 | 0 | 1 | 0 |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 2 | 25958 | 7.45022 | 9.00895 | 1.47554 | -1.25593 | 0.101362 | 2615.04 | 56516 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | uniform | gmm_12 | 2 | 25958 | 7.41726 | 8.98693 | 1.47754 | -1.26582 | 0.0983136 | 2539.59 | 56669.7 | 1 | 0.5 | 1 | 0 |
| basic_mvn | 0.9 | uniform | isotropic_12 | 2 | 25402.5 | 7.08301 | 8.61505 | 1.76612 | -1.75148 | 0.124496 | 3168.28 | 79234.5 | 1 | 0 | 1 | 0 |
| spike_slab | 0.9 | evidence_improving | ellipsoidal_12 | 2 | 10015 | 9.2117 | 10.8944 | 0.475129 | -0.411693 | 0.129646 | 1298.09 | 2260.87 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 2 | 10015 | 9.25513 | 10.9093 | 0.448404 | -0.379947 | 0.128544 | 1287.3 | 2013.68 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | evidence_improving | isotropic_12 | 2 | 10036 | 7.50045 | 9.17459 | 0.997302 | -0.918081 | 0.178904 | 1790.56 | 9981.92 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | posterior_improving | ellipsoidal_12 | 2 | 10015 | 9.28447 | 10.9525 | 0.429332 | -0.376969 | 0.140015 | 1402.25 | 1846.02 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | posterior_improving | gmm_12 | 2 | 10015 | 9.24413 | 10.9292 | 0.413737 | -0.359498 | 0.135084 | 1352.36 | 1714.35 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | posterior_improving | isotropic_12 | 2 | 10036 | 7.60073 | 9.26748 | 1.00576 | -0.948321 | 0.143832 | 1441.4 | 10151.9 | 1 | 0.5 | 1 | 0 |
| spike_slab | 0.9 | uniform | ellipsoidal_12 | 2 | 10015 | 8.76114 | 10.4194 | 0.454186 | -0.396272 | 0.125687 | 1258.75 | 2065.94 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | uniform | gmm_12 | 2 | 10015 | 8.83209 | 10.5022 | 0.407644 | -0.344457 | 0.13986 | 1400.6 | 1664.23 | 1 | 1 | 1 | 0 |
| spike_slab | 0.9 | uniform | isotropic_12 | 2 | 10036 | 6.99114 | 8.71067 | 0.967902 | -0.912707 | 0.166485 | 1667.29 | 9402.08 | 1 | 1 | 1 | 0 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok | min_samples_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | uniform | isotropic_12 | 0 | 26030 | 6.99608 | 8.54486 | -24.6065 | -26.131 | 0.857322 | 0.365647 | -1.52455 | 3480.15 | 60500.3 | 4.16946 | false | true | true |
| basic_mvn | 0.9 | uniform | isotropic_12 | 17 | 24775 | 7.16994 | 8.68525 | -24.6065 | -26.5849 | 0.868576 | 0.33955 | -1.9784 | 2856.41 | 96971.1 | 5.82654 | false | true | true |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 0 | 27555 | 7.36923 | 8.91372 | -24.6065 | -26.6369 | 0.85652 | 0.302104 | -2.03043 | 2514.85 | 113599 | 6.72097 | false | true | true |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 17 | 24361 | 7.53121 | 9.10418 | -24.6065 | -25.0879 | 0.78093 | 0.333853 | -0.481429 | 2715.23 | 5646.24 | 1.44204 | true | true | true |
| basic_mvn | 0.9 | uniform | gmm_12 | 0 | 27555 | 7.33085 | 8.92883 | -24.6065 | -26.6344 | 0.85652 | 0.300882 | -2.02794 | 2494.55 | 113321 | 6.74 | false | true | true |
| basic_mvn | 0.9 | uniform | gmm_12 | 17 | 24361 | 7.50367 | 9.04502 | -24.6065 | -25.1102 | 0.78093 | 0.325726 | -0.503699 | 2584.64 | 6180.7 | 1.54639 | true | true | true |
| basic_mvn | 0.9 | evidence_improving | isotropic_12 | 0 | 26030 | 7.38808 | 8.9126 | -24.6065 | -26.0644 | 0.857322 | 0.349374 | -1.45795 | 3177.28 | 55329.5 | 4.17302 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | isotropic_12 | 17 | 24775 | 7.63052 | 9.17526 | -24.6065 | -26.595 | 0.868576 | 0.268948 | -1.98858 | 1792.05 | 97971.1 | 7.39391 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | ellipsoidal_12 | 0 | 27555 | 7.67602 | 9.19991 | -24.6065 | -26.6512 | 0.85652 | 0.325932 | -2.04475 | 2927.21 | 115207 | 6.27354 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | ellipsoidal_12 | 17 | 24361 | 7.90692 | 9.4656 | -24.6065 | -25.164 | 0.78093 | 0.321477 | -0.557515 | 2517.64 | 7571.95 | 1.73423 | true | true | true |
| basic_mvn | 0.9 | evidence_improving | gmm_12 | 0 | 27555 | 7.65808 | 9.18289 | -24.6065 | -26.6727 | 0.85652 | 0.283725 | -2.06625 | 2218.18 | 117643 | 7.28257 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | gmm_12 | 17 | 24361 | 7.95699 | 9.51602 | -24.6065 | -25.0896 | 0.78093 | 0.356428 | -0.483125 | 3094.84 | 5686.09 | 1.35546 | true | true | true |
| basic_mvn | 0.9 | posterior_improving | isotropic_12 | 0 | 26030 | 7.45186 | 9.00671 | -24.6065 | -26.0992 | 0.857322 | 0.338902 | -1.49271 | 2989.66 | 57999.5 | 4.40455 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | isotropic_12 | 17 | 24775 | 7.68115 | 9.232 | -24.6065 | -26.6023 | 0.868576 | 0.312802 | -1.99586 | 2424.11 | 98690 | 6.38059 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | ellipsoidal_12 | 0 | 27555 | 7.80428 | 9.35106 | -24.6065 | -26.6777 | 0.85652 | 0.347073 | -2.07127 | 3319.27 | 118216 | 5.96783 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | ellipsoidal_12 | 17 | 24361 | 8.00448 | 9.56472 | -24.6065 | -25.1047 | 0.78093 | 0.349883 | -0.498284 | 2982.23 | 6048.51 | 1.42414 | true | true | true |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 0 | 27555 | 7.84717 | 9.37135 | -24.6065 | -26.6972 | 0.85652 | 0.287674 | -2.0907 | 2280.35 | 120444 | 7.2676 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 17 | 24361 | 8.04695 | 9.58438 | -24.6065 | -25.1575 | 0.78093 | 0.365337 | -0.551031 | 3251.49 | 7396.86 | 1.50828 | true | true | true |
| spike_slab | 0.9 | uniform | isotropic_12 | 0 | 10263 | 6.95754 | 8.66691 | -18.3959 | -18.9864 | 0.69059 | 0.388388 | -0.590524 | 1548.12 | 3578.9 | 1.52045 | true | true | true |
| spike_slab | 0.9 | uniform | isotropic_12 | 17 | 9809 | 7.02475 | 8.75442 | -18.3959 | -19.6308 | 0.764772 | 0.426761 | -1.23489 | 1786.46 | 14958.3 | 2.89363 | true | true | true |
| spike_slab | 0.9 | uniform | ellipsoidal_12 | 0 | 9995 | 8.7935 | 10.448 | -18.3959 | -19.0141 | 0.691539 | 0.355291 | -0.618202 | 1261.68 | 3819.83 | 1.73999 | true | true | true |
| spike_slab | 0.9 | uniform | ellipsoidal_12 | 17 | 10035 | 8.72878 | 10.3909 | -18.3959 | -18.5702 | 0.68635 | 0.353756 | -0.174342 | 1255.81 | 305.017 | 0.492832 | true | true | true |
| spike_slab | 0.9 | uniform | gmm_12 | 0 | 9995 | 8.73428 | 10.4089 | -18.3959 | -18.9583 | 0.691539 | 0.380709 | -0.562455 | 1448.67 | 3161.98 | 1.47739 | true | true | true |
| spike_slab | 0.9 | uniform | gmm_12 | 17 | 10035 | 8.92991 | 10.5954 | -18.3959 | -18.5223 | 0.68635 | 0.367126 | -0.126459 | 1352.53 | 160.479 | 0.344457 | true | true | true |
| spike_slab | 0.9 | evidence_improving | isotropic_12 | 0 | 10263 | 7.52326 | 9.18186 | -18.3959 | -18.9244 | 0.69059 | 0.396513 | -0.528545 | 1613.57 | 2867.07 | 1.33298 | true | true | true |
| spike_slab | 0.9 | evidence_improving | isotropic_12 | 17 | 9809 | 7.47764 | 9.16732 | -18.3959 | -19.7035 | 0.764772 | 0.447869 | -1.30762 | 1967.55 | 16772.1 | 2.91964 | true | true | true |
| spike_slab | 0.9 | evidence_improving | ellipsoidal_12 | 0 | 9995 | 9.30311 | 10.9912 | -18.3959 | -19.0448 | 0.691539 | 0.381727 | -0.648878 | 1456.42 | 4208.32 | 1.69985 | true | true | true |
| spike_slab | 0.9 | evidence_improving | ellipsoidal_12 | 17 | 10035 | 9.12029 | 10.7976 | -18.3959 | -18.5704 | 0.68635 | 0.337013 | -0.174508 | 1139.75 | 305.597 | 0.517809 | true | true | true |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 0 | 9995 | 9.35102 | 11.0222 | -18.3959 | -19.014 | 0.691539 | 0.363461 | -0.618078 | 1320.38 | 3818.3 | 1.70054 | true | true | true |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 17 | 10035 | 9.15924 | 10.7964 | -18.3959 | -18.5377 | 0.68635 | 0.353532 | -0.141816 | 1254.22 | 201.822 | 0.401141 | true | true | true |
| spike_slab | 0.9 | posterior_improving | isotropic_12 | 0 | 10263 | 7.60209 | 9.25784 | -18.3959 | -19.0092 | 0.69059 | 0.366914 | -0.613312 | 1381.66 | 3860.44 | 1.67154 | true | true | true |
| spike_slab | 0.9 | posterior_improving | isotropic_12 | 17 | 9809 | 7.59936 | 9.27711 | -18.3959 | -19.6792 | 0.764772 | 0.3912 | -1.28333 | 1501.14 | 16154.8 | 3.2805 | false | true | true |
| spike_slab | 0.9 | posterior_improving | ellipsoidal_12 | 0 | 9995 | 9.27356 | 10.9268 | -18.3959 | -18.9783 | 0.691539 | 0.374409 | -0.582444 | 1401.12 | 3390.71 | 1.55563 | true | true | true |
| spike_slab | 0.9 | posterior_improving | ellipsoidal_12 | 17 | 10035 | 9.29539 | 10.9782 | -18.3959 | -18.5674 | 0.68635 | 0.373963 | -0.171495 | 1403.38 | 295.133 | 0.458587 | true | true | true |
| spike_slab | 0.9 | posterior_improving | gmm_12 | 0 | 9995 | 9.24061 | 10.9341 | -18.3959 | -18.9602 | 0.691539 | 0.400451 | -0.56429 | 1602.81 | 3182.64 | 1.40913 | true | true | true |
| spike_slab | 0.9 | posterior_improving | gmm_12 | 17 | 10035 | 9.24764 | 10.9244 | -18.3959 | -18.5506 | 0.68635 | 0.331371 | -0.154707 | 1101.91 | 240.181 | 0.46687 | true | true | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: ellipsoidal_12/uniform.
- `spike_slab` best rows by target were: gmm_12/uniform.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.
- At least one row reached the uncertainty target while missing the analytic accuracy gate for every seed. Use the minimum-sample guard or stricter accuracy-gated tables when comparing `evals * variance`.
- GMM settings won 1 problem/target groups; compare against ellipsoidal rows to decide whether GMM fitting cost is buying lower variance.