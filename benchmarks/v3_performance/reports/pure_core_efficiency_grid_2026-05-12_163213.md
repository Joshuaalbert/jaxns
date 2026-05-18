# Pure-Core Efficiency Grid

Generated: `2026-05-12_163213 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn`
- Allocation targets: `uniform, evidence_improving, posterior_improving`
- LogZ uncertainty targets: `0.7, 0.5`
- Seeds: `0, 17`
- Target live points: `80`
- Max samples: `2400`
- Shell size: `20`
- Minimum samples before uncertainty stopping: `960`
- delta_K: `20`
- MC shrinkage samples: `128`
- Row isolation: `True`
- JAX backend: `cpu`
- JAX devices: `cpu:0`

## Best By Problem And Target

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.5 | evidence_improving | gmm_12 | 249616 | 0.584806 | 85368.2 | 14180.1 | 0 | 1 |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 41039.5 | 1.34997 | 74790.8 | 2190.68 | 1 | 0 |

## Best Usable By Problem And Target

Rows here require `accuracy_success_fraction >= 0.5`; this prevents biased early stops from winning solely because the MC variance estimate is small.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.5 | evidence_improving | gmm_12 | 249616 | 0.584806 | 85368.2 | 14180.1 | 0 | 1 |

## Best Strict By Problem And Target

Rows here require both `accuracy_success_fraction == 1` and `target_success_fraction == 1`.

_No rows._

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | min_sample_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 2 | 249616 | 28.1391 | 29.9156 | 0.613058 | -0.610959 | 0.0543654 | 13498.4 | 93815.6 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.5 | evidence_improving | gmm_12 | 2 | 249616 | 28.3864 | 30.165 | 0.584806 | -0.58228 | 0.0573157 | 14180.1 | 85368.2 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.5 | posterior_improving | ellipsoidal_12 | 2 | 249616 | 28.3732 | 30.1197 | 0.600487 | -0.598601 | 0.0532421 | 13195.1 | 90007.6 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.5 | posterior_improving | gmm_12 | 2 | 249616 | 28.5052 | 30.3379 | 0.61782 | -0.61686 | 0.0530622 | 13034.6 | 95278.8 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.5 | uniform | ellipsoidal_12 | 2 | 249616 | 27.653 | 29.4549 | 0.612658 | -0.610696 | 0.0503855 | 12300.1 | 93693.3 | 0 | 0.5 | 1 | 1 |
| basic_mvn | 0.5 | uniform | gmm_12 | 2 | 249616 | 27.902 | 29.7573 | 0.593235 | -0.591164 | 0.0581085 | 14442.7 | 87846.7 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 2 | 41039.5 | 11.0984 | 12.7982 | 1.34997 | -1.34951 | 0.0545348 | 2190.68 | 74790.8 | 1 | 0 | 1 | 0 |
| basic_mvn | 0.7 | evidence_improving | gmm_12 | 2 | 41039.5 | 11.0641 | 12.7442 | 1.35835 | -1.35731 | 0.0461194 | 1880.63 | 75722.8 | 1 | 0 | 1 | 0 |
| basic_mvn | 0.7 | posterior_improving | ellipsoidal_12 | 2 | 41039.5 | 11.3266 | 13.0166 | 1.39083 | -1.39009 | 0.0554289 | 2216.12 | 79386.9 | 1 | 0 | 1 | 0 |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 2 | 41039.5 | 11.189 | 12.9122 | 1.36764 | -1.3675 | 0.0518632 | 2093.29 | 76761.4 | 1 | 0 | 1 | 0 |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 2 | 41039.5 | 10.627 | 12.3268 | 1.39212 | -1.39178 | 0.0558702 | 2289.57 | 79534.6 | 1 | 0 | 1 | 0 |
| basic_mvn | 0.7 | uniform | gmm_12 | 2 | 41039.5 | 10.8402 | 12.5349 | 1.38776 | -1.38646 | 0.051733 | 2086.43 | 79036.6 | 1 | 0 | 1 | 0 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok | min_samples_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 0 | 45313 | 9.93802 | 11.6747 | -24.6065 | -26.0289 | 0.593277 | 0.234721 | -1.4224 | 2496.48 | 91678.5 | 6.05996 | false | true | true |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 17 | 36766 | 11.316 | 12.9789 | -24.6065 | -25.9676 | 0.5886 | 0.238005 | -1.36117 | 2082.66 | 68119.1 | 5.71907 | false | true | true |
| basic_mvn | 0.7 | uniform | gmm_12 | 0 | 45313 | 10.2178 | 11.9432 | -24.6065 | -26.0529 | 0.593277 | 0.207732 | -1.4464 | 1955.36 | 94798.3 | 6.96284 | false | true | true |
| basic_mvn | 0.7 | uniform | gmm_12 | 17 | 36766 | 11.4626 | 13.1265 | -24.6065 | -25.933 | 0.5886 | 0.245588 | -1.32652 | 2217.49 | 64695.5 | 5.4014 | false | true | true |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 0 | 45313 | 10.3087 | 12.0143 | -24.6065 | -25.991 | 0.593277 | 0.208428 | -1.38454 | 1968.5 | 86862.7 | 6.64276 | false | true | true |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 17 | 36766 | 11.888 | 13.5822 | -24.6065 | -25.9209 | 0.5886 | 0.256178 | -1.31449 | 2412.85 | 63527 | 5.13114 | false | true | true |
| basic_mvn | 0.7 | evidence_improving | gmm_12 | 0 | 45313 | 10.329 | 12.0354 | -24.6065 | -26.017 | 0.593277 | 0.208065 | -1.41052 | 1961.65 | 90153.2 | 6.77922 | false | true | true |
| basic_mvn | 0.7 | evidence_improving | gmm_12 | 17 | 36766 | 11.7993 | 13.453 | -24.6065 | -25.9106 | 0.5886 | 0.221241 | -1.3041 | 1799.61 | 62527 | 5.89447 | false | true | true |
| basic_mvn | 0.7 | posterior_improving | ellipsoidal_12 | 0 | 45313 | 10.7427 | 12.4568 | -24.6065 | -26.0419 | 0.593277 | 0.204213 | -1.43544 | 1889.68 | 93366.5 | 7.02912 | false | true | true |
| basic_mvn | 0.7 | posterior_improving | ellipsoidal_12 | 17 | 36766 | 11.9105 | 13.5765 | -24.6065 | -25.9512 | 0.5886 | 0.262973 | -1.34474 | 2542.55 | 66485 | 5.1136 | false | true | true |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 0 | 45313 | 10.4896 | 12.265 | -24.6065 | -25.9932 | 0.593277 | 0.208896 | -1.38675 | 1977.36 | 87140.1 | 6.63845 | false | true | true |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 17 | 36766 | 11.8884 | 13.5593 | -24.6065 | -25.9547 | 0.5886 | 0.24513 | -1.34825 | 2209.22 | 66832.6 | 5.50015 | false | true | true |
| basic_mvn | 0.5 | uniform | ellipsoidal_12 | 0 | 271378 | 27.9915 | 29.7972 | -24.6065 | -25.2661 | 0.510046 | 0.194058 | -0.659684 | 10219.7 | 118099 | 3.39942 | false | false | true |
| basic_mvn | 0.5 | uniform | ellipsoidal_12 | 17 | 227854 | 27.3146 | 29.1126 | -24.6065 | -25.1682 | 0.522607 | 0.251222 | -0.561709 | 14380.4 | 71891.7 | 2.23591 | true | false | true |
| basic_mvn | 0.5 | uniform | gmm_12 | 0 | 271378 | 28.2952 | 30.1112 | -24.6065 | -25.2472 | 0.510046 | 0.23506 | -0.64069 | 14994.5 | 111396 | 2.72565 | true | false | true |
| basic_mvn | 0.5 | uniform | gmm_12 | 17 | 227854 | 27.5087 | 29.4035 | -24.6065 | -25.1481 | 0.522607 | 0.246908 | -0.541637 | 13890.8 | 66845.6 | 2.19367 | true | false | true |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 0 | 271378 | 28.6426 | 30.4042 | -24.6065 | -25.2681 | 0.510046 | 0.225955 | -0.661643 | 13855.4 | 118801 | 2.92821 | true | false | true |
| basic_mvn | 0.5 | evidence_improving | ellipsoidal_12 | 17 | 227854 | 27.6357 | 29.4271 | -24.6065 | -25.1667 | 0.522607 | 0.240157 | -0.560275 | 13141.5 | 71525.2 | 2.33296 | true | false | true |
| basic_mvn | 0.5 | evidence_improving | gmm_12 | 0 | 271378 | 28.7552 | 30.5548 | -24.6065 | -25.243 | 0.510046 | 0.226912 | -0.636575 | 13973 | 109970 | 2.80538 | true | false | true |
| basic_mvn | 0.5 | evidence_improving | gmm_12 | 17 | 227854 | 28.0177 | 29.7752 | -24.6065 | -25.1344 | 0.522607 | 0.251281 | -0.527985 | 14387.2 | 63518.5 | 2.10117 | true | false | true |
| basic_mvn | 0.5 | posterior_improving | ellipsoidal_12 | 0 | 271378 | 28.8335 | 30.5597 | -24.6065 | -25.2526 | 0.510046 | 0.221086 | -0.646159 | 13264.7 | 113306 | 2.92266 | true | false | true |
| basic_mvn | 0.5 | posterior_improving | ellipsoidal_12 | 17 | 227854 | 27.9129 | 29.6798 | -24.6065 | -25.1575 | 0.522607 | 0.240011 | -0.551042 | 13125.6 | 69187.2 | 2.29591 | true | false | true |
| basic_mvn | 0.5 | posterior_improving | gmm_12 | 0 | 271378 | 28.8853 | 30.6986 | -24.6065 | -25.2578 | 0.510046 | 0.208293 | -0.651288 | 11774 | 115112 | 3.12678 | false | false | true |
| basic_mvn | 0.5 | posterior_improving | gmm_12 | 17 | 227854 | 28.1252 | 29.9772 | -24.6065 | -25.1889 | 0.522607 | 0.250476 | -0.582432 | 14295.2 | 77294.3 | 2.3253 | true | false | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: gmm_12/evidence_improving, ellipsoidal_12/evidence_improving.
- Some rows did not reach the requested uncertainty. Check `sample_cap_fraction` and `max_goal_iterations` before drawing strong efficiency conclusions.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.
- At least one row reached the uncertainty target while missing the analytic accuracy gate for every seed. Use the minimum-sample guard or stricter accuracy-gated tables when comparing `evals * variance`.
- GMM settings won 1 problem/target groups; compare against ellipsoidal rows to decide whether GMM fitting cost is buying lower variance.