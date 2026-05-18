# Pure-Core Efficiency Grid

Generated: `2026-05-13_113631 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds. Compare `mean_log_Z_uncert`, `mean_mc_logZ_std`, and `rmse_mc_logZ` to check uncertainty calibration.

## Configuration

- Problems: `basic_mvn`
- Allocation targets: `uniform`
- LogZ uncertainty targets: `0.5, 0.1, 0.05`
- Seeds: `0, 17, 29, 43, 71`
- Phantom modes: `on, off`
- Target live points: `40`
- Max samples: `10000000`
- Shell size: `10`
- Minimum samples before uncertainty stopping: `480`
- delta_K: `10`
- MC shrinkage samples: `128`
- Row isolation: `False`
- JAX backend: `cpu`
- JAX devices: `cpu:0`

## Best By Problem And Target

| problem | logz_uncert_target | allocation_target | sampler_setting | phantoms_enabled | mean_likelihood_evaluations | mean_log_Z_uncert | mean_mc_logZ_std | mean_mc_logZ_mean | mc_bias_logZ | mean_expectation_logZ_mean | expectation_bias_logZ | rmse_mc_logZ | rmse_over_mean_log_Z_uncert | rmse_over_mean_mc_logZ_std | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | true | 178916 | 0.746527 | 0.186014 | -25.4366 | -0.830179 | -38.6212 | -14.0147 | 0.83979 | 1.12493 | 4.51466 | 126180 | 6206.73 | 0 | 0 |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | true | 178916 | 0.746527 | 0.181225 | -25.43 | -0.823536 | -38.6212 | -14.0147 | 0.833874 | 1.117 | 4.60131 | 124409 | 5930.79 | 0 | 0 |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | true | 178916 | 0.746527 | 0.183509 | -25.4283 | -0.821882 | -38.6212 | -14.0147 | 0.833534 | 1.11655 | 4.5422 | 124307 | 6050.75 | 0 | 0.2 |

## Best Usable By Problem And Target

Rows here require `accuracy_success_fraction >= 0.5`; this prevents biased early stops from winning solely because the MC variance estimate is small.

_No rows._

## Best Strict By Problem And Target

Rows here require both `accuracy_success_fraction == 1` and `target_success_fraction == 1`.

_No rows._

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | phantoms_enabled | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | mean_log_Z_uncert | mean_mc_logZ_std | mean_mc_logZ_mean | mc_bias_logZ | mean_abs_mc_logZ_error | rmse_mc_logZ | rmse_over_mean_log_Z_uncert | rmse_over_mean_mc_logZ_std | mean_expectation_logZ_mean | expectation_bias_logZ | rmse_expectation_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | mean_evals_times_expectation_mse | target_success_fraction | accuracy_success_fraction | min_sample_fraction | sample_cap_fraction | goal_iteration_limit_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | false | 5 | 178156 | 2.28666 | 2.5407 | 0.746527 | 0.726842 | -38.6196 | -14.0132 | 14.0132 | 14.0235 | 18.785 | 19.2937 | -38.6212 | -14.0147 | 14.0248 | 0.530985 | 94561.6 | 3.50359e+07 | 3.50422e+07 | 0 | 0 | 1 | 0 | 1 |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | true | 5 | 178916 | 2.44202 | 2.81257 | 0.746527 | 0.186014 | -25.4366 | -0.830179 | 0.830179 | 0.83979 | 1.12493 | 4.51466 | -38.6212 | -14.0147 | 14.0248 | 0.0346821 | 6206.73 | 126180 | 3.51917e+07 | 0 | 0 | 1 | 0 | 1 |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | false | 5 | 178156 | 2.26849 | 2.51173 | 0.746527 | 0.769573 | -38.5868 | -13.9803 | 13.9803 | 13.9916 | 18.7422 | 18.181 | -38.6212 | -14.0147 | 14.0248 | 0.592481 | 105558 | 3.48767e+07 | 3.50422e+07 | 0 | 0 | 1 | 0 | 1 |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | true | 5 | 178916 | 2.34614 | 2.71572 | 0.746527 | 0.181225 | -25.43 | -0.823536 | 0.823536 | 0.833874 | 1.117 | 4.60131 | -38.6212 | -14.0147 | 14.0248 | 0.0331433 | 5930.79 | 124409 | 3.51917e+07 | 0 | 0 | 1 | 0 | 1 |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | false | 5 | 178156 | 4.02923 | 4.5579 | 0.746527 | 0.752247 | -38.6227 | -14.0162 | 14.0162 | 14.0267 | 18.7892 | 18.6464 | -38.6212 | -14.0147 | 14.0248 | 0.569099 | 101402 | 3.50518e+07 | 3.50422e+07 | 0 | 0 | 1 | 0 | 1 |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | true | 5 | 178916 | 5.6598 | 6.44973 | 0.746527 | 0.183509 | -25.4283 | -0.821882 | 0.821882 | 0.833534 | 1.11655 | 4.5422 | -38.6212 | -14.0147 | 14.0248 | 0.0338063 | 6050.75 | 124307 | 3.51917e+07 | 0 | 0.2 | 1 | 0 | 1 |

## Per-Seed Records

| problem | target | allocation | setting | phantoms | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | mc_bias | expectation_logZ_mean | expectation_bias | logZ_uncert | mc_logZ_std | evals_x_var | evals_x_sqerr | evals_x_exp_sqerr | err_over_std | accuracy_ok | target_ok | min_samples_ok | goal_limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | on | 0 | 178614 | 19.0744 | 21.5781 | -24.6065 | -25.5007 | -0.894268 | -38.8319 | -14.2254 | 0.757738 | 0.163378 | 4767.65 | 142840 | 3.61446e+07 | 5.4736 | false | false | true | true |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | on | 17 | 179797 | 2.30294 | 2.66681 | -24.6065 | -25.1638 | -0.557333 | -37.667 | -13.0606 | 0.730979 | 0.187533 | 6323.2 | 55848.5 | 3.06696e+07 | 2.97192 | true | false | true | true |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | on | 29 | 177529 | 2.30675 | 2.67724 | -24.6065 | -25.443 | -0.836573 | -39.2268 | -14.6204 | 0.74744 | 0.18737 | 6232.63 | 124244 | 3.79478e+07 | 4.46481 | false | false | true | true |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | on | 43 | 178075 | 2.31205 | 2.6642 | -24.6065 | -25.4658 | -0.859386 | -38.492 | -13.8855 | 0.734142 | 0.181183 | 5845.74 | 131516 | 3.43342e+07 | 4.74318 | false | false | true | true |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | on | 71 | 180566 | 2.30288 | 2.66229 | -24.6065 | -25.5683 | -0.961852 | -38.8881 | -14.2817 | 0.762337 | 0.198079 | 7084.55 | 167052 | 3.68293e+07 | 4.85591 | false | false | true | true |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | off | 0 | 177854 | 11.2695 | 12.9065 | -24.6065 | -38.8904 | -14.284 | -38.8319 | -14.2254 | 0.757738 | 0.851766 | 129034 | 3.62879e+07 | 3.59908e+07 | 16.7698 | false | false | true | true |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | off | 17 | 179037 | 2.24935 | 2.50354 | -24.6065 | -37.6502 | -13.0437 | -37.667 | -13.0606 | 0.730979 | 0.777918 | 108345 | 3.0461e+07 | 3.05399e+07 | 16.7674 | false | false | true | true |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | off | 29 | 176769 | 2.19288 | 2.45045 | -24.6065 | -39.2263 | -14.6199 | -39.2268 | -14.6204 | 0.74744 | 0.709626 | 89015.5 | 3.77827e+07 | 3.77853e+07 | 20.6022 | false | false | true | true |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | off | 43 | 177315 | 2.20804 | 2.46261 | -24.6065 | -38.4746 | -13.8682 | -38.492 | -13.8855 | 0.734142 | 0.698069 | 86405.7 | 3.41024e+07 | 3.41877e+07 | 19.8665 | false | false | true | true |
| basic_mvn | 0.5 | uniform | isotropic_24_stepout | off | 71 | 179806 | 2.22635 | 2.46635 | -24.6065 | -38.8719 | -14.2654 | -38.8881 | -14.2817 | 0.762337 | 0.723853 | 94211.8 | 3.65908e+07 | 3.66743e+07 | 19.7076 | false | false | true | true |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | on | 0 | 178614 | 2.33848 | 2.7349 | -24.6065 | -25.5274 | -0.920966 | -38.8319 | -14.2254 | 0.757738 | 0.200192 | 7158.25 | 151497 | 3.61446e+07 | 4.60043 | false | false | true | true |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | on | 17 | 179797 | 2.33046 | 2.70092 | -24.6065 | -25.1867 | -0.580211 | -37.667 | -13.0606 | 0.730979 | 0.156056 | 4378.66 | 60527.7 | 3.06696e+07 | 3.71798 | false | false | true | true |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | on | 29 | 177529 | 2.32777 | 2.69134 | -24.6065 | -25.4464 | -0.839904 | -39.2268 | -14.6204 | 0.74744 | 0.165533 | 4864.5 | 125236 | 3.79478e+07 | 5.07394 | false | false | true | true |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | on | 43 | 178075 | 2.35351 | 2.70805 | -24.6065 | -25.4303 | -0.823826 | -38.492 | -13.8855 | 0.734142 | 0.188672 | 6338.98 | 120858 | 3.43342e+07 | 4.36644 | false | false | true | true |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | on | 71 | 180566 | 2.38046 | 2.74338 | -24.6065 | -25.5592 | -0.952774 | -38.8881 | -14.2817 | 0.762337 | 0.195674 | 6913.54 | 163914 | 3.68293e+07 | 4.8692 | false | false | true | true |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | off | 0 | 177854 | 2.26478 | 2.53014 | -24.6065 | -38.8763 | -14.2698 | -38.8319 | -14.2254 | 0.757738 | 0.752301 | 100658 | 3.62161e+07 | 3.59908e+07 | 18.9683 | false | false | true | true |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | off | 17 | 179037 | 2.26932 | 2.50643 | -24.6065 | -37.553 | -12.9465 | -37.667 | -13.0606 | 0.730979 | 0.754629 | 101955 | 3.00087e+07 | 3.05399e+07 | 17.1561 | false | false | true | true |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | off | 29 | 176769 | 2.24069 | 2.46542 | -24.6065 | -39.1643 | -14.5579 | -39.2268 | -14.6204 | 0.74744 | 0.765676 | 103633 | 3.7463e+07 | 3.77853e+07 | 19.0131 | false | false | true | true |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | off | 43 | 177315 | 2.25572 | 2.50528 | -24.6065 | -38.4797 | -13.8732 | -38.492 | -13.8855 | 0.734142 | 0.787399 | 109935 | 3.41271e+07 | 3.41877e+07 | 17.619 | false | false | true | true |
| basic_mvn | 0.1 | uniform | isotropic_24_stepout | off | 71 | 179806 | 2.31195 | 2.55141 | -24.6065 | -38.8608 | -14.2543 | -38.8881 | -14.2817 | 0.762337 | 0.787861 | 111610 | 3.65339e+07 | 3.66743e+07 | 18.0924 | false | false | true | true |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | on | 0 | 178614 | 2.60317 | 2.98782 | -24.6065 | -25.529 | -0.92253 | -38.8319 | -14.2254 | 0.757738 | 0.192875 | 6644.57 | 152012 | 3.61446e+07 | 4.78305 | false | false | true | true |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | on | 17 | 179797 | 2.41548 | 2.77614 | -24.6065 | -25.2032 | -0.596744 | -37.667 | -13.0606 | 0.730979 | 0.179311 | 5780.91 | 64026.3 | 3.06696e+07 | 3.32798 | false | false | true | true |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | on | 29 | 177529 | 2.38042 | 2.76514 | -24.6065 | -25.4459 | -0.839434 | -39.2268 | -14.6204 | 0.74744 | 0.187358 | 6231.81 | 125096 | 3.79478e+07 | 4.48037 | false | false | true | true |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | on | 43 | 178075 | 2.40344 | 2.76015 | -24.6065 | -25.4375 | -0.831063 | -38.492 | -13.8855 | 0.734142 | 0.172833 | 5319.35 | 122990 | 3.43342e+07 | 4.80846 | false | false | true | true |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | on | 71 | 180566 | 2.40757 | 2.77361 | -24.6065 | -25.5676 | -0.961123 | -38.8881 | -14.2817 | 0.762337 | 0.197693 | 7057 | 166799 | 3.68293e+07 | 4.86169 | false | false | true | true |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | off | 0 | 177854 | 2.28056 | 2.54918 | -24.6065 | -38.8178 | -14.2113 | -38.8319 | -14.2254 | 0.757738 | 0.754026 | 101120 | 3.59198e+07 | 3.59908e+07 | 18.8473 | false | false | true | true |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | off | 17 | 179037 | 2.27127 | 2.51198 | -24.6065 | -37.6611 | -13.0546 | -37.667 | -13.0606 | 0.730979 | 0.772575 | 106862 | 3.05119e+07 | 3.05399e+07 | 16.8975 | false | false | true | true |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | off | 29 | 176769 | 2.30123 | 2.53351 | -24.6065 | -39.2226 | -14.6161 | -39.2268 | -14.6204 | 0.74744 | 0.776421 | 106562 | 3.77633e+07 | 3.77853e+07 | 18.825 | false | false | true | true |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | off | 43 | 177315 | 2.29155 | 2.5591 | -24.6065 | -38.4621 | -13.8556 | -38.492 | -13.8855 | 0.734142 | 0.683487 | 82833.5 | 3.40405e+07 | 3.41877e+07 | 20.2719 | false | false | true | true |
| basic_mvn | 0.05 | uniform | isotropic_24_stepout | off | 71 | 179806 | 2.28871 | 2.54975 | -24.6065 | -38.9347 | -14.3283 | -38.8881 | -14.2817 | 0.762337 | 0.647698 | 75430.9 | 3.6914e+07 | 3.66743e+07 | 22.1218 | false | false | true | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: isotropic_24_stepout/uniform/on, isotropic_24_stepout/uniform/on, isotropic_24_stepout/uniform/on.
- Some rows did not reach the requested uncertainty. Check `sample_cap_fraction`, `goal_iteration_limit_fraction`, and `max_goal_iterations` before drawing strong efficiency conclusions.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.