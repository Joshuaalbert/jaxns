# Pure-Core Efficiency Grid

Generated: `2026-05-12_171945 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn`
- Allocation targets: `uniform`
- LogZ uncertainty targets: `0.0`
- Seeds: `17, 29`
- Target live points: `30`
- Max samples: `1200`
- Shell size: `15`
- Minimum samples before uncertainty stopping: `1200`
- delta_K: `1`
- MC shrinkage samples: `1000`
- Row isolation: `False`
- JAX backend: `cpu`
- JAX devices: `cpu:0`

## Best By Problem And Target

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0 | uniform | isotropic_24 | 483440 | 0.51117 | 126320 | 19592.7 | 0 | 1 |

## Best Usable By Problem And Target

Rows here require `accuracy_success_fraction >= 0.5`; this prevents biased early stops from winning solely because the MC variance estimate is small.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0 | uniform | isotropic_24 | 483440 | 0.51117 | 126320 | 19592.7 | 0 | 1 |

## Best Strict By Problem And Target

Rows here require both `accuracy_success_fraction == 1` and `target_success_fraction == 1`.

_No rows._

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | min_sample_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0 | uniform | isotropic_24 | 2 | 483440 | 7.16162 | 8.83881 | 0.51117 | -0.511003 | 0.0404646 | 19592.7 | 126320 | 0 | 1 | 1 | 1 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok | min_samples_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0 | uniform | isotropic_24 | 17 | 494447 | 10.6834 | 13.061 | -24.6065 | -25.1305 | 0.81278 | 0.207935 | -0.524051 | 21378.4 | 135790 | 2.52026 | true | false | true |
| basic_mvn | 0 | uniform | isotropic_24 | 29 | 472433 | 3.63987 | 4.61658 | -24.6065 | -25.1044 | 0.853626 | 0.194145 | -0.497956 | 17807.1 | 117145 | 2.56487 | true | false | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: isotropic_24/uniform.
- Some rows did not reach the requested uncertainty. Check `sample_cap_fraction` and `max_goal_iterations` before drawing strong efficiency conclusions.