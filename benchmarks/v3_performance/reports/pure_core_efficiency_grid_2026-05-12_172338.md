# Pure-Core Efficiency Grid

Generated: `2026-05-12_172338 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn`
- Allocation targets: `uniform`
- LogZ uncertainty targets: `0.0`
- Seeds: `0`
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
| basic_mvn | 0 | uniform | isotropic_12 | 243131 | 0.7815 | 148490 | 21101.5 | 0 | 1 |

## Best Usable By Problem And Target

Rows here require `accuracy_success_fraction >= 0.5`; this prevents biased early stops from winning solely because the MC variance estimate is small.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0 | uniform | isotropic_12 | 243131 | 0.7815 | 148490 | 21101.5 | 0 | 1 |

## Best Strict By Problem And Target

Rows here require both `accuracy_success_fraction == 1` and `target_success_fraction == 1`.

_No rows._

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | min_sample_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0 | uniform | isotropic_12 | 1 | 243131 | 9.36985 | 11.645 | 0.7815 | -0.7815 | 0.0867907 | 21101.5 | 148490 | 0 | 1 | 1 | 1 |
| basic_mvn | 0 | uniform | isotropic_24 | 1 | 431101 | 6.20913 | 7.84174 | 0.726271 | -0.726271 | 0.0421113 | 18154.2 | 227392 | 0 | 0 | 1 | 1 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok | min_samples_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0 | uniform | isotropic_12 | 0 | 243131 | 9.36985 | 11.645 | -24.6065 | -25.388 | 0.825248 | 0.294603 | -0.7815 | 21101.5 | 148490 | 2.65272 | true | false | true |
| basic_mvn | 0 | uniform | isotropic_24 | 0 | 431101 | 6.20913 | 7.84174 | -24.6065 | -25.3327 | 0.832574 | 0.20521 | -0.726271 | 18154.2 | 227392 | 3.53915 | false | false | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: isotropic_12/uniform.
- Some rows did not reach the requested uncertainty. Check `sample_cap_fraction` and `max_goal_iterations` before drawing strong efficiency conclusions.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.