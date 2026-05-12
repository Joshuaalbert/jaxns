# Pure-Core Efficiency Grid

Generated: `2026-05-12_172455 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn`
- Allocation targets: `uniform`
- LogZ uncertainty targets: `0.0`
- Seeds: `0`
- Target live points: `30`
- Max samples: `2000`
- Shell size: `15`
- Minimum samples before uncertainty stopping: `2000`
- delta_K: `1`
- MC shrinkage samples: `1000`
- Row isolation: `False`
- JAX backend: `cpu`
- JAX devices: `cpu:0`

## Best By Problem And Target

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0 | uniform | isotropic_24 | 1.04784e+06 | 0.764676 | 612706 | 44126.9 | 0 | 0 |

## Best Usable By Problem And Target

Rows here require `accuracy_success_fraction >= 0.5`; this prevents biased early stops from winning solely because the MC variance estimate is small.

_No rows._

## Best Strict By Problem And Target

Rows here require both `accuracy_success_fraction == 1` and `target_success_fraction == 1`.

_No rows._

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | min_sample_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0 | uniform | isotropic_24 | 1 | 1.04784e+06 | 20.2072 | 23.2418 | 0.764676 | -0.764676 | 0.042112 | 44126.9 | 612706 | 0 | 0 | 1 | 1 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok | min_samples_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0 | uniform | isotropic_24 | 0 | 1047845 | 20.2072 | 23.2418 | -24.6065 | -25.3711 | 0.833922 | 0.205212 | -0.764676 | 44126.9 | 612706 | 3.72627 | false | false | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: isotropic_24/uniform.
- Some rows did not reach the requested uncertainty. Check `sample_cap_fraction` and `max_goal_iterations` before drawing strong efficiency conclusions.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.