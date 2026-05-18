# Pure-Core Efficiency Grid

Generated: `2026-05-12_112018 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn`
- Allocation targets: `uniform`
- LogZ uncertainty targets: `0.85`
- Seeds: `0`
- Target live points: `40`
- Max samples: `240`
- Shell size: `10`
- delta_K: `10`
- MC shrinkage samples: `64`
- JAX backend: `cpu`
- JAX devices: `cpu:0`

## Best By Problem And Target

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.85 | uniform | isotropic_6 | 240 | 70.8766 | 1.20564e+06 | 142.225 | 1 | 0 |

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.85 | uniform | isotropic_6 | 1 | 240 | 17.1743 | 18.4818 | 70.8766 | -70.8766 | 0.592602 | 142.225 | 1.20564e+06 | 1 | 0 | 0 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.85 | uniform | isotropic_6 | 0 | 240 | 17.1743 | 18.4818 | -24.6065 | -95.483 | 0.714261 | 0.769807 | -70.8766 | 142.225 | 1.20564e+06 | 92.0706 | false | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: isotropic_6/uniform.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.