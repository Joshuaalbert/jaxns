# Pure-Core Efficiency Grid

Generated: `2026-05-12_105404 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

## Configuration

- Problems: `basic_mvn`
- Allocation targets: `uniform`
- LogZ uncertainty targets: `0.5`
- Seeds: `0`
- Target live points: `200`
- Max samples: `1200`
- Shell size: `15`
- MC shrinkage samples: `128`
- JAX backend: `cpu`
- JAX devices: `cpu:0`

## Best By Problem And Target

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.5 | uniform | isotropic_12 | 29031 | 3.46506 | 348565 | 5058.25 | 0 | 0 |

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.5 | uniform | isotropic_12 | 1 | 29031 | 84.3933 | 3.46506 | -3.46506 | 0.174236 | 5058.25 | 348565 | 0 | 0 | 1 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.5 | uniform | isotropic_12 | 0 | 29031 | 84.3933 | -24.6065 | -28.0715 | 0.763299 | 0.417416 | -3.46506 | 5058.25 | 348565 | 8.30121 | false | false |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: isotropic_12/uniform.
- Some rows hit the sample cap before the requested uncertainty; increase `max_samples` for precise targets before drawing strong efficiency conclusions.
- Some rows met the requested uncertainty but failed the analytic logZ accuracy gate; treat those settings as biased rather than efficient.