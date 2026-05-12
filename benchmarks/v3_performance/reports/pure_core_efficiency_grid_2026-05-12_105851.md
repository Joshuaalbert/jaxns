# Pure-Core Efficiency Grid

Generated: `2026-05-12_105851 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn`
- Allocation targets: `uniform, evidence_improving, posterior_improving`
- LogZ uncertainty targets: `0.9`
- Seeds: `0`
- Target live points: `30`
- Max samples: `120`
- Shell size: `15`
- MC shrinkage samples: `64`
- JAX backend: `cpu`
- JAX devices: `cpu:0`

## Best By Problem And Target

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | posterior_improving | gmm_6 | 180 | 52.2619 | 491635 | 184.92 | 1 | 0 |

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | evidence_improving | gmm_6 | 1 | 180 | 11.8478 | 11.8615 | 52.4648 | -52.4648 | 1.50424 | 270.764 | 495460 | 1 | 0 | 0 |
| basic_mvn | 0.9 | evidence_improving | isotropic_6 | 1 | 180 | 11.3177 | 11.3316 | 71.8043 | -71.8043 | 1.24233 | 223.62 | 928055 | 1 | 0 | 0 |
| basic_mvn | 0.9 | posterior_improving | gmm_6 | 1 | 180 | 11.8825 | 11.8976 | 52.2619 | -52.2619 | 1.02733 | 184.92 | 491635 | 1 | 0 | 0 |
| basic_mvn | 0.9 | posterior_improving | isotropic_6 | 1 | 180 | 11.1514 | 11.1655 | 71.6173 | -71.6173 | 0.990055 | 178.21 | 923226 | 1 | 0 | 0 |
| basic_mvn | 0.9 | uniform | gmm_6 | 1 | 180 | 12.694 | 12.7092 | 52.4425 | -52.4425 | 1.21266 | 218.279 | 495039 | 1 | 0 | 0 |
| basic_mvn | 0.9 | uniform | isotropic_6 | 1 | 180 | 13.1536 | 14.5311 | 71.2027 | -71.2027 | 1.1015 | 198.27 | 912569 | 1 | 0 | 0 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | uniform | isotropic_6 | 0 | 180 | 13.1536 | 14.5311 | -24.6065 | -95.8092 | 0.865103 | 1.04952 | -71.2027 | 198.27 | 912569 | 67.8429 | false | true |
| basic_mvn | 0.9 | uniform | gmm_6 | 0 | 180 | 12.694 | 12.7092 | -24.6065 | -77.049 | 0.865107 | 1.10121 | -52.4425 | 218.279 | 495039 | 47.6227 | false | true |
| basic_mvn | 0.9 | evidence_improving | isotropic_6 | 0 | 180 | 11.3177 | 11.3316 | -24.6065 | -96.4108 | 0.865103 | 1.1146 | -71.8043 | 223.62 | 928055 | 64.4216 | false | true |
| basic_mvn | 0.9 | evidence_improving | gmm_6 | 0 | 180 | 11.8478 | 11.8615 | -24.6065 | -77.0713 | 0.865107 | 1.22648 | -52.4648 | 270.764 | 495460 | 42.7769 | false | true |
| basic_mvn | 0.9 | posterior_improving | isotropic_6 | 0 | 180 | 11.1514 | 11.1655 | -24.6065 | -96.2237 | 0.865103 | 0.995015 | -71.6173 | 178.21 | 923226 | 71.9761 | false | true |
| basic_mvn | 0.9 | posterior_improving | gmm_6 | 0 | 180 | 11.8825 | 11.8976 | -24.6065 | -76.8684 | 0.865107 | 1.01357 | -52.2619 | 184.92 | 491635 | 51.5619 | false | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: gmm_6/posterior_improving.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.
- GMM settings won 1 problem/target groups; compare against ellipsoidal rows to decide whether GMM fitting cost is buying lower variance.