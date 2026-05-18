# Pure-Core Efficiency Grid

Generated: `2026-05-12_104633 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

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

## Best By Problem And Target

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | evidence_improving | gmm_6 | 180 | 52.3528 | 493347 | 196.18 | 1 |

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | evidence_improving | gmm_6 | 1 | 180 | 12.1058 | 52.3528 | -52.3528 | 1.08989 | 196.18 | 493347 | 1 | 0 |
| basic_mvn | 0.9 | evidence_improving | isotropic_6 | 1 | 180 | 11.574 | 71.431 | -71.431 | 1.1119 | 200.141 | 918429 | 1 | 0 |
| basic_mvn | 0.9 | posterior_improving | gmm_6 | 1 | 180 | 11.8459 | 52.3528 | -52.3528 | 1.08989 | 196.18 | 493347 | 1 | 0 |
| basic_mvn | 0.9 | posterior_improving | isotropic_6 | 1 | 180 | 11.2157 | 71.431 | -71.431 | 1.1119 | 200.141 | 918429 | 1 | 0 |
| basic_mvn | 0.9 | uniform | gmm_6 | 1 | 180 | 12.8734 | 52.3528 | -52.3528 | 1.08989 | 196.18 | 493347 | 1 | 0 |
| basic_mvn | 0.9 | uniform | isotropic_6 | 1 | 180 | 14.5527 | 71.431 | -71.431 | 1.1119 | 200.141 | 918429 | 1 | 0 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | target_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | uniform | isotropic_6 | 0 | 180 | 14.5527 | -24.6065 | -96.0374 | 0.865103 | 1.05446 | -71.431 | 200.141 | 918429 | true |
| basic_mvn | 0.9 | uniform | gmm_6 | 0 | 180 | 12.8734 | -24.6065 | -76.9593 | 0.865107 | 1.04398 | -52.3528 | 196.18 | 493347 | true |
| basic_mvn | 0.9 | evidence_improving | isotropic_6 | 0 | 180 | 11.574 | -24.6065 | -96.0374 | 0.865103 | 1.05446 | -71.431 | 200.141 | 918429 | true |
| basic_mvn | 0.9 | evidence_improving | gmm_6 | 0 | 180 | 12.1058 | -24.6065 | -76.9593 | 0.865107 | 1.04398 | -52.3528 | 196.18 | 493347 | true |
| basic_mvn | 0.9 | posterior_improving | isotropic_6 | 0 | 180 | 11.2157 | -24.6065 | -96.0374 | 0.865103 | 1.05446 | -71.431 | 200.141 | 918429 | true |
| basic_mvn | 0.9 | posterior_improving | gmm_6 | 0 | 180 | 11.8459 | -24.6065 | -76.9593 | 0.865107 | 1.04398 | -52.3528 | 196.18 | 493347 | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: gmm_6/evidence_improving.
- GMM settings won 1 problem/target groups; compare against ellipsoidal rows to decide whether GMM fitting cost is buying lower variance.