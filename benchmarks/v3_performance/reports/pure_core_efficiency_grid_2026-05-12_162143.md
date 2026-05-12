# Pure-Core Efficiency Grid

Generated: `2026-05-12_162143 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn`
- Allocation targets: `uniform, evidence_improving, posterior_improving`
- LogZ uncertainty targets: `0.7, 0.6`
- Seeds: `0, 17`
- Target live points: `40`
- Max samples: `1600`
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
| basic_mvn | 0.6 | uniform | ellipsoidal_12 | 252347 | 0.508846 | 65338.8 | 30022.3 | 0 | 1 |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 207424 | 0.471733 | 46158.6 | 20507.2 | 0.5 | 1 |

## Best Usable By Problem And Target

Rows here require `accuracy_success_fraction >= 0.5`; this prevents biased early stops from winning solely because the MC variance estimate is small.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.6 | uniform | ellipsoidal_12 | 252347 | 0.508846 | 65338.8 | 30022.3 | 0 | 1 |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 207424 | 0.471733 | 46158.6 | 20507.2 | 0.5 | 1 |

## Best Strict By Problem And Target

Rows here require both `accuracy_success_fraction == 1` and `target_success_fraction == 1`.

_No rows._

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | min_sample_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.6 | evidence_improving | ellipsoidal_12 | 2 | 252347 | 29.0782 | 30.8802 | 0.542051 | -0.455259 | 0.111036 | 28026.4 | 74144.4 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.6 | evidence_improving | gmm_12 | 2 | 252347 | 29.1187 | 30.8611 | 0.532935 | -0.442687 | 0.118725 | 29973.2 | 71671.4 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.6 | posterior_improving | ellipsoidal_12 | 2 | 252347 | 29.1377 | 30.8742 | 0.540971 | -0.451986 | 0.100422 | 25354.4 | 73849.3 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.6 | posterior_improving | gmm_12 | 2 | 252347 | 29.2411 | 31 | 0.531791 | -0.447205 | 0.108848 | 27473.6 | 71364.1 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.6 | uniform | ellipsoidal_12 | 2 | 252347 | 28.3309 | 30.1213 | 0.508846 | -0.413526 | 0.11894 | 30022.3 | 65338.8 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.6 | uniform | gmm_12 | 2 | 252347 | 28.4775 | 30.2339 | 0.529483 | -0.444333 | 0.0951911 | 24029.9 | 70746 | 0 | 1 | 1 | 1 |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 2 | 207424 | 27.0653 | 28.8055 | 0.524305 | -0.432426 | 0.109347 | 21747.2 | 57020 | 0.5 | 1 | 1 | 0.5 |
| basic_mvn | 0.7 | evidence_improving | gmm_12 | 2 | 207424 | 26.9534 | 28.6983 | 0.550647 | -0.455491 | 0.115123 | 22621.5 | 62893.6 | 0.5 | 1 | 1 | 0.5 |
| basic_mvn | 0.7 | posterior_improving | ellipsoidal_12 | 2 | 207424 | 27.059 | 28.7675 | 0.529724 | -0.430217 | 0.120216 | 23688.1 | 58204.9 | 0.5 | 1 | 1 | 0.5 |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 2 | 207424 | 27.1423 | 28.879 | 0.471733 | -0.400848 | 0.101802 | 20507.2 | 46158.6 | 0.5 | 1 | 1 | 0.5 |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 2 | 207424 | 26.4214 | 28.0606 | 0.504569 | -0.419211 | 0.116063 | 22260.1 | 52808.3 | 0.5 | 1 | 1 | 0.5 |
| basic_mvn | 0.7 | uniform | gmm_12 | 2 | 207424 | 26.5369 | 28.1758 | 0.49174 | -0.415518 | 0.128228 | 24185.7 | 50156.9 | 0.5 | 1 | 1 | 0.5 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok | min_samples_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 0 | 162993 | 24.1507 | 25.7841 | -24.6065 | -25.3065 | 0.694514 | 0.396097 | -0.700017 | 25572.4 | 79870.6 | 1.76729 | true | true | true |
| basic_mvn | 0.7 | uniform | ellipsoidal_12 | 17 | 251856 | 28.6922 | 30.3371 | -24.6065 | -24.7449 | 0.720523 | 0.274285 | -0.138405 | 18947.7 | 4824.56 | 0.504603 | true | false | true |
| basic_mvn | 0.7 | uniform | gmm_12 | 0 | 162993 | 24.2033 | 25.8081 | -24.6065 | -25.2849 | 0.694514 | 0.427216 | -0.678487 | 29748.4 | 75033 | 1.58816 | true | true | true |
| basic_mvn | 0.7 | uniform | gmm_12 | 17 | 251856 | 28.8706 | 30.5435 | -24.6065 | -24.759 | 0.720523 | 0.271925 | -0.15255 | 18623 | 5861.04 | 0.560999 | true | false | true |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 0 | 162993 | 24.8227 | 26.5225 | -24.6065 | -25.3354 | 0.694514 | 0.361067 | -0.728911 | 21249.3 | 86599.9 | 2.01877 | true | true | true |
| basic_mvn | 0.7 | evidence_improving | ellipsoidal_12 | 17 | 251856 | 29.3079 | 31.0884 | -24.6065 | -24.7424 | 0.720523 | 0.297195 | -0.135941 | 22245.1 | 4654.28 | 0.457414 | true | false | true |
| basic_mvn | 0.7 | evidence_improving | gmm_12 | 0 | 162993 | 24.6347 | 26.3528 | -24.6065 | -25.3714 | 0.694514 | 0.378725 | -0.76491 | 23378.5 | 95365.1 | 2.0197 | true | true | true |
| basic_mvn | 0.7 | evidence_improving | gmm_12 | 17 | 251856 | 29.2721 | 31.0438 | -24.6065 | -24.7525 | 0.720523 | 0.294641 | -0.146072 | 21864.4 | 5373.88 | 0.495764 | true | false | true |
| basic_mvn | 0.7 | posterior_improving | ellipsoidal_12 | 0 | 162993 | 24.7607 | 26.4781 | -24.6065 | -25.3457 | 0.694514 | 0.385088 | -0.739282 | 24170.7 | 89081.8 | 1.91977 | true | true | true |
| basic_mvn | 0.7 | posterior_improving | ellipsoidal_12 | 17 | 251856 | 29.3573 | 31.0569 | -24.6065 | -24.7276 | 0.720523 | 0.303543 | -0.121152 | 23205.6 | 3696.66 | 0.399125 | true | false | true |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 0 | 162993 | 24.9676 | 26.788 | -24.6065 | -25.256 | 0.694514 | 0.339869 | -0.649551 | 18827.5 | 68769.3 | 1.91118 | true | true | true |
| basic_mvn | 0.7 | posterior_improving | gmm_12 | 17 | 251856 | 29.317 | 30.97 | -24.6065 | -24.7586 | 0.720523 | 0.296806 | -0.152145 | 22187 | 5830.01 | 0.512608 | true | false | true |
| basic_mvn | 0.6 | uniform | ellipsoidal_12 | 0 | 252838 | 28.0098 | 29.8708 | -24.6065 | -25.3165 | 0.692103 | 0.368119 | -0.71004 | 34262.5 | 127470 | 1.92883 | true | false | true |
| basic_mvn | 0.6 | uniform | ellipsoidal_12 | 17 | 251856 | 28.6521 | 30.3719 | -24.6065 | -24.7235 | 0.720523 | 0.319951 | -0.117012 | 25782.1 | 3448.36 | 0.365718 | true | false | true |
| basic_mvn | 0.6 | uniform | gmm_12 | 0 | 252838 | 28.0493 | 29.8299 | -24.6065 | -25.3388 | 0.692103 | 0.336003 | -0.732291 | 28544.9 | 135585 | 2.17942 | true | false | true |
| basic_mvn | 0.6 | uniform | gmm_12 | 17 | 251856 | 28.9057 | 30.6378 | -24.6065 | -24.7628 | 0.720523 | 0.27836 | -0.156375 | 19514.9 | 6158.69 | 0.561773 | true | false | true |
| basic_mvn | 0.6 | evidence_improving | ellipsoidal_12 | 0 | 252838 | 28.6224 | 30.4318 | -24.6065 | -25.3559 | 0.692103 | 0.353362 | -0.749467 | 31570.5 | 142019 | 2.12096 | true | false | true |
| basic_mvn | 0.6 | evidence_improving | ellipsoidal_12 | 17 | 251856 | 29.534 | 31.3287 | -24.6065 | -24.7675 | 0.720523 | 0.311781 | -0.16105 | 24482.3 | 6532.4 | 0.516547 | true | false | true |
| basic_mvn | 0.6 | evidence_improving | gmm_12 | 0 | 252838 | 28.8648 | 30.7046 | -24.6065 | -25.3459 | 0.692103 | 0.382135 | -0.739415 | 36921.1 | 138235 | 1.93496 | true | false | true |
| basic_mvn | 0.6 | evidence_improving | gmm_12 | 17 | 251856 | 29.3725 | 31.0176 | -24.6065 | -24.7524 | 0.720523 | 0.302362 | -0.145958 | 23025.3 | 5365.5 | 0.482728 | true | false | true |
| basic_mvn | 0.6 | posterior_improving | ellipsoidal_12 | 0 | 252838 | 28.7311 | 30.5043 | -24.6065 | -25.3557 | 0.692103 | 0.357167 | -0.749237 | 32254.1 | 141932 | 2.09772 | true | false | true |
| basic_mvn | 0.6 | posterior_improving | ellipsoidal_12 | 17 | 251856 | 29.5442 | 31.2441 | -24.6065 | -24.7612 | 0.720523 | 0.270693 | -0.154735 | 18454.7 | 6030.18 | 0.571625 | true | false | true |
| basic_mvn | 0.6 | posterior_improving | gmm_12 | 0 | 252838 | 29.0406 | 30.8259 | -24.6065 | -25.3414 | 0.692103 | 0.348462 | -0.73497 | 30701.1 | 136578 | 2.10918 | true | false | true |
| basic_mvn | 0.6 | posterior_improving | gmm_12 | 17 | 251856 | 29.4415 | 31.1742 | -24.6065 | -24.7659 | 0.720523 | 0.310274 | -0.15944 | 24246.1 | 6402.5 | 0.51387 | true | false | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: ellipsoidal_12/uniform, gmm_12/posterior_improving.
- Some rows did not reach the requested uncertainty. Check `sample_cap_fraction` and `max_goal_iterations` before drawing strong efficiency conclusions.
- GMM settings won 1 problem/target groups; compare against ellipsoidal rows to decide whether GMM fitting cost is buying lower variance.