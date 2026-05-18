# Pure-Core Efficiency Grid

Generated: `2026-05-12_145000 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn, spike_slab`
- Allocation targets: `uniform, evidence_improving, posterior_improving`
- LogZ uncertainty targets: `1.2, 0.9, 0.7`
- Seeds: `0, 17`
- Target live points: `40`
- Max samples: `960`
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
| basic_mvn | 0.7 | evidence_improving | gmm_6 | 36706 | 1.80158 | 119136 | 6099.67 | 0 | 0 |
| basic_mvn | 0.9 | posterior_improving | gmm_6 | 36706 | 1.77808 | 116049 | 5530.9 | 1 | 0 |
| basic_mvn | 1.2 | uniform | gmm_6 | 36706 | 1.75316 | 112818 | 6299.79 | 1 | 0 |
| spike_slab | 0.7 | posterior_improving | gmm_6 | 13590.5 | 1.12539 | 17212.4 | 2378.26 | 1 | 0.5 |
| spike_slab | 0.9 | evidence_improving | gmm_6 | 13590.5 | 1.12542 | 17213.3 | 2438.6 | 1 | 0.5 |
| spike_slab | 1.2 | evidence_improving | gmm_6 | 13590.5 | 1.12035 | 17058.5 | 2415.11 | 1 | 0.5 |

## Best Usable By Problem And Target

Rows here require `accuracy_success_fraction >= 0.5`; this prevents biased early stops from winning solely because the MC variance estimate is small.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spike_slab | 0.7 | posterior_improving | gmm_6 | 13590.5 | 1.12539 | 17212.4 | 2378.26 | 1 | 0.5 |
| spike_slab | 0.9 | evidence_improving | gmm_6 | 13590.5 | 1.12542 | 17213.3 | 2438.6 | 1 | 0.5 |
| spike_slab | 1.2 | evidence_improving | gmm_6 | 13590.5 | 1.12035 | 17058.5 | 2415.11 | 1 | 0.5 |

## Best Strict By Problem And Target

Rows here require both `accuracy_success_fraction == 1` and `target_success_fraction == 1`.

_No rows._

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | min_sample_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.7 | evidence_improving | gmm_6 | 2 | 36706 | 31.9979 | 33.5017 | 1.80158 | -1.78536 | 0.168224 | 6099.67 | 119136 | 0 | 0 | 1 | 1 |
| basic_mvn | 0.7 | posterior_improving | gmm_6 | 2 | 36706 | 32.1406 | 33.634 | 1.83915 | -1.8253 | 0.161111 | 5795.87 | 124157 | 0 | 0 | 1 | 1 |
| basic_mvn | 0.7 | uniform | gmm_6 | 2 | 36706 | 31.2804 | 32.8678 | 1.82021 | -1.80338 | 0.148018 | 5259.93 | 121613 | 0 | 0 | 1 | 1 |
| basic_mvn | 0.9 | evidence_improving | gmm_6 | 2 | 36706 | 31.8216 | 33.2979 | 1.81969 | -1.79992 | 0.157174 | 5731.08 | 121543 | 1 | 0 | 1 | 1 |
| basic_mvn | 0.9 | posterior_improving | gmm_6 | 2 | 36706 | 32.1701 | 33.6986 | 1.77808 | -1.76395 | 0.152806 | 5530.9 | 116049 | 1 | 0 | 1 | 1 |
| basic_mvn | 0.9 | uniform | gmm_6 | 2 | 36706 | 31.0801 | 32.676 | 1.79551 | -1.78488 | 0.14901 | 5370.26 | 118334 | 1 | 0 | 1 | 1 |
| basic_mvn | 1.2 | evidence_improving | gmm_6 | 2 | 36706 | 31.7419 | 33.2376 | 1.79583 | -1.78127 | 0.166866 | 6057.07 | 118376 | 1 | 0 | 1 | 1 |
| basic_mvn | 1.2 | posterior_improving | gmm_6 | 2 | 36706 | 32.0039 | 33.4973 | 1.78377 | -1.76477 | 0.136986 | 4991.27 | 116792 | 1 | 0 | 1 | 1 |
| basic_mvn | 1.2 | uniform | gmm_6 | 2 | 36706 | 31.4509 | 33.0451 | 1.75316 | -1.72985 | 0.17374 | 6299.79 | 112818 | 1 | 0 | 1 | 1 |
| spike_slab | 0.7 | evidence_improving | gmm_6 | 2 | 13590.5 | 35.6148 | 37.3105 | 1.13849 | 0.624961 | 0.169423 | 2294.04 | 17615.5 | 1 | 0.5 | 1 | 1 |
| spike_slab | 0.7 | posterior_improving | gmm_6 | 2 | 13590.5 | 36.1001 | 37.8728 | 1.12539 | 0.609081 | 0.174906 | 2378.26 | 17212.4 | 1 | 0.5 | 1 | 1 |
| spike_slab | 0.7 | uniform | gmm_6 | 2 | 13590.5 | 34.962 | 36.7694 | 1.15182 | 0.616394 | 0.149519 | 2020.3 | 18030.5 | 1 | 0.5 | 1 | 1 |
| spike_slab | 0.9 | evidence_improving | gmm_6 | 2 | 13590.5 | 35.6586 | 37.3651 | 1.12542 | 0.608345 | 0.180125 | 2438.6 | 17213.3 | 1 | 0.5 | 1 | 1 |
| spike_slab | 0.9 | posterior_improving | gmm_6 | 2 | 13590.5 | 35.8493 | 37.5414 | 1.12573 | 0.602997 | 0.185535 | 2511.76 | 17222.9 | 1 | 0.5 | 1 | 1 |
| spike_slab | 0.9 | uniform | gmm_6 | 2 | 13590.5 | 34.8243 | 36.6589 | 1.13954 | 0.645352 | 0.170723 | 2307.27 | 17647.9 | 1 | 0.5 | 1 | 1 |
| spike_slab | 1.2 | evidence_improving | gmm_6 | 2 | 13590.5 | 36.0277 | 37.753 | 1.12035 | 0.616918 | 0.177179 | 2415.11 | 17058.5 | 1 | 0.5 | 1 | 1 |
| spike_slab | 1.2 | posterior_improving | gmm_6 | 2 | 13590.5 | 36.1181 | 37.8087 | 1.16569 | 0.614367 | 0.191773 | 2607.49 | 18467.2 | 1 | 0.5 | 1 | 1 |
| spike_slab | 1.2 | uniform | gmm_6 | 2 | 13590.5 | 34.8818 | 36.7011 | 1.15298 | 0.636943 | 0.173779 | 2362.87 | 18066.7 | 1 | 0.5 | 1 | 1 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok | min_samples_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 1.2 | uniform | gmm_6 | 0 | 41051 | 30.5365 | 32.1809 | -24.6065 | -26.6212 | 0.719761 | 0.394846 | -2.01476 | 6400 | 166637 | 5.10264 | false | true | true |
| basic_mvn | 1.2 | uniform | gmm_6 | 17 | 32361 | 32.3654 | 33.9093 | -24.6065 | -26.0514 | 0.73178 | 0.437694 | -1.44494 | 6199.59 | 67565 | 3.30126 | false | true | true |
| basic_mvn | 1.2 | evidence_improving | gmm_6 | 0 | 41051 | 30.7444 | 32.233 | -24.6065 | -26.6159 | 0.719761 | 0.388885 | -2.00946 | 6208.21 | 165761 | 5.16723 | false | true | true |
| basic_mvn | 1.2 | evidence_improving | gmm_6 | 17 | 32361 | 32.7394 | 34.2422 | -24.6065 | -26.1595 | 0.73178 | 0.427201 | -1.55308 | 5905.92 | 78056.5 | 3.63547 | false | true | true |
| basic_mvn | 1.2 | posterior_improving | gmm_6 | 0 | 41051 | 31.4208 | 32.9417 | -24.6065 | -26.6309 | 0.719761 | 0.358448 | -2.02439 | 5274.43 | 168233 | 5.64765 | false | true | true |
| basic_mvn | 1.2 | posterior_improving | gmm_6 | 17 | 32361 | 32.587 | 34.0529 | -24.6065 | -26.1116 | 0.73178 | 0.381428 | -1.50516 | 4708.12 | 73314.2 | 3.94612 | false | true | true |
| basic_mvn | 0.9 | uniform | gmm_6 | 0 | 41051 | 30.082 | 31.7294 | -24.6065 | -26.5864 | 0.719761 | 0.355185 | -1.97997 | 5178.86 | 160931 | 5.57446 | false | true | true |
| basic_mvn | 0.9 | uniform | gmm_6 | 17 | 32361 | 32.0782 | 33.6225 | -24.6065 | -26.1962 | 0.73178 | 0.414564 | -1.58978 | 5561.67 | 81789.7 | 3.83483 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | gmm_6 | 0 | 41051 | 31.0221 | 32.4661 | -24.6065 | -26.6738 | 0.719761 | 0.385215 | -2.06737 | 6091.57 | 175453 | 5.36681 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | gmm_6 | 17 | 32361 | 32.6212 | 34.1296 | -24.6065 | -26.1389 | 0.73178 | 0.40738 | -1.53248 | 5370.59 | 75999.2 | 3.76178 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | gmm_6 | 0 | 41051 | 31.4323 | 32.9441 | -24.6065 | -26.5941 | 0.719761 | 0.367224 | -1.98767 | 5535.88 | 162186 | 5.41269 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | gmm_6 | 17 | 32361 | 32.9078 | 34.4531 | -24.6065 | -26.1467 | 0.73178 | 0.41323 | -1.54023 | 5525.92 | 76770.2 | 3.7273 | false | true | true |
| basic_mvn | 0.7 | uniform | gmm_6 | 0 | 41051 | 30.2421 | 31.9097 | -24.6065 | -26.6568 | 0.719761 | 0.328866 | -2.0503 | 4439.78 | 172568 | 6.23447 | false | false | true |
| basic_mvn | 0.7 | uniform | gmm_6 | 17 | 32361 | 32.3186 | 33.8259 | -24.6065 | -26.1629 | 0.73178 | 0.433454 | -1.55646 | 6080.07 | 78396.4 | 3.59082 | false | false | true |
| basic_mvn | 0.7 | evidence_improving | gmm_6 | 0 | 41051 | 31.2962 | 32.7737 | -24.6065 | -26.633 | 0.719761 | 0.388489 | -2.02652 | 6195.58 | 168588 | 5.21642 | false | false | true |
| basic_mvn | 0.7 | evidence_improving | gmm_6 | 17 | 32361 | 32.6996 | 34.2297 | -24.6065 | -26.1507 | 0.73178 | 0.430726 | -1.5442 | 6003.76 | 77166.7 | 3.58512 | false | false | true |
| basic_mvn | 0.7 | posterior_improving | gmm_6 | 0 | 41051 | 31.3512 | 32.853 | -24.6065 | -26.6571 | 0.719761 | 0.36604 | -2.05061 | 5500.23 | 172620 | 5.60216 | false | false | true |
| basic_mvn | 0.7 | posterior_improving | gmm_6 | 17 | 32361 | 32.93 | 34.4149 | -24.6065 | -26.2064 | 0.73178 | 0.433862 | -1.59998 | 6091.51 | 82841.8 | 3.68776 | false | false | true |
| spike_slab | 1.2 | uniform | gmm_6 | 0 | 13937 | 35.4722 | 37.3723 | -18.3959 | -16.7979 | 0.621414 | 0.420769 | 1.59802 | 2467.49 | 35590.4 | 3.79786 | false | true | true |
| spike_slab | 1.2 | uniform | gmm_6 | 17 | 13244 | 34.2914 | 36.03 | -18.3959 | -18.72 | 0.642516 | 0.41293 | -0.324134 | 2258.25 | 1391.45 | 0.784961 | true | true | true |
| spike_slab | 1.2 | evidence_improving | gmm_6 | 0 | 13937 | 36.6829 | 38.4192 | -18.3959 | -16.8438 | 0.621414 | 0.44482 | 1.55212 | 2757.65 | 33575.1 | 3.48931 | false | true | true |
| spike_slab | 1.2 | evidence_improving | gmm_6 | 17 | 13244 | 35.3725 | 37.0869 | -18.3959 | -18.7142 | 0.642516 | 0.395591 | -0.318279 | 2072.58 | 1341.64 | 0.804567 | true | true | true |
| spike_slab | 1.2 | posterior_improving | gmm_6 | 0 | 13937 | 36.4097 | 38.1063 | -18.3959 | -16.7909 | 0.621414 | 0.441875 | 1.60502 | 2721.24 | 35902.8 | 3.63229 | false | true | true |
| spike_slab | 1.2 | posterior_improving | gmm_6 | 17 | 13244 | 35.8265 | 37.5111 | -18.3959 | -18.7722 | 0.642516 | 0.433927 | -0.376282 | 2493.75 | 1875.19 | 0.867156 | true | true | true |
| spike_slab | 0.9 | uniform | gmm_6 | 0 | 13937 | 35.4221 | 37.3225 | -18.3959 | -16.8114 | 0.621414 | 0.365231 | 1.58454 | 1859.11 | 34992.5 | 4.33846 | false | true | true |
| spike_slab | 0.9 | uniform | gmm_6 | 17 | 13244 | 34.2265 | 35.9952 | -18.3959 | -18.6897 | 0.642516 | 0.456127 | -0.293834 | 2755.44 | 1143.47 | 0.644194 | true | true | true |
| spike_slab | 0.9 | evidence_improving | gmm_6 | 0 | 13937 | 36.0773 | 37.7772 | -18.3959 | -16.8407 | 0.621414 | 0.391196 | 1.55518 | 2132.84 | 33707.6 | 3.97544 | false | true | true |
| spike_slab | 0.9 | evidence_improving | gmm_6 | 17 | 13244 | 35.24 | 36.953 | -18.3959 | -18.7344 | 0.642516 | 0.45521 | -0.338485 | 2744.37 | 1517.39 | 0.74358 | true | true | true |
| spike_slab | 0.9 | posterior_improving | gmm_6 | 0 | 13937 | 36.3909 | 38.0938 | -18.3959 | -16.8423 | 0.621414 | 0.396719 | 1.55361 | 2193.49 | 33639.8 | 3.91615 | false | true | true |
| spike_slab | 0.9 | posterior_improving | gmm_6 | 17 | 13244 | 35.3077 | 36.9889 | -18.3959 | -18.7435 | 0.642516 | 0.462259 | -0.347616 | 2830.02 | 1600.36 | 0.751993 | true | true | true |
| spike_slab | 0.7 | uniform | gmm_6 | 0 | 13937 | 35.2848 | 37.1468 | -18.3959 | -16.8065 | 0.621414 | 0.340077 | 1.58941 | 1611.85 | 35207.8 | 4.67366 | false | true | true |
| spike_slab | 0.7 | uniform | gmm_6 | 17 | 13244 | 34.6393 | 36.392 | -18.3959 | -18.7525 | 0.642516 | 0.428235 | -0.35662 | 2428.75 | 1684.34 | 0.832767 | true | true | true |
| spike_slab | 0.7 | evidence_improving | gmm_6 | 0 | 13937 | 36.3337 | 38.063 | -18.3959 | -16.8193 | 0.621414 | 0.380624 | 1.57659 | 2019.12 | 34642.1 | 4.14211 | false | true | true |
| spike_slab | 0.7 | evidence_improving | gmm_6 | 17 | 13244 | 34.896 | 36.5579 | -18.3959 | -18.7226 | 0.642516 | 0.440423 | -0.326663 | 2568.97 | 1413.25 | 0.741703 | true | true | true |
| spike_slab | 0.7 | posterior_improving | gmm_6 | 0 | 13937 | 36.6548 | 38.4971 | -18.3959 | -16.8405 | 0.621414 | 0.422361 | 1.5554 | 2486.2 | 33717.4 | 3.68264 | false | true | true |
| spike_slab | 0.7 | posterior_improving | gmm_6 | 17 | 13244 | 35.5454 | 37.2484 | -18.3959 | -18.7331 | 0.642516 | 0.414032 | -0.337239 | 2270.32 | 1506.24 | 0.814524 | true | true | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: gmm_6/evidence_improving, gmm_6/posterior_improving, gmm_6/uniform.
- `spike_slab` best rows by target were: gmm_6/posterior_improving, gmm_6/evidence_improving, gmm_6/evidence_improving.
- Some rows did not reach the requested uncertainty. Check `sample_cap_fraction` and `max_goal_iterations` before drawing strong efficiency conclusions.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.
- At least one row reached the uncertainty target while missing the analytic accuracy gate for every seed. Use the minimum-sample guard or stricter accuracy-gated tables when comparing `evals * variance`.
- GMM settings won 6 problem/target groups; compare against ellipsoidal rows to decide whether GMM fitting cost is buying lower variance.