# Pure-Core Efficiency Grid

Generated: `2026-05-12_151142 UTC`

## Objective

Compare pure-JAX nested-sampling settings by likelihood work needed for fixed log-evidence precision. Lower values are better for both `mean_evals_times_variance` and `mean_evals_times_mse`.

The stopping target is `result.log_Z_uncert`; MC shrinkage variance and analytic-reference RMSE are reported separately. `mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` across seeds.

## Configuration

- Problems: `basic_mvn, spike_slab`
- Allocation targets: `uniform, evidence_improving, posterior_improving`
- LogZ uncertainty targets: `0.9`
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
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 77199 | 0.34806 | 9352.33 | 8110.47 | 1 | 1 |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 26762 | 0.769487 | 15846 | 2960.04 | 1 | 1 |

## Best Usable By Problem And Target

Rows here require `accuracy_success_fraction >= 0.5`; this prevents biased early stops from winning solely because the MC variance estimate is small.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 77199 | 0.34806 | 9352.33 | 8110.47 | 1 | 1 |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 26762 | 0.769487 | 15846 | 2960.04 | 1 | 1 |

## Best Strict By Problem And Target

Rows here require both `accuracy_success_fraction == 1` and `target_success_fraction == 1`.

| problem | logz_uncert_target | allocation_target | sampler_setting | mean_likelihood_evaluations | rmse_logZ | mean_evals_times_mse | mean_evals_times_variance | target_success_fraction | accuracy_success_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 77199 | 0.34806 | 9352.33 | 8110.47 | 1 | 1 |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 26762 | 0.769487 | 15846 | 2960.04 | 1 | 1 |

## Rollups

| problem | logz_uncert_target | allocation_target | sampler_setting | num_seeds | mean_likelihood_evaluations | mean_run_seconds | mean_wall_seconds | rmse_logZ | bias_logZ | mean_mc_logZ_variance | mean_evals_times_variance | mean_evals_times_mse | target_success_fraction | accuracy_success_fraction | min_sample_fraction | sample_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | evidence_improving | ellipsoidal_12 | 2 | 77199 | 32.5152 | 34.0316 | 0.363939 | -0.20726 | 0.0904231 | 6973.15 | 10225.1 | 1 | 1 | 1 | 1 |
| basic_mvn | 0.9 | evidence_improving | gmm_12 | 2 | 77199 | 32.8573 | 34.382 | 0.354077 | -0.192644 | 0.0899168 | 6957.43 | 9678.46 | 1 | 1 | 1 | 1 |
| basic_mvn | 0.9 | evidence_improving | isotropic_12 | 2 | 105125 | 21.5198 | 23.0516 | 1.6836 | -1.56572 | 0.0831031 | 8749.76 | 297976 | 1 | 0 | 1 | 1 |
| basic_mvn | 0.9 | posterior_improving | ellipsoidal_12 | 2 | 77199 | 32.5244 | 34.0344 | 0.358488 | -0.203275 | 0.0998932 | 7712.78 | 9921.1 | 1 | 1 | 1 | 1 |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 2 | 77199 | 32.9163 | 34.4965 | 0.34806 | -0.201871 | 0.105039 | 8110.47 | 9352.33 | 1 | 1 | 1 | 1 |
| basic_mvn | 0.9 | posterior_improving | isotropic_12 | 2 | 105125 | 21.4684 | 22.9758 | 1.66095 | -1.56038 | 0.074708 | 7991.89 | 290015 | 1 | 0 | 1 | 1 |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 2 | 77199 | 32.0994 | 33.6634 | 0.360925 | -0.218007 | 0.109844 | 8497 | 10056.5 | 1 | 1 | 1 | 1 |
| basic_mvn | 0.9 | uniform | gmm_12 | 2 | 77199 | 32.0287 | 33.57 | 0.370265 | -0.223698 | 0.0988311 | 7614.04 | 10583.7 | 1 | 1 | 1 | 1 |
| basic_mvn | 0.9 | uniform | isotropic_12 | 2 | 105125 | 20.933 | 22.4532 | 1.67076 | -1.55511 | 0.0600685 | 6346.94 | 293449 | 1 | 0 | 1 | 1 |
| spike_slab | 0.9 | evidence_improving | ellipsoidal_12 | 2 | 26762 | 33.4163 | 35.1147 | 0.80895 | 0.776152 | 0.114228 | 3063.7 | 17513.1 | 1 | 1 | 1 | 1 |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 2 | 26762 | 33.4303 | 35.2202 | 0.769487 | 0.750838 | 0.11014 | 2960.04 | 15846 | 1 | 1 | 1 | 1 |
| spike_slab | 0.9 | evidence_improving | isotropic_12 | 2 | 28692 | 22.5495 | 24.2495 | 1.34298 | 0.955564 | 0.116617 | 3343.53 | 51748.8 | 1 | 0.5 | 1 | 1 |
| spike_slab | 0.9 | posterior_improving | ellipsoidal_12 | 2 | 26762 | 33.7263 | 35.4321 | 0.794221 | 0.775092 | 0.14966 | 4013.94 | 16881.1 | 1 | 1 | 1 | 1 |
| spike_slab | 0.9 | posterior_improving | gmm_12 | 2 | 26762 | 33.4875 | 35.234 | 0.785539 | 0.764027 | 0.109181 | 2924.15 | 16514 | 1 | 1 | 1 | 1 |
| spike_slab | 0.9 | posterior_improving | isotropic_12 | 2 | 28692 | 22.6924 | 24.4078 | 1.33537 | 0.943018 | 0.130822 | 3747.6 | 51163.8 | 1 | 0.5 | 1 | 1 |
| spike_slab | 0.9 | uniform | ellipsoidal_12 | 2 | 26762 | 32.8448 | 34.6059 | 0.811998 | 0.777619 | 0.124955 | 3342.73 | 17645.3 | 1 | 1 | 1 | 1 |
| spike_slab | 0.9 | uniform | gmm_12 | 2 | 26762 | 32.7602 | 34.493 | 0.832087 | 0.808349 | 0.122312 | 3279.22 | 18529.2 | 1 | 1 | 1 | 1 |
| spike_slab | 0.9 | uniform | isotropic_12 | 2 | 28692 | 22.3779 | 24.1549 | 1.3165 | 0.935079 | 0.121959 | 3499.57 | 49727.8 | 1 | 0.5 | 1 | 1 |

## Per-Seed Records

| problem | target | allocation | setting | seed | evals | run_s | wall_s | logZ_ref | mc_logZ_mean | logZ_uncert | mc_logZ_std | error | evals_x_var | evals_x_sqerr | err_over_std | accuracy_ok | target_ok | min_samples_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic_mvn | 0.9 | uniform | isotropic_12 | 0 | 112540 | 20.9113 | 22.4242 | -24.6065 | -26.7724 | 0.774208 | 0.253801 | -2.1659 | 7249.27 | 527940 | 8.53386 | false | true | true |
| basic_mvn | 0.9 | uniform | isotropic_12 | 17 | 97710 | 20.9547 | 22.4822 | -24.6065 | -25.5508 | 0.767198 | 0.236055 | -0.944315 | 5444.6 | 87131.1 | 4.0004 | false | true | true |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 0 | 75069 | 32.0673 | 33.6109 | -24.6065 | -25.1121 | 0.717077 | 0.319052 | -0.505652 | 7641.59 | 19194 | 1.58486 | true | true | true |
| basic_mvn | 0.9 | uniform | ellipsoidal_12 | 17 | 79329 | 32.1314 | 33.7158 | -24.6065 | -24.5368 | 0.696369 | 0.343357 | 0.0696384 | 9352.4 | 384.707 | 0.202817 | true | true | true |
| basic_mvn | 0.9 | uniform | gmm_12 | 0 | 75069 | 31.9833 | 33.5213 | -24.6065 | -25.1252 | 0.717077 | 0.32583 | -0.51875 | 7969.69 | 20201.2 | 1.59209 | true | true | true |
| basic_mvn | 0.9 | uniform | gmm_12 | 17 | 79329 | 32.0742 | 33.6188 | -24.6065 | -24.5351 | 0.696369 | 0.302485 | 0.0713548 | 7258.38 | 403.904 | 0.235895 | true | true | true |
| basic_mvn | 0.9 | evidence_improving | isotropic_12 | 0 | 112540 | 21.5978 | 23.1686 | -24.6065 | -26.7911 | 0.774208 | 0.291427 | -2.1846 | 9558.01 | 537094 | 7.4962 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | isotropic_12 | 17 | 97710 | 21.4417 | 22.9346 | -24.6065 | -25.5533 | 0.767198 | 0.28509 | -0.946848 | 7941.52 | 87599.1 | 3.32122 | false | true | true |
| basic_mvn | 0.9 | evidence_improving | ellipsoidal_12 | 0 | 75069 | 32.3285 | 33.8221 | -24.6065 | -25.1129 | 0.717077 | 0.30644 | -0.506416 | 7049.4 | 19252 | 1.65258 | true | true | true |
| basic_mvn | 0.9 | evidence_improving | ellipsoidal_12 | 17 | 79329 | 32.7018 | 34.2412 | -24.6065 | -24.5146 | 0.696369 | 0.294857 | 0.091897 | 6896.91 | 669.938 | 0.311666 | true | true | true |
| basic_mvn | 0.9 | evidence_improving | gmm_12 | 0 | 75069 | 32.4664 | 34.0011 | -24.6065 | -25.0962 | 0.717077 | 0.287116 | -0.489727 | 6188.35 | 18004 | 1.70568 | true | true | true |
| basic_mvn | 0.9 | evidence_improving | gmm_12 | 17 | 79329 | 33.2482 | 34.7629 | -24.6065 | -24.502 | 0.696369 | 0.312087 | 0.10444 | 7726.5 | 865.3 | 0.334651 | true | true | true |
| basic_mvn | 0.9 | posterior_improving | isotropic_12 | 0 | 112540 | 21.4164 | 22.9279 | -24.6065 | -26.736 | 0.774208 | 0.305528 | -2.12957 | 10505.3 | 510378 | 6.97015 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | isotropic_12 | 17 | 97710 | 21.5204 | 23.0236 | -24.6065 | -25.5976 | 0.767198 | 0.236788 | -0.991181 | 5478.48 | 95994.2 | 4.18593 | false | true | true |
| basic_mvn | 0.9 | posterior_improving | ellipsoidal_12 | 0 | 75069 | 32.334 | 33.8627 | -24.6065 | -25.105 | 0.717077 | 0.315224 | -0.498559 | 7459.32 | 18659.2 | 1.5816 | true | true | true |
| basic_mvn | 0.9 | posterior_improving | ellipsoidal_12 | 17 | 79329 | 32.7148 | 34.206 | -24.6065 | -24.5145 | 0.696369 | 0.316891 | 0.092009 | 7966.23 | 671.571 | 0.290349 | true | true | true |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 0 | 75069 | 32.8715 | 34.5369 | -24.6065 | -25.0919 | 0.717077 | 0.322983 | -0.485409 | 7831.05 | 17687.9 | 1.50289 | true | true | true |
| basic_mvn | 0.9 | posterior_improving | gmm_12 | 17 | 79329 | 32.9612 | 34.4561 | -24.6065 | -24.5248 | 0.696369 | 0.325209 | 0.0816675 | 8389.9 | 529.091 | 0.251123 | true | true | true |
| spike_slab | 0.9 | uniform | isotropic_12 | 0 | 29101 | 22.3364 | 24.0828 | -18.3959 | -16.5341 | 0.631004 | 0.350405 | 1.86179 | 3573.12 | 100871 | 5.31324 | false | true | true |
| spike_slab | 0.9 | uniform | isotropic_12 | 17 | 28283 | 22.4193 | 24.227 | -18.3959 | -18.3875 | 0.666911 | 0.348042 | 0.00837131 | 3426.02 | 1.98204 | 0.0240525 | true | true | true |
| spike_slab | 0.9 | uniform | ellipsoidal_12 | 0 | 26358 | 33.5194 | 35.272 | -18.3959 | -17.852 | 0.633273 | 0.358049 | 0.543847 | 3379.07 | 7795.9 | 1.51892 | true | true | true |
| spike_slab | 0.9 | uniform | ellipsoidal_12 | 17 | 27166 | 32.1701 | 33.9397 | -18.3959 | -17.3845 | 0.622617 | 0.348871 | 1.01139 | 3306.4 | 27788.4 | 2.89904 | true | true | true |
| spike_slab | 0.9 | uniform | gmm_12 | 0 | 26358 | 33.4865 | 35.1982 | -18.3959 | -17.7849 | 0.633273 | 0.328152 | 0.611013 | 2838.33 | 9840.41 | 1.86198 | true | true | true |
| spike_slab | 0.9 | uniform | gmm_12 | 17 | 27166 | 32.0339 | 33.7877 | -18.3959 | -17.3902 | 0.622617 | 0.370054 | 1.00568 | 3720.1 | 27475.7 | 2.71767 | true | true | true |
| spike_slab | 0.9 | evidence_improving | isotropic_12 | 0 | 29101 | 22.8386 | 24.5397 | -18.3959 | -16.4967 | 0.631004 | 0.33263 | 1.89922 | 3219.82 | 104969 | 5.70972 | false | true | true |
| spike_slab | 0.9 | evidence_improving | isotropic_12 | 17 | 28283 | 22.2605 | 23.9594 | -18.3959 | -18.384 | 0.666911 | 0.35013 | 0.0119048 | 3467.23 | 4.00836 | 0.034001 | true | true | true |
| spike_slab | 0.9 | evidence_improving | ellipsoidal_12 | 0 | 26358 | 33.9806 | 35.6839 | -18.3959 | -17.8477 | 0.633273 | 0.31233 | 0.548144 | 2571.23 | 7919.59 | 1.75502 | true | true | true |
| spike_slab | 0.9 | evidence_improving | ellipsoidal_12 | 17 | 27166 | 32.8519 | 34.5454 | -18.3959 | -17.3917 | 0.622617 | 0.361808 | 1.00416 | 3556.17 | 27392.5 | 2.7754 | true | true | true |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 0 | 26358 | 33.9321 | 35.78 | -18.3959 | -17.8134 | 0.633273 | 0.28153 | 0.582457 | 2089.11 | 8942.13 | 2.0689 | true | true | true |
| spike_slab | 0.9 | evidence_improving | gmm_12 | 17 | 27166 | 32.9285 | 34.6604 | -18.3959 | -17.4767 | 0.622617 | 0.375527 | 0.919218 | 3830.96 | 22954.3 | 2.44781 | true | true | true |
| spike_slab | 0.9 | posterior_improving | isotropic_12 | 0 | 29101 | 22.9014 | 24.6283 | -18.3959 | -16.5074 | 0.631004 | 0.341028 | 1.88849 | 3384.44 | 103786 | 5.53766 | false | true | true |
| spike_slab | 0.9 | posterior_improving | isotropic_12 | 17 | 28283 | 22.4835 | 24.1873 | -18.3959 | -18.3983 | 0.666911 | 0.38124 | -0.00245774 | 4110.76 | 0.170842 | 0.00644669 | true | true | true |
| spike_slab | 0.9 | posterior_improving | ellipsoidal_12 | 0 | 26358 | 34.3465 | 36.0503 | -18.3959 | -17.7941 | 0.633273 | 0.35782 | 0.601829 | 3374.75 | 9546.83 | 1.68193 | true | true | true |
| spike_slab | 0.9 | posterior_improving | ellipsoidal_12 | 17 | 27166 | 33.106 | 34.8139 | -18.3959 | -17.4475 | 0.622617 | 0.413866 | 0.948354 | 4653.12 | 24432.4 | 2.29145 | true | true | true |
| spike_slab | 0.9 | posterior_improving | gmm_12 | 0 | 26358 | 34.049 | 35.7989 | -18.3959 | -17.8144 | 0.633273 | 0.321879 | 0.58145 | 2730.84 | 8911.21 | 1.80642 | true | true | true |
| spike_slab | 0.9 | posterior_improving | gmm_12 | 17 | 27166 | 32.9261 | 34.669 | -18.3959 | -17.4493 | 0.622617 | 0.338756 | 0.946603 | 3117.45 | 24342.3 | 2.79435 | true | true | true |

## First-Pass Hypotheses

- `basic_mvn` best rows by target were: gmm_12/posterior_improving.
- `spike_slab` best rows by target were: gmm_12/evidence_improving.
- Some rows failed the analytic logZ accuracy gate; treat those settings as biased or under-resolved rather than efficient.
- At least one row reached the uncertainty target while missing the analytic accuracy gate for every seed. Use the minimum-sample guard or stricter accuracy-gated tables when comparing `evals * variance`.
- GMM settings won 2 problem/target groups; compare against ellipsoidal rows to decide whether GMM fitting cost is buying lower variance.