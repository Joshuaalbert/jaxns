# SS10: 30 completed continuations from 0.02 to 0.015

All 30 seeds completed without failures. The whole queue took 9 h 35 min; summed worker continuation time was 58.84 core-hours. Median worker time was 1.72 h, with a range of 0.99–3.92 h. All workers used one pinned CPU.

| Classic metric | Archived 0.02 | Efficient A+C at 0.015 |
|---|---:|---:|
| log-Z bias | -0.291138 | -0.249782 |
| log-Z RMSE | 0.526813 | 0.472594 |
| Mean spike mass (truth ≈ 0.500008) | 0.275926 | 0.312654 |
| Spike-mass RMSE | 0.322728 | 0.279686 |
| Nominal 95% log-Z interval coverage | 2/30 | 1/30 |

Recovery improves on average, but most runs still misestimate the spike mass. Reaching the reported uncertainty target does not establish correct mode weights. Phantom rows do not contribute posterior weights.

Paired-seed bootstrap 95% intervals for RMSE change (0.015 minus 0.02): log Z [-0.06688331185112142, -0.04167856986705247]; spike mass [-0.053700307756125776, -0.032914627266562725]. These use 20,000 paired resamples and describe this cohort.

Each seed added 86–129 complete goals (median 100); the cohort added 5.180 billion likelihood evaluations. Measured process peak RSS during continuation ranged from 28.52 to 55.04 GiB (median 28.92 GiB), including checkpointing. This differs from the retained-input, repeated one-goal benchmark buffer lifetime; it is a complete continuation measurement.

Sampler source is unchanged from 650ae05; runner source was frozen at 82cd8d3. The input source is frozen old A+C aa79a0d. Across the seven available old partial trajectories, 590 overlapping goals exactly matched the checked allocation/root counts, classic sample counts, likelihood counts, classic log Z, uncertainty and Kish ESS. Scientific sample hashes and continuation keys are recorded for every final state. The preliminary one-goal and checkpoint-reload smoke checks matched every scientific hash against the earlier paired benchmark.

Timing includes load, one-time index conversion, compilation, stopping metric evaluation and periodic checkpoints. Final accuracy and hashing follow that timer. The 0.02 baseline uses its original classic expected evidence and classic posterior mass. No new phantom-prefix Monte Carlo analysis is included.

All final states and checkpoint hard links remain under `results/ss10-0015-20260910/seed-XX/`. They occupy about 320.86 GiB in total. The dispatcher is finished and no jobs remain queued. [Compact evidence and provenance](ss10-0015.json).

| Seed | Added goals | Worker hours | Spike mass at 0.02 | At 0.015 | log-Z error at 0.015 |
|---|---:|---:|---:|---:|---:|
| 0 | 89 | 1.101 | 0.029279 | 0.068254 | -0.621193 |
| 1 | 89 | 1.218 | 0.039894 | 0.085426 | -0.592307 |
| 2 | 90 | 1.381 | 0.090177 | 0.119415 | -0.571481 |
| 3 | 129 | 3.918 | 0.819023 | 0.804980 | 0.933288 |
| 4 | 103 | 1.643 | 0.058341 | 0.179152 | -0.484362 |
| 5 | 99 | 1.721 | 0.233574 | 0.261921 | -0.370643 |
| 6 | 100 | 1.732 | 0.213535 | 0.258157 | -0.401555 |
| 7 | 107 | 2.000 | 0.245698 | 0.315884 | -0.327434 |
| 8 | 111 | 2.383 | 0.511138 | 0.500975 | -0.003074 |
| 9 | 110 | 2.025 | 0.265490 | 0.343301 | -0.278945 |
| 10 | 117 | 2.689 | 0.614703 | 0.600847 | 0.229048 |
| 11 | 93 | 1.431 | 0.084354 | 0.134178 | -0.544713 |
| 12 | 101 | 1.672 | 0.122689 | 0.209195 | -0.441551 |
| 13 | 94 | 1.482 | 0.118145 | 0.165884 | -0.502898 |
| 14 | 115 | 2.544 | 0.534251 | 0.541007 | 0.100178 |
| 15 | 86 | 0.990 | 0.026245 | 0.052961 | -0.651725 |
| 16 | 117 | 2.602 | 0.491307 | 0.517614 | 0.051746 |
| 17 | 100 | 1.775 | 0.165069 | 0.223664 | -0.422896 |
| 18 | 97 | 1.531 | 0.149602 | 0.221104 | -0.466852 |
| 19 | 100 | 1.611 | 0.086196 | 0.178749 | -0.483963 |
| 20 | 91 | 1.443 | 0.114560 | 0.143914 | -0.514494 |
| 21 | 118 | 2.759 | 0.588667 | 0.585762 | 0.179619 |
| 22 | 94 | 1.487 | 0.107210 | 0.159507 | -0.506543 |
| 23 | 109 | 2.488 | 0.448504 | 0.454149 | -0.086017 |
| 24 | 98 | 1.718 | 0.221254 | 0.245093 | -0.432865 |
| 25 | 96 | 1.456 | 0.072263 | 0.142558 | -0.543387 |
| 26 | 129 | 3.827 | 0.797413 | 0.788475 | 0.862765 |
| 27 | 95 | 1.613 | 0.148229 | 0.189360 | -0.472033 |
| 28 | 101 | 1.948 | 0.275678 | 0.296342 | -0.350166 |
| 29 | 117 | 2.653 | 0.605279 | 0.591800 | 0.221009 |
