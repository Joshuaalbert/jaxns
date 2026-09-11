# Memory forecasts for the next uncertainty stages

This is an operational forecast, not an experiment outcome. The snapshot uses
30 completed SS10 0.05 cores/analyses, 24 completed 0.02 cores, and 23 completed
0.02 analyses. Six 0.02 runs remain active. Final pool sizing will refit after
all 30 seeds finish each preceding stage.

| Phase / capacity | Measured peak range (GiB) |
|---|---:|
| 0.05 core / 107200 | 3.90–4.11 |
| 0.05 core / 214400 | 5.50–5.77 |
| 0.02 core / 857600 | 13.87–14.93 |
| 0.02 core / 1715200 | 26.24–27.33 |
| 0.02 analysis / 857600 | 14.02–18.53 |
| 0.02 analysis / 1715200 | 26.00–27.02 |

Fitting nonnegative fixed overhead plus terms for capacity and classic count
separates persistent storage from analysis temporaries. The sample-count
elasticity with respect to inverse achieved uncertainty is measured for each
matched seed: median 2.107, observed range 1.972–2.446. Forecasts retain a
quadratic floor, add 10% count headroom, and round capacity through the actual
power-of-two growth schedule. Each RSS model covers its largest positive
training residual and receives a further 20% memory margin.

Core RSS is explained primarily by capacity: approximately 2.51 GiB fixed
plus 14.15 GiB per million slots, with a 0.56 GiB residual allowance.
Analysis RSS also depends strongly on classic count: approximately 0.59 GiB
fixed, 5.32 GiB per million slots, and 18.90 GiB per million classic samples.
These are empirical extrapolations, not guaranteed physical bounds.

The models were tested by excluding the largest observed capacity and
predicting those held-out cores and analyses. Their reservations covered all
held-out peaks. Deterministic tests use the frozen 107-observation snapshot in
`scripts/MEMORY_CALIBRATION_BASELINE.json`, with raw-record hashes.

| Next target | Projected core reservation (GiB/run) | Initial core pool |
|---|---:|---:|
| 0.01 | 62–121 | 7 (493 GiB reserved) |
| 0.005 | 237–470 | 1–2 according to seed size |

The provisional analysis reservations are 71–171 GiB at 0.01 and 306–729 GiB
at 0.005. The latter extrapolates two stages and must be reassessed after 0.01;
some analyses may require a lower-memory reducer before they can run on this
node. Jobs are never made to fit by capping their estimates at the RAM budget.
If remaining work cannot fit, RESOURCE_WAIT.json records the requirement while
preserving all states and pending work. This does not claim experiment completion.

The dispatcher retains a 512 GiB reservation budget, a 48 GiB available-memory
margin at launch, at most 60 workers, and one distinct pinned CPU per worker.
Core and analysis phases receive separate estimates. Active forecasts update
from goal progress; fits and seed growth exponents update after complete stages.
The current 0.02 scheduler rule and all sampling trajectories are preserved.

`MEMORY_PREVIEW.json` contains provisional per-seed projections.
`MEMORY_CALIBRATION_0p01.json` and `MEMORY_CALIBRATION_0p005.json` will record
the final measured fits used to dispatch those stages. All earlier launch,
finish, and CPU-affinity events remain in the original dispatch log.
