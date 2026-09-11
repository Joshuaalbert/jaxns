# A+C evidence experiments at D=10 (2026-09-09)

Status amendment, 2026-09-10: all 90 baseline results and all 30 matched SS10
0.02 continuations and analyses completed and are included in the manuscript.
The remaining ladder was cancelled for computational cost. No 0.01 result
completed; 0.005 never started. The continuation plan below is historical.
The installed paper additionally reports paired absolute cohort-bias contrasts
using the same 10,000 whole-seed bootstrap indices as RMSE; this is an added
descriptive analysis, not a pre-specified noninferiority test. See
[SS10_CONTRASTS.json](SS10_CONTRASTS.json) and [CANCELLED.json](CANCELLED.json).

The active paper cases are G10, CG10, and the linked repository SS10.
G10 uses N(0,I) prior, mean (3,0,...,0), covariance with diagonal 1 and
off-diagonal 0.99. CG10 applies the component-centered, unit-Jacobian shear
x2 = z2 + 0.4*((z1-3)^2-1). SS10 uses U[-4,8]^10 and the sum of two
normalized Gaussian densities: means (6,6,0,...,0) and (2.5,2.5,0,...,0),
covariances 0.08I and 0.8I. REFERENCES.json records deterministic log Z.

Baseline: 30 seeds (0--29) for each case; d0=300, shell width 100,
100 isotropic no-step-out slices, 99 retained phantoms, all valid phantom
U points eligible for seeding, no bounded reservoir. First full goal uses
uniform allocation with delta_K=1; subsequent goals use evidence-improving
allocation with delta_K=300. Depth dlogZ=log(1.001). Stop at classic
expected sigma(log Z)<0.05. Posterior weights always use classics only.

SS10 extension: resume the same 30 full-capacity trees and stored random
streams to 0.02. If spike-mass recovery fails, resume to 0.01, and then
0.005 as the final attempt. Recovery is descriptively defined before these
runs as spike-mass RMSE <=0.05 and maximum absolute mass error <=0.15,
relative to the exact bounded mass 0.5000077444258325. The criterion is not
a formal hypothesis test. Stop decisions depend on posterior recovery,
not on obtaining a preferred phantom result. Publish all completed stages.
The uncertainty target may be achieved while recovery fails.

Mass uses classic posterior weights times exact component responsibilities.
For each fixed tree compute 2048 paired shrinkage draws at phantom prefix
sizes 0,10,...,90, prefix increments of 10, gate C_min=20. One gamma
weight is shared by all retained phantoms from a chain across increments. The prefix changes only
evidence conditioning; it does not change the classic posterior mass.
Evidence comparisons report empirical RMSE, bias, mean reported uncertainty,
coverage, and paired RMSE change. Bootstrap whole seeds (10,000 resamples,
seed 20260909) to quantify Monte Carlo uncertainty across runs. Preserve
the seed-level values, draws, states, manifests and worker logs.

One distinct pinned CPU per worker, numerical threads=1, CPU backend with
JAX x64 enabled; the inherited jaxctx unit-prior dtype remains float32.
At most 60 workers. Conservative memory reservations reduce concurrency
for tighter stages: 8/24/64/192 GiB per worker for targets
0.05/0.02/0.01/0.005, against a 480 GiB reservation budget. These are
scheduling bounds, not changes to the sampler. Record actual peak RSS,
likelihood calls, goal iterations, cumulative and incremental wall time.

Checkpoint every five goals and at completion. Preserve full state capacity
for resumption: a focused test found that trimming before resumption changes
the trajectory despite preserving PRNG keys. Pickle/reload without trimming
matches uninterrupted continuation exactly. Trim only for analysis.
The final checkpoint and state are hardlinks to avoid duplicate disk usage.

Validation before launch: eight focused tests passed, including model/prior
agreement with SciPy, conjugate G10 evidence, adaptive and Hermite CG10
quadrature, SS10 identity with the previous repository model, CLI smoke,
full phantom checkpoint continuation, and the existing core resume test.
Ruff and Flake8 (100-column limit) passed for the new harness and tests.
No production sampler source is modified by this experiment branch.

## Scheduling amendment before the 0.02 stage

The original fixed reservations above were replaced by estimates using measured
prior-stage peak RSS. Reserve the larger of 8 GiB and the preceding SS10
maximum core/analysis RSS multiplied by (old_target/new_target)^2 and 1.15,
rounded up to 4 GiB and capped at 480 GiB. Against the same 480 GiB reservation
budget this gives 44 GiB per worker and ten concurrent workers at 0.02.
The dispatcher records every reservation and observed peak RSS. This affects
job concurrency only; cores, random keys, and sampling settings stay fixed.

## Analysis amendment

Sampling is frozen at aa79a0dc395c0d64b85cbb88d0e3e732decd7361. SS10 seed 16
has seven identical classic points in one likelihood block; the original
singleton-only sparse report reducer rejected it after successful sampling.
Analysis commit 5667381608eced7e3fcb754f32cc6c59db185735 supports the existing
core's plateau law without changing its src tree, models, or posterior weights.
Singleton random fields remain bitwise identical. Failed and successful
attempts are retained in RETRIES.json and the dispatch log. ANALYSIS_SOURCES.json
records the source identity checks. Further analyses use the corrected reducer.

Completed full states may be moved to the Ceph archive after a verified SHA256
copy, leaving state.pkl and checkpoint.pkl as links to the identical raw pickle.
STATE_ARCHIVE.json records each transfer. Archiving never trims the state.

## Concurrency correction at 16:14 UTC

The global 44 GiB reservation unnecessarily limited 0.02 to ten workers.
Completed 0.02 cores with capacity 857600 peaked below 15 GiB. The replacement
coordinator adopts all live workers without restarting them and estimates
memory separately for each tree. Project classic sample count using the
observed uncertainty-squared scaling and 15% count headroom, then round up to
the full power-of-two state capacity. Reserve ceil(1.2*(3 + 0.0000138*capacity))
GiB per worker, with minimum 8 GiB and maximum 512 GiB. This fit uses the
observed 0.05 and 0.02 memory footprint and includes 20% per-worker headroom.
For completed trees, analysis reservations use their actual state capacity.
The shared reservation budget is 512 GiB on the 573 GiB node; launches also
require enough currently available memory to leave 48 GiB after reserving
that job. The CPU cap remains 60, with one distinct pinned CPU per worker.

The correction immediately increased concurrency from ten to 22 sampling
workers. Reservations are recomputed as goals progress. The original and
replacement dispatch events share the same append-only log, preserving each
worker's initial PID, CPU, start, and finish. The frozen sampler/model sources,
random state, tree buffers, and scientific stopping conditions are unchanged.

## Empirical calibration for 0.01 and 0.005

Future stage admission now fits separate core/analysis peak-RSS models to all
completed preceding-stage measurements. The features are fixed overhead,
full state capacity, and classic sample count, with nonnegative coefficients.
The maximum positive training residual and 20% RAM headroom are added.
Each seed's most recent matched-stage sample-count elasticity with respect
to inverse achieved uncertainty forecasts its growth (quadratic floor),
with 10% count headroom and exact power-of-two capacity rounding. The
observed 0.05-to-0.02 growth is used to choose the 0.01 pool; the complete
0.01 data refit the 0.005 pool. MEMORY_FORECAST.md explains the current fit.

The current 0.02 policy is preserved. A replacement coordinator adopts its
live workers; no sampler/model, PRNG state, or numerical data is modified.
Full calibration snapshots are written before future stages. Oversized
estimates are not capped to the machine budget; an unresolved resource
requirement is recorded in RESOURCE_WAIT.json with all pending work preserved.
