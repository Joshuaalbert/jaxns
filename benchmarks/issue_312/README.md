# Contour-covariance direction experiment

Tracks [issue #312](https://github.com/Joshuaalbert/jaxns/issues/312).
Base: `9efa455`. This is an opt-in local experiment; the released isotropic
default and GMM workflow are unchanged.

See [REPORT.md](REPORT.md) for the completed 30-seed curved comparison,
ten-seed straight controls, validation, and limitations. The covariance change
improves the curved toy substantially but does not recover its posterior.
The [evidence-allocation follow-up](EVIDENCE_REPORT.md) repeats allocation until
the classic log-evidence uncertainty reaches 0.1.
The [slice-count comparison](SLICE_COUNT_REPORT.md) tests 180 versus 90
transitions at that same goal. The [geometry review](GEOMETRY_REVIEW.md)
examines full-cube bracketing, disconnected line slices, curvature, and a
reproduced cube-face boundary bug, subsequently fixed under
[issue #313](https://github.com/Joshuaalbert/jaxns/issues/313).
The [staged GMM report](STAGED_GMM_REPORT.md) tests a ten-component fit after
isotropic evidence allocation reaches uncertainty 0.1, followed by posterior
allocation until Kish ESS doubles, with paired isotropic controls and no
covariance whitening.
The [GMM density-fit comparison](GMM_FIT_REPORT.md) examines 10 through 500 EM
iterations on independent posterior draws and the saved populations, alongside
recursive K-means splits accepted by the historical enclosing-volume criterion.
It includes all 18 marginal densities, joint-density plots, convergence traces
and portable input clouds. Ten iterations underfit the full curve; increasing
to 100 helps substantially, but neither method resolves the density mismatch.
The [100-iteration continuation](GMM_ITERATIONS_REPORT.md) tests whether that
longer fit improves sampling, using the same evidence checkpoints and ESS goals
as the ten-iteration staged experiment.
The [Dynesty control verification](DYNESTY_REPORT.md) reruns the reporter's
150-live-point, 90-slice configuration with Dynesty 3.0.0 across six seeds.
It reproduces good evidence estimates in some seeds, but also finds large
evidence misses and persistent ridge-coordinate collapse.
The [larger-population controls](LARGE_POPULATION_REPORT.md) compare six seeds
with 1,000 live points in Dynesty and a minimum of 1,000 in reactive UltraNest,
including evidence, all three flat marginals, Gaussian widths and ridge coverage.
Neither tested configuration recovers this extreme toy: evidence RMSE is 1.566
and 5.755 nats respectively, with persistent distortion of the ridge marginal.

## Direction law

At a local dispatch boundary, take the shallowest non-root parent contour
`lambda` in that batch. Fit an equally weighted, centred sample covariance `C`
from **all valid stored classic points with log L > lambda**. This selection is
for geometry only; it does not replace the existing stationary-seed eligibility
rule. There are no posterior weights or phantom coordinates in this fit.

Use

```text
s = trace(C) / D
C_regularised = C + relative_regularisation * s * I
L = cholesky(C_regularised)
direction = L @ Normal(0, I), normalised in U-space
```

The initial coefficient is `1e-8`. If `s` is zero, use the unit-cube prior
variance `1/12` for the isotropic core. With fewer than two points, or a failed
Cholesky factorisation, retain the previous geometry (initially isotropic).
Covariance arithmetic uses measure precision; directions preserve the model's
U-space dtype. JAX x64 mode does not necessarily make JAXCTX's U-space float64.

Refresh when `abs(log E[X_current] - log E[X_last_fit]) >= 1`, including a return
to a shallower contour in a later allocation round. The first constrained
batch fits immediately. Both volumes use the existing classic block law, with
no phantom conditioning. The refresh check reconstructs the current block path
each batch; that extra sorting is explicitly experimental overhead.

One factor is shared by the batch and frozen for every transition in each
chain. Parent contours within a batch can differ, so this is geometry at the
batch's shallowest contour, not a separate exact covariance for every lane.
The factor and refresh state survive resize and checkpoint/resume.

Full-cube bracketing, 90 transitions, random-key splitting, root draws, seed
selection, allocation, stopping, phantom retention, and evidence reduction
are unchanged. No independent isotropic-direction mixture is added: the
isotropic core is an additive covariance ridge.

## Running

Use the worktree source explicitly; the development environment's editable
installation otherwise points at the primary checkout.

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 conda run --no-capture-output -n jaxns_py \
  python benchmarks/issue_312/run_ridge.py \
  --mode covariance --seeds 0 1 2 --cpu 2 \
  --output benchmarks/issue_312/raw/covariance.jsonl
```

Use `--mode isotropic` for the paired baseline, `--curvature 0` for the straight
ridge control, and `--regularisation` to vary only the isotropic core.
The default performs one complete warmup at the first requested seed, then
records every requested seed, including that seed again. Warmup and subsequent
run times are reported separately; new buffer shapes can still compile on
later seeds. `--no-warmup` is suitable for smoke checks, not timing comparisons.

The default model has 150 roots, sigma_stiff `1e-5`, 14 Gaussian coordinates
with sigma `0.05`, and flat coordinates x1, x16, x17. Its log evidence is zero
up to negligible Gaussian tails beyond the cube. Healthy posterior standard
deviations are `1/sqrt(12)` for the three flat coordinates and `0.05` for x5.
The script also compares inferred volume with the analytic interior-contour
volume of a sheared 15-ball.

For repeated evidence-improving allocation to a classic uncertainty goal:

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 conda run --no-capture-output -n jaxns_py \
  python benchmarks/issue_312/run_ridge.py \
  --mode covariance --allocation-target evidence_improving --log-z-uncert 0.1 \
  --seeds 0 --cpu 2 --no-warmup \
  --checkpoint-dir /tmp/issue312-evidence-covariance-0 \
  --output benchmarks/issue_312/raw/evidence-covariance-0.jsonl
```

This retains the default per-epoch depth condition,
`dlogZ=log1p(1e-3)` and no `cummax_XL_frac` cutoff. It repeats epochs until
`state.expected_log_Z_uncert <= 0.1`, with unlimited automatic sample-buffer
growth so the default sample cap cannot substitute for the requested goal.
The sibling `.progress.jsonl` file records each completed epoch. Checkpoints
are optional and saved at each epoch; reuse the same command to resume an
interrupted run. Use a separate checkpoint directory for each configuration.
`--no-warmup` avoids duplicating the full multi-epoch run; its elapsed time
includes compilation and should not be read as steady-state throughput.

To compare `10D` with `5D` on the 18-dimensional toy, use `--slices 180`
with otherwise identical covariance/evidence settings. Keep the checkpoints
separate from the 90-slice runs:

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 conda run --no-capture-output -n jaxns_py \
  python benchmarks/issue_312/run_ridge.py \
  --mode covariance --slices 180 \
  --allocation-target evidence_improving --log-z-uncert 0.1 \
  --seeds 0 --cpu 2 --no-warmup \
  --checkpoint-dir /tmp/issue312-evidence-covariance-180-0 \
  --output benchmarks/issue_312/raw/evidence-covariance-180-0.jsonl
```

Repeat for seeds 1 and 2, then compare with the existing 90-slice records:

```bash
conda run -n jaxns_py python benchmarks/issue_312/summarise_evidence.py \
  --comparison slice_count
```

The equivalent application-side opt-in is:

```python
state = runner.initialise(key)
state = state.covariance_directions(
    relative_regularisation=1e-8, refresh_efolds=1.0,
)
state = runner.resume_until_goal(
    state, lambda current: int(current.goal_loop_iter) > 0,
)
```

This experiment does not support periodic priors, simultaneous GMM geometry,
or distributed continuation. Disable covariance directions with
`state.iso_directions()` before merging runs or handing them to a distributed
runner. These limits keep the experiment within the local direction-law path.

## Staged GMM comparison

The subsequent experiment uses no covariance whitening. For each seed, start
with isotropic directions, 150 roots, 180 slices, and `evidence_improving`
allocation until the classic log-evidence uncertainty is at most 0.1. It uses
the cube-face fix from `f72a696`, the default depth condition, no phantoms,
and unlimited sample-buffer growth, matching the earlier goal experiments.

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 conda run --no-capture-output -n jaxns_py \
  python benchmarks/issue_312/run_ridge.py \
  --mode isotropic --slices 180 --allocation-target evidence_improving \
  --log-z-uncert 0.1 --seeds 0 --cpu 2 --no-warmup \
  --checkpoint-dir /tmp/issue312-gmm10-180-0/evidence \
  --output benchmarks/issue_312/raw/staged-gmm10-180-0.evidence.jsonl
```

At that boundary, fit ten full-covariance GMM components using the existing
posterior-weighted fit. Retain its defaults: ten EM iterations, relative
regularisation `1e-6`, and isotropic probability 0.01. Fit once and freeze the
mixture for the subsequent `posterior_improving` allocation stage, stopping
at the first depth boundary where classic posterior-weight Kish ESS is at
least twice its value before fitting.

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 conda run --no-capture-output -n jaxns_py \
  python benchmarks/issue_312/run_gmm_stage.py \
  --evidence-record benchmarks/issue_312/raw/staged-gmm10-180-0.evidence.jsonl \
  --evidence-checkpoint /tmp/issue312-gmm10-180-0/evidence/seed-0 \
  --checkpoint-dir /tmp/issue312-gmm10-180-0/gmm \
  --mode gmm --components 10 --ess-multiple 2 --cpu 2 \
  --output benchmarks/issue_312/raw/staged-gmm10-180-0.gmm.jsonl
```

Repeat with `--mode isotropic` and separate output/checkpoint paths for the
paired continuation control. Both arms start from the exact same saved state,
random keys and ESS target; their records carry its checkpoint checksum. Repeat
for seeds 1 and 2. Restart an interrupted continuation with its same command;
the runner restores the saved state, including its frozen fitted geometry.

The direction component is selected by contour-trimmed ellipsoid volume,
independently of the chain location. This tests the existing position-independent
mixture, not nearest-component tangent selection. The goal ESS only measures
posterior weight concentration; it is not corrected for chain dependence.
Evidence error, posterior widths, flat-coordinate Wasserstein distances, and
actual isotropic-direction usage are therefore recorded alongside ESS.

The completed three-seed results are in [STAGED_GMM_REPORT.md](STAGED_GMM_REPORT.md).
Regenerate the summary tables and figures with:

```bash
MPLCONFIGDIR=/tmp/issue312-matplotlib conda run --no-capture-output -n jaxns_py \
  python benchmarks/issue_312/summarise_gmm_stage.py
```

## Staged GMM with 100 EM iterations

Reuse each saved evidence-stage checkpoint above. It already satisfies
`expected_log_Z_uncert <= 0.1` after isotropic `evidence_improving` allocation,
with 150 initial roots and 180 slices. Using the identical checkpoint preserves
the fitting population, sampler random keys and ESS target across the comparison.
Fit ten components once with 100 EM iterations, then use `posterior_improving`
allocation to twice that baseline Kish ESS:

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 conda run --no-capture-output -n jaxns_py \
  python benchmarks/issue_312/run_gmm_stage.py \
  --evidence-record benchmarks/issue_312/raw/staged-gmm10-180-0.evidence.jsonl \
  --evidence-checkpoint /tmp/issue312-gmm10-180-0/evidence/seed-0 \
  --checkpoint-dir /tmp/issue312-gmm100-180-0/gmm \
  --mode gmm --components 10 --fit-iterations 100 --ess-multiple 2 --cpu 2 \
  --output benchmarks/issue_312/raw/staged-gmm100-180-0.gmm.jsonl
```

Repeat for seeds 1 and 2 with their corresponding evidence checkpoints and
separate continuation paths. The `gmm100` filename denotes the iteration count;
the component count remains ten. All other settings, including the one-percent
isotropic mixture, regularisation, default depth condition and disabled
covariance whitening, match the earlier continuation. Geometry remains frozen
throughout the posterior stage. The earlier ten-iteration and isotropic controls
are reused. Generate the paired tables and trajectories with:

```bash
MPLCONFIGDIR=/tmp/issue312-matplotlib conda run --no-capture-output -n jaxns_py \
  python benchmarks/issue_312/summarise_gmm_iterations.py
```
