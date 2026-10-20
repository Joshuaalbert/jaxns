# Contour-covariance direction experiment

Tracks [issue #312](https://github.com/Joshuaalbert/jaxns/issues/312).
Base: `9efa455`. This is an opt-in local experiment; the released isotropic
default and GMM workflow are unchanged.

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
