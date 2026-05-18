# Ticket 0014: Posterior-Weighted GMM Non-Isotropic Direction Kernel

Branch: `feature/v3-gmm-non-isotropic-direction-kernel`
Priority: after baseline direction kernels, gamma-weighted phantom conditioning,
and execution diagnostics
Depends on: Ticket 0002, Ticket 0004, Ticket 0005, Ticket 0007, Ticket 0008,
Ticket 0010, Ticket 0011, Ticket 0013
Design docs:

- `docs/design/jaxns-v3-constrained-sampling.md`
- `docs/design/jaxns-v3-validation-plan.md`
- `docs/design/paper.tex`

## Goal

Replace the accepted Ticket 0008 one-component history ellipsoid adaptation
with the v3 target for GMM-based non-isotropic direction kernels: a
coordinator-owned, posterior-weighted Gaussian mixture fit from accumulated
accepted classic samples, with bounded fitting data, volume-proportional
component sampling, frozen kernel snapshots for each chain and shell epoch,
explicit diagnostics, and robust fallback to the previous valid kernel or the
isotropic kernel.

The paper-level requirement is intentionally small: the constrained sampler may
use an ellipsoidal Gaussian direction kernel where a Gaussian mixture model of
the posterior is constructed from samples collected so far, and a component is
selected proportional to integrated volume. The five-shell update cadence,
posterior-weight construction details, fitting thresholds, bounded resampling
policy, covariance regularization, diagnostics, and fallback behavior are v3
implementation policy from the design docs.

## Current Code Context

This context is from direct inspection of the current repository state.

- `src/jaxns/constrained_sampler.py` already defines
  `EllipsoidalGaussianDirectionKernel` with component means, radii, rotations,
  and probabilities. The sampler freezes the selected direction kernel at chain
  start and samples directions by choosing a component from
  `component_probabilities`, drawing a zero-mean Gaussian through that
  component's radii/rotation, and normalizing to a unit direction.
- `_build_ellipsoidal_direction_kernel(...)` accepts precomputed component
  fields from `adaptation_context`, or builds components from sample history.
  Component probabilities are normalized from integrated ellipsoid volumes,
  with non-positive volumes clipped to zero.
- `_build_ellipsoidal_components_from_history(...)` is the accepted Ticket 0008
  minimal path: it flattens `samples_U`, applies an optional `valid_mask`, then
  builds one bounding ellipsoid with `ellipsoid_params(...)`. It does not fit a
  posterior-weighted GMM and does not implement update cadence, bounded fitting
  rows, posterior weights, weighted EM, or resampling.
- `src/jaxns/em_gmm.py` currently exposes an unweighted EM path. It initializes
  means by random data selection, uses a diagonal empirical variance
  covariance initialization, computes unweighted responsibilities, and applies
  a fixed `1e-4 * I` covariance regularization in `m_step(...)`. It has a mask
  but no posterior fitting weights, no bounded resampling helper, no
  responsibility effective-size filtering, no covariance floor based on global
  weighted covariance, and no integrated-volume probability output.
- `src/jaxns/constrained_sampler.py` has no coordinator-owned fitting dataset
  or versioned active-kernel state. Workers and sampler calls are consumers of
  the immutable snapshot provided through `adaptation_context`.
- Ticket 0013 completed the gamma-weighted phantom-conditioning target. This
  matters because posterior fitting weights must use the same current posterior
  weight convention as result construction, including plateau equality-mass
  splitting and any shrinkage Monte Carlo mean used by the active result path.

## Required Behavior

Direction-kernel fitting is owned by the coordinator or execution layer. The
constrained sampler and workers remain pure consumers of an immutable direction
kernel snapshot. They must not append to history, refit mixtures, mutate
probabilities, or adapt to the current Markov-chain point.

Build the posterior fitting dataset from accumulated accepted classic samples:

- rows are flattened prior-space coordinates;
- retained phantom likelihood clusters are excluded because v3 discards
  phantom coordinates and phantoms are shrinkage diagnostics, not posterior
  coordinates for fitting `D`;
- rows with non-finite coordinates or non-positive posterior fitting weight are
  excluded from fitting, and the excluded counts are diagnostic fields;
- posterior fitting weights are the same normalized posterior weights used for
  result construction at that state;
- if shrinkage is represented by Monte Carlo draws, use the mean posterior
  weight per accepted classic sample across shrinkage draws;
- for plateau blocks, use the same plateau posterior-mass convention as
  results: equality atom mass is divided equally across samples in the plateau
  block.

The fitting target must satisfy:

```text
D_dim = flattened prior-space dimension
N_pos = number of classic samples with positive fitting weight
N_eff = 1 / sum_i w_i^2
```

Do not attempt a non-isotropic fit unless `N_pos >= D_dim + 1`,
`N_eff >= max(20, 2 * (D_dim + 1))`, and at least `D_dim + 1` unique finite
coordinate rows remain. If these thresholds fail, keep the previous successful
non-isotropic kernel; if none exists, use the isotropic kernel.

Use at most `M_D = 20000` fitting rows:

- if `N_pos <= M_D`, run weighted EM on the finite positive-weight rows;
- if `N_pos > M_D`, form a bounded posterior fitting dataset by systematic
  resampling exactly `M_D` rows from normalized posterior fitting weights, keep
  duplicate rows, and run unweighted EM on the resampled rows;
- record the resampling seed, `M_D`, original `N_pos`, original `N_eff`,
  resampled row count, and whether the fit used the weighted-EM or bounded
  resampling path.

Weighted EM is the preferred path. If implementation risk makes direct weighted
EM too large, the bounded systematic-resampling path may be used as the first
production implementation for both small and large datasets, provided the
ticket branch explicitly records that choice, preserves posterior weighting in
the resampling probabilities, and keeps the weighted-EM API/test surface
available for a later exact path. A hybrid implementation may support both.

Set the component count by:

```text
K_D = min(8, max(1, floor(N_eff / (2 * (D_dim + 1)))))
```

Then cap `K_D` by the number of unique fitting rows. If the cap drives `K_D`
below one, use the fallback path instead of fitting.

For each fitted component:

- compute responsibilities with posterior fitting weights in the weighted-EM
  path, or with duplicate rows preserved in the bounded resampling path;
- compute component means for diagnostics and interpretation;
- compute covariance in flattened coordinates from the component's weighted
  responsibilities;
- compute the full fitting dataset weighted covariance `C_global`;
- define

```text
sigma2_floor = max(1e-12, 1e-6 * trace(C_global) / D_dim)
C_k_reg = C_k + sigma2_floor I
```

- obtain component rotations and radii from eigendecomposition or SVD of
  `C_k_reg`, with `radii = sqrt(max(eigenvalue, sigma2_floor))`;
- compute integrated volume proportional to `prod(radii)`;
- drop invalid components with non-finite parameters, non-positive integrated
  volume, or effective responsibility size

```text
N_eff,k = (sum_i r_ik w_i)^2 / sum_i (r_ik w_i)^2
```

  less than `D_dim + 1`.

Normalize component probabilities from valid component integrated volumes:

```text
q_k = V_k / sum_j V_j
```

If no valid component remains, use the previous successful non-isotropic kernel
or the isotropic fallback. Any EM failure, covariance failure, invalid volume,
or probability-normalization failure is a direction-adaptation failure only; it
must not fail the nested-sampling run unless the user explicitly requested a
hard direction-adaptation error.

The active kernel update cadence is:

- initial kernel is isotropic;
- attempt the first non-isotropic fit after root/bootstrap samples produce a
  valid posterior fitting dataset;
- after that, attempt a fit after every `S_D = 5` newly completed distinct
  likelihood shells since the last successful direction-kernel update;
- a shell is a distinct sorted likelihood block, and a plateau block counts as
  one shell;
- all chain dispatches in the same shell epoch use the same active kernel
  version;
- a successful fit receives a monotonically increasing version identifier and
  becomes active only for future dispatches;
- chains already in flight continue with the kernel snapshot selected at
  dispatch.

Direction draws from the fitted kernel remain centered at zero. Component means
are retained for diagnostics and fit interpretation, but are not added to raw
direction draws. This preserves symmetry `D(n) = D(-n)` and keeps the direction
kernel independent of the current chain point.

Diagnostics must include enough information to audit fitting and fallback:

- active kernel mode and version;
- shell epoch and distinct shell count since last successful update;
- attempted update count and successful update count;
- fallback activation and fallback reason;
- fitting path (`weighted_em` or `bounded_resampling`);
- `D_dim`, `N_pos`, `N_eff`, unique finite row count, excluded row counts, and
  posterior weight normalization checks;
- `M_D`, resampling seed, and original/resampled fitting sizes when bounded
  resampling is used;
- requested and final `K_D`;
- component means, radii summaries, integrated volumes, probabilities, and
  dropped-component counts/reasons;
- EM iteration count, convergence status, and regularization floor;
- kernel version used by each dispatched chain or sampler diagnostic record.

## Standard-Problem Relevance

The standard-problem suite is relevant as a regression and calibration signal:
switching from isotropic to posterior-weighted non-isotropic directions should
not measurably bias known-reference evidence estimates, and any benefit should
show up as better constrained-prior exploration, posterior representation,
mixing, or likelihood-evaluation efficiency. This ticket should add focused
unit and integration coverage first; standard-problem evidence should be used
as supporting validation evidence, not as a substitute for deterministic tests
of the fitting target, cadence, snapshot freezing, and fallback behavior.

## Out Of Scope

- Changing the shrinkage law, posterior-weight definitions, or plateau
  posterior-mass convention outside what is required to consume the current
  result weights for fitting.
- Using phantom coordinates in direction-kernel fitting.
- Changing the constrained-prior target or evidence accounting. The direction
  kernel affects proposal geometry only.
- Implementing a new trajectory method. Straight-line and Galilean trajectories
  consume the same frozen direction snapshots.
- Final expensive benchmark campaigns or new benchmark pass thresholds.
- Rewriting accepted Ticket 0008 history. This ticket supersedes only the
  minimal one-component adaptation behavior as the target implementation.

## Test Plan

Write tests before implementation and review them against
`docs/implementation_orchestration/UNIT_TEST_STANDARDS.md` and
`docs/implementation_orchestration/PERFORMANCE_TEST_STANDARDS.md`.

Required unit tests:

- Hand-built `NamedTuple` or frozen-dataclass fixtures for accepted classic
  sample coordinates, block likelihoods, posterior weights, and plateau blocks.
  Fixtures must make the posterior-weighted fitting target obvious without
  reimplementing nested-sampling result construction in the assertion body.
- Posterior fitting weight tests proving that high posterior-weight rows drive
  the fit even when they are outnumbered by low-weight rows. Include a plateau
  fixture where equality atom mass is divided equally across plateau samples.
- Shrinkage-draw aggregation tests proving that mean posterior weight across
  shrinkage draws is used when MC shrinkage weights are supplied.
- Dataset validation tests for non-finite coordinates, zero/negative weights,
  all-zero weights, insufficient `N_pos`, insufficient `N_eff`, insufficient
  unique rows, and dimension-one behavior.
- Bounded fitting-row tests proving `M_D = 20000` is enforced, systematic
  resampling preserves duplicate rows, the resampling seed is deterministic,
  and diagnostics record original and bounded sizes. Use small configurable
  limits in tests rather than allocating 20000 rows where possible.
- Weighted-EM tests with explicit weights. If the implementation chooses the
  resampling path initially, keep these tests at the helper/API boundary or
  mark the exact weighted path as a targeted expected-failure only if the branch
  explicitly documents that choice before implementation review.
- Weighted-EM versus bounded-resampling agreement tests on simple separated
  Gaussian fixtures, within Monte Carlo tolerance and with fixed seeds.
- `K_D` policy tests for the floor formula, cap at eight, cap by unique fitting
  rows, and fallback when the effective cap leaves no valid component.
- Covariance regularization tests proving `sigma2_floor` uses
  `max(1e-12, 1e-6 * trace(C_global) / D_dim)`, degenerate covariance receives
  positive radii, and non-finite covariance fails into fallback diagnostics.
- Component validation tests for non-finite means/radii/rotations, non-positive
  integrated volumes, low component responsibility effective size, and
  renormalization after dropping invalid components.
- Volume-probability tests proving component probabilities are proportional to
  integrated volume, normalized to one, and reject the all-invalid/all-zero
  volume case.
- Direction-draw tests proving frozen GMM kernels produce unit directions,
  centered-zero symmetric draws, and no dependence on the current chain point
  except tree shape/unflattening.
- Snapshot tests proving all directions in one chain use the same frozen kernel
  version and that later shell updates do not affect in-flight chains.
- Update-cadence tests proving initial isotropic mode, first attempted fit
  after a valid bootstrap fitting dataset, `S_D = 5` distinct-shell cadence,
  plateau blocks counting as one shell, and failed attempts preserving the
  previous active version.
- Diagnostics tests for fallback reasons, version increments, fitting counts,
  `N_eff`, requested/final `K_D`, component drop reasons, EM convergence,
  bounded-resampling metadata, and chain-dispatch kernel version.
- Runtime or coordinator tests proving workers receive immutable component
  fields and do not refit or mutate direction-kernel state.
- Regression tests proving retained phantom likelihood clusters are excluded
  from the fitting dataset.
- Integration tests comparing isotropic and GMM non-isotropic settings on a
  small anisotropic problem for valid evidence calibration and improved or
  unchanged likelihood-evaluation efficiency. These tests must not encode a
  noisy performance win as a hard unit-test requirement.

Required performance checks:

- Keep fitting-data construction vectorized over accepted classic samples.
  Avoid per-sample Python loops in the hot execution path.
- Bound EM input rows by `M_D = 20000` before fitting work that scales worse
  than linearly in row count.
- Any performance assertion must include a measured threshold and rationale, as
  required by `PERFORMANCE_TEST_STANDARDS.md`.

Suggested focused acceptance commands after implementation:

```bash
conda run -n jaxns_py pytest \
  tests/test_em_gmm.py \
  tests/test_v3_direction_gmm.py \
  tests/test_v3_direction_trajectories.py \
  tests/test_v3_sampler_contract.py \
  tests/test_constrained_sampler.py \
  tests/test_runtime.py \
  tests/test_v3_run_pattern.py \
  tests/test_v3_execution_diagnostics.py
conda run -n jaxns_py ruff check \
  src/jaxns/em_gmm.py \
  src/jaxns/constrained_sampler.py \
  src/jaxns/core.py \
  src/jaxns/runtime.py
conda run -n jaxns_py flake8 \
  src/jaxns/em_gmm.py \
  src/jaxns/constrained_sampler.py \
  src/jaxns/core.py \
  src/jaxns/runtime.py
```

Broaden the pytest and lint commands if the implementation touches result
construction, validation producers, or public diagnostics outside these files.

## Implementation Notes

- Prefer a small, explicit fitting helper API over burying posterior fitting
  logic inside sampler proposal construction. The sampler should continue to
  accept precomputed component fields through `adaptation_context`.
- Keep a pure NumPy/reference helper for systematic resampling, weighted
  covariance, `K_D`, and component filtering where it makes deterministic tests
  easier to read. Use JAX for production paths only where it matches existing
  execution boundaries.
- Extend `src/jaxns/em_gmm.py` carefully. Existing unweighted tests should keep
  passing; weighted EM should be additive rather than a silent behavior change
  for callers that do not pass weights.
- Use log-space calculations for posterior fitting weights where result
  construction already exposes log weights or where underflow is plausible.
- Treat fallback as a normal adaptation outcome. It should be visible in
  diagnostics but should not interrupt nested sampling unless a hard-error
  option is explicitly enabled.
- Keep component means in diagnostics and kernel snapshots, but sample
  directions from zero-mean component covariances to preserve kernel symmetry.
- Preserve the accepted Ticket 0008 ability to consume explicitly supplied
  component fields. Tests should cover both directly supplied fields and
  coordinator-fitted fields.
- Do not update `LEARNINGS.md` during implementation unless the branch uncovers
  a durable repo-specific convention or pitfall not already captured there.

## Implementation Review Follow-Up Actions

Status: resolved in the accepted implementation.

Implementation review found these blocking issues before acceptance; they have
been resolved and verified by the final focused and standard-problem gates:

- Enforce `N_eff >= max(20, 2 * (D_dim + 1))` unconditionally. Do not keep
  test-only or small-fixture bypasses, and do not inflate component
  responsibility effective sizes to pass component filtering.
- Compute MC-shrinkage posterior fitting weights as the mean of per-draw
  normalized posterior weights. Do not average raw likelihood/shrinkage masses
  first and normalize only once afterward, because that differs when evidence
  varies by shrinkage draw.
- Wire coordinator-owned GMM adaptation into the v3 execution path. The run
  loop must build fitting rows/weights from accumulated accepted classic
  samples, apply the five-distinct-shell cadence since the last successful
  update, fit/update the active kernel, and pass frozen direction snapshots to
  sampler calls.
- Wire local runtime worker dispatch to receive per-dispatch direction
  snapshots instead of hard-coding `direction_adaptation_context=None`.
  Runtime compile/payload caching must not hide newer kernel snapshots.
- Ensure standard-problem coverage actually runs with the GMM non-isotropic
  direction path enabled for `uniform`, `evidence_improving`, and
  `posterior_improving`; passing those cases with default isotropic directions
  is not Ticket 0014 acceptance.
- Preserve required diagnostics when rows are filtered out before fitting,
  including nonfinite and nonpositive-weight exclusion counts.

## Acceptance Criteria

- Test-first draft is accepted against `UNIT_TEST_STANDARDS.md` and
  `PERFORMANCE_TEST_STANDARDS.md` before implementation starts.
- The active non-isotropic direction-kernel target fits from accumulated
  accepted classic samples using posterior fitting weights that match current
  result posterior weights, including MC-shrinkage averaging and plateau
  equality-mass splitting.
- Retained phantom likelihood clusters are excluded from the direction-kernel
  fitting dataset.
- Fitting data is bounded by `M_D = 20000`; implementations use weighted EM for
  `N_pos <= M_D` or an explicitly documented posterior-resampling path, and use
  systematic posterior resampling before unweighted EM when `N_pos > M_D`.
- The `K_D` policy follows
  `min(8, max(1, floor(N_eff / (2 * (D_dim + 1)))))`, capped by unique fitting
  rows, with tested fallback when thresholds fail.
- Covariance regularization, radii extraction, integrated-volume computation,
  component filtering, and volume-proportional component probabilities match
  the v3 constrained-sampling design.
- The update cadence attempts the first valid non-isotropic fit after bootstrap
  data, then after every `S_D = 5` newly completed distinct likelihood shells,
  with plateau blocks counting as one shell.
- Kernel snapshots are frozen per chain and shell epoch; in-flight chains are
  unaffected by later fitting attempts or successful version updates.
- Invalid datasets, invalid components, EM failures, and probability failures
  fall back to the previous valid non-isotropic kernel or the isotropic kernel
  with explicit diagnostics and without failing the run by default.
- Diagnostics expose fitting path, fitting sizes, `N_eff`, `K_D`,
  regularization, component probabilities, dropped-component reasons, fallback
  activation, active kernel version, and per-dispatch kernel version.
- Focused unit/integration tests and lint checks for touched modules pass under
  the repository's `conda run -n jaxns_py ...` workflow.
