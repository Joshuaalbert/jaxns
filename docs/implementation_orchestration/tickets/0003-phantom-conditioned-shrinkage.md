# Ticket 0003: Phantom-Conditioned Shrinkage

Branch: `feature/v3-phantom-conditioned-shrinkage`
Priority: 3
Depends on: Ticket 0001, Ticket 0002
Design docs:

- `docs/design/jaxns-v3-phantom-conditioning.md`
- `docs/design/jaxns-v3-statistical-core.md`
- `docs/design/jaxns-v3-constrained-sampling.md`

## Goal

Implement v3 phantom conditioning as effective multinomial observations of the
block probability vector `(p_{>g}, p_{=g}, p_{<g})`, preserving phantom cluster
structure and calibrating `rho_g` from cluster-bootstrap covariance.

## Current Code Context

Relevant files likely include:

- `src/jaxns/phantom_eval.py`
- `src/jaxns/phantom_eval_ref.py`
- `src/jaxns/state.py`
- `src/jaxns/results.py`
- `src/jaxns/samples.py`
- `tests/test_phantom_eval_jax.py`
- `tests/test_phantom_eval_ref.py`
- `tests/test_state.py`

Current phantom evaluation code already has cluster-shaped phantom arrays and
some `A`, `B`, `E` counting. It appears to use an older/global-rho beta
shrinkage approximation rather than the paper's per-block Dirichlet update with
`m_g`, `epsilon_g`, equality atom mass, and the rank/trace covariance estimator.
The minimal path is to evolve the existing JAX and NumPy phantom evaluators and
their tests, preserving cluster-shaped inputs and the no-phantom fallback.

## Required Behavior

For each block `g`, compute phantom counts:

```text
A_g = sum_{(x,L) in P_eligible(g)} 1{L(x) > L_{g-1}}
B_g = sum_{(x,L) in P_eligible(g)} 1{L(x) > L_g} 1{L(x) > L_{g-1}}
E_g = sum_{(x,L) in P_eligible(g)} 1{L(x) = L_g} 1{L(x) > L_{g-1}}
```

`P_eligible(g)` contains only phantom clusters whose parent/constraint contour
is no stricter than `lambda_{g-1}`. In log-likelihood code this means the
cluster's stored `log_L_constraint <= log_L_{g-1}`. Stricter-contour phantoms
must not be pooled into lower-contour probability estimates.

Use effective count:

```text
A_g^eff = rho_g A_g
0 < rho_g <= 1
```

and posterior:

```text
p_g | K_g, m_g, rho_g, P ~ Dir(b_{>g}, b_{=g}, b_{<g})
b_{>g} = K_g - m_g + 1 + rho_g B_g
b_{=g} = m_g + epsilon_g + rho_g E_g
b_{<g} = 1 - epsilon_g + rho_g (A_g - B_g - E_g)
```

Estimate `rho_g` by cluster bootstrap:

- resample phantom clusters with replacement;
- compute bootstrap covariance of
  `q_g = (hat p_{>g}, hat p_{=g})^T`;
- match it to the iid multinomial covariance;
- use the paper estimator:

```text
hat rho_g =
    rank(Sigma_g) / trace(Sigma_g^+ Sigma_g^boot)
```

Fit a default low-order function against normalized race time:

```text
rho_g = c_0 + c_1 (s_g / s_G) + c_2 (s_g / s_G)^2
```

The v3 implementation must expose both raw per-block `hat rho_g` estimates and
the fitted/smoothed `rho_g` curve used for conditioning and diagnostics. The
fit must be constrained or transformed so the resulting values satisfy
`0 < rho_g <= 1`. Degenerate cases must use an explicit documented fallback,
not an accidental curve fit.

## Out Of Scope

- Sampler generation of phantoms. Covered by Ticket 0007.
- Public result API polish. Covered by Ticket 0004.
- Benchmark calibration. Covered by Ticket 0010.

## Test Plan

Write tests before implementation.

Required unit tests:

- Direct `A_g`, `B_g`, `E_g` counts on small hand-built clustered phantom sets,
  including equality atoms.
- Parent-contour eligibility tests: clusters generated under stricter contours
  are excluded from lower-boundary counts, and clusters with
  `log_L_constraint <= log_L_{g-1}` are eligible.
- Counts preserve cluster identity and do not treat individual phantoms as
  bootstrap units.
- Effective Dirichlet concentrations reduce to Ticket 0002 when no phantom
  information is available according to the design decision for `A_g = 0`.
- `rho_g` estimator uses the first two components and Moore-Penrose inverse.
- `rho_g` is clipped or constrained to `0 < rho_g <= 1`.
- Default `rho_g` fitting produces a finite per-block curve with the same block
  alignment as the raw estimates and is tested separately from the raw
  bootstrap estimator.
- Degenerate `rho_g` cases are explicit and finite: rank-zero `Sigma_g`,
  zero-trace or non-positive trace denominator, `nan`/`inf` estimator values,
  bootstrap resamples with no active phantom counts, and empty or single-cluster
  bootstrap inputs.
- Iid synthetic phantom clusters produce `rho_g` near one within tolerance.
- Strongly correlated/repeated clusters produce `rho_g < 1`.
- Non-stationarity is not silently "corrected"; tests must ensure burn-in or
  valid-mask controls are explicit inputs rather than inferred from values.
- Equality phantom counts influence `p_{=g}` concentration, not just `p_{>g}`.
- Invalid or malformed count relations fail explicitly, including
  `B_g + E_g > A_g`, mismatched phantom cluster masks, stale
  phantom-to-classic associations, stale or misaligned `log_L_constraints`, and
  shape mismatches between per-phantom and per-cluster validity.
- `State.to_result()` and result conversion preserve the per-cluster validity
  semantics needed by bootstrap; any collapse from per-phantom masks to
  cluster-level masks is deliberate, documented, and tested.

Recommended integration tests:

- Compare JAX and NumPy reference implementations for the same hand-built
  plateau and non-plateau scenarios.
- Verify deterministic output for identical PRNG keys and ordered inputs.

## Implementation Notes

- The paper does not specify behavior when `A_g = 0`; decide this explicitly in
  this ticket before implementation and document the choice. A common safe
  choice is to leave the block at the non-phantom posterior, but reviewers must
  see that this is an implementation convention, not a paper formula. The chosen
  convention must have direct tests before implementation.
- Do not allow `rho_g = 0` as a model value. If a limiting fallback is needed,
  keep it separate from the paper parameter.
- Do not rely on clipping to repair `nan` or `inf` `rho_g` estimates; handle
  degenerate covariance cases before applying bounds.
- Treat the fitted curve as part of v3 phantom conditioning, not a plotting-only
  optional feature. Ticket 0004 only exposes the shapes and public result API.
- Store only likelihoods if that is enough for conditioning, but preserve cluster
  boundaries, association with the classic sample, and parent-contour
  provenance through `log_L_constraints` or an equivalent derived field.
- Keep a pure NumPy/reference path for numerical review.
- Keep the public `sample_mc_shrinkage` entry point compatible where possible;
  add v3-specific outputs or options explicitly instead of silently changing
  legacy diagnostics before Ticket 0004 exposes the final API.
- Reuse current count/precompute code only where it already gates clusters by
  their parent constraints and has v3 tests. Do not carry forward the old
  global-rho beta posterior as the v3 phantom-conditioned model.

## Acceptance Criteria

- Phantom conditioning updates all three Dirichlet components.
- Raw `rho_g` estimation is cluster-based, the default fitted `rho_g` curve is
  tested, and both are finite with `0 < rho_g <= 1`.
- Plateau equality counts improve atom-mass inference in toy tests.
- The classic-only path remains available and unchanged when no phantom
  information is present.

## Current Review Follow-Up

The first test-first draft was reviewed and must be revised before
implementation. Required fixes:

- Use one public phantom-conditioned Dirichlet API name/signature consistently
  across unit and validation tests.
- Add negative tests for mismatched phantom masks, stale classic associations,
  stale or misaligned `log_L_constraints`, per-phantom/per-cluster validity
  shape mismatches, and `State.to_result()` cluster-validity preservation.
- Avoid reimplementing the full `rho_g` estimator in the test oracle; keep the
  oracle focused on hand-computable fixtures or independent reference outputs.
- Add an iid synthetic cluster case where raw `rho_g` is near one.
