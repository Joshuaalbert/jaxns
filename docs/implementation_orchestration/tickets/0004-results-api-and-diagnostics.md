# Ticket 0004: Shrinkage Results API and Posterior Weights

Branch: `feature/v3-results-api-and-diagnostics`
Priority: 4
Depends on: Ticket 0001, Ticket 0002, Ticket 0003
Design docs:

- `docs/design/jaxns-v3-statistical-core.md`
- `docs/design/jaxns-v3-phantom-conditioning.md`
- `docs/design/jaxns-v3-validation-plan.md`

## Goal

Expose v3 shrinkage, posterior-weight, and phantom-conditioning outputs through
stable result APIs so downstream users and validation scripts do not depend on
private helper internals. This ticket covers the block/shrinkage result surface;
execution/sampler/distributed diagnostics are integrated later in Ticket 0011.

## Current Code Context

Relevant files likely include:

- `src/jaxns/results.py`
- `src/jaxns/state.py`
- `src/jaxns/phantom_eval.py`
- `src/jaxns/utils.py`
- `docs/examples/`
- `tests/test_results_plotting.py`
- `tests/test_phantom_eval_jax.py`
- `tests/test_state.py`
- `tests/test_utils.py`

Current `NestedSamplerResults` exposes `log_dp`, `log_X_mean`, classic
evidence estimates, phantom likelihood arrays, and `sample_mc_shrinkage`.
The v3 design needs results that can represent plateau atom weights,
phantom-conditioned shrinkage samples, shrinkage-sample `rho_g` diagnostics, and
block-level metadata.
Prefer extending `NestedSamplerResults` and its helpers in place. Add a small
v3 block/shrinkage result container only if it avoids overloading legacy fields
such as `log_dp`.

## Required Behavior

Result objects or helper APIs must expose:

- block likelihood levels as `log_L_blocks` plus likelihood-scale equivalents
  where useful;
- block sizes `m_g`;
- incoming active lineage counts `K_g`;
- posterior-weight outputs that distinguish plateau atom mass from non-plateau
  shell mass;
- Dirichlet concentration arrays for classic and phantom-conditioned shrinkage;
- sampled or expected `p_{>g}` and `p_{=g}` paths, with naming that makes the
  summary type explicit;
- evidence samples from the Bayesian shrinkage model;
- phantom count summaries `A_g`, `B_g`, `E_g` where phantoms are present;
- `sample_mc_shrinkage(...)` returns an evidence/shrinkage sample container that
  includes raw `rho_g` estimates and fitted/smoothed `rho_g` values for each
  shrinkage draw or for the fitted conditioning model, with shapes documented;
- enough block/shrinkage metadata to reproduce validation diagnostics in
  `docs/design/jaxns-v3-validation-plan.md`.

The main `NestedSamplerResults` object may expose input block metadata and
phantom counts, but raw/fitted `rho_g` curves are known only once shrinkage
sampling or fitting has been performed. Do not expose `results.rho_values` or
`results.rho_fit` as eager result fields unless they are explicit cached outputs
of `sample_mc_shrinkage(...)`.

For the target interface, the object returned by `sample_mc_shrinkage(...)` is
publicly typed as `jaxns.phantom_eval.EvidenceSamples` and exposes:

- `log_Z_samples`;
- `rho_values` for raw per-block `rho_g` estimates aligned with
  `NestedSamplerResults.log_L_blocks`;
- `rho_fit` for fitted or smoothed per-block `rho_g` values aligned with
  `NestedSamplerResults.log_L_blocks`.

If implementation keeps legacy names such as `rho_samples`, add explicit aliases
or update the interface doc and tests together.

Existing APIs may remain, but must not silently expose legacy weights as v3
weights.

## Out Of Scope

- Implementing the shrinkage math itself. Covered by Tickets 0002 and 0003.
- Core execution and allocation. Covered by Ticket 0005.
- Benchmark suite generation. Covered by Ticket 0010.

## Test Plan

Write tests before implementation.

Required unit tests:

- `to_result().trim()` preserves block metadata and phantom cluster summaries.
- `sample_evidence` and `sample_mc_shrinkage` return v3 evidence samples with
  deterministic shapes and finite values on simple problems.
- `log_dp` or replacement posterior-weight fields use plateau atom rules.
- No-phantom runs expose zero or empty phantom diagnostics without pretending
  `rho_g = 0`.
- Phantom-present runs expose cluster-level diagnostics and block counts.
- `sample_mc_shrinkage(...)` exposes `rho_g` raw/fitted diagnostics in its
  returned evidence-sample object; plotting code that shows rho curves consumes
  that returned object rather than eager fields on `NestedSamplerResults`.
- `EvidenceSamples` exposes `rho_values` and `rho_fit` fields or properties with
  documented shapes aligned to the result block likelihoods.
- `docs/design/interface/run_pattern.py` and any docs/examples plot rho curves
  against `results.log_L_blocks`, not per-sample likelihood arrays.
- Resampling uses v3 posterior weights and remains deterministic for a fixed
  key.
- Result resampling keeps all per-sample provenance arrays aligned, including
  `log_L_constraints`, `log_L_phantom`, `valid_phantom`, and any retained
  phantom cluster summaries.
- Plotting/summary code handles no phantoms, phantoms, plateaus, and padded
  samples without crashing.
- Diagnostics, posterior weights, and validation inputs can be reconstructed
  from public result objects only.
- Malformed or misaligned per-block/per-sample arrays fail explicitly.
- Public resampling utilities and docs/example-facing usage do not silently treat
  legacy `log_dp` as v3 plateau-aware weights.

## Implementation Notes

- If backward compatibility requires retaining `log_dp`, document whether it is
  an expected posterior weight, a sampled posterior weight summary, or legacy
  output.
- Keep `State.to_result()`, plotting, and resampling changes incremental. Each
  existing public method should either preserve old behavior or expose a clearly
  named v3 path with tests.
- Audit current `_resample` behavior before extending it. Per-sample sample
  arrays, constraints, phantom likelihoods, validity masks, and diagnostics must
  be resampled together; per-block arrays should be trimmed and validated
  separately.
- Avoid hiding block-level arrays inside plotting-only code. Validation and
  reviewers need machine-checkable outputs.
- Keep result trimming consistent across all per-sample and per-block arrays.
- If there are both log-likelihood and likelihood notations, expose names that
  make the scale explicit.

## Acceptance Criteria

- Users can compute evidence uncertainty, posterior weights, plateau diagnostics,
  and phantom diagnostics without private imports.
- Result APIs make legacy versus v3 semantics explicit.
- Tests cover no-phantom, phantom, singleton-block, and plateau-block cases.
