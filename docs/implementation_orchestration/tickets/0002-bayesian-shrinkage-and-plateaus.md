# Ticket 0002: Bayesian Shrinkage and Plateau Weights

Branch: `feature/v3-bayesian-shrinkage-and-plateaus`
Priority: 2
Depends on: Ticket 0001
Design docs:

- `docs/design/jaxns-v3-statistical-core.md`
- `docs/design/jaxns-v3-phantom-conditioning.md`
- `docs/design/jaxns-v3-validation-plan.md`

## Goal

Implement the v3 Bayesian shrinkage model for classic race-tree samples,
including plateau censorship, Dirichlet block probabilities, evidence sampling,
and posterior weights for plateau and non-plateau blocks.

## Current Code Context

Relevant files likely include:

- `src/jaxns/evidence_calculation.py`
- `src/jaxns/state.py`
- `src/jaxns/results.py`
- `src/jaxns/phantom_eval.py`
- `src/jaxns/phantom_eval_ref.py`
- `tests/test_state.py`
- `tests/test_results_plotting.py`

Current evidence code uses classic shrinkage expectations and Monte Carlo
shrinkage paths. It does not yet implement the v3 Dirichlet model over
`(p_{>g}, p_{=g}, p_{<g})`, and posterior weights do not yet assign plateau
equality atom mass separately. The migration should extend the existing
`evidence_calculation.py`, `phantom_eval.py`, and result conversion paths where
their boundaries remain useful; avoid replacing the evidence stack wholesale.
Do not treat current `sample_mc_shrinkage` or `State.sample_logZ` as safe v3
math foundations as-is: they use legacy beta shrinkage assumptions and do not
model `(p_{>g}, p_{=g}, p_{<g})` block Dirichlet probabilities.

## Required Behavior

For each block `g`, compute the classic-observation posterior:

```text
p_g | K_g, m_g ~ Dir(a_{>g}, a_{=g}, a_{<g})
a_{>g} = K_g - m_g + 1
a_{=g} = m_g + epsilon_g
a_{<g} = 1 - epsilon_g
```

with the paper's default epsilon policy:

```text
epsilon_g = 1e-6  if m_g = 1
epsilon_g = 1/2   if m_g > 1
```

where `epsilon_g` represents belief about
`E[p_{=g} / (1 - p_{>g})]`.

Evidence samples must use:

```text
X_0 = 1
X_g = X_{g-1} p_{>g}
Z ~= sum_g lambda_g (X_{g-1} - X_g)
```

This v3 evidence sampler must consume the canonical block state from Ticket
0001 and the Dirichlet parameters above. It should replace or bypass
`State.sample_logZ` for v3 evidence uncertainty.

Posterior weights must distinguish:

```text
w_g^{(k)} = lambda_g X_{g-1} p_{=g} / m_g
```

for `k in \mathcal{B}_g` on plateau blocks, and:

```text
w_g = lambda_g X_{g-1} (1 - p_{>g})
```

for non-plateau samples.

## Out Of Scope

- Phantom counts and `rho_g` conditioning. Covered by Ticket 0003.
- Dynamic allocation and parent selection. Covered by Ticket 0005.
- Benchmark suite. Covered by Ticket 0010.

## Test Plan

Write tests before implementation.

Required unit tests:

- Dirichlet concentration parameters for singleton and plateau blocks.
- `p_{>g}` marginal matches `Beta(a_{>g}, a_{=g} + a_{<g})`.
- Plateau censorship is represented by separating `p_{>g}` and `p_{=g}`; tests
  must not treat the final in-atom rank as the strict endpoint.
- Evidence samples follow `X_g = X_{g-1} p_{>g}`.
- No-phantom runs still construct canonical likelihood blocks from classic
  samples; they must not fall back to one per-sample block when plateaus exist.
- Singleton non-plateau blocks follow the v3 Dirichlet posterior exactly. Do not
  assert legacy `Beta(K_g, 1)` behavior; under the v3 singleton model,
  `p_{>g}` has marginal `Beta(K_g, 2)`.
- Plateau atom mass is split equally across samples in the classic plateau
  block.
- Non-plateau posterior mass uses `1 - p_{>g}`.
- Constant-likelihood or all-plateau toy cases remain finite and explicit.
- Invalid `K_g < m_g` fails before sampling.
- Log-domain numerical tests cover `logL = -inf`, very small shell mass, extreme
  `p_{>g}` near one, and stable `logZ` aggregation without catastrophic
  cancellation.

Recommended reference tests:

- A small deterministic NumPy reference for block concentration parameters,
  evidence sampling shapes, and posterior mass normalization.
- Repeated-seed calibration smoke test where known simple likelihoods produce
  finite `log_Z` samples and nonnegative posterior weights.

## Implementation Notes

- Keep the classic-only Bayesian shrinkage path independent from phantom
  conditioning so Ticket 0003 can layer on effective counts.
- Add narrow helper functions for block Dirichlet concentrations and evidence
  sampling before changing public result fields. This keeps legacy expected
  evidence paths available while v3 sampling is validated.
- Implement v3 Dirichlet helpers separately from the legacy
  `sample_mc_shrinkage` beta path, reusing only block/count precompute pieces
  that are explicitly validated against the v3 design.
- Do not migrate v3 evidence uncertainty through `State.sample_logZ`; provide a
  v3 block evidence sampler and route public v3 APIs through it.
- Do not silently replace equality comparisons with tolerances unless the design
  is updated. The paper's plateau model is about exact equality atoms.
- If log-likelihoods are stored, document the conversion between likelihood
  notation in the design and log-likelihood arrays in code.
- Use stable log-domain operations for `X_{g-1} - X_g`, evidence summation, and
  posterior weights; avoid naive subtraction when shrinkage is near one.
- Avoid overloading existing fields such as `log_dp` if they cannot represent
  sampled plateau equality mass correctly.

## Acceptance Criteria

- The classic Bayesian shrinkage model is available without phantom data.
- Plateau and non-plateau posterior weights are tested and normalized.
- Evidence uncertainty can be sampled from the Bayesian shrinkage model.
- Existing result conversion uses the v3 weight semantics or explicitly exposes
  a separate v3 result path.

## Current Review Follow-Up

Implementation review found blockers that must be resolved before acceptance:

- v3 Dirichlet/evidence sampling is not wired into public result/evidence paths;
  `State.to_result()`, `State.sample_logZ`, and public shrinkage sampling still
  use legacy evidence code.
- No-phantom `sample_mc_shrinkage` falls back to one block per sample instead of
  canonical likelihood blocks when plateaus exist.
- Public posterior weights still come from legacy `log_dp` semantics rather
  than v3 plateau atom/shell rules.
