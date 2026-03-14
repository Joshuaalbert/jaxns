
# Design notes (v3): global bootstrap + rho sampling for `sample_mc_shrinkage`

This document accompanies `phantom_eval_ref.py` and focuses on the mathematics and design decisions
for the improved Monte Carlo evidence sampler `sample_mc_shrinkage`.

It assumes the base evaluation function `evaluate_phantoms` described earlier.

---

## 0. Scope and invariants

- `log_L_blocks` is strictly increasing and sorted, with no root block (`log_L_root=-∞`).
- Current reference implementation requires an **unaugmented** schedule:
  `log_L_blocks == np.unique(log_L_classic)`.

The unaugmented requirement ensures every block corresponds to a true classic plateau level, allowing a
well-defined mapping from blocks to the classic live-count `K`.

---

## 1. What we are trying to improve relative to the baseline

Baseline `compute_mc_shrinkage` samples each boundary shrinkage independently:

- per boundary `g`, sample `r_g ~ Beta(alpha_g, beta_g)` using a fixed global `rho_global`.

This misses two important uncertainty effects when using loose phantom reuse:

### 1.1 Cross-boundary dependence

When loose samples are reused, the **same phantom clusters** contribute to multiple boundaries.
That means the boundary estimates `{(A_g,B_g,E_g)}` are statistically dependent across `g`
(e.g. exceedances are nested: `1{L > b_{g+1}}` implies `1{L > b_g}`).

Sampling `r_g` independently from marginal posteriors ignores this coupling, typically leading to
mischaracterized uncertainty in `log Z`.

### 1.2 Uncertainty in rho

`rho_global` is estimated from a candlestick likelihood. Treating it as fixed neglects estimation uncertainty,
which can matter when phantom information is moderate and/or mixing quality varies.

---

## 2. Data structure: clusters and loose reuse

We treat each replacement run (one constrained sampling call) as a **cluster**:

- cluster `i` has constraint `c_i = log_L_constraint[i]`
- cluster provides a correlated set of phantom draws `{L_{i,t}}`

For a boundary `a -> b` (with `b>a`), clusters with `c_i <= a` are eligible.
Define pooled phantom counts among eligible clusters:

- `A = #{L > a}` (qualifiers; represent draws from π_a after conditioning)
- `B = #{L > b}` (successes)
- `E = #{L = b}` (equality/atom; mainly for candlestick calibration)

Estimator:
- `p_hat_> = B/A` estimates `r(a->b) = X(b)/X(a)`
- `p_hat_= = E/A` estimates equality mass at `b`

---

## 3. Candlestick likelihood for rho (summary)

We embed the classic shrinkage model as a Dirichlet prior on
`p = (p_>, p_=, p_<)`:
- `p ~ Dirichlet(K, ε, 1-ε)`
so that `p_> ~ Beta(K,1)` marginally.

For iid phantom draws, conditional on `A`, proportions satisfy:
- `Cov(p_hat | A) = Σ * (1 + α0/A)`, with `α0=K+1`.

To account for correlation, define an efficiency `rho ∈ (0,1]`:
- `A_eff = rho * A`
- `Cov(p_hat | A) ≈ Σ * (1 + α0/(rho*A))`.

Using a Gaussian approximation and 2D Mahalanobis distances:
- `d^2 = (p_hat-μ)^T Σ^{-1} (p_hat-μ)`,
we approximate:
- `d^2 ~ κ * χ^2_2`, where `κ = 1 + α0/(rho*A)`.

This yields an approximate likelihood `L(rho)` which we evaluate on a discrete grid.

---

## 4. `sample_mc_shrinkage`: global bootstrap + rho sampling

### 4.1 Global cluster bootstrap

For each evidence draw `s`:

1. Resample clusters with replacement from the set of valid phantom clusters.
   This produces a bootstrap replicate dataset that preserves within-cluster correlation
   and induces realistic dependence across boundaries because the same cluster replicate
   is used for all boundaries.

2. On that replicate, compute pooled boundary counts `(A_g,B_g,E_g)` for every boundary `g`.

### 4.2 Sample rho from its likelihood

3. Compute candlestick distances `d^2_g` (for boundaries with `A_g>0`) and sample `rho_s`
   from the normalized likelihood over the grid.

By default we sample from **likelihood only** (no prior) because the user requested:
- `p(rho) ∝ L(rho)`.

A simple extension is a log-uniform prior `p(rho) ∝ 1/rho`, which is implemented as an option.

### 4.3 Sample boundary shrinkages and integrate evidence

4. Given `rho_s` and bootstrap counts, form per-boundary Beta posteriors:
   - `alpha_g = K_g + rho_s * B_g`
   - `beta_g = 1 + rho_s * (A_g - B_g)`
   (fallback to `Beta(K_g,1)` when `A_g=0`).

5. Sample shrinkages `r_g ~ Beta(alpha_g, beta_g)` and compute survival masses:
   - `X_-1 = 1`
   - `X_g = X_{g-1} * r_g`
   and integrate evidence by a right-Riemann sum:
   - `Z = Σ_g (X_{g-1}-X_g) * exp(log_L_blocks[g])`.

This yields a sample of `log Z` and per-block `log X`.

---

## 5. Why this is (often) better than baseline Beta sampling

Relative to sampling each `r_g` independently from a fixed posterior:

- The bootstrap couples the set `{(A_g,B_g)}` across boundaries, preserving the nested structure
  induced by reusing the same clusters for multiple boundaries.
- Sampling rho injects uncertainty in the phantom-information rate.
- The method is still relatively simple and does not require explicit time-series models.

---

## 6. Limitations

- The candlestick likelihood is approximate (Gaussian + scaled chi-square).
- The Beta update uses "effective counts" `rho*A`; it does not explicitly model temporal autocorrelation.
- This is still not the exact joint posterior over all shrinkages; it is a practical stochastic approximation.

---

## 7. Extension notes

### 7.1 ρ as a smooth function ρ(a)

Instead of sampling a single `rho`, treat `rho` as a smooth function of the previous boundary level `a`:
- parameterize `log rho(a)` with splines, constrain to (0,1]
- sample / fit using the same candlestick likelihood over all boundaries jointly.

### 7.2 Supporting augmented schedules

When blocks include intermediate contours not present among classic likelihood values, the classic
order-statistic prior `Beta(K,1)` does not directly specify a prior strength at those intermediate boundaries.

A plausible future direction:
- abandon per-boundary order-statistics for augmented levels and instead learn `X(ℓ)` on the augmented grid
  via **joint multinomial / Dirichlet models** using all loose samples:
  - for each base constraint `a`, bin qualifying samples into intervals between augmented contours
  - infer conditional bin probabilities under `π_a`, yielding `X(t_k)/X(a)` for all intermediate contours.
- evidence integration then uses the inferred `X` values on the fine grid.

This shifts the model from "shrinkage is an order statistic" to "shrinkage is learned from conditional
likelihood distribution under constraints", which naturally supports empty blocks.

---

## 8. Practical debugging invariants

- `log_X_per_block` must be non-increasing in block index for each sample.
- `log_Z_samples` should be finite for typical inputs (unless all contributions are zero).
- If there are no valid phantom clusters, v2 falls back to baseline sampling.
