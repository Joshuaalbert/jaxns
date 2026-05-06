# JAXNS v3 Phantom Conditioning

Status: paper-derived design draft.
Source: `docs/design/paper.tex`, Sections 1, 3, and 4.

This document defines how JAXNS v3 should use phantom samples to condition the
Bayesian shrinkage model. It intentionally treats current code as irrelevant.

## Role of Phantom Samples

A phantom cluster `P_i` is the set of retained post-burn-in constrained-sampler
transition states produced while realising classic sample `i`. These states are
correlated draws from the strict constrained prior of the classic sample's
parent:

```text
P_i contains states from pi_{lambda_{p(i)}}.
```

Phantom samples are not independent race participants. They must not be inserted
into the ordered classic nested-sampling sequence and must not change parent
out-degrees or active lineage counts.

Do not treat arbitrary intermediate likelihood evaluations as phantoms. In
particular, Galilean trajectory/bracketing/reflection points are excluded
because the paper states they do not uniformly sample `pi_lambda`.

They are valid only as Monte Carlo observations of constrained-prior
probabilities such as

```text
P_{x ~ p(. | L > a)}(L(x) > b) = X_b / X_a.
```

The equality component matters because likelihood atoms censor the strict
endpoint of a plateau block: the final plateau rank lies between `p_{>g}` and
`p_{>g} + p_{=g}` without revealing `p_{>g}` itself.

## Epsilon Prior

The phantom-conditioned posterior extends the paper's Bayesian shrinkage prior

```text
p(p_g) = Dir(1, epsilon_g, 1 - epsilon_g).
```

The paper interprets `epsilon_g` as prior belief about
`E[p_{=g} / (1 - p_{>g})]`, and suggests:

```text
epsilon_g = 1e-6  if m_g = 1
epsilon_g = 1/2   if m_g > 1.
```

## Phantom Counts

Let `P = union_i P_i` be the retained phantom set. For each block `g`, define:

```text
A_g = sum_{(x,L) in P} 1{L(x) > L_{g-1}}
B_g = sum_{(x,L) in P} 1{L(x) > L_g} 1{L(x) > L_{g-1}}
E_g = sum_{(x,L) in P} 1{L(x) = L_g} 1{L(x) > L_{g-1}}
```

The paper's Monte Carlo estimate of the block probability vector is

```text
hat p_g = (
    B_g / A_g,
    E_g / A_g,
    (A_g - B_g - E_g) / A_g
).
```

The three components estimate `(p_{>g}, p_{=g}, p_{<g})`.

The paper does not specify behavior when `A_g = 0`; in that case `hat p_g` is
undefined and the block has no phantom Monte Carlo estimate in the stated
formula.

## Effective Multinomial Likelihood

If the phantom points were independent, `A_g hat p_g` would follow a multinomial
likelihood. Because Markov-chain phantom samples are correlated, the paper uses
an effective count

```text
A_g^eff = rho_g A_g
0 < rho_g <= 1.
```

The effective-count posterior is

```text
p_g | K_g, m_g, rho_g, P ~ Dir(b_{>g}, b_{=g}, b_{<g})
b_{>g} = K_g - m_g + 1 + rho_g B_g
b_{=g} = m_g + epsilon_g + rho_g E_g
b_{<g} = 1 - epsilon_g + rho_g (A_g - B_g - E_g)
```

The marginals for `p_{>g}` and `p_{=g}` follow the same Dirichlet marginal rules
used by the statistical core.

The role of `rho_g` is variance calibration:

- `rho_g = 1` for independent phantom samples.
- `rho_g < 1` for correlated Markov-chain phantom samples, which have inflated
  covariance.

## Estimating `rho_g`

The paper estimates `rho_g` by cluster bootstrap. The bootstrap unit is the
phantom cluster, not an individual phantom sample, so the estimate can capture
correlation both within and between clusters.

For each block:

1. Resample phantom clusters with replacement.
2. Recompute `hat p_g`.
3. Let `q_g = (hat p_{>g}, hat p_{=g})^T`.
4. Compute the bootstrap covariance `Sigma_g^boot` of `q_g`.
5. Match it to the analytic iid multinomial covariance.

The analytic covariance for the first two components is

```text
Sigma_g = (1 / A_g) [
    [hat p_{>g} (1 - hat p_{>g}), -hat p_{>g} hat p_{=g}],
    [-hat p_{>g} hat p_{=g}, hat p_{=g} (1 - hat p_{=g})],
]
```

The effective-count model predicts `Sigma_g / rho_g`. The paper proposes

```text
hat rho_g =
    rank(Sigma_g) / trace(Sigma_g^+ Sigma_g^boot)
```

where `Sigma_g^+` is the Moore-Penrose inverse.

Because individual `hat rho_g` estimates are noisy, the paper suggests fitting a
low-order function of normalized race time:

```text
rho_g = c_0 + c_1 (s_g / s_G) + c_2 (s_g / s_G)^2.
```

## Stationarity Requirement

Phantom conditioning only corrects variance inflation from correlation. It does
not correct bias from non-stationary phantom states.

JAXNS v3 should therefore:

- retain phantom states only after a burn-in period;
- associate retained phantoms with the classic sample whose generation produced
  them;
- preserve clusters for bootstrap resampling;
- exclude non-uniform trajectory construction states, including Galilean
  reflection/bracketing points;
- use `rho_g` estimates with the understanding that non-stationary phantoms are
  outside the correction supplied by the paper.

## Downstream Use

Conditioned block probabilities enter the same evidence and posterior-weight
calculation as the non-phantom Bayesian model:

```text
X_g = X_{g-1} p_{>g}.
```

For plateau samples in the classic block `\mathcal{B}_g`, the equality atom mass
is split equally:

```text
w_g^{(k)} = lambda_g X_{g-1} p_{=g} / m_g.
```

For a non-plateau sample, the paper uses:

```text
w_g = lambda_g X_{g-1} (1 - p_{>g}).
```

## Data Requirements

The conditioning formulas require, for each retained phantom state:

- likelihood value `L(x)`;
- the cluster identity `P_i`;
- the associated classic sample `i`;
- the generation constraint likelihood used for that cluster, usually available
  as the accepted sample's strict `log_L_constraint` audit field.

The paper's constrained sampler returns phantom pairs `(x^k, L(x^k))`, but the
conditioning counts above use likelihood values and cluster identity. JAXNS v3
should therefore retain only phantom likelihoods, cluster boundaries, the
associated classic sample, and the generation constraint likelihood. Phantom
coordinates are discarded and are not added to posterior samples.

The generation constraint is provenance for validating eligibility of phantom
clusters in block counts. It is not a persisted parent id, and it does not make
phantom records part of the race tree.

## Invariants

- Phantom samples never increment out-degree.
- Phantom samples never change `K_g` or `m_g`.
- Phantom contributions enter only through the Dirichlet concentration
  parameters.
- The paper's effective-count model uses `0 < rho_g <= 1`.
- Retaining non-stationary phantom samples can bias shrinkage and is outside the
  correction supplied by the paper.
