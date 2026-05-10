# JAXNS v3 Statistical Core

Status: paper-derived design draft.
Source: `docs/design/paper.tex`, Sections 1-4.

This document defines the statistical contract for JAXNS v3 independently of any
current implementation. It covers strict constrained priors, race trees, plateau
censorship, Bayesian shrinkage, evidence calculation, and posterior weights.

## Probability Model

Let `(A, Sigma, p)` be the prior probability space and let
`L: A -> [0, infinity)` be the likelihood or likelihood-like score. The evidence
is

```text
Z = integral_A L(x) dp(x).
```

For a threshold `lambda`, define the strict constrained set, its prior volume,
and the strict constrained prior:

```text
A_lambda = {x in A: L(x) > lambda}
X_lambda = p(A_lambda)
pi_lambda(dx) = 1{L(x) > lambda} p(dx) / X_lambda
```

when `X_lambda > 0`.

The sentinel likelihood `lambda_0` is the contour with `X_{lambda_0} = 1`.
For increasing distinct likelihood levels

```text
lambda_0 < lambda_1 < ... < lambda_G
X_g = X_{lambda_g}
```

the block quadrature estimate is

```text
Z ~= sum_{g=1}^G lambda_g (X_{g-1} - X_g).
```

## Race Tree

A classic sample is a pair `(x_i, L_i)` drawn from a strict constrained prior.
Classic samples and the sentinel form a race tree when every non-sentinel sample
has a parent sample whose contour generated it:

```text
for i in 1..N:
    exists p(i) in 0..N such that x_i ~ pi_{lambda_{p(i)}}
```

The parent relation is written `p(i) -> i`. A lineage is a maximal chain of
parent links. The out-degree of sample `i` is

```text
d_i = #{j: p(j) = i}.
```

The sentinel must have `d_0 > 0`. The initial active lineage count is

```text
K_1 = d_0.
```

For a race from sample `i - 1` to `i`, `K_i` active lineages race with unit-rate
exponential clocks:

```text
s_{i-1 -> i} ~ Exp(K_i)
r_{i-1 -> i} = exp(-s_{i-1 -> i}) ~ Beta(K_i, 1)
```

After sample `i` arrives, its incoming lineage terminates and its children begin:

```text
K_{i+1} = K_i - 1 + d_i.
```

This recurrence is the core accounting rule. It makes no reference to live
points or to the order in which children were generated, only to the final race
tree out-degrees. The parent relation is the mathematical construction that
justifies the out-degrees; the core algorithm only needs to persist out-degree
counts, and parent blocks can be reconstructed from sorted likelihoods and
out-degrees if they are needed for diagnostics.

If a selected parent has no strict seed, for example on a plateau, the core
algorithm changes that parent to the sentinel before generation. The out-degree
increment belongs to the actual parent after this fallback, and any derived
parent diagnostics must be reconstructed from those final out-degree counts.

## Blocks and Plateaus

Sort classic samples by nondecreasing likelihood:

```text
lambda_0 = L_0 < L_1 <= ... <= L_N.
```

For each distinct likelihood level `lambda_g`, define the block

```text
\mathcal{B}_g = {i: L_i = lambda_g}
m_g = |\mathcal{B}_g|.
```

When `m_g > 1`, the block is a plateau. The ordering of samples inside a
plateau is not unique. A valid block must have enough incoming racing lineages
to account for all plateau winners, so the shrinkage model requires
`K_g >= m_g`.

For block `g`, define the probability vector under the previous strict contour,
`x ~ pi_{lambda_{g-1}}`:

```text
p_g = (p_{>g}, p_{=g}, p_{<g})
p_{<g} = P(L_g > L > L_{g-1} | L > L_{g-1})
p_{=g} = P(L = L_g | L > L_{g-1})
p_{>g} = P(L > L_g | L > L_{g-1}) = X_g / X_{g-1}
```

These three components sum to one. The strict endpoint of the race is `p_{>g}`.

In a plateau, all winners finish the race, but the final in-atom rank does not
reveal the strict endpoint. The final plateau rank has distribution

```text
r^{(m_g)}_{g-1 -> g} ~ Beta(K_g - m_g + 1, m_g)
```

and is bounded by

```text
p_{>g} <= r^{(m_g)}_{g-1 -> g} <= p_{>g} + p_{=g}.
```

This final in-atom rank is not the strict endpoint `p_{>g}`. JAXNS v3 must
therefore model the equality atom separately from the strict endpoint.

## Bayesian Shrinkage Without Phantoms

For a block with incoming active lineage count `K_g` and block size `m_g`, the
paper interprets the classic race observations as:

- `K_g - m_g` samples lie past the equality atom.
- `m_g` samples fall in the equality atom.
- `0` samples fall in the open race interval.

Use the Dirichlet prior

```text
p(p_g) = Dir(1, epsilon_g, 1 - epsilon_g).
```

The paper suggests a neutral epsilon policy:

```text
epsilon_g = 1e-6  if m_g = 1
epsilon_g = 1/2   if m_g > 1
```

The paper interprets `epsilon_g` as prior belief about
`E[p_{=g} / (1 - p_{>g})]`, the atom mass fraction conditional on not passing
the strict endpoint.

The conjugate posterior is

```text
p_g | K_g, m_g ~ Dir(a_{>g}, a_{=g}, a_{<g})
a_{>g} = K_g - m_g + 1
a_{=g} = m_g + epsilon_g
a_{<g} = 1 - epsilon_g
```

with marginals

```text
p_{>g} | K_g, m_g ~ Beta(a_{>g}, a_{=g} + a_{<g})
p_{=g} | K_g, m_g ~ Beta(a_{=g}, a_{>g} + a_{<g})
```

For `p_{>g}`, this differs from the final plateau rank distribution by one unit
in the second beta shape, which is the paper's rigorous expression of plateau
censorship.

## Phantom-Conditioned Extension

The same block probability vector can be conditioned on retained post-burn-in
phantom samples when those states are stationary enough under their parent
contours. Phantom samples are organised as clusters `P_c`. Clusters are assumed
approximately independent of each other, while retained states within one
cluster may be arbitrarily correlated. A cluster can be the phantom set
associated with a classic sample, or any smaller disjoint subset if
between-cluster independence remains approximately valid. Cluster membership
and the generation parent contour must be retained for phantom conditioning.
Pre-burn-in states and non-uniform trajectory/bracketing/reflection points are
not phantom-conditioning observations.

For each block `g`, define per-cluster counts gated by parent contour
eligibility:

```text
A_{cg} = sum_{(x,L) in P_c} 1{L(x) > L_{g-1}} 1{L_{p(c)} <= L_{g-1}}
B_{cg} = sum_{(x,L) in P_c} 1{L(x) > L_g} 1{L_{p(c)} <= L_{g-1}}
E_{cg} = sum_{(x,L) in P_c} 1{L(x) = L_g} 1{L_{p(c)} <= L_{g-1}}
R_{cg} = A_{cg} - B_{cg} - E_{cg}.
```

The paper target is no longer an effective-count `rho_g` Dirichlet posterior.
Instead, each shrinkage draw starts with independent same-scale draws from the
non-phantom race posterior in gamma form:

```text
M_{>g} ~ Gamma(a_{>g}, 1), independently
M_{=g} ~ Gamma(a_{=g}, 1), independently
M_{<g} ~ Gamma(a_{<g}, 1), independently.
```

For that joint shrinkage draw, draw one independent cluster weight per phantom
cluster,

```text
v_c ~ Gamma(1, 1),
```

independent over clusters and independent of the race gammas. Reuse the same
`v_c` for cluster `c` across all blocks and components in that joint draw. Let

```text
C_g^Kish = (sum_c A_{cg})^2 / sum_c A_{cg}^2
I_g = 1{C_g^Kish >= C_min}.
```

`C_min` defaults to `20` unless explicitly overridden by the implementation
ticket or public API. If there are no participating clusters, `I_g = 0`. The
phantom-conditioned block draw is

```text
M'_{>g} = M_{>g} + I_g sum_c v_c B_{cg}
M'_{=g} = M_{=g} + I_g sum_c v_c E_{cg}
M'_{<g} = M_{<g} + I_g sum_c v_c R_{cg}

p_g = (M'_{>g}, M'_{=g}, M'_{<g})
      / (M'_{>g} + M'_{=g} + M'_{<g}).
```

Singleton independent phantom clusters recover the iid Dirichlet posterior by
gamma additivity. Correlated multi-sample clusters preserve the expected count
contribution but inflate variance by sharing one cluster weight. This correction
does not remove bias from non-stationary phantoms.

## Evidence Sampling

Each draw from the shrinkage model produces a volume path by recurrence:

```text
X_0 = 1
X_g = X_{g-1} p_{>g}
```

The evidence draw is then computed with block quadrature:

```text
Z ~= sum_{g=1}^G lambda_g (X_{g-1} - X_g).
```

This recurrence admits producing samples of the evidence. Expected-path
summaries are additional estimator choices, but the paper's uncertainty
discussion is based on Monte Carlo shrinkage samples.

## Posterior Weights

The paper distinguishes plateau and non-plateau posterior mass assignment.

For plateau samples in block `\mathcal{B}_g`, equality atom mass is divided
equally:

```text
w_g^{(k)} = lambda_g X_{g-1} p_{=g} / m_g,
k in \mathcal{B}_g.
```

For a non-plateau sample, the mass mirrors classic block quadrature:

```text
w_g = lambda_g X_{g-1} (1 - p_{>g}).
```

The design implication is that posterior weighting must have access to sampled
or expected `p_{=g}` as well as `p_{>g}`. A representation that stores only
classic shrinkage ratios cannot correctly handle equality atoms.

## Statistical Invariants

- Strict endpoint shrinkage is always `X_g / X_{g-1} = p_{>g}`.
- Equality atom mass is not part of the strict endpoint.
- Stationary phantom samples can condition `p_g` through gamma-weighted
  per-cluster counts, but do not alter `K_g`, `m_g`, or parent out-degrees.
- Race-tree validity depends on classic samples being independent constrained
  prior draws from their parents.
- Any generation strategy is valid only if the final sample set satisfies the
  race-tree parent relation and the constrained-sampling approximation.
