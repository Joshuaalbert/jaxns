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
contours. Phantom samples are organised as clusters `P_i` associated with the
classic sample whose generation produced them. Cluster membership must be
retained for the paper's bootstrap estimate of phantom correlation.
Pre-burn-in states and non-uniform trajectory/bracketing/reflection points are
not phantom-conditioning observations. Define phantom counts over
`P = union_i P_i`:

```text
A_g = sum_{(x,L) in P} 1{L(x) > L_{g-1}}
B_g = sum_{(x,L) in P} 1{L(x) > L_g} 1{L(x) > L_{g-1}}
E_g = sum_{(x,L) in P} 1{L(x) = L_g} 1{L(x) > L_{g-1}}.
```

Because phantom samples are correlated, the paper uses an effective count
`A_g^eff = rho_g A_g` with `0 < rho_g <= 1`. The conditioned posterior is

```text
p_g | K_g, m_g, rho_g, P ~ Dir(b_{>g}, b_{=g}, b_{<g})
b_{>g} = K_g - m_g + 1 + rho_g B_g
b_{=g} = m_g + epsilon_g + rho_g E_g
b_{<g} = 1 - epsilon_g + rho_g (A_g - B_g - E_g).
```

The paper estimates `rho_g` by cluster-bootstrap covariance matching. It uses
the covariance of the first two components
`q_g = (hat p_{>g}, hat p_{=g})^T`, the iid multinomial covariance `Sigma_g`,
and the bootstrap covariance `Sigma_g^boot`:

```text
hat rho_g =
    rank(Sigma_g) / trace(Sigma_g^+ Sigma_g^boot).
```

Because per-block estimates are noisy, the paper suggests fitting a low-order
function of normalized race time:

```text
rho_g = c_0 + c_1 (s_g / s_G) + c_2 (s_g / s_G)^2.
```

This variance correction cannot remove bias from non-stationary phantoms.

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
- Stationary phantom samples can condition `p_g` through effective-count
  Dirichlet updates, but do not alter `K_g`, `m_g`, or parent out-degrees.
- Race-tree validity depends on classic samples being independent constrained
  prior draws from their parents.
- Any generation strategy is valid only if the final sample set satisfies the
  race-tree parent relation and the constrained-sampling approximation.
