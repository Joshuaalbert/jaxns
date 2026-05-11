# JAXNS v3 Phantom Conditioning

Status: paper-derived design draft.
Source: `docs/design/paper.tex`, Sections 1, 3, and 4.

This document defines how JAXNS v3 should use phantom samples to condition the
Bayesian shrinkage model. It intentionally treats current code as irrelevant,
and it describes the statistical target rather than a required implementation
layout.

## Role of Phantom Samples

A phantom cluster `P_c` is a set of retained post-burn-in constrained-sampler
transition states produced under one generation contour. In the common case the
cluster is the phantom set associated with classic sample `i`, but the design
only requires a disjoint decomposition whose clusters are approximately
independent of each other. States within one cluster may be arbitrarily
correlated.

Each cluster has a parent contour `L_{p(c)}`. Its retained states are correlated
draws from the strict constrained prior of that contour:

```text
P_c contains states from pi_{L_{p(c)}}.
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

When a retained sampler trajectory can be split into smaller disjoint clusters
that are still approximately independent between clusters, the paper favors
using the smaller clusters. This increases information use while preserving the
assumption that only between-cluster independence is relied on.

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

## Per-Cluster Counts

For each block `g` and phantom cluster `c`, define counts with a parent-contour
eligibility gate:

```text
A_{cg} = sum_{(x,L) in P_c} 1{L(x) > L_{g-1}} 1{L_{p(c)} <= L_{g-1}}
B_{cg} = sum_{(x,L) in P_c} 1{L(x) > L_g} 1{L_{p(c)} <= L_{g-1}}
E_{cg} = sum_{(x,L) in P_c} 1{L(x) = L_g} 1{L_{p(c)} <= L_{g-1}}
R_{cg} = A_{cg} - B_{cg} - E_{cg}
```

The gate ensures that cluster `c` only conditions block `g` if its retained
states were generated from a contour no stricter than the block's parent
contour. Then `A_{cg}` counts participating retained states, `B_{cg}` counts
states beyond the strict endpoint, `E_{cg}` counts states in the equality atom,
and `R_{cg}` counts states in the open interval
`L_{g-1} < L(x) < L_g`.

The corresponding Monte Carlo estimate of the block probability vector is

```text
hat p_g = (
    sum_c B_{cg} / sum_c A_{cg},
    sum_c E_{cg} / sum_c A_{cg},
    sum_c R_{cg} / sum_c A_{cg}
).
```

The three components estimate `(p_{>g}, p_{=g}, p_{<g})`.

The paper does not specify behavior when `sum_c A_{cg} = 0`; in that case
`hat p_g` is undefined and the block has no phantom Monte Carlo estimate in the
stated formula.

## Gamma-Weighted Conditioning Target

The old design target was an effective-count Dirichlet posterior using a
per-block `rho_g`. That is no longer the paper target. The target is now a
generative shrinkage draw built from gamma variables.

First draw independent, same-scale race-induced posterior gammas:

```text
M_{>g} ~ Gamma(a_{>g}, 1), independently
M_{=g} ~ Gamma(a_{=g}, 1), independently
M_{<g} ~ Gamma(a_{<g}, 1), independently

a_{>g} = K_g - m_g + 1
a_{=g} = m_g + epsilon_g
a_{<g} = 1 - epsilon_g
```

Then draw one independent cluster weight per phantom cluster for the joint
shrinkage draw:

```text
v_c ~ Gamma(1, 1), independently over clusters and independent of the race gammas.
```

The same `v_c` must be reused for cluster `c` across all blocks and all
components within that joint shrinkage draw. A new joint shrinkage draw gets a
new independent set of race gammas and cluster weights. Equivalent
implementations are acceptable if they preserve the same distributional target.

Before adding phantom counts, require enough independent participating clusters.
The paper uses a Kish-like participating cluster count:

```text
C_g^Kish = (sum_c A_{cg})^2 / sum_c A_{cg}^2.
```

If there are no participating clusters, the denominator is zero and the
criterion fails. The canonical minimum threshold name is `C_min`, with default
`C_min = 20`. Implementations should expose `C_min` as a configurable parameter
with this default unless the implementation ticket or public API explicitly
chooses a different default.

For a block with `C_g^Kish >= C_min`, add weighted per-cluster counts and
normalize:

```text
I_g = 1{C_g^Kish >= C_min}

M'_{>g} = M_{>g} + I_g sum_c v_c B_{cg}
M'_{=g} = M_{=g} + I_g sum_c v_c E_{cg}
M'_{<g} = M_{<g} + I_g sum_c v_c R_{cg}

p_g = (M'_{>g}, M'_{=g}, M'_{<g})
      / (M'_{>g} + M'_{=g} + M'_{<g})
```

If the threshold is not met, `I_g = 0` and the block uses the non-phantom
race-induced posterior.

In public v3 result and MC-shrinkage outputs, "active" block probability
statistics mean this mixed target: gamma-weighted phantom conditioning for
valid blocks whose Kish gate is active, and the classic race posterior for
blocks with no phantoms, no participating clusters, or insufficient Kish
participation. `C_min` defaults to `20`. A block with zero participating
clusters fails the gate even if a caller supplies `C_min <= 0`; implementations
may still reject non-positive public `C_min` values at validation boundaries.
Public summaries named like `p_gt_mean`, `p_eq_mean`, and `p_lt_mean` should be
means of the active probability samples returned by the same call. Classic-only
probability summaries should be exposed only with explicit classic-only names.

This construction has two required limiting behaviors:

- If all phantom clusters are singleton independent observations, then
  `A_{cg}`, `B_{cg}`, `E_{cg}`, and `R_{cg}` are category indicators. Gamma
  additivity with common scale recovers the iid Dirichlet posterior exactly.
- If a cluster contains correlated multiple observations, the shared `v_c`
  preserves the expected count contribution but ties those observations
  together, inflating variance relative to singleton independent observations.

The third component is expressed with `R_{cg} = A_{cg} - B_{cg} - E_{cg}` so the
normalized vector remains categorical over strict endpoint, equality atom, and
open interval.

## Stationarity Requirement

Phantom conditioning only accounts for within-cluster correlation through the
cluster-weight construction. It does not correct bias from non-stationary
phantom states.

JAXNS v3 should therefore:

- retain phantom states only after a burn-in period;
- associate retained phantoms with the classic sample whose generation produced
  them;
- preserve cluster boundaries for per-cluster counts and cluster weights;
- exclude non-uniform trajectory construction states, including Galilean
  reflection/bracketing points;
- treat non-stationary phantoms as outside the correction supplied by the
  paper.

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
- the cluster identity `P_c`;
- the associated classic sample `i`;
- the generation constraint likelihood `L_{p(c)}` used for that cluster,
  usually available as the accepted sample's strict `log_L_constraint` audit
  field.

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
- Phantom contributions enter only through the gamma-weighted shrinkage draw.
- The old `rho_g` effective-count Dirichlet posterior is not the paper target.
- Singleton independent phantom clusters recover the iid Dirichlet posterior by
  gamma additivity.
- Correlated multi-sample clusters inflate shrinkage variance while preserving
  the expected count contribution.
- Retaining non-stationary phantom samples can bias shrinkage and is outside the
  correction supplied by the paper.

## Paper-Review Notes

- Earlier paper drafts alternated between `C_min` and `C_0` for the minimum
  Kish participating-cluster threshold. The current paper and this design use
  `C_min` as the canonical name.
- Earlier paper drafts added raw `A_{cg}` in the displayed `M'_{<g}` update.
  The required target is the open-interval count
  `R_{cg} = A_{cg} - B_{cg} - E_{cg}`, matching the Monte Carlo vector and the
  singleton Dirichlet-recovery property; the current paper has been edited to
  that target in this repo.
