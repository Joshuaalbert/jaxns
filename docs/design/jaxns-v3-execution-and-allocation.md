# JAXNS v3 Execution and Allocation

Status: paper-derived design draft.
Source: `docs/design/paper.tex`, Section 5.

This document defines the intended JAXNS v3 execution model, dynamic lineage
allocation, depth and goal conditions, parent selection, and load-balanced
worker runtime contract. It is independent of the current code state.

## Execution Model

The race tree permits new children to be generated from any existing strict
sample contour. This separates statistical lineage accounting from the strategy
used to decide which contour receives more work.

The algorithm has two loops:

- An outer loop chooses a lineage allocation target function
  `K_*^k: R_+ -> N`.
- An inner loop generates samples under the current target until a depth
  condition says there is little utility in continuing with that target.

The outer loop stops when a user-specified goal condition is met.

JAXNS v3 has one statistical core. Local runs, batched local runs, and
distributed runs should all drive the same core state transitions and acceptance
logic. Deployment changes how constrained-sampler work and likelihood
evaluations are executed; it must not create a second nested-sampling algorithm
with separate semantics.

## Core State

The coordinator must maintain enough state to recover:

- sentinel likelihood `lambda_0`;
- root lineage count `d_0`;
- classic samples `(x_i, L_i)`;
- out-degree `d_i` for every sample;
- phantom likelihood cluster `P_i` for every classic sample;
- sorted blocks and active lineage counts `K_i` or `K_g`;
- shrinkage posterior state for volumes, evidence, and posterior weights.

The selected parent contour is necessary while a child is being generated, and
the chosen parent's out-degree is incremented when that child is accepted. The
paper explicitly states that persistent per-sample parent storage is unnecessary
because the algorithm only needs out-degrees. If parent blocks are desired for
analysis or diagnostics, they can be uniquely reconstructed from sorted
likelihoods and out-degrees.

## Statistical Accounting Used by Execution

The execution layer preserves the paper's race-tree quantities. The sentinel
out-degree is the initial active lineage count:

```text
K_1 = d_0.
```

For a race with `K_i` active lineages:

```text
s_{i-1 -> i} ~ Exp(K_i)
r_{i-1 -> i} ~ Beta(K_i, 1)
K_{i+1} = K_i - 1 + d_i.
```

For block-level shrinkage with possible plateaus, execution supplies `K_g` and
`m_g` to the Bayesian model over

```text
p_g = (p_{>g}, p_{=g}, p_{<g})
X_g = X_{g-1} p_{>g}.
```

Plateau atom mass `p_{=g}` is tracked separately from the strict endpoint
`p_{>g}`. Posterior mass in a plateau block is divided equally across the
samples in that block, while non-plateau mass uses `1 - p_{>g}`.

When phantom clusters are retained, execution preserves likelihood values,
cluster boundaries, and generation constraint likelihoods so the shrinkage layer
can compute parent-contour-gated per-cluster counts `A_{cg}`, `B_{cg}`,
`E_{cg}`, and `R_{cg}`. The shrinkage layer draws race posterior gamma
variables, draws independent `Gamma(1, 1)` weights per phantom cluster, applies
the Kish participating-cluster gate, records gate activation diagnostics, and
normalizes to `p_g`. Execution must preserve enough cluster identity and parent
contour provenance for that calculation while respecting the paper's burn-in
and stationarity warning. Phantom coordinates are not retained by the v3
execution state.

## Core Algorithm

Paper-equivalent pseudocode:

```text
input: root lineage count d_0, sentinel likelihood lambda_0

sample (x_i, L_i, P_i) ~ pi_{lambda_0} for i in 1..d_0
set N = d_0
set outer iteration k = 0

while not goal:
    choose allocation target K_*^k

    while not depth:
        count active lineages for current samples
        choose parent work set Q from under-allocated contours

        for each parent i in Q:
            set N = N + 1
            set j = N
            choose seed x_j^0 with L(x_j^0) > L_i
            sample (x_j, L_j, P_j) ~ pi_{L_i} from x_j^0
            set d_i = d_i + 1

        compute depth

    set k = k + 1
    compute goal

return {(x_i, L_i, P_i, d_i): i in 1..N}
```

For a parent with no possible seed, for example because of a plateau, the paper
states that the parent is changed to the sentinel.

The returned race state contains classic samples, phantom clusters, and
out-degrees. It does not need to contain stored parent indices.

## Parent Selection

During the inner loop, samples are identified whose active lineage counts do not
meet the allocation target. When targeting allocation to block `g'`, the paper
gives a strong parent-block choice:

```text
P(parent block = g | target block = g') proportional to
    (X_{g'} / X_g) 1{X_{g'} < X_g}.
```

The design intent is to choose a parent contour with high probability of
increasing active lineage count at the under-allocated target block.

## Depth Conditions

The depth condition captures when the current allocation target has little
remaining utility. The paper lists two available conditions.

Small remaining evidence:

```text
L_G X_G / (sum_g^G L_g (X_{g-1} - X_g) + L_G X_G) < tau_Z
```

Small remaining posterior mass:

```text
L_g X_g / max_{g'}(L_{g'} X_{g'}) < tau_post
```

Implementations should evaluate these from the current shrinkage state. If using
Monte Carlo shrinkage draws, the paper does not specify whether the condition is
evaluated with a mean, quantile, or other summary.

## Goal Conditions

The goal condition is user-specified and may be any condition computable from
the samples. The paper gives examples:

- target effective sample size;
- target evidence uncertainty.

The goal condition belongs to the outer loop. It decides whether another
allocation target should be attempted.

The public execution API should pass the current `State` to a user-supplied
goal condition:

```text
goal_cond(state: State) -> bool
```

The goal condition may call `state.to_result()` when it needs evidence,
posterior, or diagnostic summaries. Implementations should document and test the
cost of this conversion, because goal checks happen at outer-loop boundaries.

## Allocation Targets

Allocation targets define how likelihood evaluation work is dedicated per outer
loop. The paper lists three target families:

```text
Depth-first uniform:
K_*^k(L) = d_0 + k DeltaK

Evidence improving:
K_*^k(L) = d_0 + k DeltaK Ubar^Z(L)

Posterior improving:
K_*^k(L) = d_0 + k DeltaK Ubar^P(L)
```

where `DeltaK > 0`, and utilities are normalized to unit peak and represented as
functions of likelihood by linear interpolation.

Depth-first uniform allocation explores higher likelihood structure before
shallower contours are over-refined. The paper states that each outer loop costs
similar amounts of work, progressively better resolves modes, and manages cost
without choosing a large initial root lineage count.

The public string values are:

```text
"uniform"              -> depth-first uniform
"evidence_improving"  -> evidence-improving utility
"posterior_improving" -> posterior-improving utility
```

Unknown allocation targets must fail explicitly.

## Evidence-Improving Utility

Evidence-improving allocation seeks the contour where one more child has the
largest expected reduction in log-evidence variance.

```text
T^Z_{g h} = P_{x ~ pi_{L_g}}(L(x) > L_h)
          = (X_h / X_g) 1{h > g}
```

A child generated from parent block `g` changes the parent out-degree
`d_g -> d_g + 1`. If that child reaches block `h`, it contributes one
additional active lineage to the intervening strict shrinkage races.

Use the block quadrature evidence `Z` and define the first-order sensitivity

```text
B_h = d log Z / d log p_{>h}
    = (L_h X_h - sum_{j > h} L_j (X_{j-1} - X_j)) / Z.
```

For `p_{>h} ~ Beta(alpha_h, beta_h)`,

```text
Var[log p_{>h}] = psi_1(alpha_h) - psi_1(alpha_h + beta_h),
```

where `psi_1` is the trigamma function. Adding one active lineage gives
`alpha_h -> alpha_h + 1`, so the delta approximation to the log-evidence
variance reduction is

```text
Delta Var[log Z; h] ~= B_h^2 (
    1 / alpha_h^2 - 1 / (alpha_h + beta_h)^2
).
```

The evidence-improving utility is the transition expectation of this reduction:

```text
U^Z_g = (1 / X_g) sum_{h > g} X_h B_h^2 (
    1 / alpha_h^2 - 1 / (alpha_h + beta_h)^2
).
```

The paper notes this can be computed efficiently with two cumulative reductions.

## Posterior-Improving Utility

Posterior-improving allocation seeks contours likely to place a new child in
low-information posterior shells.

```text
T^P_{g h} = P_{x ~ pi_{L_g}}(L_{h-1} < L(x) <= L_h)
          ~= ((X_{h-1} - X_h) / X_g) 1{h > g}.
```

Use Kish's effective sample size as the proxy:

```text
ESS_Kish = W^2 / Q
W = sum_j w_j
Q = sum_j w_j^2.
```

If a new sample lands in shell `h`, it splits shell mass `w_h`. With
`u ~ Uniform(0, 1)`, the old `w_h^2` contribution to `Q` is replaced by
`u^2 w_h^2 + (1 - u)^2 w_h^2`, while `W` is unchanged. Marginalizing over `u`
gives

```text
Delta ESS_Kish_h =
    W^2 [
        atan(sqrt(w_h^2 / (2Q - w_h^2)))
        / sqrt((Q - w_h^2 / 2)(w_h^2 / 2))
        - 1 / Q
    ],
```

with continuous limiting value `0` as `w_h -> 0`.

The paper also gives a conservative approximation:

```text
Delta ESS_Kish_h ~= W^2 / (Q - w_h^2 / 3) - W^2 / Q.
```

The posterior-improving utility is

```text
U^P_g = (1 / X_g) sum_{h > g}
    (X_{h-1} - X_h) Delta ESS_Kish_h.
```

## Load-Balanced Worker Runtime Contract

JAXNS v3 is designed for heterogeneous parallel and distributed likelihood
evaluation, from a laptop to clusters with thousands of nodes.

The paper's runtime contract is:

- a central process coordinates the core algorithm;
- workers perform likelihood evaluations;
- workers receive serialized likelihood models;
- scheduling uses load balancing;
- communication uses ZMQ;
- serialization uses Python `pickle`;
- likelihood-model arguments should be arbitrarily nested trees of literals and
  arrays;
- constrained-sampling calls from different parent contours can overlap in
  wall-clock time.

Ticket 0018 refines the ordinary remote work unit to likelihood evaluation
only. Constrained samplers remain local to the runner, may run in parallel where
execution policy allows, and send one proposed `U` per likelihood probe. Identity
registration may happen during runner creation or before ordinary work, but
worker-local JIT compilation happens on first matching work for a registered
identity/device class.

The intended user-facing runtime has these roles:

- A load balancer owns worker registration, fair sharing, compile-identity
  registration/cache coordination, and nested-sampler runner creation.
- Workers join the load balancer and establish compute sectors such as CPU or
  GPU device pools. Each likelihood worker process handles at most one active
  likelihood evaluation at a time.
- A nested-sampler runner owns one v3 core state for one submitted model and
  dispatches likelihood-eval work through the load balancer.
- Multiple clients may submit different models to the same worker pool; the load
  balancer schedules fairly without changing statistical semantics.

This runtime is not a layer over a single-tenant, lease-first, point-to-point
transport. If existing transport code is useful as a utility it may be reused,
but the v3 runtime contract is the load-balancer/worker/runner model above.

The race-tree formulation makes asynchronous execution natural because a
completed child only updates the out-degree of its known effective parent.

## Invariants

- Generation order is not the statistical ordering. Statistical ordering is by
  likelihood blocks.
- Dispatching a child reserves no shrinkage contribution by itself. The race tree
  changes when the child is accepted and the parent's out-degree is updated.
- Allocation targets can change future work but must not change the shrinkage
  law for existing race-tree samples.
- Load-balanced execution must preserve the in-flight parent target until
  completion, because that out-degree is updated on acceptance. Persistent race
  state still only needs out-degrees, not per-sample parent fields.
- Load-balanced execution must preserve phantom cluster identity across
  serialization boundaries.
