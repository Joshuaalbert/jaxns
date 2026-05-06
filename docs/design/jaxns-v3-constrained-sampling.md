# JAXNS v3 Constrained Sampling

Status: paper-derived design draft.
Source: `docs/design/paper.tex`, Section 5.

This document defines the constrained-prior sampler contract for JAXNS v3. It is
agnostic of the current implementation.

## Statistical Contract

The statistical theory requires each classic child to be approximately an
independent sample from the strict constrained prior of its parent:

```text
x_j ~ pi_lambda
pi_lambda(dx) = 1{L(x) > lambda} p(dx) / X_lambda
```

for the parent contour `lambda`.

The constrained sampler may also return a phantom cluster. Retained phantom
states must have marginal expectation under the same parent contour after
burn-in. Correlated phantom states are allowed, but non-stationary phantom states
can bias the phantom-conditioned shrinkage model.

## Slice-Sampling Algorithm

The paper's constrained sampler uses one-dimensional slice-sampling transitions
from a seed point.

Inputs:

- direction kernel `D`;
- contour `lambda`;
- seed point `x^0`;
- steps per acceptance `T`;
- burn-in `B`.

Paper-equivalent pseudocode:

```text
set i = 0

while i < T:
    choose direction n_hat ~ D
    construct trajectory I through x^i along n_hat
        with endpoints outside the slice
        and endpoints independent from x^i
    uniformly sample x^{i+1} ~ pi_lambda along I
    set i = i + 1
    set done = L(x^i) > lambda

set phantom cluster P = {L(x^k): k in B..T-1}
return (x^T, L(x^T), P)
```

The final classic sample is `x^T`. Phantom collection starts after burn-in and
uses intermediate chain states, but the v3 sampler result retains only their
likelihood values and cluster boundary. Phantom coordinates are discarded after
their likelihood has been recorded.

## Direction Kernels

The paper lists two direction-kernel choices:

- isotropic Gaussian;
- ellipsoidal Gaussian.

For the ellipsoidal Gaussian, a Gaussian mixture model of the posterior is built
from samples collected so far. The direction kernel samples a Gaussian component
proportional to integrated volume, then samples the direction from that
component.

The direction distribution should be symmetric and independent of the current
chain point. All directions in a chain should come from the same distribution.

## Straight-Line Perfect Bracketing

The first trajectory method is perfect bracketing along a straight line.

Design properties from the paper:

- endpoints are the intersections with the unit-hypercube bounds representing
  the prior homogeneous measure;
- the interval is the maximal possible straight-line interval;
- interval construction needs no likelihood evaluations;
- constrained sampling along the interval uses uniform proposals;
- if a proposal falls outside the slice, greedy trajectory shrinkage is used;
- shrinkage exponentially contracts toward the current chain point, where the
  slice is necessarily satisfied.

The paper notes this trades zero-cost interval selection for extra likelihood
evaluations during shrinkage. It also notes the extra cost is expected to be
small and compensated by better prior exploration.

The paper identifies a risk: straight-line moves can increase mode evaporation
because a chain can move between modes. Adaptive lineage allocation can
ameliorate this, and depth-first uniform growth can rediscover modes.

## Galilean Trajectories

The second trajectory method is Galilean sampling.

Design properties from the paper:

- Hausdorff reflections off hard likelihood constraints preserve reversibility
  and detailed balance;
- the trajectory is kept inside the slice;
- the sampler proceeds from `x^i` along both `n_hat` and `-n_hat`;
- it stops when the momentum vector is anti-aligned with the initial direction,
  a U-turn;
- the resulting trajectory is non-straight;
- since the trajectory should remain in the slice, uniform sampling is used
  without trajectory shrinkage;
- the uniform point is chosen by sampling a trajectory segment proportional to
  its length, then sampling uniformly within that segment.

The paper specifies a single-side trajectory builder. Starting from a point
`x^0` with `L(x^0) > lambda`, an initial unit direction `n^0`, and a step size
`Delta t`, repeat while `n^i . n^0 >= 0`:

```text
propose x' = x^i + Delta t n^i

if L(x') <= lambda:
    halve Delta t until the proposal is back inside the contour
    reflect from the inside point using the likelihood gradient
    normalize the new direction
    append the inside point to the trajectory
else:
    double Delta t until the next proposal leaves the contour
    reflect from the last inside point using the likelihood gradient
    normalize the new direction
    append the last inside point to the trajectory
```

The reflection has the form

```text
n_new = n_old - 2 (n_old . grad L(x_reflect)) grad L(x_reflect),
```

followed by normalization. A full Galilean trajectory is built from the seed in
both `n_hat` and `-n_hat` directions, then sampled uniformly by path length.

## Seed Requirements

The core algorithm chooses a seed `x_j^0` with

```text
L(x_j^0) > L_i
```

for parent contour `L_i`. If no possible seed exists from a parent, for example
because of a plateau, the paper states that the parent is changed to the
sentinel. A valid constrained-sampler call therefore assumes its supplied seed
satisfies the strict parent contour.

## Phantom Collection Requirements

To support phantom-conditioned shrinkage:

- retain likelihoods for chain states `k in B..T-1`;
- retain cluster boundaries by classic sample;
- associate each cluster with the classic sample generated under the parent
  contour;
- retain the strict generation constraint likelihood for cluster eligibility;
- do not include burn-in states in the phantom cluster;
- do not treat phantom states as classic race samples.

Galilean trajectory-building points are not retained as phantom samples. The
paper explicitly warns that those internal points do not uniformly sample
`pi_lambda`. Only accepted Markov chain states from the generic constrained
sampler can contribute to the retained post-burn-in phantom cluster.

The retained phantom likelihoods must support the paper's block counts:

```text
A_g = sum 1{L(x) > L_{g-1}}
B_g = sum 1{L(x) > L_g} 1{L(x) > L_{g-1}}
E_g = sum 1{L(x) = L_g} 1{L(x) > L_{g-1}}.
```

The shrinkage layer then uses `A_g^eff = rho_g A_g`, Dirichlet concentration
updates scaled by `rho_g`, and cluster-bootstrap covariance matching for
`rho_g`. The paper also suggests a smoothed `rho_g` fit against normalized race
time. These requirements are why cluster boundaries must survive sampler output.

## Open Implementation Choices

The paper leaves these details to implementation:

- exact stepping and termination rules for greedy interval shrinkage;
- exact construction and update schedule for the posterior Gaussian mixture;
- Galilean initial step-size policy and safeguards for numerical gradient or
  reflection failures;
- how the burn-in `B` is chosen before phantom states are retained.

These choices should be made without altering the statistical contract: the
returned classic child must approximate a strict constrained-prior draw, and
retained phantoms must be stationary enough for unbiased constrained-prior
expectations.
