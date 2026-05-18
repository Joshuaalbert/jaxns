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

## Runtime Boundary For Likelihood Evaluation

The constrained sampler owns local chain state. It chooses seeds, directions,
slice or trajectory proposals, strict-contour checks, phantom-retention
boundaries, and retry/failure policy for a parent task. In the process-isolated
runtime, only deterministic likelihood probes are dispatched to remote worker
processes: the sampler sends one proposed prior-space coordinate `U` and
receives one scalar `log_L`.

This means constrained samplers may run locally and in parallel for many parent
tasks, while likelihood evaluations overlap through the worker pool. Sampler
state, phantom buffers, direction snapshots, and parent metadata do not cross
the worker boundary for every proposal. The runner remains responsible for
accepting completed child samples and mutating race-tree state exactly once.

Sampler implementations should avoid exposing dynamic JAX shapes to the
likelihood worker path. Variable candidate lists, active-parent pools, phantom
buffers, and trajectory segments should be trimmed in NumPy before JAX calls or
represented with fixed shapes and explicit masks. A likelihood worker may JIT
its model on first work for a static `U` tree shape, but it must not silently
recompile for variable proposal shapes under the same runtime identity.

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

The direction distribution must be symmetric and independent of the current
chain point. All directions in a chain must come from the same frozen
distribution. Direction-kernel adaptation is therefore an execution-level update
between chains and shell epochs, never a mutation during a chain.

### Isotropic Direction Kernel

The isotropic baseline draws a standard normal vector in the flattened
prior-space coordinate system and normalizes it to unit length. In one
dimension this reduces to a random sign. This kernel is always available and is
the fallback whenever non-isotropic adaptation is unavailable or invalid.

### GMM-Based Non-Isotropic Direction Kernel

The ellipsoidal, non-isotropic direction kernel `D` is a Gaussian-mixture
direction kernel fit to the current posterior approximation from samples
collected so far, matching the paper's statement that the GMM is a posterior
GMM. The fit is used only to choose directions; it does not alter the strict
constrained-prior target and it does not enter shrinkage accounting.

Adaptation state:

- The coordinator owns the posterior fitting dataset. Its rows are flattened
  prior-space coordinates of accumulated accepted classic samples whose
  likelihoods and posterior weights are known at the time of the update.
- Posterior fitting weights are the same active normalized posterior weights
  used for result construction at that state. "Active" means
  phantom-conditioned where retained phantom metadata and the Kish gate make
  that the result target, and classic race posterior otherwise. For plateau
  blocks, use the same plateau posterior-mass convention as results: equality
  atom mass is divided equally across samples in the plateau block.
- Conceptually, for shrinkage draw `s` and likelihood block `g`, use raw
  fitting mass

```text
a_i^(s) = L_g X_{g-1}^{(s)} (1 - p_{>g}^{(s)})
          for singleton/non-plateau block member i in g,
a_i^(s) = L_g X_{g-1}^{(s)} p_{=g}^{(s)} / m_g
          for plateau block member i in g.
```

  Normalize within each shrinkage draw first,

```text
w_i^(s) = a_i^(s) / sum_j a_j^(s),
```

  then average the normalized per-sample weights over shrinkage draws to obtain
  `w_i`, with a final normalization only to remove numerical drift. Do not
  average raw masses over draws and normalize once; that is a different target
  when the evidence varies by draw. Implementations should evaluate these
  expressions in log space when needed.
- Weighted EM is the preferred fit. An implementation may instead form an
  equivalent weighted posterior approximation by systematic resampling from the
  normalized posterior weights and running unweighted EM on the resampled rows,
  provided the resampling seed and effective sample size are recorded.
- Retained phantom likelihood clusters are not used to fit `D`. This is both a
  v3 storage policy and a statistical policy: v3 discards phantom coordinates,
  and phantoms are used only as shrinkage diagnostics.
- The constrained sampler and workers are pure consumers of an immutable
  direction-kernel snapshot. They must not update the sample history, refit the
  mixture, or change component probabilities.
- Each fitted kernel has a monotonically increasing version identifier for
  diagnostics and for reproducing which kernel was used by an in-flight chain.

Update schedule:

- The initial kernel is isotropic.
- The paper does not name an update cadence. JAXNS v3 policy is to attempt the
  first non-isotropic fit after root/bootstrap samples produce a valid
  posterior fitting dataset, and then after every `S_D = 5` newly completed
  distinct likelihood shells since the last successful direction-kernel update.
- A shell is a distinct sorted likelihood block; plateau blocks count as one
  shell for the update schedule.
- A successful fit becomes the active kernel only for future chain dispatches.
  Chains already in flight continue with the snapshot selected at dispatch.
- All chain dispatches in the same shell epoch use the same active kernel
  version. The kernel is not updated inside the chain, between Markov steps, or
  for individual directions.

Fitting target:

- Fit a Gaussian mixture in flattened prior-space coordinates using the
  posterior-weighted accumulated accepted classic samples.
- Let `D_dim` be the flattened prior-space dimension, `N_pos` the number of
  classic samples with positive posterior fitting weight, and
  `N_eff = 1 / sum_i w_i^2` the Kish effective size of the normalized fitting
  weights.
- Do not attempt a non-isotropic fit unless `N_pos >= D_dim + 1`,
  `N_eff >= max(20, 2 * (D_dim + 1))`, and at least `D_dim + 1` unique
  coordinate rows remain after removing non-finite rows.
- Use at most `M_D = 20000` fitting rows. If at most `M_D` samples have positive
  posterior weight, run weighted EM on those rows. If more than `M_D` samples
  have positive posterior weight, first form a bounded posterior fitting dataset
  by systematic resampling `M_D` rows from the normalized posterior weights,
  then run unweighted EM on the resampled rows with duplicate rows preserved.
  Record the resampling seed, `M_D`, original `N_pos`, and original `N_eff`.
- Use

```text
K_D = min(8, max(1, floor(N_eff / (2 * (D_dim + 1)))))
```

  Gaussian components, further capped by the number of unique fitting rows.
- Each component supplies a mean, covariance-derived radii, and rotation. The
  mean is retained for diagnostics and for interpreting the fitted component,
  but direction draws are centered at zero so that `D(n) = D(-n)`.
- For component `k`, compute the weighted covariance in flattened coordinates
  from its responsibilities. Let `C_global` be the weighted covariance of the
  full fitting dataset and define

```text
sigma2_floor = max(1e-12, 1e-6 * trace(C_global) / D_dim).
C_k_reg = C_k + sigma2_floor I.
```

- Obtain the component rotation and radii by eigendecomposition or SVD of
  `C_k_reg`; radii are `sqrt(max(eigenvalue, sigma2_floor))`.
- Component integrated volume is the ellipsoidal volume implied by the
  covariance radii, proportional to `prod(radii)` up to the common
  dimension-dependent constant.
- Component selection probabilities are normalized integrated volumes:

```text
q_k = V_k / sum_j V_j.
```

- To draw a direction, sample `k ~ Categorical(q)`, draw a zero-mean Gaussian
  vector with component `k`'s radii and rotation, then normalize the result to a
  unit direction in the flattened prior-space coordinate system.

Fallbacks and safeguards:

- If the fitting dataset does not pass the `N_pos`, `N_eff`, uniqueness, or
  finiteness thresholds, keep the previous successful non-isotropic kernel; if
  none exists, use isotropic directions.
- Drop invalid components with non-finite parameters or non-positive integrated
  volume. Also drop components whose effective responsibility size

```text
N_eff,k = (sum_i r_ik w_i)^2 / sum_i (r_ik w_i)^2
```

  is less than `D_dim + 1`. If no valid component remains, use the previous
  successful kernel or the isotropic fallback.
- If EM fitting, covariance regularization, volume normalization, or component
  sampling fails, the failure is a direction-adaptation failure only; it must
  not fail the nested-sampling run unless the user explicitly requested a hard
  adaptation error.
- Fallback activation, fitting `N_pos`, fitting `N_eff`, component counts,
  component probabilities, dropped-component counts, and kernel version changes
  should be recorded as diagnostics.

Validation expectations:

- Unit tests should verify that fitted component probabilities are proportional
  to integrated volume, normalize to one, and reject or drop invalid volumes.
- Tests should verify that posterior fitting weights, not unweighted classic
  sample counts, define the GMM fitting target; weighted EM and systematic
  posterior resampling should agree within Monte Carlo tolerance on simple
  fixtures.
- Direction draws from a frozen non-isotropic kernel should be symmetric around
  zero, unit-normalized, and independent of the current chain point except for
  tree shape/unflattening.
- Tests should verify that all directions in one chain use the same frozen
  kernel snapshot and that in-flight chains are not affected by later shell
  updates.
- Schedule tests should verify the initial isotropic kernel, attempted
  non-isotropic updates under the v3 `S_D = 5` policy, plateau blocks counting
  as one shell, and fallback to the previous valid kernel or isotropic kernel.
- Statistical validation should compare isotropic and non-isotropic settings on
  constrained-prior exploration without changing evidence calibration: any
  improvement should appear in mixing, likelihood-evaluation efficiency, or
  posterior exploration, not as a change to the shrinkage law.

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
- retain the strict generation parent contour for cluster eligibility;
- do not include burn-in states in the phantom cluster;
- do not treat phantom states as classic race samples.

Galilean trajectory-building points are not retained as phantom samples. The
paper explicitly warns that those internal points do not uniformly sample
`pi_lambda`. Only accepted Markov chain states from the generic constrained
sampler can contribute to the retained post-burn-in phantom cluster.

The retained phantom likelihoods must support the paper's block counts:

```text
A_{cg} = sum_{(x,L) in P_c} 1{L(x) > L_{g-1}} 1{L_{p(c)} <= L_{g-1}}
B_{cg} = sum_{(x,L) in P_c} 1{L(x) > L_g} 1{L_{p(c)} <= L_{g-1}}
E_{cg} = sum_{(x,L) in P_c} 1{L(x) = L_g} 1{L_{p(c)} <= L_{g-1}}
R_{cg} = A_{cg} - B_{cg} - E_{cg}.
```

The shrinkage layer then draws race posterior gamma variables, draws one
`Gamma(1, 1)` weight per phantom cluster, adds weighted per-cluster counts only
when the Kish participating-cluster threshold is met, and normalizes to `p_g`.
The open-interval component uses `R_{cg}` rather than raw `A_{cg}`.
These requirements are why cluster boundaries and generation parent contours
must survive sampler output.

## Open Implementation Choices

The paper leaves these details to implementation:

- exact stepping and termination rules for greedy interval shrinkage;
- Galilean initial step-size policy and safeguards for numerical gradient or
  reflection failures;
- how the burn-in `B` is chosen before phantom states are retained.

These choices should be made without altering the statistical contract: the
returned classic child must approximate a strict constrained-prior draw, and
retained phantoms must be stationary enough for unbiased constrained-prior
expectations.
