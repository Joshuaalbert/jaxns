# JAXNS v3 Design Overview

Status: paper-derived design draft.
Source: `docs/design/paper.tex`.

This document set describes the intended JAXNS v3 design from the paper
"Phantom-Boosted Nested Sampling". It is deliberately agnostic of the current
code state. Treat these documents as paper-derived implementation guidance, not
as an audit of what the repository currently implements.

## Design Documents

| Document | Scope |
| --- | --- |
| `jaxns-v3-statistical-core.md` | Race-tree nested sampling, plateaus, Bayesian shrinkage, evidence and posterior weights. |
| `jaxns-v3-phantom-conditioning.md` | Phantom sample clusters, effective multinomial conditioning, `rho_g` estimation, and stationarity requirements. |
| `jaxns-v3-execution-and-allocation.md` | Single core outer and inner algorithm, lineage allocation targets, depth and goal conditions, parent choice, and load-balanced worker execution. |
| `jaxns-v3-constrained-sampling.md` | Constrained-prior sampler contract, slice-sampling transitions, direction kernels, trajectory construction, and phantom collection; includes the Galilean rule that trajectory-building points are not retained as phantoms. |
| `jaxns-v3-validation-plan.md` | Correctness, calibration, evidence-efficiency, and posterior-quality benchmark protocol. |

## Paper-Derived Scope

JAXNS v3 is intended to implement phantom-boosted nested sampling. The paper
frames the method around these ideas:

1. Evidence estimation is a one-dimensional compression problem once likelihood
   contours have prior volumes.
2. The volume sequence is treated as a random variable with a race-induced prior.
3. Classic nested-sampling samples form a race tree of independent constrained
   prior draws.
4. Phantom samples are correlated intermediate constrained-prior states. They
   must not be inserted into the classic race sequence, but they can condition
   shrinkage as Monte Carlo observations.
5. Plateaus are handled by a Bayesian categorical model over strict endpoint,
   equality atom, and open race interval probabilities.
6. The race-tree formulation decouples valid shrinkage accounting from the order
   and strategy used to generate children, enabling dynamic allocation and
   asynchronous execution.

## Global Design Principles

- All shrinkage quantities are strict: contours are defined by `L(x) > lambda`.
  Inclusive constraints may be useful inside a sampler, but they do not define
  the evidence shrinkage law in the paper.
- A classic child sample must be approximately independent from the strict
  constrained prior of its parent contour.
- In the paper's shrinkage model, a phantom sample acts as a Monte Carlo
  observation of constrained-prior probabilities. It is not a race participant
  and does not change lineage counts.
- The sentinel sample defines the initial likelihood contour with prior volume
  one. Its out-degree is the root lineage count.
- Parent out-degrees, not live-point membership, are the primitive bookkeeping
  needed to infer active lineage counts. The core algorithm does not need to
  persist a parent field for each sample; if parent blocks are needed later,
  they can be reconstructed from sorted likelihoods and out-degrees.
- Phantom clusters contain retained post-burn-in constrained-sampler states,
  not arbitrary intermediate likelihood evaluations. Galilean trajectory
  bracketing/reflection points are excluded because they are not uniform draws
  from the constrained prior.
- Plateaus are represented as blocks of equal likelihood. Equality atom mass is
  inferred explicitly rather than folded into the strict endpoint. The atom
  censors the strict endpoint: the final plateau rank lies in
  `p_{>g} <= r^{(m_g)} <= p_{>g} + p_{=g}` rather than revealing `p_{>g}`.
- Evidence uncertainty should be obtained by sampling the Bayesian shrinkage
  model, with `X_g = X_{g-1} p_{>g}` at each block.
- The Bayesian shrinkage model uses `Dir(1, epsilon_g, 1 - epsilon_g)` before
  classic and phantom observations. The paper suggests `epsilon_g = 1e-6` for
  singleton blocks and `epsilon_g = 1/2` for plateau blocks.
- Phantom conditioning uses block counts `A_g`, `B_g`, and `E_g`, scaled by an
  effective-count factor `rho_g`, to update the Dirichlet concentrations.
- Plateau posterior mass is assigned from the equality atom and divided equally
  across samples in the plateau block.
- Dynamic allocation is an experimental-design layer over the race tree. It may
  choose where to spend future likelihood evaluations without changing the
  shrinkage accounting.
- JAXNS v3 should have one statistical execution core. Local, batched,
  threaded, and distributed runs are deployment choices for evaluating the same
  core work items, not separate nested-sampling algorithms to maintain.
- The worker runtime is a load-balanced, multi-tenant execution service: workers
  establish compute sectors, and the load balancer creates nested-sampler
  runners for submitted models. This is a design requirement, not a compatibility
  constraint with any existing point-to-point or lease-based transport.

## Conceptual Records

An implementation can choose any concrete storage model, but the paper uses the
following conceptual quantities, so an implementation should represent or derive
them:

- Sentinel record:
  `sample_index = 0`, `L_0 = lambda_0`, `X_{lambda_0} = 1`, and out-degree
  `d_0 > 0`.
- Classic sample record:
  sample index, point `x_i`, likelihood `L_i`, out-degree `d_i`, and
  associated phantom cluster `P_i`. The generating parent contour is transient
  execution context, not required persisted race state in the paper algorithm.
- Block record:
  distinct likelihood level `lambda_g`, sample set
  `\mathcal{B}_g = {i: L_i = lambda_g}`, block size `m_g`, incoming active
  lineage count `K_g`, and shrinkage
  probabilities `(p_{>g}, p_{=g}, p_{<g})`.
- Phantom cluster record:
  the classic sample it belongs to, retained phantom likelihood values, cluster
  boundaries, and the generation constraint likelihood. The paper writes
  phantom pairs `(x, L(x))`, but the shrinkage model uses only likelihoods and
  cluster identity; v3 does not retain phantom coordinates and does not add
  phantoms to posterior samples.

## Paper Status Notes

The current paper draft gives explicit formulas for the evidence-improving
utility `U^Z_g`, the posterior-improving utility `U^P_g`, and the Galilean
trajectory construction. The experimental results are still placeholders for
future benchmarks, so validation docs describe the required protocol without
inventing numeric pass thresholds.
