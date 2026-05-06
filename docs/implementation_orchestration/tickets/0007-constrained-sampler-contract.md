# Ticket 0007: Constrained Sampler Contract and Phantom Collection

Branch: `feature/v3-constrained-sampler-contract`
Priority: 5
Depends on: Ticket 0001, Ticket 0003
Design docs:

- `docs/design/jaxns-v3-constrained-sampling.md`
- `docs/design/jaxns-v3-phantom-conditioning.md`
- `docs/design/jaxns-v3-statistical-core.md`

## Goal

Align the constrained sampler interface and baseline slice sampler with the v3
statistical contract: each classic child approximates an independent strict
constrained-prior draw from its parent, while retained phantom states are
clustered post-burn-in Markov states used only for Monte Carlo conditioning.

## Current Code Context

Relevant files likely include:

- `src/jaxns/constrained_sampler.py`
- `src/jaxns/constrained_sampler_distributed.py`
- `src/jaxns/samples.py`
- `src/jaxns/core.py`
- `tests/test_constrained_sampler.py`

Current `UniDimSliceSampler` supports slice proposals, phantom collection, and a
burn-in parameter. The contract needs stronger v3 tests around strict seed
constraints, child independence expectations, cluster boundaries, and no-seed
sentinel fallback integration. Keep the existing sampler interface and return
type unless a test proves it cannot represent the v3 contract.

Known current-code gaps to address in this ticket:

- `UniDimSliceSampler.get_sample()` does not validate `seed_point.log_L0` against
  the strict constraint at the sampler boundary.
- `num_phantom()` can become negative when `phantom_burn_in > num_slices - 1`;
  validate `0 <= burn_in <= num_slices - 1` in local and worker-backed sampler
  paths.
- The local sampler's validation hook must actually run during construction.
- Direct `_new_proposal` tests should be kept passing while sampler contract
  checks are tightened.

## Required Behavior

- Sampler calls require a seed satisfying the strict parent contour:

```text
L(x_j^0) > L_i
```

- If no seed exists from a chosen parent, the execution layer changes the parent
  to the sentinel before calling the sampler.
- Each classic child returned by the sampler approximates an independent draw
  from `pi_lambda`.
- Retained phantom states are chain states from after burn-in `B`, associated
  with the classic sample whose generation produced them.
- Phantom cluster output preserves likelihood values and cluster boundaries
  needed for `A_g`, `B_g`, `E_g`, and cluster-bootstrap `rho_g`. Phantom
  coordinates are not retained or added to the posterior.
- V3 removes or bypasses legacy `store_phantom_samples` behavior. Collected
  phantom information is limited to likelihood values, validity/cluster
  structure, associated classic sample, and generation constraint provenance.
- Phantom coordinates must never feed posterior samples, resampling, MAP or
  supremum point estimates, summary point estimates, or result plotting.
- Sampler/execution output preserves parent-contour provenance for each phantom
  cluster, typically via the accepted sample's `log_L_constraint`, so Ticket
  0003 can enforce parent-contour eligibility.
- Phantom states never become classic race samples and never increment
  out-degree.

## Out Of Scope

- Ellipsoidal direction kernels. Covered by Ticket 0008.
- Galilean/gradient-informed trajectories. Covered by Ticket 0012.
- Phantom conditioning math. Covered by Ticket 0003.
- Load-balanced worker runtime. Covered by Ticket 0009.

## Test Plan

Write tests before implementation.

Required unit tests:

- A supplied seed equal to or below the strict constraint is rejected or never
  passed to the sampler, according to the final execution API boundary.
- No-seed parent fallback happens in the execution layer before sampler call.
- Local and worker-backed sampler construction rejects invalid
  `phantom_burn_in` values and proves `num_phantom()` cannot be negative.
- Returned final sample satisfies `L(x^T) > lambda`.
- Every retained phantom state satisfies `L(x^k) > lambda` for the parent
  contour.
- Phantom cluster contains states `k in B..T-1`, not burn-in states and not the
  final classic sample `x^T`.
- Post-burn-in phantom marginal behavior is validated on a simple constrained
  prior where an empirical expectation can be checked; this test must keep
  phantoms clustered and must not treat them as iid race samples.
- `num_phantom()` equals retained post-burn-in phantom capacity and cannot be
  negative.
- Phantom valid masks preserve cluster validity without collapsing to individual
  iid assumptions.
- Parent-contour provenance remains aligned with phantom clusters across
  sampler output, state append/sort, result conversion, and resampling.
- Result conversion and resampling prove that phantom coordinates are unavailable
  and cannot affect posterior, MAP, supremum, or point-estimate outputs.
- Sampler output has stable shapes for `num_phantom = 0` and `num_phantom > 0`.
- Identical key, seed, and model produce deterministic output.

Recommended stochastic tests:

- For a simple uniform constrained prior, many sampler calls produce empirical
  marginal behavior consistent with the constrained region.
- Multiple chains from the same contour are treated as independent classic child
  attempts, while phantoms inside one chain remain clustered.

## Implementation Notes

- Keep the baseline straight-line slice sampler simple and auditable.
- Reuse `PhantomSamples`, `SeedPoint`, `UniDimSliceSampler.num_phantom()`, and
  the current `get_sample(...)` shape where possible. Tighten validation and
  documentation before introducing new sampler abstractions.
- If `_new_proposal` remains JIT-compiled, make Python/function and branch
  arguments static as needed; otherwise remove the JIT. Do not leave direct
  proposal tests broken as part of the v3 migration.
- Make burn-in defaults explicit and report them in diagnostics where useful.
- Do not infer stationarity from values. The paper only says burn-in is
  suggested and non-stationary phantoms cannot be corrected.
- Keep log-likelihood scale names explicit.
- Audit current `store_phantom_samples`, `U_supremum`, and `X_supremum` paths
  during this ticket. Preserve supremum behavior only for classic accepted
  samples; do not let phantom coordinates participate in v3 result point
  estimates.

## Acceptance Criteria

- The sampler interface documents and tests strict constrained-prior semantics.
- Phantom cluster boundaries are preserved end to end.
- V3 phantom collection is likelihood-only; phantom coordinates are discarded and
  cannot affect posterior, MAP, resampling, supremum, or plotting outputs.
- Execution and sampler responsibilities are separated: execution handles
  no-seed sentinel fallback; sampler samples from a valid strict contour.
