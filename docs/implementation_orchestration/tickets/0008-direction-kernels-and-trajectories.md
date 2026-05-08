# Ticket 0008: Baseline Direction Kernels and Straight-Line Trajectories

Branch: `feature/v3-baseline-directions-straight-line`
Priority: after depth-first execution and sampler contract
Depends on: Ticket 0005, Ticket 0007
Design docs:

- `docs/design/jaxns-v3-constrained-sampling.md`
- `docs/design/jaxns-v3-validation-plan.md`

## Goal

Implement and test the v3 baseline constrained-sampler direction and trajectory
choices. Isotropic Gaussian directions and straight-line perfect bracketing can
proceed directly. Ellipsoidal Gaussian directions and greedy-shrinkage details
require a short design-resolution gate inside this ticket before implementation,
because the design leaves the exact GMM construction/update schedule and greedy
shrinkage stepping/termination rules open.

Galilean trajectories remain out of scope here because Ticket 0012 implements
the now-specified gradient-informed Galilean method.

## Current Code Context

Relevant files likely include:

- `src/jaxns/constrained_sampler.py`
- `src/jaxns/constrained_sampler_distributed.py`
- `src/jaxns/core.py`
- `src/jaxns/core_distributed.py` as a migration reference only
- `src/jaxns/multi_ellipsoid_utils.py`
- `src/jaxns/em_gmm.py`
- `tests/test_constrained_sampler.py`
- `tests/test_multi_ellipsoid.py`
- `tests/test_em_gmm.py`

Current code has isotropic direction sampling and straight-line slice utilities.
Default sampler construction is currently split across local and distributed
paths; Ticket 0005 should consolidate execution into a single core before this
ticket wires v3 sampler modes. `gradient_guided` and worker-backed
gradient-guided paths are marked not implemented. The minimal path is to reuse
`_sample_direction`, `_slice_bounds`,
`_pick_point_in_interval`, `_shrink_interval`, `UniDimSliceSampler`, and
`DistributedUniDimSliceSampler` where compatible, adding narrow mode/config
hooks rather than a new sampler hierarchy.

Current `_sample_direction()` is effectively isotropic with unused
`radii`/`rotation` hooks. This ticket must add the missing adaptation context,
explicit modes, frozen per-chain direction configuration, and tests before
calling the ellipsoidal path v3-complete.

## Required Behavior

Direction kernels:

- isotropic Gaussian direction kernel;
- ellipsoidal Gaussian direction kernel after the GMM construction/update
  schedule is documented and reviewed;
- ellipsoidal kernel builds a Gaussian mixture from samples collected so far,
  using an adaptation-context hook supplied by execution;
- component selection is proportional to integrated volume;
- direction is sampled from the selected Gaussian component;
- directions are symmetric and independent of the current chain point;
- the chosen direction-kernel configuration is fixed for the chain so all
  directions in that chain come from the same distribution.

Trajectory methods:

- straight-line perfect bracketing uses unit-hypercube intersections as
  endpoints;
- straight-line interval construction uses no likelihood evaluations;
- uniform proposals plus greedy shrinkage are used when a proposal falls outside
  the slice;
- greedy shrinkage rules are documented and reviewed before implementation,
  including the stepping/termination rule and the exponential contraction
  behavior described by the design.

## Out Of Scope

- Race-tree execution loop. Covered by Ticket 0005.
- Phantom-conditioned shrinkage. Covered by Ticket 0003.
- Load-balanced worker runtime. Covered by Ticket 0009.
- Galilean and gradient-informed trajectory behavior. Covered by Ticket 0012.

## Test Plan

Write tests before implementation.

Required unit tests:

- Isotropic directions have correct shape, finite values, and symmetric sign
  behavior under controlled seeds.
- Ellipsoidal directions choose components according to integrated volume in a
  deterministic toy GMM.
- The sampler API receives the adaptation context needed to build/freeze the
  ellipsoidal kernel from samples collected so far.
- Direction sampling is independent of current point except through the selected
  configured kernel.
- Direction-kernel configuration is snapshotted for a chain; adaptive updates
  between chains do not change the distribution mid-chain.
- Straight-line bracket endpoints lie on unit-hypercube boundaries and bracket
  the current point.
- Straight-line bracket construction performs no likelihood calls.
- Greedy shrinkage follows the documented rule and preserves the current point
  as an in-slice fallback.
- Direct `_new_proposal` tests cover the JIT/static-argument boundary or the
  deliberate removal of JIT from proposal construction.
- Greedy shrinkage returns a point satisfying the strict contour.
- Mode-tunneling or mode-evaporation behavior is surfaced in diagnostics or
  tests where the design mentions risk.
- Sampler modes are selectable through user-facing constructors or documented as
  manual sampler-object configuration only.
- Unsupported Galilean/gradient-informed modes fail clearly until Ticket 0012
  implements them.

Recommended integration tests:

- Compare isotropic and ellipsoidal kernels on a small anisotropic Gaussian
  problem for likelihood-evaluation count trends.
- Straight-line methods produce valid constrained samples on simple convex
  slices.

## Implementation Notes

- Do not call the current `gradient_guided` flag v3 Galilean unless it satisfies
  the design behavior and Ticket 0012 tests.
- Keep a pure-Python or NumPy reference for geometric pieces where JAX
  transformations make debugging difficult.
- Do not implement ellipsoidal GMM directions or greedy-shrinkage variants until
  the open choices listed in this ticket are documented and reviewed.

## Acceptance Criteria

- Baseline direction kernels and straight-line trajectory methods are selectable
  by explicit mode.
- Any design-resolution notes for GMM adaptation and greedy shrinkage are
  reviewed before code is enabled.
- Unsupported modes fail clearly rather than silently falling back.
- Tests prove shape, strict-contour validity, frozen direction-kernel behavior,
  and straight-line bracketing invariants.

## Current Review Follow-Up

Implementation review rejected the first implementation pass. Required fixes:

- Wire the documented greedy shrink/fallback behavior into production
  straight-line proposal construction, or otherwise bound the JAX shrink loop so
  a valid current point is returned when no other strict point is found.
- Make one-dimensional isotropic directions symmetric instead of always `+1`.
- Resolve ellipsoidal adaptation by building/freezing a mixture from sample
  history through the adaptation context, using existing ellipsoid/GMM utilities
  where possible, and documenting the update schedule enabled by this ticket.
- Make worker-backed/distributed unsupported gradient or Galilean modes fail at
  construction with a clear message until Ticket 0012 owns the Galilean path.

Resolution notes:

- Straight-line shrinkage in production is bounded by `max_shrinkage_steps`
  (default 32). If the bounded retry budget is exhausted, the sampler returns
  the current strict seed point for that transition rather than entering an
  unbounded shrink loop.
- Ellipsoidal adaptation is frozen at chain start. Execution supplies
  `adaptation_context` to `get_sample`; when component fields are absent, the
  sampler builds a one-component bounding ellipsoid from `samples_U` and
  `valid_mask` using existing ellipsoid utilities, then reuses that frozen
  kernel for every direction in the chain. Future adaptation-context updates
  apply only to later chains.

Review status: accepted after remediation. Independent review found no blocking
issues after the bounded shrinkage, one-dimensional symmetry, sample-history
ellipsoidal adaptation, distributed unsupported-mode validation, and legacy
metadata-regression fixes. Focused acceptance run:
`conda run -n jaxns_py pytest tests/test_v3_direction_trajectories.py tests/test_v3_sampler_contract.py tests/test_constrained_sampler.py tests/test_distributed_core.py`
passed with 53 tests.
