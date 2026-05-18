# Ticket 0006: Evidence and Posterior Allocation Utilities

Branch: `feature/v3-allocation-utilities`
Priority: after depth-first execution
Depends on: Ticket 0005
Design docs:

- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-design-overview.md`
- `docs/design/jaxns-v3-validation-plan.md`

## Goal

Implement the paper-specified evidence-improving and posterior-improving
allocation utilities on top of the allocation framework from Ticket 0005.

This ticket must be a minimal extension of the v3 execution path. Do not rewrite
the sampler loop or result pipeline to add these modes; add utility helpers,
tests, and explicit mode wiring around the existing depth-first allocation
abstraction.

## Current Code Context

Relevant files likely include:

- `src/jaxns/core.py`
- `src/jaxns/core_distributed.py` as a migration reference only
- `src/jaxns/state.py`
- `src/jaxns/evidence_calculation.py`
- `src/jaxns/phantom_eval.py`
- allocation helper modules introduced by Ticket 0005
- tests added by Ticket 0005

Ticket 0005 should already expose block summaries, volume summaries, and an
allocation target interface. Use those interfaces. If the data needed by the
utility formulas is missing, extend the block/allocation helper boundary rather
than reaching into unrelated result or sampler internals.

## Required Behavior

Evidence-improving allocation uses:

```text
T^Z_{g h} = P_{x ~ pi_{L_g}}(L(x) > L_h)
          = (X_h / X_g) 1{h > g}
```

Define:

```text
B_h = d log Z / d log p_{>h}
    = (L_h X_h - sum_{j > h} L_j (X_{j-1} - X_j)) / Z
```

For `p_{>h} ~ Beta(alpha_h, beta_h)`:

```text
Delta Var[log Z; h] ~= B_h^2 (
    1 / alpha_h^2 - 1 / (alpha_h + beta_h)^2
)
```

`alpha_h` and `beta_h` must come from the v3 block shrinkage summaries produced
by Tickets 0002 and 0003 and surfaced to allocation by Ticket 0005. Do not infer
these parameters from legacy per-sample `EvidenceCalculation` state.

The utility is:

```text
U^Z_g = (1 / X_g) sum_{h > g} X_h B_h^2 (
    1 / alpha_h^2 - 1 / (alpha_h + beta_h)^2
)
```

Posterior-improving allocation uses:

```text
T^P_{g h} = P_{x ~ pi_{L_g}}(L_{h-1} < L(x) <= L_h)
          ~= ((X_{h-1} - X_h) / X_g) 1{h > g}
```

Use Kish ESS:

```text
ESS_Kish = W^2 / Q
W = sum_j w_j
Q = sum_j w_j^2
```

For shell `h`, implement the exact marginalized change:

```text
Delta ESS_Kish_h =
    W^2 [
        atan(sqrt(w_h^2 / (2Q - w_h^2)))
        / sqrt((Q - w_h^2 / 2)(w_h^2 / 2))
        - 1 / Q
    ]
```

with continuous limiting value `0` as `w_h -> 0`. Also implement the paper's
conservative approximation as an explicit option:

```text
Delta ESS_Kish_h ~= W^2 / (Q - w_h^2 / 3) - W^2 / Q
```

The utility is:

```text
U^P_g = (1 / X_g) sum_{h > g}
    (X_{h-1} - X_h) Delta ESS_Kish_h
```

Convert block utilities to allocation targets by normalizing to unit peak,
linearly interpolating to `bar U(L)`, and using:

```text
K_*^k(L) = d_0 + k DeltaK bar U(L)
```

The resulting real-valued target must pass through the same integer
rounding/clipping policy defined and tested in Ticket 0005.

Public allocation-target literals:

- `"evidence_improving"` selects `U^Z_g`.
- `"posterior_improving"` selects `U^P_g`.

The `"uniform"` literal is implemented by Ticket 0005. All three literals must
share the same validation and unknown-string failure path.

Define and test deterministic fallback behavior when utility values are all
zero, non-finite, or tied. Invalid `DeltaK <= 0` must fail explicitly.

## Out Of Scope

- Depth-first uniform allocation. Covered by Ticket 0005.
- New constrained-sampler modes. Covered by Tickets 0008 and 0012.
- Benchmark claims. Covered by Ticket 0010 after feature implementation.

## Test Plan

Write tests before implementation.

Required unit tests:

- `U^Z_g` matches hand-computed toy schedules, including downstream cumulative
  sensitivity terms.
- `U^P_g` exact and conservative forms match hand-computed toy schedules.
- The exact posterior utility limit is finite and returns zero as `w_h -> 0`.
- Utility normalization produces unit peak and deterministic interpolation.
- Utility targets use the same integer conversion, rounding, and clipping rule
  as depth-first targets from Ticket 0005.
- `"evidence_improving"` and `"posterior_improving"` wire to the correct utility
  families and are accepted by `run_until_goal(...)` and
  `resume_until_goal(...)`.
- All-zero, tied, negative after numerical error, and non-finite utility inputs
  use the documented fallback or fail explicitly.
- Evidence-improving allocation prefers contours with higher expected
  log-evidence variance reduction in toy schedules.
- Posterior-improving allocation prefers contours that improve Kish ESS in toy
  schedules.
- Plateau blocks, zero shell mass, tiny `X_g`, and `Z <= 0` edge cases are
  explicit and tested.
- Parent-selection integration uses the selected utility target without changing
  race-tree shrinkage accounting or adding stored parent fields.

Recommended implementation checks:

- Compare cumulative-reduction implementation against a simple reference loop.
- Verify identical ordered inputs and PRNG keys produce deterministic targets.
- Verify utility helpers accept the same block summaries used by depth/goal
  conditions rather than recomputing inconsistent volume paths.
- Verify `alpha_h` and `beta_h` are read from v3 block posterior summaries and
  are not reconstructed from legacy `EvidenceCalculation`.

## Implementation Notes

- Keep utility computation in a small helper module or in the allocation helper
  introduced by Ticket 0005. Avoid broad changes to `NestedSamplerResults`.
- Use stable log-domain or guarded linear-domain operations for tiny volumes and
  nearly zero shell masses.
- The paper says `U^Z_g` can be computed with two cumulative reductions; keep a
  readable reference path for tests even if the production path is vectorized.
- Make expected, sampled, and conservative summaries explicit in names. Do not
  silently mix sampled shrinkage draws with expected-path utilities.

## Acceptance Criteria

- Both utility modes have formula-level unit tests tied to the design docs.
- The v3 execution path can select depth-first, evidence-improving, or
  posterior-improving allocation modes explicitly.
- Utility target construction is deterministic, numerically guarded, and
  reviewable without private sampler state.

## Review Status

Accepted in the implementation loop. The accepted implementation builds
allocation plans from v3 block summaries, uses stable likelihood scaling for
evidence and posterior utilities, shares integer target construction across
uniform/evidence/posterior modes, supports the conservative posterior option
through allocation and run APIs, schedules multiple work items for remaining
deficits, and still schedules zero-utility blocks when they are below the
base `d_0` target.
