# Ticket 0001: Race-Tree State and Block Model

Branch: `feature/v3-race-tree-state-and-blocks`
Priority: 1
Depends on: none
Design docs:

- `docs/design/jaxns-v3-statistical-core.md`
- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-design-overview.md`

## Goal

Make the current sample/state layer explicitly support the v3 race-tree and
block model used by all downstream shrinkage, phantom conditioning, allocation,
and load-balanced worker execution work.

The implementation must be agnostic to sample generation order. Statistical
ordering is by likelihood blocks; lineage accounting is derived from the
sentinel/root out-degree and sample out-degrees.

## Current Code Context

Relevant files likely include:

- `src/jaxns/samples.py`
- `src/jaxns/state.py`
- `src/jaxns/core.py`
- `src/jaxns/core_distributed.py` as a migration reference only
- `src/jaxns/evidence_calculation.py`
- `src/jaxns/phantom_eval.py`
- `tests/test_state.py`

The current code already has `root_out_degree`, `Samples.out_degree`,
`Samples.log_L_constraints`, `compute_num_live_points_per_sample`, and derived
parent graph logic. This is close to the paper's compact state: parent indices
are transient append/execution metadata, while persisted race state is
out-degree based. It does not yet expose a canonical v3 block model with plateau
sizes, incoming block lineage counts, and strict endpoint bookkeeping as a
first-class contract. Current consumers are mostly per-sample, and equal
likelihood sorting is not a valid way to define plateau order for v3 shrinkage.

## Required Behavior

- Represent or derive the sentinel sample with `X_{lambda_0} = 1` and
  out-degree `d_0 > 0`.
- Preserve the race-tree recurrence:

```text
K_1 = d_0
K_{i+1} = K_i - 1 + d_i
```

At block level, plateau ties must be handled as an aggregate transition:

```text
K_{g+1} = K_g - m_g + sum_{i in \mathcal{B}_g} d_i
```

Children of samples in a plateau must not affect the incoming `K_g` used for
that same plateau block.

- Derive blocks from sorted classic likelihoods:

```text
\mathcal{B}_g = {i: L_i = lambda_g}
m_g = |\mathcal{B}_g|
```

- Derive incoming block lineage counts `K_g` before the block contributes to
  Bayesian shrinkage.
- Provide a canonical derived block state as the first shared dependency for v3
  shrinkage paths. It must include at least `log_L_g`, `m_g`, sample
  indices/ranges, incoming `K_g`, and segment-summed out-degree
  `sum_{i in \mathcal{B}_g} d_i`.
- Ensure v3 shrinkage consumers use this block state rather than inferring
  plateau behavior from per-sample order or unstable equal-likelihood sorting.
- Treat all shrinkage constraints as strict, `L(x) > lambda`.
- Do not add a persisted `parent_idx` field for every classic sample. The paper
  explicitly says the algorithm only needs out-degrees; if parent blocks are
  required for diagnostics, derive them from sorted likelihoods and out-degrees.
- Preserve the selected parent contour as transient execution metadata and, for
  accepted samples, keep enough contour information to audit strict generation
  constraints without making parent links part of the statistical state.
- Validate strict contour consistency for generated samples:
  `L_child > L_parent_constraint`.
- Validate that each block has enough incoming active lineages for its plateau
  winners, `K_g >= m_g`.
- Preserve phantom cluster identity per classic sample without allowing phantom
  samples to alter out-degree or active lineage counts.

## Out Of Scope

- Bayesian shrinkage formulas and evidence sampling. Covered by Ticket 0002.
- Phantom effective-count conditioning. Covered by Ticket 0003.
- Dynamic allocation policy. Covered by Ticket 0005 and Ticket 0006.
- Load-balanced worker runtime changes. Covered by Ticket 0009.

## Test Plan

Write tests before implementation.

Required unit tests:

- `K_i` recurrence from `root_out_degree` and `out_degree` for hand-built race
  trees.
- Derived parent blocks or graph-like diagnostics for linear, branching, and
  sentinel-rooted out-degree schedules. These diagnostics must be derived, not
  stored as the source of truth.
- Out-degree state survives sorting, block construction, append/resize, and
  result conversion without adding a persisted parent field.
- Stable block construction for distinct likelihoods and likelihood plateaus.
- Permutation-invariance tests: permute the same generated race tree and assert
  identical canonical likelihood-block order, `K_g`, `m_g`, derived parent-block
  diagnostics, and plateau membership sets.
- `K_g` and `m_g` computation for plateau blocks, including `K_g >= m_g`.
- Aggregate plateau transition
  `K_{g+1} = K_g - m_g + sum_{i in \mathcal{B}_g} d_i`.
- Canonical block helper emits `log_L_g`, sample indices/ranges, `m_g`, `K_g`,
  and segment-summed out-degree for downstream Tickets 0002, 0003, 0005, and
  0006.
- Every accepted child satisfies `L_child > L_parent_constraint`; a child below
  or equal to its parent contour fails explicitly.
- Strict constraint semantics: equality with a parent contour is not considered
  inside the child constrained prior.
- Phantom clusters do not affect `K_i`, `K_g`, `m_g`, derived parent diagnostics, or
  out-degree.
- Invalid graph cases fail explicitly: broken lineage, negative/insufficient
  active lineage count, strict-contour parent violation, and inconsistent
  out-degree sum.
- Determinism: identical ordered inputs produce identical block and lineage
  outputs.
- `Samples.resize()` preserves `log_L_constraints` and all per-sample
  provenance fields; add a regression test for the current fill-atom gap.

Use small fixtures with clear schemas, preferably `NamedTuple` or frozen
dataclasses. Do not test by mirroring private helper internals.

## Implementation Notes

- Prefer a small race-tree/block helper module if it keeps the statistical
  contract clearer than extending unrelated result or sampler types. Keep it
  focused on derived block state; avoid a new tree/state rewrite.
- Keep existing public APIs working where possible, but do not let legacy
  "live point" terminology obscure the v3 `K_i` lineage-count contract.
- Existing `Samples.compute_num_live_points_per_sample` may be kept if its
  behavior is renamed, wrapped, or documented as active lineage count.
- Existing `Samples.log_L_constraints` can serve as the accepted sample's parent
  contour audit trail where that is sufficient; do not replace it with a stored
  parent id unless a later reviewed design proves out-degrees are insufficient.
- Sorting equal likelihoods must not pretend a unique within-plateau order is
  known.
- Treat the block helper as the migration boundary for v3 math. Existing
  per-sample `K_i` helpers may remain for legacy code, but v3 Tickets 0002,
  0003, 0005, and 0006 should consume canonical block state.

## Acceptance Criteria

- Tests prove the v3 race-tree and block invariants without relying on current
  implementation accidents.
- Downstream tickets can consume a block representation containing likelihood
  level, block membership or size, incoming `K_g`, sample indices, and
  segment-summed out-degree.
- No phantom record can change lineage accounting.
- Reviewers can trace each invariant to the design docs above.

## Review Status

Accepted with a narrow follow-up. The previous blocker list is resolved:
default block construction keeps valid `logL=-inf` blocks distinct from
padding, canonical block state exposes sample membership, public evidence/result
paths validate strict contour and graph consistency before jitted use, and
`Samples.resize()` preserves `log_L_constraints` without the ordinary traced
`max_samples` failure.

Remaining non-blocking follow-up:

- decide whether `build_block_state(..., validate=False)` should remain the
  default low-level behavior or whether public callers need a validated wrapper;
- add a plateau-stable block diagnostic for reconstructed parent information if
  later diagnostics require it, rather than relying on per-sample parent-edge
  ordering inside plateaus.
