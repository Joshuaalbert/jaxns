# Ticket 0021: Efficient seed selection from all stationary phantoms

Status: exact implementation validated; prepared for review on develop.
Evidence: [implementation, parity, and performance report](../../../benchmarks/phantom_seeding/README.md).
The original scope and gates below remain the design record.
Branch: `feature/efficient-phantom-seeding-pr`
Priority: establish selector parity and measured scaling before further long
SS10 precision runs.
Tracking context: [Allow stationary phantom samples to seed constrained chains
(#292)](https://github.com/Joshuaalbert/jaxns/issues/292).
Design: [Efficient all-phantom seed selection](../../design/efficient-phantom-seeding.md).

## Problem and intended behavior

The experimental A intervention makes all eligible stationary phantom points
available as seeds. Its scientific benefit over classic seeding is modest and
problem dependent, but its implementation repeatedly queries the full allocated
history and resolves candidates that are never used. Large query buffers,
coordinate storage, growth, and checkpoint copies also limit concurrency.

Implement an exact block eligibility index and lazy candidate lookup, initially
retaining every phantom U coordinate. Given identical state and random keys,
the optimized selector must return exactly the same seed identities as A.
This is a performance change to the demonstrated intervention, not a new
reservoir or allocation experiment.

This ticket revises #292's proposed bounded reservoir, one representative per
chain, delayed publication, and cluster reservation approach for the first
implementation candidate. Preserve the experimental A population, immediate
phantom visibility, and individual-identity reservations instead. The goal of
using stationary phantoms remains; #292 is historical design context and is
not closed by this ticket. Eligibility alone does not prove independence or
new guarantees about adaptive reseeding.

## Reference and implementation boundary

The behavioral reference is `paper/evidence10-AC300` at
`aa79a0dc395c0d64b85cbb88d0e3e732decd7361`, in the adjacent
`jaxns-paper-evidence10-AC300` worktree. That experimental source contains A;
the main checkout has a different core layout. Preserve the frozen reference
and map its behavior into the chosen implementation branch explicitly.
Ticket 0020 supplies core architecture context, not a requirement to repeat
the historical migration before profiling this reference.

Relevant paths in the experimental source:

- `src/jaxns/sampling/phantom_seeds.py`: cumulative counts and rank lookup.
- `src/jaxns/algorithm/depth.py`: `_sample_stationary_seeds`, effective-contour
  fallback, reservations, and classic seed indexing.
- `src/jaxns/samples.py`: full coordinates, likelihoods, masks, and seed cache.
- `src/jaxns/sampling/slice.py` and `sampling/continuation.py`: later replay
  feasibility only; no sampler change in the first implementation.

All experiments and automatic report helpers were cancelled on 2026-09-10.
Completed 0.05/0.02 cohorts and partial 0.01 checkpoints remain preserved.
This design ticket does not authorize restarting the ladder or implementing
the proposal before the user requests implementation.

## Required invariants

For chain i and phantom j, eligibility at contour lambda remains
`valid_ij AND birth_i <= lambda AND lambda < log_L_ij`.
Likelihood determines membership only; its height must not weight seeds.

- Retain all valid phantoms, including those from accepted classics not yet
  published as classic seed sources. Do not retire a phantom after a deeper
  contour: later allocation may request a shallower one.
- Preserve population order: frozen classics, the existing recent-classic
  reservoir, then phantoms in append-row and original transition order.
  Preserve the logical identity `row * (P + 1) + slot`, with slot zero classic.
- Preserve the random draw shapes and keys, lane permutation, rotated strata,
  proposal order, retry counters, exhaustion policy, and same-contour
  reservations, including unfinished groups and in-flight work.
- Preserve classic publication, effective-contour fallback, scheduling,
  allocation, stopping, and likelihood accounting.
- Preserve phantom transition order and cluster membership for evidence
  conditioning. Phantoms do not become posterior observations or race nodes.

Matching only a marginally uniform distribution is insufficient: A uses
systematic stratification and collision retries, so ordered rank lookup and
the random schedule are part of the reference behavior.

## First implementation candidate

### Exact block counts

Partition append-order rows into fixed blocks of B rows; benchmark B=64, 256,
and 1024. These are storage blocks, not geometric clusters.

For each sealed block, cache sorted birth contours with cumulative valid
phantom counts and sorted valid phantom likelihoods. For valid intervals with
`birth_i < log_L_ij`, the eligible count is exactly

```
sum_i valid_count_i * 1[birth_i <= lambda]
    - sum_(i,j)_valid 1[log_L_ij <= lambda].
```

Two binary searches supply each block count. Locate the block containing the
existing proposal rank from cumulative block counts, then inspect only that
block in original row/transition order to recover the same identity. Gather
coordinates only for an accepted candidate.

Maintain an exact unsealed tail of fewer than B rows. Include it immediately
in queries; sealing a block must not change visibility. Handle invalid root
slots, nonmonotone births, ties, infinities, padding, growth, and resume.
Replace the row-sorted likelihood cache with the block-sorted cache rather
than retaining a third likelihood array. Use the same count query to remove
the separate capacity-wide phantom-existence scan in `_effective_parent_contour`.

For shell width S and row capacity C, the main cumulative table changes from
`[S,C]` to `[S,ceil(C/B)]`, plus bounded tail and local lookup scratch. At
C=1,715,200, S=100, B=256, an int64 table falls from 1.278 GiB to 5.11 MiB.
This is an array-size calculation, not a measured speedup or RSS forecast.
Avoid an `[S, proposals, B, P]` temporary that recreates the memory problem.

Query work remains `O(S (C/B) log(BP) + S BP)`, where P is phantoms per row;
construction across N accepted rows costs `O(NP log(BP))`. The query still
depends linearly on C/B, with a quadratic accumulated term over an expanding
run. Record the range where this is practical rather than claiming unlimited
history scaling.

### Lazy candidate resolution

Keep the existing 64-value random draw per lane/attempt and the first-proposal
strata override. Resolve proposal zero in its actual population and check its
reservation. Resolve subsequent proposals in their original order only if
needed; after all 64 collide, use the same next attempt and key as A.

A scalar random draw is not an equivalent replacement for a 64-value JAX
draw. Likewise, a vmapped conditional can evaluate both branches. Verify
that the lowered scalar control flow skips unused rank lookups. Benchmark
fallback tiles only if collisions justify them, keeping proposal order exact.

### Full coordinates and measured buffer ownership

Retain full U for the first candidate. Measure persistent arrays, compiler
temporaries, growth copies, classic-only views, and checkpoint serialization
separately. Check actual dtypes and peak simultaneously live buffers. Any
buffer donation must have an explicit ownership contract; ordinary state
reuse must not silently become invalid.

## Validation and acceptance

1. Keep an independent NumPy enumeration reference for membership, counts,
   and rank lookup. Cover empty and partial blocks, ties and plateaus,
   invalid/root slots, nonmonotone births, extreme contours, and growth.
2. Compare every proposal/selected identity against frozen A for identical
   keys, including classic-only, phantom-only, mixed populations, collisions,
   exhausted pools, in-flight reservations, padding, and resumed groups.
   Selected coordinates and likelihoods must be bitwise equal.
3. Check short complete runs and checkpoint/resume parity: race state,
   likelihood counts, stopping, posterior diagnostics, and paired evidence
   reductions must agree. Investigate any deviation before performance claims.
4. Benchmark four variants: reference, lazy lookup only, blocks only, and
   both. Pin one core, synchronize JAX timings, separate compilation from warm
   execution, and vary N, C/N, P, D, S, block size, and collision rate.
5. Report selector timings, end-to-end time, persistent bytes, HLO memory
   analysis, measured peak RSS, index build/growth, and checkpoint peaks.
   Establish where the combined change improves runtime and memory; retain
   separate variants so any regression has an attributable cause.
6. Run the affected package checks required by AGENTS.md. Keep implementation
   and benchmark code linearly readable, with shapes at array transformations.
   Scientific acceptance tolerances must not be weakened to admit an
   optimization. Large new cohorts require a separate decision after parity
   and bounded performance evidence.

## Later candidates, gated by evidence

If coordinates dominate after these changes, evaluate compact isotropic
replay using the chain key, accepted scalar steps, and a fixed number of
literal anchors. Require zero additional likelihood calls and numerical
parity before replacing U storage; preserve a full-U fallback for old states
and unsupported kernels. Floating-point reconstruction is not assumed exact.

If block scans still dominate at large N, evaluate a hierarchical packed
birth/death rank index. The existing classic rank index supplies ideas and
tests, but copying its dense representation over NP slots or changing the
phantom identity order would defeat this design. These follow-ups require
their own measurements and design review; neither is a first-phase dependency.

## Deliverables

- Exact block and lazy selectors with an independent reference and parity
  evidence, integrated into the chosen implementation branch.
- Reproducible four-variant performance report with measured scaling and
  peak-memory limits, plus a decision on replay or hierarchical follow-up.
- Documentation of the preserved all-phantom semantics and the revised
  relationship to #292; no reservoir, geometric allocation, or MCMC changes
  bundled into this optimization.
