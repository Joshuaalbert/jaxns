# Ticket 0005: Core Execution Loop and Depth-First Allocation

Branch: `feature/v3-core-execution-depth-first`
Priority: 6
Depends on: Ticket 0001, Ticket 0002, Ticket 0007
Design docs:

- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-statistical-core.md`
- `docs/design/jaxns-v3-constrained-sampling.md`

## Goal

Migrate the sampler execution loop to a single v3 core with the initialization
and outer/inner structure, then implement the depth-first uniform allocation
target. This ticket should establish the allocation framework while leaving the
paper-specified evidence-improving and posterior-improving utility modes to
Ticket 0006.

## Current Code Context

Relevant files likely include:

- `src/jaxns/core.py`
- `src/jaxns/state.py`
- `src/jaxns/samples.py`
- `src/jaxns/termination_condition.py`
- `tests/test_ns_standard_problems.py`
- `tests/test_gh108.py`

The current `_run_ns` loop chooses under-target contours against a fixed
`target_num_live_points`. `core_distributed.py` contains related batching and
transient-parent ideas, but v3 should not leave users with two maintained core
algorithms. The v3 design has an outer allocation-target loop and an inner depth
loop, with a user-specified goal condition. The existing local and distributed
core paths are starting points for extracting useful pieces: sorted `Samples`,
transient `parent_idxs`, no-seed sentinel fallback, `Samples.append_samples(...)`,
phantom likelihood-cluster propagation, and
`TerminationCondition`/`TerminationRegister` inputs. The migration should
consolidate those pieces into one core rather than replacing the sampler from
scratch or preserving parallel core implementations.

## Required Behavior

Implement the v3 control skeleton:

- initialize by drawing `d_0` classic samples from the sentinel contour and
  setting `N = d_0`;
- outer loop chooses `K_*^k`;
- inner loop generates children until depth condition is met;
- outer loop stops when goal condition is met;
- completed child updates only its known parent's out-degree, or sentinel
  out-degree on the no-seed fallback;
- local batched generation does not mutate `K`, shrinkage inputs, out-degree, or
  phantom clusters before the batch result is accepted;
- sample generation may occur in batches or asynchronously through deployment
  backends, but statistical accounting is independent of generation order and
  uses the same core acceptance path.
- parent links are not persisted in the accepted sample state. Block-level
  targeting must resolve to a concrete parent sample index, or the sentinel
  fallback, before dispatch. That exact in-flight index is preserved until
  acceptance and used to update the correct out-degree.

Add the public execution methods:

```python
run_until_goal(
    goal_cond: Callable[[State], bool],
    depth_cond: TerminationCondition,
    allocation_target: str = "uniform",
    **options,
) -> State

resume_until_goal(
    state: State,
    goal_cond: Callable[[State], bool],
    depth_cond: TerminationCondition,
    allocation_target: str = "uniform",
    **options,
) -> State
```

`goal_cond` receives the current `State` and is evaluated only at outer-loop
boundaries. It may call `state.to_result()`; the implementation must keep the
summary choice deterministic for fixed state. `depth_cond` controls the inner
loop and must not be confused with the outer goal condition.

Implement depth-first uniform allocation:

```text
K_*^k(L) = d_0 + k DeltaK
```

The public allocation-target literal for this mode is `"uniform"`. Unknown
target strings must fail explicitly.

Parent choice must target under-allocated active lineage counts. When targeting
block `g'`, support the paper's parent-block weighting:

```text
P(parent block = g | target block = g') proportional to
    (X_{g'} / X_g) 1{X_{g'} < X_g}
```

For this ticket, if `X_g` is sampled or uncertain, define and test the summary
used for parent selection, depth conditions, and goal conditions.

Depth conditions available in JAXNS v3:

```text
small remaining evidence:
L_G X_G / (sum_g^G L_g (X_{g-1} - X_g) + L_G X_G) < tau_Z

small remaining posterior mass:
L_g X_g / max_{g'}(L_{g'} X_{g'}) < tau_post
```

Goal conditions must be user-specified and support at least target ESS and
target evidence uncertainty.

Integer allocation targets must be explicit. Define and test how real-valued
interpolated targets are rounded or clipped to `N`, and reject invalid
`DeltaK <= 0`.

## Out Of Scope

- Evidence-improving and posterior-improving utilities. Covered by Ticket 0006.
- Load-balanced worker runtime. Covered by Ticket 0009.
- New trajectory methods. Covered by Ticket 0008.

## Test Plan

Write tests before implementation.

Required unit tests:

- Outer loop increments `k` and changes `K_*^k` for depth-first uniform.
- `run_until_goal(...)` and `resume_until_goal(state=...)` accept the public
  keyword arguments used in `docs/design/interface/run_pattern.py`.
- `goal_cond` receives `State`, may call `state.to_result()`, and is evaluated
  only at outer-loop boundaries.
- Unknown `allocation_target` strings fail explicitly, and `"uniform"` selects
  depth-first uniform.
- Initial root generation draws exactly `d_0` classic children from the sentinel,
  sets `N = d_0`, records sentinel constraints, and associates initial phantom
  clusters with those samples.
- Inner loop stops on depth conditions and does not evaluate goal prematurely.
- Goal condition stops the outer loop.
- Stochastic shrinkage summaries for parent selection, depth conditions, and
  goal conditions are deterministic and documented for fixed inputs.
- Parent selection only chooses valid strict parent contours or sentinel fallback
  when no seed exists.
- Out-degree updates are applied to the selected parent only; no other lineage
  accounting changes.
- Local/batched in-flight parent targets have exact state-snapshot tests: no
  lineage or phantom mutation before batch acceptance, stale allocation targets
  cannot change already selected parent indices, and unsupported depth/goal
  transitions fail explicitly.
- No-seed plateau fallback increments sentinel/root out-degree and records the
  effective root constraint.
- Identical ordered inputs and PRNG keys produce deterministic parent choices.
- Batch generation preserves the selected parent-index array through acceptance
  and applies out-degree updates to those exact indices.

Recommended integration tests:

- A small toy model reaches a goal condition with depth-first uniform allocation.
- Existing standard-problem smoke tests still run with the v3 execution loop.

## Implementation Notes

- Keep the allocation target abstraction small. It only needs depth-first
  uniform in this ticket, plus a narrow extension point for Ticket 0006.
- Preserve the existing `NestedSampler` user path where possible, but make it
  drive the single v3 core once this ticket lands. Add v3 allocation controls as
  explicit options, and treat fixed `target_num_live_points` behavior as a
  compatibility wrapper only if tests prove the mapping is correct.
- Use `core_distributed.py` only as a source of migration details or regression
  cases. Do not create a v3 local core and a v3 distributed core.
- Do not call old `target_num_live_points` behavior v3 allocation unless tests
  prove it is equivalent to `K_*^k(L) = d_0 + k DeltaK`.
- Make depth/goal condition inputs explicit so validation can reproduce them.

## Acceptance Criteria

- The v3 outer/inner execution shape is tested independently from old
  termination behavior.
- Local `NestedSampler.run_until_goal(...)` and
  `NestedSampler.resume_until_goal(state=...)` are public, tested, and driven by
  the single v3 core.
- Depth-first uniform allocation is functional and design-aligned.
- Evidence-improving and posterior-improving modes are not enabled in this
  ticket.

## Review Status

Accepted in the implementation loop. The accepted implementation exposes the
v3 `run_until_goal(...)` / `resume_until_goal(...)` run pattern, initializes
root samples before the first goal check, keeps the inner loop running until
the depth condition is met, uses fixed initial `d_0` and outer iteration `k`
for `K_*^k`, preserves in-flight parent indices through acceptance, implements
weighted strict parent-block selection with sentinel fallback, and supports v3
`dlogZ` depth handling while failing explicitly for unsupported legacy depth
fields. Legacy `run()` / `resume()` remain documented fixed-live-point
compatibility paths.
