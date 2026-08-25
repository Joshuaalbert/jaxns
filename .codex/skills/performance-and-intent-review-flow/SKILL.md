---
name: performance-and-intent-review-flow
description: Review JAX library code for tracing, compilation, device-runtime, memory, batching, and numerical-performance regressions together with loss of code-intent readability. Use when the user asks for a JAX performance or hotpath review, HLO or memory-plan comparison, Pytree design review, maintainability or comment review, or whether a change introduced retracing, materialisation, batching overhead, large closures, code sprawl, or obscured physical and algorithmic intent.
---

# Performance And Intent Review Flow

## Objective

Review changes for three coupled failure modes:

1. Unnecessary work during tracing, compilation, or repeated device execution.
2. Loss of maintainability because contracts, shapes, physical intent, or sequencing became harder to understand.
3. Code sprawl or reinvention of an existing model, operator, utility, or test pattern.

Use a code-review stance unless the user explicitly asks for implementation. Preserve numerical correctness and precision
while assessing performance. Lead with findings ordered by severity and grounded in file/line references.

## Review Setup

1. Read the user request and any local repo instructions before judging the code.
2. Inspect the changed files, nearby code, downstream consumers, shape conventions, tests, and existing utilities. Do not
   infer the execution grid or reduced direction set without tracing the real call path.
3. Identify representative production shapes and repeated axes: chunks, integrations, channels, baselines, antennas,
   directions, facets, stencil points, solver iterations, or another domain-specific loop.
4. Classify work by execution layer:
   - Python construction and orchestration;
   - JAX tracing and specialisation;
   - XLA compilation;
   - asynchronous dispatch and host/device transfer;
   - repeated device execution and synchronization.
5. Establish the correctness and precision reference before comparing plans. Avoid performance recommendations that
   silently change the model, approximation, dtype, or accepted error.
6. Search for an existing implementation before proposing a helper, model, operator, interpolator, batching utility,
   or diagnostic.

## Contract And Tracing Pass

- Keep static contract checks close to the jitted function that owns the contract. Shape, rank, static option, and
  auxiliary-metadata checks that execute during tracing provide useful fail-fast behaviour and are not repeated device
  work.
- Distinguish static checks from checks that depend on traced array values. Do not use Python truth-value conversion on
  tracers. Decide deliberately whether a dynamic invariant belongs in the mathematical construction, a checkify-style
  diagnostic path, or an external boundary.
- Do not move validation out of a traced program merely because it appears inside the Python function body. First
  determine whether it runs once while tracing or lowers into every device execution.
- Avoid redundant conversion of traced arguments, such as calling `jnp.asarray` on inputs already governed by the JAX
  contract.
- Watch for changing shapes, Python values, static arguments, object identities, or closures that create avoidable
  recompilation and cache fragmentation.

## Pytree And Modular JAX Pass

Prefer a registered Pytree, or `PureDataclassPytree` where available, when a model needs to carry traceable data,
static auxiliary configuration, and operations into compiled programs.

- Use slotted dataclasses and place traceable parameters in Pytree children. Put genuinely static configuration in the
  flatten auxiliary data.
- Avoid mutable list/dict fields, dataclass inheritance, custom `__init__` methods, and `@property` interfaces on these
  model objects.
- Make a public JAX-executing method a thin forwarder to a module-level implementation with a leading underscore. Type
  the `self` argument as the class defined above it and use the repository pattern:

```python
def evaluate(self, inputs: FloatArray, mode: str) -> FloatArray:
    return _evaluate(self, inputs, mode)


@partial(jax.jit, inline=True, static_argnames=["mode"])
def _evaluate(self: SomeModel, inputs: FloatArray, mode: str) -> FloatArray:
    ...
```

- Use `inline=True` so composition into a larger jitted program retains optimisation opportunities. Mark arguments
  static only when they are small, hashable compile-time choices and the resulting specialisations are intentional.
- Do not accidentally close over large arrays or changing constants. Carry model data as Pytree children or explicit
  arguments, and inspect whether static objects inflate HLO, compile time, executable size, or cache keys.

## Device Performance Pass

- Look for large intermediates, explicit replication, unnecessary transposes/copies, host callbacks, synchronization,
  dtype promotion, repeated specialisation, and accidental Cartesian products of independent batch axes.
- Prefer broadcasting, fusion-friendly elementwise work, contractions, or a reformulated algorithm before reaching for
  explicit batching.
- Treat `lax.map(..., batch_size=...)`, manual chunking, and similar batching as memory/performance trade-offs, not
  default optimisations. They often reduce peak memory while increasing launches, compilation, or repeated work.
  Compare credible alternatives and choose with evidence at representative shapes.
- Reuse stable geometry, interpolation support, plans, and specialised model state when their physical inputs are
  unchanged. Do not retain a cache whose key omits a dependency that changes its result.
- Use Python control flow for genuinely static branches. Use `lax.cond` for scalar dynamic predicates whose branches do
  substantially different work, and `lax.switch` for dynamic multi-way dispatch. Remember that vmapping these control
  operators may turn them into select-style execution and remove the branch-skipping benefit.
- Prefer a clear `jnp.where` for simple elementwise selection; do not wrap straightforward arithmetic in `lax.cond`.
- Consider whether a custom primitive, fused kernel, changed data layout, or moved specialisation boundary is warranted
  when evidence shows that batching or composition cannot meet the target. Do not force every finding into a tiny local
  edit when the interface is the cause.

## Measurement Pass

For material performance claims:

1. Record device, JAX/XLA configuration, dtype/precision mode, representative shapes, and relevant chunk sizes.
2. Separate lowering/compilation from steady-state execution.
3. Warm up the exact specialised call and use `block_until_ready()` before and during timing so asynchronous dispatch
   is not mistaken for execution time.
4. Use repeated measurements and report a robust statistic plus variability, not one timing.
5. Compare numerical outputs against the reference at an explicit tolerance.
6. Measure or estimate peak live memory, including persistent operands, outputs, compiler temporaries, and library
   workspaces. State clearly whether a value is measured, compiler-reported, or analytic.
7. Inspect lowered HLO/compiler diagnostics when comparing plans. Look for fusion, duplicated computations, broadcasts
   that materialise, transposes/copies, collectives, maps/loops, and large constants. Use HLO to explain measurements,
   not as a substitute for them.

Do not infer production performance from a tiny mock without a justified scaling argument. Distinguish a measured
regression from a plausible risk that still needs profiling.

## Intent Pass

Check whether the change made the code harder to understand:

- Preserve the knowledge in important comments, but revalidate and rewrite comments when behaviour changes instead of
  restoring stale wording.
- Comments should explain intent and invariants, not narrate obvious assignments.
- Record array shapes at important transformations and state units, axis meaning, precision choices, static/dynamic
  Pytree boundaries, approximation assumptions, and ownership where they are otherwise easy to misread.
- Explain why a batching, specialisation, interpolation, caching, or control-flow choice exists and what downstream
  assumptions would be invalidated by changing it.
- Cite a design document, paper, benchmark artifact, issue, or test where that reference materially supports the choice
  and will help a future developer reassess it.
- Prefer linear, local code when the repeated logic is small. Do not ask for abstraction just to reduce line count.
- Flag abstractions that add mental plumbing, spread one concept across files, obscure ownership boundaries, or hide a hotpath behind generic helpers.
- Flag duplicated implementations and compatibility layers that have no concrete remaining consumer. Reuse existing
  utilities when their contract genuinely matches; do not contort code merely to claim reuse.
- Check names and local aliases for physical and mathematical meaning. Do not recommend aliases solely for speculative
  Python attribute-lookup savings.
- Verify the reader can understand behavior from code plus comments without reconstructing history from the PR discussion.

## Test Pass

- Test static contracts through the jitted entry point and confirm invalid shapes/options fail while tracing or
  specialising at the intended boundary.
- Test numerical equivalence, gradients where relevant, dtype/shape contracts, Pytree flatten/unflatten and
  serialisation, and supported static specialisations.
- Add a focused regression test for a removed or relocated check at its new owner.
- Use a benchmark or profiler for performance claims; functional tests and call-counting fakes are not runtime evidence
  for compiled JAX code.
- If comments document sequencing or recovery semantics, tests should cover the behavior, not the comment text.
- Prefer focused regression tests over broad integration tests unless the risk crosses module boundaries.

## Reporting

When reviewing, report in this order:

1. Findings first, ordered by severity.
2. Include file/line, affected execution layer, production shape/path, what changed, why it matters, evidence, and the
   smallest fix that addresses the root cause.
3. Open questions or assumptions only when they affect correctness.
4. Brief summary and test gaps after findings.

Label claims as measured, compiler-observed, analytically estimated, or inferred. If there are no findings, say that
clearly and still mention residual risks, production shapes not exercised, or tests not run.

## Implementation Mode

If the user asks to implement the review findings:

1. Make the smallest changes that address the demonstrated root cause without obscuring the model or changing numerical
   semantics unintentionally.
2. Keep comments concise and focused on invariants or ordering.
3. Reuse existing abstractions when their contracts match and remove superseded paths when the migration is complete.
4. Add or adjust contract, correctness, gradient, and benchmark coverage in proportion to the change.
5. Run focused tests and linters for touched files, plus representative profiling when making a performance claim.
6. Report the execution-layer change, numerical verification, performance and memory evidence, intent comments added or
   revised, and exact commands run.
