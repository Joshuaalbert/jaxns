# Design Requirements

This file contains the current JAXNS implementation, architecture, API,
performance, and test-harness requirements. Scientific and user-visible
properties that must hold independently of implementation live in
`INVARIANTS.md`.

## Design Sources

- Requirement: The JAXNS v3 algorithm is defined by `docs/papers/paper.tex`; code comments and
  tests may clarify it but may not silently replace its scientific model.
- Requirement: The intended run orchestration is documented in
  `docs/design/interface/run_pattern.py` and must be updated when the accepted local or
  distributed run pattern changes.
- Requirement: Design requirements and invariants use unversioned scientific names; source
  files, classes, and variables do not acquire `v3` prefixes or suffixes.
- Requirement: Distributed execution remains out of release scope until its process,
  serialization, failure, scheduling, and reproducibility design has been validated explicitly.

## Package And Public API

- Requirement: Production code lives under `src/jaxns`, tests remain outside `src`, and the
  package supports Python 3.10 and newer.
- Requirement: State, samples, result, block-data, shrinkage-data, and scheduler-data containers
  use frozen, slotted dataclasses registered through `PureDataclassPytree` where their contents
  are JAX-compatible data.
- Requirement: Every array-valued dataclass field has an adjacent shape comment using symbolic
  axis names such as `# [N, D]` or `# []`.
- Requirement: User-facing state and result objects carry thin methods for operations naturally
  applied to their data, so scientific users can work object-orientedly without moving core
  orchestration into those containers.
- Requirement: `NestedSamplerResults` keeps the commonly used evidence, posterior, and sample
  fields visually primary and stores block-aligned implementation detail in one `BlockData`
  field.
- Requirement: Persistent samples store parent likelihood contours and out-degrees, not parent
  storage indices.
- Requirement: Parent indices may exist only as transient scheduling or scatter operands and
  must not survive into persistent state or results.
- Requirement: A compatible parent graph is reconstructed as a derived diagnostic when a caller
  requests it; parent storage indices are not added to scientific state to make reconstruction
  convenient.
- Requirement: Result block data is derived consistently from classic likelihoods, parent
  contours, out-degrees, and the valid-sample count.
- Requirement: End-user state and result methods compute from the data carried by that object and
  do not depend on an unrelated global run.
- Requirement: Public methods validate incompatible shapes, missing metadata, and unsupported
  modes before launching expensive compiled work.
- Requirement: Public APIs have concise Google-style docstrings that state non-obvious
  scientific behavior, arguments, returns, and failures.
- Requirement: Serialization is supported through the shared pytree serialization surface rather
  than bespoke serialization hidden in individual algorithms.
- Requirement: Serializing and restoring a supported state or result preserves its scientific
  arrays, model association, and posterior and evidence behavior.

## Core Run Architecture

- Requirement: The outer goal loop is Pythonic and calls a user-provided condition on `State`.
- Requirement: One depth epoch is a pure JAX computation suitable for JIT compilation and
  returns control at a depth boundary, capacity boundary, or explicit no-progress boundary.
- Requirement: The compiled depth loop computes allocation gaps, selects parents, selects
  stationary seeds, reparents seedless work, samples a fixed replacement width, and accepts that
  work without host callbacks.
- Requirement: Parallel replacement sampling uses `jax.vmap`; `jax.lax.map` is not used for the
  replacement batch.
- Requirement: Replacement width is static for a compiled depth epoch, while a validity mask
  prevents unused lanes from changing scientific state or likelihood counts.
- Requirement: The default allocation increment fills one replacement batch so ordinary outer
  iterations do not leave most compiled sampler lanes idle.
- Requirement: The depth hot path does not repeatedly sort all stored samples; append-order
  samples are paired with a lightweight likelihood-order index and block reductions.
- Requirement: Sample storage remains append ordered, and derived likelihood order is updated
  without rewriting existing sample payload rows.
- Requirement: Core functions remain linear and intent-commented; every non-obvious selection,
  mask, lineage update, or numerical construction explains the scientific consequence that would
  break if changed.
- Requirement: Accelerated block, shrinkage, and phantom calculations retain a clear NumPy or
  pure-Python reference implementation for correctness comparison.

## Capacity And Return State

- Requirement: A finite `max_samples` is a scientific capacity limit; unlimited automatic growth
  is an explicit opt-in mode rather than the default meaning of an omitted limit.
- Requirement: The default finite maximum capacity is proportional to the root out-degree, with
  a target default of approximately one thousand samples per root lineage unless the user
  supplies a limit.
- Requirement: The initial physical allocation is smaller than a large finite maximum and is
  large enough for the root batch plus at least one replacement batch.
- Requirement: When unlimited growth is enabled and a depth epoch fills storage, the Python goal
  loop doubles physical capacity sufficiently to fit the next full replacement batch and resumes
  transparently after recompilation.
- Requirement: `State` carries `needs_growth`, `depth_reached`, and a termination reason code so
  callers do not infer return causes from unrelated counters or buffer sizes.
- Requirement: Returning from a depth epoch solely because storage filled does not increment
  `goal_loop_iter`.
- Requirement: Resizing preserves every existing pytree leaf, likelihood-order entry, count,
  diagnostic, and random key exactly.
- Requirement: Padded storage entries contribute no prior volume, evidence, posterior mass,
  block multiplicity, or lineage count.
- Requirement: A result is built only after block capacity and lineage consistency have been
  validated over the valid sample prefix.

## Allocation And Seed Scheduling

- Requirement: Supported allocation targets are uniform, evidence improving, and posterior
  improving.
- Requirement: Allocation utilities use expectation-based block volumes and cumulative
  reductions rather than Monte Carlo evidence draws inside the depth loop.
- Requirement: Work planning selects at most the fixed replacement width of parent contours per
  depth iteration without a full sort of sample payloads.
- Requirement: Stationary seed eligibility is determined from each candidate's likelihood and
  recorded parent contour.
- Requirement: Seed reuse is prevented only among simultaneous children of the same parent
  contour; seeds for distinct parent contours do not need to be globally distinct.
- Requirement: If a requested parent has no stationary seed, work is reassigned to the closest
  valid contour that has an eligible stationary seed and the child records that actual contour.
- Requirement: Seed-reuse and reparenting event counters are not part of the user-facing result
  schema.

## Constrained Sampler

- Requirement: The release sampler uses isotropic directions and perfect unit-hypercube
  bracketing with greedy interval shrinkage.
- Requirement: Gradient-guided directions, ellipsoidal directions, Galilean trajectories, and
  step-out bracketing are not accepted by the release core until separately implemented and
  validated.
- Requirement: Constrained chains are sampled in parallel at fixed replacement width even though
  their likelihood-evaluation counts may differ.
- Requirement: Parallelizing individual likelihood calls independently of chain sampling is
  future work and must preserve the non-deterministic number of evaluations used by each chain.
- Requirement: Retained phantoms are the earliest eligible post-burn-in intermediate states of a
  chain, are ordered as generated, and exclude the final classic child.
- Requirement: Merely enabling phantom retention cannot change the number of slice transitions
  or random choices that determine the final classic child.

## Shrinkage And Evidence

- Requirement: Singleton blocks set equality concentration and equality phantom contribution to
  exactly zero and implement their complement as `A - B`.
- Requirement: Plateau blocks use the paper's three-class concentrations with the configured
  equality prior, whose current neutral value is epsilon equal to one half.
- Requirement: The classic expectation calculation is the online source used for depth
  conditions and inexpensive state summaries.
- Requirement: Final user-facing evidence uncertainty and evidence draws use Monte Carlo
  shrinkage and expose classic and phantom-conditioned modes explicitly.
- Requirement: The default phantom gate uses the Kish participating-cluster count with
  `C_min = 20`, while public final-inference APIs may accept an explicit alternative threshold.
- Requirement: One Gamma(1, 1) weight is drawn per phantom cluster and Monte Carlo draw and is
  reused for all counts and blocks from that cluster.
- Requirement: Monte Carlo shrinkage supports bounded batching so requested draw count does not
  require materializing every draw and every block at once.
- Requirement: Phantom coordinates are not required for evidence conditioning once likelihoods,
  validity, cluster identity, and parent-contour metadata have been retained.

## Defaults And Scientific Comparability

- Requirement: Unless explicitly overridden, root out-degree is thirty times unit-hypercube
  dimensionality, slice transitions are five times dimensionality, and replacement width is at
  most the root degree and targets ten times dimensionality.
- Requirement: Default evidence stopping uses `log1p(1e-3)` so v2 and v3 scientific comparisons
  use the same remaining-evidence condition.
- Requirement: Benchmark comparisons explicitly report and match termination condition, root or
  live-point count, slice count, phantom setting, hardware, precision, and compilation
  treatment.
- Requirement: Standard-problem acceptance retains the established tolerances in
  `tests/test_ns_standard_problems.py`; v3 may not pass by loosening those tolerances.
- Requirement: Standard-problem tests exercise phantom collection both disabled and enabled and
  assess Monte Carlo evidence estimates against known expectations.
- Requirement: Release evidence uses approximately thirty independent seeds per standard problem
  and reports each problem separately rather than pooling unlike problems into one row.
- Requirement: Per-problem release evidence reports log-evidence bias, root-mean-square error,
  normalized-error mean and dispersion, reported uncertainty, likelihood evaluations, wall time,
  and posterior diagnostics where a reference posterior exists.
- Requirement: v2 and v3 performance comparisons use the same scientific stopping rule and
  equivalent sampler budgets; defaults are never assumed silently in the report.
- Requirement: Phantom-off and phantom-on comparisons report both how many phantoms were
  retained and how many blocks passed the conditioning gate.
- Requirement: Difficult multimodal problems, including spike-slab mode-weight recovery, are
  judged on posterior mode weights as well as aggregate evidence error.
- Requirement: Diagnostic artifacts for maintained benchmark runs include corner plots and
  evidence or shrinkage diagnostics sufficient to investigate anomalous per-problem results.
- Requirement: JAXNS v3 must be at least as accurate and performant as the latest released v2 on
  the maintained standard-problem benchmark, with regressions explained and resolved before
  release.

## CI/CD Test Ownership

- Requirement: `docs/design/INVARIANTS.md` is authored before existing tests are mapped to it so
  the test suite cannot define the scientific contract retrospectively.
- Requirement: `cicd/coverage_record.json` maps exact invariant text to non-empty lists of
  uniquely named tests that genuinely exercise that invariant.
- Requirement: Every discovered unit test is classified exactly once as invariant coverage or in
  `cicd/non_invariant_test_coverage.json` with a path, reason, and note.
- Requirement: Reviewer autochecks validate coverage-record integrity, test ownership,
  repository structure, and review policy without pretending to be scientific invariant tests.
- Requirement: Feature pull requests into `develop` run unit tests and reviewer autochecks,
  while `develop` to `main` additionally runs system tests and the all-invariants-covered
  pre-release gate.
- Requirement: The unit-test workflow is a required pass for Python 3.10, 3.11, 3.12, 3.13, and
  3.14.
- Requirement: Pre-release autochecks fail when any design invariant lacks recorded test
  coverage, even if all existing tests pass.
- Requirement: System tests have an explicit design specification and validate composed public
  behavior rather than private implementation state.
- Requirement: Demos are executable, deterministic, network-independent examples of supported
  user workflows and contain enough assertions to fail when the demonstrated contract drifts.
- Requirement: Demos run automatically on `develop` after feature integration.
- Requirement: Maintained benchmarks exercise production entry points, keep validation and
  compilation outside timed steady-state regions, and record enough metadata for comparisons
  across commits.
- Requirement: Benchmark programs are not ordinary unit tests and do not make a pull request
  fail solely because shared-runner wall time fluctuates.
