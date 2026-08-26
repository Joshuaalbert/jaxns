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
- Requirement: Distributed execution follows the accepted process, serialization, failure,
  scheduling, and reproducibility decision recorded on issue 252; changes to those boundaries
  require new evidence and an updated design decision.

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
- Requirement: Runtime and optional dependencies are declared only in `pyproject.toml`, with the
  rationale and public entry-point audit recorded in `docs/design/DEPENDENCIES.md`.
- Requirement: The base install includes JAX, JAXCTX, NumPy, SciPy, and TFP because these support
  the documented model-authoring, local nested-sampling, and public diagnostic workflow.
- Requirement: JAX selects its compatible `jaxlib`; JAXNS does not independently constrain that
  transitive runtime or interfere with accelerator-specific JAX installations.
- Requirement: Matplotlib is supplied by the base installation but imported only when a plotting
  operation is requested; extras do not repeat dependencies already supplied by the base.
- Requirement: ZeroMQ and Cloudpickle are isolated in the `distributed` extra, while the
  installed `jaxns-cli` configuration-validation command remains usable without importing JAX,
  ZeroMQ, or model-serialization support.

## Distributed Execution

- Requirement: Distributed execution is an opt-in `DistributedNestedSampler`; the established
  local `NestedSampler` remains the compiled, dependency-light execution path.
- Requirement: `jaxns-cli` owns one named coordinator or worker node and provides idempotent
  `config validate`, `up`, `status`, and `down` operations from a TOML configuration, plus
  explicit CurveZMQ certificate creation.
- Requirement: A coordinator publishes a versioned same-user ownership manifest containing its
  IPC endpoint, advertised TCP endpoint, protocol version, process identity, and resolved
  configuration fingerprint. Remote nodes use an explicit TOML endpoint when they cannot read
  that local discovery record.
- Requirement: The scientific client and same-machine workers discover and use same-user IPC;
  remote workers use encrypted CurveZMQ TCP and only public keys installed in the coordinator's
  authorization directory may connect. Pickle and Cloudpickle are never exposed over
  unauthenticated TCP or presented as safe input from an untrusted node.
- Requirement: The supervisor imports no JAX or model code and routes opaque registration,
  request, and result bytes between scientific clients and workers.
- Requirement: Protocol headers and payloads have explicit size limits, and model registration
  rejects workers whose Python, JAXNS, JAX/JAXLIB, x64, or measure-dtype semantics differ from
  the scientific process. Device platform may differ intentionally.
- Requirement: Each worker is one OS process with one configured JAX device and one fixed batch
  capacity; size one calls a scalar chain directly and larger compatible groups use `jax.vmap`,
  never `jax.lax.map`.
- Requirement: Distributed scientific scheduling has no shell size. The main process submits
  scalar logical lineage threads continuously to fill compatible live pool lanes from currently
  known allocation gaps; worker batch size is only a device-execution choice and cannot change
  allocation targets or random task identity.
- Requirement: Task credits are bounded by compatible live pool capacity. Every lane can begin
  immediately, but the scheduler does not speculate beyond the currently measured pool.
- Requirement: The coordinator groups only tasks from the same session and direction-state
  specialization. A partial group may run after a short bounded fill interval so a wide worker
  cannot deadlock a sparsely populated queue.
- Requirement: A worker executes a complete constrained-sampling chain or vmapped chain batch;
  individual likelihood evaluations inside data-dependent sampler loops never cross IPC.
- Requirement: The first distributed release bootstraps the finite root prior batch in the main
  process and distributes complete constrained replacement chains; root-batch distribution is
  not silently claimed as part of its measured scaling.
- Requirement: Model, sampler, arguments, and parameters are registered once per scientific
  session; task messages contain only task-specific keys, strict contours, stationary seeds,
  validity, and direction state.
- Requirement: Each worker keeps a small bounded cache of compiled session programs keyed by the
  exact registration payload so releasing and later resuming the same model does not force an
  avoidable recompilation or create unbounded executable retention.
- Requirement: The main scientific process exclusively owns nested-sampling state, parent
  out-degree commits, PRNG assignment, task identity, allocation, growth, and goal evaluation.
- Requirement: Dispatch creates a separate provisional lineage reservation before transport;
  reservations influence allocation planning but are not observations in user-facing `State`.
- Requirement: The asynchronous pool consumes the shared logical-thread queue without a wave
  barrier; an unknown pending endpoint is conservatively reserved beyond known contours and a
  newly joined worker can immediately consume already queued work.
- Requirement: Physical capacity accounts for committed and reserved rows, and capacity growth
  preserves the active depth epoch and already dispatched immutable task payloads.
- Requirement: Task submission is idempotent by session and task identity, a retry retains the
  exact scientific payload and keys, and a completed payload remains replayable until the client
  acknowledges its exactly-once commit.
- Requirement: Results are committed to scientific state in stable task-ID order while workers
  continue asynchronously. Completion latency, including phantom payload size, cannot select a
  different race tree for the same assigned random stream.
- Requirement: Worker death or task timeout requeues an unchanged task when compatible capacity
  remains, reports degraded capacity visibly, and does not hide failure through automatic worker
  restart.
- Requirement: Coordinator restart is explicit rather than transparent. Its ownership lock fences
  a second local owner; old worker leases become invalid, workers self-quarantine, node stacks are
  restarted by the operator, and the scientific process resumes its immutable checkpoint.
- Requirement: Every worker has a coordinator-issued lease and sends heartbeats independently of
  data-dependent JAX sampling. Two missed heartbeats fence and drop the worker; two missed lease
  acknowledgements make the worker self-quarantine, and assignment generations reject late
  results from an old lease.
- Requirement: A worker node may join an active session and receives its immutable registration
  before work. Removing a node drains running work before stopping its processes; an ungraceful
  loss remains bounded by heartbeat detection and exact task requeue.
- Requirement: Coordinator status exposes node identity, worker identity, device specialization,
  readiness, busy/draining/dropped state, compile time, execution time, and memory high-water
  mark without exposing authentication secrets.
- Requirement: The supervisor fairly rotates among registered sessions with dispatchable work;
  one client cannot permanently monopolize every compatible idle worker.
- Requirement: Goal conditions run only at a drained depth boundary; pending results and
  provisional reservations are committed or explicitly retained in a resumable distributed
  checkpoint before results can be exposed.
- Requirement: Exact samples may differ when worker topology or completion latency changes, but
  every topology preserves the constrained-prior and race-tree laws; topology invariance is not
  promised as bitwise reproducibility.

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
- Requirement: The initial physical allocation is the root batch plus approximately 64
  replacement batches, clamped to a finite maximum; it is smaller than a large finite maximum
  so unused padding does not dominate compiled block operations.
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

- Requirement: The release sampler defaults to isotropic directions and perfect unit-hypercube
  bracketing with greedy interval shrinkage, preserving isotropic directions as an explicit
  correctness reference and fallback.
- Requirement: Opt-in ellipsoidal directions use persistent warm-started GMM geometry, select
  contour-eligible components by ellipsoidal volume, and independently retain an isotropic
  transition probability of one percent by default.
- Requirement: A direction-geometry update occurs at most once before a replacement batch, and
  the fit and parent contour remain fixed for each complete constrained chain in that batch.
- Requirement: Gradient-guided directions, Galilean trajectories, and step-out bracketing are
  not accepted by the release core until separately implemented and validated.
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
- Requirement: Final evidence sampling defaults to an economical result without per-draw block
  diagnostics and evaluates at most 64 independent draws per batch unless explicitly changed.
- Requirement: Evidence batches execute sequentially at device runtime, or synchronize at a
  host batching boundary, so asynchronous dispatch cannot make several nominally bounded
  workspaces live concurrently.
- Requirement: Batched per-block evidence moments are merged from first and second sufficient
  statistics; batches are not averaged with equal weight when the final batch is partial.
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
  `cicd/tests/test_ns_standard_problems.py`; v3 may not pass by loosening those tolerances.
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
