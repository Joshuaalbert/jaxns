# Scientific API cleanup (#288)

The runner owns the model, requested sampler configuration, and a default
`DepthCondition`. State owns the model inputs needed to resume and
interpret that run. A constrained sampler describes transitions only, and
receives the owning run's model explicitly when asked to sample.
New runs use the runner's model. Once a state exists, its saved model and
inputs are authoritative for continuation, including worker registration.

## Results and inference

- `State.expected_log_Z_mean` and `State.expected_log_Z_uncert` provide the
  inexpensive classic expectation calculation used by goal conditions.
- `NestedSamplerResults.log_Z_mean` and `log_Z_uncert` carry that expectation
  calculation. They do not silently become Monte Carlo summaries.
- Both objects expose `sample_evidence(num_samples, *, key,
  phantom_conditioning=False, num_phantoms=None, batch_size=None, C_min=20,
  diagnostics=False)`. Classic conditioning is the default. Phantom
  conditioning is an explicit opt-in. The returned `EvidenceSamples` owns
  the Monte Carlo summaries. Random evidence draws require a key.
- `results.resample(num_samples, *, key, replace=True)` returns
  `PosteriorSamples`, containing aligned U coordinates, X coordinates, and
  likelihoods. Its `integrate_fn_over_posterior` averages over these equally
  weighted observations. It carries no run evidence, uncertainty, ESS, or
  race metadata. The weighted result retains its own integration method.
- `BlockData.incoming_K` owns lineage counts. An expanded copy is not stored
  on result rows. The MC adapter derives this view only when needed by the
  existing shrinkage kernel.

## Run configuration and constrained sampling

Both runners accept `args` and `params` only when starting a run through
`initialise`, `run`, or `run_until_goal`. Local `run_single_iteration` accepts
them when starting without a state too. These inputs are stored on State,
never on the runner. An existing state or checkpoint takes precedence over
new-run inputs before any model-dependent defaults are resolved.

Both runners use one shared default-resolution function. It returns an
immutable, transient execution configuration at initialization or resumption.
Dimension, periodic topology, and dependent defaults are derived from the
active model inputs. Requested settings on the runner stay unchanged, so
reusing it for different input shapes cannot inherit a previous run's defaults.
The distributed runner no longer constructs or retains a local runner.
Execution continues to use its explicitly owned default depth condition.
Workers receive an immutable session containing the active inputs once per
registration. This transport copy is derived from State on resumption.

`replacement_width` replaces `shell_size` on the local runner. Distributed
sampling has no replacement-width setting: workers own batching. Local and
distributed allocation increments and initial capacities keep their existing
defaults. Shared resolution must not make the distributed defaults depend on
worker topology or execute a likelihood in the scientific process.

`UniDimSliceSampler(num_slices, collect_phantom_samples=False,
max_phantom_samples=None)` retains the direct phantom-prefix capacity and
collection switch. The deprecated inverse `phantom_burn_in` spelling and the
unsupported step-out switch are removed. Perfect unit-cube bracketing is the
implemented transition. The existing random split schedule is preserved.

## Removed surface and migration

- Import `Prior` from `jaxns.priors`. It re-exports the JAXCTX class directly,
  so ordinary model definitions no longer need its dependency's import path.
- `target_num_live_points` becomes `root_allocation_degree`.
- `shell_size` becomes `replacement_width` for local execution.
- `sample_logZ`, `sample_evidence_mc`, and the former result shrinkage alias are
  replaced by the single `sample_evidence` method.
- Move constructor `args` and `params` to the run-start call. Resumption
  reads them from State. Runner attributes retain requested settings rather
  than exposing model-dependent resolved defaults before initialization.
- `conditioning="phantom"` becomes `phantom_conditioning=True` on that method.
- Result `expected_log_Z_*` aliases are removed. Use its `log_Z_*` fields.
- `num_live_points_per_sample` and `evidence_equivalent_live_points` are
  removed. Block incoming lineage counts remain available.
- Pass the model to sampling operations, not to `UniDimSliceSampler`.
  Custom constrained samplers accept the keyword-only `model` argument too.
- Manually constructed results supply `BlockData` explicitly. Saved result
  objects using the retired schema should be regenerated from their State.
- Request and worker-execution types belong to `jaxns.sampling.protocol` and
  `jaxns.sampling.batching`, rather than being sampler-module re-exports.

This change does not regroup State storage, change the statistical model,
change seed selection, or alter the distributed wire protocol and retry law.

## Performance and intent review

Review compared the implementation with develop at `317cec5`. The MC entry
point now validates metadata once through the owning shrinkage API. Lineage
expansion compiles from four required arrays, so changing a phantom prefix or
the transformed parameter tree cannot retrace this independent operation.
Distributed resumption registers the checkpoint model, matching the model
used by local resumption, rather than reading a separate runner model. The
sampler retains the existing random split schedule after removing
step-out bracketing, including the unused reserved stream.
An explicitly supplied phantom capacity must be positive: zero previously
fell through to full retention instead of expressing a valid memory bound.

A fixed-key local run produced identical classic and phantom samples,
out-degrees, likelihood counts, posterior weights, expectation summaries, and
classic and phantom Monte Carlo evidence draws before and after the change.
A compiled constrained-sampling comparison used CPU, JAX 0.11.1, x64, ten
dimensions, 100 lanes, 100 transitions, and all 99 phantom states. Compiler
cost estimates were identical. Both programs used 10,500 argument bytes,
892,364 output bytes, and 1,810,520 temporary bytes, with zero aliased bytes.
This is a compiler and numerical comparison, not a wall-time speedup claim.

The follow-up input-ownership change resolves configuration once at each
Python initialization or resume boundary, after checkpoint precedence has
been decided. It does not introduce resolution inside a compiled depth loop
or per worker task. Resolved settings retain no model-input arrays, and the
default depth condition remains owned by the runner. A second fixed-key
comparison against `25806f1` again matches all of the scientific outputs
listed above exactly. Regression tests cover reuse across input dimensions,
all run-start methods, and state-owned inputs and topology on resumption.

## Execution handoff and documentation

`DistributedState.from_state(state)` starts a fresh runtime session from a
completed local goal boundary, preserving the exact scientific state and keys.
`distributed.to_state()` returns that full state only after pending tasks and
active schedules have drained. Neither conversion transforms posterior samples
or repeats likelihood evaluations. The laptop-to-cluster user guide covers both
directions, delayed CPU/GPU worker arrival, checkpoints, and automatic growth.

The composed handoff test exposed a pre-existing unlimited-growth bug: the
distributed status classifier treated physical buffer capacity as a scientific
hard limit. It now distinguishes the two, preserving finite limits while
allowing unlimited runs to resize. The real TCP test exercises growth, saved
state transfer, late worker arrival, checkpoint precedence, and local return.

All maintained evidence-sampling helpers, reference functions, benchmarks, and
tests now use the `sample_evidence` name, including their compiled and batched
variants. This is a symbol-only change to the evidence kernels.

## Progress and interruption

Both runners accept `verbose=False`. Progress uses existing scalar fields,
sample capacity, task identifiers, and Python monotonic times. It neither
constructs results nor evaluates evidence or posterior summaries for logging.

The requested SIGINT behavior extends the distributed lifecycle decision from
issue #252: protocol 6 adds an explicit cancellation flag to session release.
A cancelled session's busy assignments are fenced, its queued/completed task
records are removed, and the existing node supervisor restarts affected worker
processes. Other sessions and idle workers remain available. Local checkpointed
runs return to Python at most every 32 replacement batches without callbacks
inside JAX. Control returns retain the active schedule, random keys, and goal
iteration counters. They do not become scientific goal boundaries.

After Ctrl-C, the newest coherent state is saved when checkpointing is enabled,
then KeyboardInterrupt propagates. Distributed cleanup also runs without a
checkpoint. Signal deferral restores the caller's handler and is installed only
on the main thread. Compilation and a currently executing batch must complete
before the local continuation can be saved, so the batch bound is not a deadline
in seconds. Initialisation interrupted before a complete state exists can only
cancel the session, not invent a resumable state.

The performance/intent review keeps progress outside compiled loops, adds no
host callbacks, and restricts the extra compiled stopping condition to a scalar
batch counter on the checkpointed path. A fixed-key SIGINT test resumes an
unfinished local depth and reproduces every uninterrupted state leaf exactly,
including the goal-call sequence. Real-process tests cover distributed SIGINT
with and without checkpoints, zero-worker registration, replay of pending work,
and preservation of another registered session. No wall-time speedup is claimed.
