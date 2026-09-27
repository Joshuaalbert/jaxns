# Scientific API cleanup (#288)

The runner owns the model, model inputs, sampler configuration, and a default
`DepthCondition`. State retains the model and inputs needed to resume and
interpret that run. A constrained sampler describes transitions only, and
receives the owning run's model explicitly when asked to sample.

## Results and inference

- `State.expected_log_Z_mean` and `State.expected_log_Z_uncert` provide the
  inexpensive classic expectation calculation used by goal conditions.
- `NestedSamplerResults.log_Z_mean` and `log_Z_uncert` carry that expectation
  calculation. They do not silently become Monte Carlo summaries.
- Both objects expose `sample_evidence_mc(num_samples, *, key,
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

Both runners use one shared default-resolution function. It returns an
immutable resolved configuration which each runner consumes at construction.
The distributed runner no longer constructs or retains a local runner.
Each runner remains the single owner of its resolved configuration, and
execution continues to use its explicitly owned default depth condition.

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

- `target_num_live_points` becomes `root_allocation_degree`.
- `shell_size` becomes `replacement_width` for local execution.
- `sample_logZ`, `sample_evidence`, and result `sample_mc_shrinkage` are
  replaced by the single `sample_evidence_mc` method.
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
The sampler retains the existing random split schedule after removing
step-out bracketing, including the unused reserved stream.

A fixed-key local run produced identical classic and phantom samples,
out-degrees, likelihood counts, posterior weights, expectation summaries, and
classic and phantom Monte Carlo evidence draws before and after the change.
A compiled constrained-sampling comparison used CPU, JAX 0.11.1, x64, ten
dimensions, 100 lanes, 100 transitions, and all 99 phantom states. Compiler
cost estimates were identical. Both programs used 10,500 argument bytes,
892,364 output bytes, and 1,810,520 temporary bytes, with zero aliased bytes.
This is a compiler and numerical comparison, not a wall-time speedup claim.
