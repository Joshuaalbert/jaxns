# Phantom-conditioning paper benchmark

This directory preserves the early **8D** benchmark used during development
of the phantom-conditioning paper. It tests prefixes through `9D` and is not
the final G10/CG10/SS10 protocol with full `10D-1` retention. Later cohort
analysis and rendering scripts are documented in
[`docs/papers/EXPERIMENT_CODE.md`](../../docs/papers/EXPERIMENT_CODE.md).
Some historical protocol identifiers and default directory names contain
`10d`, but the model definitions in `cases.py` use dimension 8.

This benchmark isolates the paper's main statistical claim. Each seed creates
one race tree with either isotropic directions or an explicitly staged GMM
direction fit and retained phantom clusters.
Each chain uses `T=10D` slice transitions and therefore has `T-1` eligible
phantom states before the final classic sample. The completed tree is evaluated
with classic shrinkage and leading prefixes of `D, ..., 9D` phantoms. A prefix
of `pD` is the fraction `pD/(T-1)` of all eligible phantom states. This paired
design gives every sweep entry the same classic samples, stopping point, and
likelihood evaluations.

The Python goal is expected classic `log_Z_uncert < 0.05`. The compiled depth
loop uses only `dlogZ = log1p(1e-3)`, so final Monte Carlo shrinkage is never
placed in the sampling hot path. Problems use 30 paired seeds (`0` through
`29`). Every completed tree uses 2048 MC shrinkage draws for each prefix.
This makes each tree's reported MC uncertainty precise enough to standardise
its own evidence error before calibration is assessed across the 30 trees.
For the two spike--slab cases, the same shrinkage draws also give posterior
mode masses at every phantom prefix.

Reported RMS standard errors use 10,000 non-parametric bootstrap resamples of
the 30 completed tree indices. The same indices are shared across all phantom
prefixes, so standard errors and percentile intervals for RMS changes retain
the paired design. Selected-prefix intervals are descriptive because the best
prefix is chosen from the same sweep.

The sweep reduction shares classic gamma races and cluster weights between
prefixes and processes each phantom interval event once per draw. On CPU the
benchmark encodes those events in SciPy CSR difference matrices: sparse matrix
multiplication preserves each cluster's shared gamma weight while avoiding
millions of slow XLA scatter updates. Exact prefix-specific Kish gates are
computed from interval intersections. The host reference matches the JAX
kernel draw-for-draw to floating-point precision; only the coupling between
prefix columns is deliberately paired. This avoids repeating shorter-prefix
events in every longer-prefix calculation.

The four paper cases are `basic_mvn`, `weak_curved_mvn8`, `spike_slab`, and
`correlated_spike_slab8`. They are purpose-built continuous, known-evidence
8D stress tests and are deliberately separate from the fixed standard-problem
release gate.

Each raw record separates core `run_seconds` from
`evidence_sweep_seconds`. The cells may be executed concurrently, so those
wall times are provenance diagnostics rather than publication comparisons;
the paper's performance measure is the exact likelihood-evaluation count,
which is identical across all prefix entries on a completed tree.

Run this resumable benchmark through its entrypoint with explicit writable
checkpoint and output directories:

```bash
conda run -n jaxns_py python docs/papers/run_experiments.py \
  --processes 8 --state-root /path/to/states --output-root /path/to/results
```

The entrypoint separates production, analysis, independent checkpoint
verification, summary, and figure phases. Select phases explicitly when a
machine needs different concurrency for sampling and Monte Carlo analysis:

```bash
conda run -n jaxns_py python docs/papers/run_experiments.py \
  --phase core --processes 8 \
  --state-root /path/to/states --output-root /path/to/results

conda run -n jaxns_py python docs/papers/run_experiments.py \
  --phase analysis --processes 4 --mc-workers 2 \
  --state-root /path/to/states --output-root /path/to/results

conda run -n jaxns_py python docs/papers/run_experiments.py \
  --phase verify --phase summary --phase figures \
  --state-root /path/to/states --output-root /path/to/results
```

Add `--dry-run` to inspect commands without launching sampling. The default
seed population contains only classic samples. `--phantom-seeding on`
requires the separate experimental implementation that accepts this option,
selected through `PYTHONPATH`. That implementation is not added to `develop`
by these scripts. Checkpoint analysis likewise requires the implementation
and model definitions used to create the checkpoint.
