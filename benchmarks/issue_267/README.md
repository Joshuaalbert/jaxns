# Issue 267 distributed topology benchmark

This maintained comparison isolates the scheduling and transport change on an
analytic one-dimensional problem. Local and distributed runs use the same
model, run keys, root degree 6, allocation increment 6, three perfect-bracket
slice steps, `dlogZ=0.1`, and finite sample budget. The distributed setting has
three logical host CPU devices with one scalar worker each, so worker-local
batching cannot change the chain law.

The seed-zero compile/warm run is recorded but excluded from steady summaries.
Thirty measured seeds are run with phantom collection off and on:

```bash
PYTHONPATH=src conda run -n jaxns_py python \
  benchmarks/issue_267/run_comparison.py \
  --seeds 30 \
  --output benchmarks/issue_267/results.json
```

This cheap model is a transport-overhead stress case, not the workload for
which distributed execution is recommended. The unchanged-tolerance standard
problem suite remains the release accuracy gate.

Complete-chain pool throughput and the matched local-core `vmap` scaling are
reproduced with:

```bash
PYTHONPATH=src conda run -n jaxns_py python \
  benchmarks/issue_267/run_throughput.py \
  --tasks 300 \
  --repeats 7 \
  --output benchmarks/issue_267/throughput.json
```

That command also measures the GMM batch-compatibility rule. It gives each
task byte-distinct fit bookkeeping while holding every direction-defining
array, key, seed, and contour fixed, then compares the old full-state hash with
the execution-state hash used by the worker pool.

The representative scientific comparison uses the maintained 8D spike--slab
problem rather than the analytic transport control. Both runners start with
exact isotropic directions, stop at the user-selected expected-log-evidence
uncertainty, call `fit_gmm_directions(iso_prob=1e-2)` on the drained state, and
explicitly resume with that frozen fit. The local and distributed runners use
the same 240 root lineages, unit uniform-allocation multiplier, replacement
width 80, 40 slice transitions, finite sample limit, and termination
condition. This host's measured topology is one CPU worker with
`batch_size = 3`; a second identical worker contended for the same cores and
did not improve the smoke run.

```bash
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/jaxns-matplotlib-267 \
  conda run -n jaxns_py python \
  benchmarks/issue_267/run_standard_comparison.py \
  --case spike_slab \
  --seeds 30 \
  --runner both \
  --workers 1 \
  --batch-size 3 \
  --fit-log-z-uncert 0.2 \
  --iso-prob 1e-2 \
  --output benchmarks/issue_267/standard_spike_slab.json

# Repeat with retained phantoms and phantom-conditioned final MC evidence.
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/jaxns-matplotlib-267 \
  conda run -n jaxns_py python \
  benchmarks/issue_267/run_standard_comparison.py \
  --case spike_slab \
  --seeds 30 \
  --runner both \
  --workers 1 \
  --batch-size 3 \
  --fit-log-z-uncert 0.2 \
  --iso-prob 1e-2 \
  --phantoms \
  --output benchmarks/issue_267/standard_spike_slab_phantoms.json

PYTHONPATH=src:. conda run -n jaxns_py python \
  benchmarks/issue_267/summarise_standard.py \
  benchmarks/issue_267/standard_spike_slab.json
```

The standard runner checkpoints after every completed seed because a scalar or
small-width worker comparison is intentionally much slower than the local
width-80 `vmap`. Add `--resume` with the same scientific configuration to skip
already-checkpointed runner/seed pairs after an interruption. The harness
rejects mismatched configuration metadata before combining records. GMM update
counts and isotropic-direction fractions are part of every record, so a
scheduling candidate cannot appear faster merely by skipping direction
fitting. Evidence records contain both the expectation estimate and a
1,000-draw final Monte Carlo calculation; phantom runs use phantom conditioning
for that final calculation while the depth loop remains classic and
expectation based.

The worker-only root-likelihood boundary was also replayed on spike--slab seed
0 after the 30-seed artifacts were collected. Reproduce the focused comparison
with the first standard command changed to `--seeds 1` and output
`standard_worker_likelihood_smoke.json`; the report lists the scientific fields
checked bitwise against `standard_spike_slab.json`.

See `REPORT.md` for the reviewed tables and interpretation.
