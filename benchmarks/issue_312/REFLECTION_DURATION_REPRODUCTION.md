# Frozen reflection duration comparison

This benchmark starts from published commit
`295893708805ff668178fc443cbda63faf2ac75d`. The executed sampling/reduction source is
`724d5e9a94d2db417573cd8cc473224e82a60a13`; report-only changes are committed after
all workers finish. The protocol and schedule were frozen before validation.
No production sampler or manuscript change is part of this experiment.

The three arms are `fixed`, `short`, and `duration`. Each runs both `uniform`
and `evidence_improving` allocation with fresh seeds 30–59. The duration arm
uses the inherited step schedule as desired resolution and chooses
`B = ceil(0.03 / h_old)`, then `h = 0.03 / B`. The common early region uses
B=10/h=0.003. This is a duration test, with no new acceptance calibration.
Read `REFLECTION_DURATION_PROTOCOL.md` for the complete scientific contract.

## Execution and provenance

All original outputs remain at
`/largedata/albert/jaxns-reflection-duration-20261027` on dorrie. Source freeze,
protocol, schedule, validation and independent audit checksums are bound by
`EVALUATION_RELEASE.json`. Exact source archives include the shared core,
benchmark entrypoints, trajectory primitive, observer and evidence backend.
The earlier attempts are retained in `attempt-a59a4a7/` and `attempt-b1c2351/`;
they are not evaluation runs. Coordinator audit scripts and receipts are in
`coordinator-audits/`.

Use the `jaxns_py` conda environment. The CPU-only launch environment is:

```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=''
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=.:src
```

Pin before importing JAX. For example, from the executed source worktree:

```bash
taskset -c 2 conda run --no-capture-output -n jaxns_py python \
  -m benchmarks.issue_312.run_reflection_duration \
  --cpu 2 --arm duration --allocation uniform --seed 30 \
  --schedule /largedata/albert/jaxns-reflection-duration-20261027/SCHEDULE.json \
  --output /tmp/new-duration-seed30
```

The runner checks the frozen source/release and refuses an existing output
directory. Reproduction requires the exact executed commit and original
release artifacts. Never replace a completed directory. The independent
controller source and launch log are archived with the experiment. It starts
four workers, requires both full duration pilots to exercise B>10 with exact
replay, and then permits at most 24 workers subject to memory and disk gates.
CPU70 is reserved for coordinator audits. The controller adopts completion
markers on restart and fails rather than overwriting an incomplete run.

For a completed run, use `analyse_reflection_duration` with `--cpu`, `--run`
and `--schedule`. It replays all saved chains before doing the evidence
reductions. Analyse archived records at their recorded source revision;
new schema-2 arrays are required by this analyser.

## Raw record contract

Each run has two immutable snapshots at the first complete batch crossing
2M/4M logical model calls. `CORE-{budget}.json` records actual work, overshoot,
posterior quantities, elapsed time, keys and array hashes. The final complete
`state.pkl` is resumable and has a SHA256 sidecar. `MANIFEST.json` records
source/dependency/host/CPU provenance. `progress.jsonl` records completed
allocation epochs and snapshots.

`calls-{budget}.npz` contains every classic unit/physical point, likelihood,
constraint, posterior log weight, seed/parent/batch index and exact chain key.
It also records per-classic `bounce_counts`, `nominal_steps`,
`gradient_evaluations_per_sample`, and `value_only_evaluations_per_sample`.
Every root, including later allocated roots, has B/h/gradient zero and one
value call. Each nonroot uses exactly 9B gradients, counted once in its total
logical calls. Masked vmap hardware work is not inferred from these counters.

`inputs-{budget}.npz` retains the complete evidence inputs, including all eight
complete-cycle phantom likelihoods and their cluster validity. Phantoms
receive no posterior weights and never participate in seed selection.
`evidence-{budget}.npz` retains all 4,096 classic and phantom-conditioned log-Z
draws and conditioning diagnostics. `chain-diagnostics.npz` retains exact
replayed path acceptance and per-cycle realised steps/durations. Reports use
ESJD per gradient as sum squared displacement divided by sum actual gradients;
RMS divided by a gradient count is not used in the duration analysis.

`ANALYSIS-{budget}.json` binds exact replay, independent likelihood/count
checks, posterior diagnostics, movement bins and evidence draws. The
remaining-tail diagnostic is stored in finite log form; a linear overflow is
explicit null plus an overflow flag. `ANALYSIS-COMPLETE.json` requires both
snapshots and both evidence reductions to pass.

## Final report and archives

After all 180 analyses complete, the shared aggregation path audits all 360
snapshots and produces `SUMMARY.json`, `ALL_RECORDS.csv`, `REPORT.md` and
PDF/PNG figures. It uses all seeds and the prespecified 100,000 paired
bootstrap resamples; no preferred estimator, allocation or snapshot is chosen
from the observed outcomes. The coordinator independently checks the raw
records, evidence estimates and intervals before publication.

```bash
taskset -c 27 conda run --no-capture-output -n jaxns_py python \
  -m benchmarks.issue_312.summarise_reflection_duration --cpu 27 \
  --root /largedata/albert/jaxns-reflection-duration-20261027
```

Full states/raw arrays remain outside Git. Compact records, review artifacts
and the incremental Git bundle are listed with SHA256s in `HANDOFF_FINAL.json`.
The coordinator owns GitHub publication; the numerical worker does not push.

The completed cohort also has a post-analysis diagnostic correction, recorded in
`DIAGNOSTIC_CORRECTION.json`: six movement-bin B/time descriptors were rebuilt
from unchanged raw counts and replay steps. Original JSON/aggregate outputs remain
in `pre-movement-diagnostic-correction/`. Run the archived correction script at
the final report revision after reproducing original analyses at724d5e9; it checks
independent gradient counts and binds every corrected-file checksum. No sampler,
evidence or replay work is repeated. The original source freeze/release stays intact.
