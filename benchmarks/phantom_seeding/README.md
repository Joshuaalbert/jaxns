# Exact efficient phantom seeding: implementation and evidence

The combined block index and lazy candidate lookup passes the exact-A gates
and reduces measured warm whole-goal time by **4.08× on G10** and **3.64× on
SS10**. Every tested scientific trajectory and likelihood count is preserved.
This is an implementation improvement to full A+C, without a new sampling law.

Worktree: `jaxns-efficient-phantom-A-20260910`.
Branch: `feature/v3-efficient-all-phantom-seeding`.
Reference: frozen A+C commit `aa79a0dc395c0d64b85cbb88d0e3e732decd7361`
(`paper/evidence10-AC300`). Related tracking: [#292](https://github.com/Joshuaalbert/jaxns/issues/292).
The shared checkout, paper, frozen source, and original experiment archives
were left untouched. The cancelled precision ladder was not restarted.

[measurements.json](measurements.json) contains 45 measured cases, raw timing
series, quartiles, array sizes, compiler memory plans, process RSS, source
hash manifests, and checked scientific fingerprints. Large states, individual
selected-point arrays, HLO, logs, and prior harness versions remain locally
under `results/`, excluded from version control.

## Thirty-seed SS10 follow-up at 0.02

[The paired cohort report](cohort-ss10-002.md), [raw evidence](cohort-ss10-002.json)
and [figure](cohort-ss10-002.png) measure one further complete goal iteration
from each of 30 archived completed 0.02 trees. They do not measure fresh
initialization-to-0.02 runtime. Both variants use the same pinned CPU per pair,
balanced execution order and three warm repeats after a cold call.

Every accepted scientific field, continuation key, likelihood count and fresh
classic evidence/posterior metric is exactly equal in all 30 pairs. Geometric
mean speedup is **5.53x** (paired-seed bootstrap 95% interval 5.39–5.68x), with
**2.8% lower median paired warm peak RSS**. At capacity 857,600 (21 seeds),
median goal time is 208.84→39.28 s; at capacity 1,715,200 (9 seeds), it is
446.90→71.64 s. The corresponding median warm RSS is 17.05→16.57 GiB and
32.99→32.01 GiB. Host load varied as the cohort drained, so the bootstrap
interval does not quantify that systematic timing effect.

All outputs report classic sigma(log Z) below 0.02, but the common log-Z RMSE
is 0.52608, mean recovered spike mass is 0.27642 (truth 0.50001), and spike-mass
RMSE is 0.32213. Only 2/30 nominal classic 95% evidence intervals contain truth.
The efficient implementation preserves the old recovery, including its errors.
Archived input-tree phantom analyses are retained separately in the evidence;
no new output-tree phantom Monte Carlo reducer was run.

The follow-up performance/intent review fixed two benchmark-only issues after
freezing the cohort: lifetime RSS now accumulates across per-run marker resets,
and hashing contiguous sample arrays avoids a second full byte-string copy.
The cohort uses the valid per-run warm RSS field; its legacy lifetime field is
explicitly excluded. A fresh seed-0 0.05 smoke run of the revised harness
matches every scientific hash and accuracy metric from the cohort control.
No production sampler source changed after 650ae05.

Rebuild the compact report and figure from the retained local records:

```bash
conda run -n jaxns_py python benchmarks/phantom_seeding/report_cohort_goals.py \
  --input benchmarks/phantom_seeding/results/cohort-ss10-002-20260910 \
  --output benchmarks/phantom_seeding/cohort-ss10-002
conda run -n jaxns_py python benchmarks/phantom_seeding/plot_cohort_goals.py \
  --input benchmarks/phantom_seeding/cohort-ss10-002.json \
  --output benchmarks/phantom_seeding/cohort-ss10-002
```

## What changed

- `sampling/phantom_index.py` stores birth endpoints with cumulative valid
  interval counts and sorted likelihood endpoints for blocks of 256 chains.
  Births need not be monotone. Counts implement exactly
  `valid & (birth <= contour) & (contour < likelihood)`; zero-width or reversed
  intervals never qualify. Likelihood information determines membership only.
- The index belongs to `State`, outside the sample pytree used by sorted
  posterior views. New full blocks are sealed at acceptance. The unfinished
  tail is scanned immediately, including unpublished classic rows.
  Growth and trim preserve the cache; merge rebuilds it in the merged row order.
- Cumulative block counts locate the original proposal rank. A bounded local
  scan returns the original `row*(P+1)+slot` identity, retaining original
  transition order. Every phantom U point is still retained.
- Candidate lookup keeps the exact 64-value random draw, stratum override,
  attempt keys and first-admissible rule. Scalar conditionals resolve only the
  relevant population; a scalar retry loop stops at the first admissible rank.
  Same-contour reservations, retained start groups, exhaustion, fallback,
  scheduling, allocation and stopping are unchanged.
- New states replace the row-sorted seed cache with the block cache. Old
  checkpoints remain loadable and use the legacy row cache with lazy lookup.
  The legacy path has a concrete consumer and is retained. Index conversion
  in the benchmark is explicit; loading an old checkpoint alone does not
  promise the combined speedup.

There is no reservoir, coordinate replay, mode weighting, precision reduction,
new buffer-donation contract, or posterior use of phantom rows.

## Whole-goal results

Each measurement starts from the same immutable in-memory copy of the
completed seed-0 0.05 tree and advances exactly one goal. Reference and candidate
run serially on CPU 68 of dorrie. Three warm repeats follow one cold invocation;
every returned array is synchronized. Each repeat starts with identical keys,
accepted rows and spare capacity. Timing includes allocation, classic
publication, constrained sampling, phantom-index maintenance and state buffers.
No checkpoint writing or posterior/evidence Monte Carlo analysis is timed here.

| Problem | Reference warm median (IQR), s | Combined warm median (IQR), s | Speedup | Added likelihood calls, both |
|---|---:|---:|---:|---:|
| G10 | 51.312 (51.081–51.800) | 12.564 (12.563–12.578) | 4.08× | 2,176,542 |
| SS10 | 10.003 (9.983–10.010) | 2.745 (2.744–2.746) | 3.64× | 467,738 |

Cold calls including compilation were 65.99→28.48 s for G10 and 24.58→18.15 s
for SS10. The candidate's one-time conversion of an old checkpoint is separate
in the raw records. New runs build their index incrementally.
Measured total process high-water RSS, including loading, compilation and
fingerprint verification, was 5.630→5.381 GB for G10 and 3.397→3.330 GB for
SS10. This is a modest whole-process memory reduction, not a many-fold worker
memory reduction. Source states remain reusable; the benchmark does not donate
or invalidate them.

Fresh seed-0 runs from initialization also reproduced all three complete 0.05
paper trajectories, not just their summaries:

| Problem | Goals | Accepted classics | Likelihood calls | Classic expected log Z | Classic sigma(log Z) |
|---|---:|---:|---:|---:|---:|
| G10 | 28 | 187,833 | 60,532,854 | -14.475368641829233 | 0.04999351018297042 |
| CG10 | 29 | 195,547 | 64,690,886 | -14.58863537715857 | 0.04930224158416203 |
| SS10 | 18 | 72,357 | 16,573,991 | -24.74418864564433 | 0.04963884388154501 |

SHA-256 hashes match the archived reference for every accepted scientific
sample field, including every classic/phantom U coordinate, likelihood,
validity flag, birth, out-degree and per-row likelihood count. Continuation
keys, goal/allocation/depth counters, classic evidence summary and Kish ESS
also match. Consequently these exact trees preserve the inputs to the paper's
posterior and phantom-conditioned evidence analyses, including SS10's missed
spike mass. Those expensive analyses were not rerun.

Separate three-goal runs against the actual clean frozen source, with a
checkpoint round trip after the first candidate goal, matched the same fields
on G10, CG10 and SS10. Their cold total-time changes were only about 5%, 4%
and -1%, respectively: early compilation and likelihood work dilute the
selector gain. Full-run archive wall times are not a controlled speed baseline.

## Selector ablation and scaling

These are synchronized selector-and-coordinate-gather timings, not sampler
speedups. Each row uses the same saved tree and keys, 100 contour lanes, 99
phantom slots, dimension 10, and 11 warm repetitions. Mixed contours are birth
quantiles over the accepted history. Units below are milliseconds.

| Snapshot | Accepted N / capacity C | Reference | Lazy only | Blocks only | Combined | Combined speedup |
|---|---:|---:|---:|---:|---:|---:|
| G10 0.05 | 187,833 / 214,400 | 516.72 | 504.57 | 1,271.19 | 29.62 | 17.45× |
| CG10 0.05 | 195,547 / 214,400 | 514.34 | 510.31 | 1,281.50 | 28.83 | 17.84× |
| SS10 0.05 | 72,357 / 107,200 | 264.39 | 253.88 | 1,253.00 | 24.59 | 10.75× |
| SS10 0.02 | 468,908 / 857,600 | 1,956.82 | 1,951.85 | 1,307.28 | 53.10 | 36.85× |

Blocks alone regress on the smaller trees because eager lookup still scans
64 candidate blocks per lane. Lazy lookup alone leaves the capacity-wide
count table. The combined change removes both costs. At G10, block sizes
64/256/1024 gave 32.11/29.62/85.60 ms; 256 is the measured default among these.

| Snapshot | Compiler temporary MB, reference→combined | Measured warm RSS MB, reference→combined | Persistent state MB, reference→combined |
|---|---:|---:|---:|
| G10 0.05 | 343.24→3.39 | 2,374.57→2,034.83 | 1,227.01→1,230.55 |
| CG10 0.05 | 343.24→3.39 | 2,376.53→2,035.42 | 1,227.01→1,230.55 |
| SS10 0.05 | 171.67→1.77 | 1,536.82→1,371.41 | 613.51→615.27 |
| SS10 0.02 | 693.08→13.44 | 6,578.44→5,931.66 | 4,908.04→4,921.77 |

MB/GB are decimal. Warm RSS uses Linux VmHWM after resetting only the worker's
own high-water marker. Loading peaks are kept separately: for example, the
large SS10 pickle load peaks near 9.97 GB in both variants. Compiler temporary
bytes are XLA estimates, not total process memory. The extra persistent bytes
are birth/count metadata and block padding, not duplicate row-sorted caches.
Selector compilation remains around 2.2–2.3 s. Building a candidate index from
an existing tree, including its compilation, takes 1.40–6.87 s in this matrix.

Additional G10 probes use seven warm repeats and the same keys within each
reference/candidate pair:

| Probe | Reference ms | Combined ms | Speedup |
|---|---:|---:|---:|
| Root contours, S=100 | 522.41 | 27.83 | 18.77× |
| Recent contours, S=100 | 518.45 | 27.79 | 18.66× |
| Mixed contours, S=10 | 68.55 | 3.21 | 21.36× |
| Mixed contours, S=300 | 1,527.88 | 87.69 | 17.42× |
| P=9 prefix, D=10 | 341.59 | 9.06 | 37.71× |
| Repeated coordinates, D=20 | 512.11 | 28.42 | 18.02× |
| Doubled C=428,800, unchanged N | 979.66 | 35.11 | 27.90× |

The last three are explicitly synthetic selector-only views; they are not
new scientific runs or a 20D sampling claim. All 21 comparisons, covering
21,070 selected identities with their coordinates, stored likelihoods and
constraints, pass byte-for-byte equality. Raw quartiles are retained rather
than inferring a statistical confidence interval from seven or eleven repeats.

## Performance and code-intent review

The requested performance-and-intent skill was applied before handoff.
Review fixes included explicit common-width IDs at scalar conditionals (the
reference's `where` implicitly promoted mixed int32/int64 fixtures), preserving
Python-scalar contour inputs, cache ownership outside sample-row operations,
and rebuilding after legacy/new state merges. Static invalid block sizes fail
at JIT tracing. Comments explain the unchanged random-vector shape, why the
conditional must remain scalar, and immediate partial-tail visibility.

The optimized HLO retains scalar `conditional` nodes inside the scalar
proposal loop. At G10 it carries an `s64[100,838]` cumulative table instead of
`s64[100,214400]`. Local eligibility scans have shape `[256,99]`; candidate
batching does not materialize `[64,256,99]`. Index arrays are dynamic state
arguments, not large captured constants. Capacity growth intentionally changes
shapes; sealing another block does not. The measured executable reports no
input/output aliasing, consistent with retained immutable-state semantics.

Precision is unchanged: these paper trees use float32 U coordinates,
float64 likelihoods/contours and int64 counts/identities, with JAX x64 enabled.
No differentiated quantities or precision boundaries were changed.

Relevant tests cover 163 cases across core, distributed core, state, phantom
results, evidence10, legacy phantom seeds and the new index suite. They cover
literal Python enumeration, ties/infinities, invalid and empty populations,
nonmonotone births, partial tails, incremental sealing, growth/trim, pickle
resume, merge, mixed populations, unpublished sources, reservations and
exhaustion. A 400-identity fixture with 397 reservations exercises multiple
64-proposal retry batches against the frozen selector.

One full-process core test run hit LLVM's accumulated compilation-memory
failure after 47 passes; the remaining six core cases, including the GMM
staging case, passed in a fresh process. A phantom-result assertion that
expected U to be discarded was reproduced failing on the frozen reference;
it now checks A's retained coordinates. The associated public-result test
still verifies that phantom coordinates do not enter posterior outputs.
Ruff passes all touched Python files. Flake8 (100-column limit) passes new
files and reports no introduced findings in existing files; those existing
files retain 21 baseline findings. `git diff --check` passes.

No remaining correctness findings were identified in the exercised paths.
Performance is measured on dorrie's aarch64 CPU with JAX 0.10.0, one pinned
core per process, `jaxns_py`, CPU backend, and one OpenMP/OpenBLAS thread.
GPU performance and heavily reservation-saturated production workloads were
not measured. Near-exhaustion correctness is tested; scalar retries could
have a different performance crossover there.

The query still costs O(S C/B) block searches and full coordinates still cost
O(C P D) storage. Across an ever-growing history the accumulated query term
remains quadratic. This implementation does not solve unlimited-history
memory, checkpoint serialization peaks, or the separate evidence reducer.
The evidence supports adopting this exact implementation as the next baseline;
it does not justify raising the worker pool or restarting the cancelled ladder.
It also does not establish a new scientific advantage over R: it preserves
full A's existing, problem-dependent behavior.

## Reproduction

Use explicit `PYTHONPATH`: the shared conda environment otherwise imports a
different checkout. Run commands from the implementation worktree. The archive
paths below are read-only inputs; choose fresh output names.

```bash
export PHANTOM_WORKTREE="$PWD"
export PHANTOM_REFERENCE=/largedata/albert/git/jaxns-paper-evidence10-AC300
export PHANTOM_ARCHIVE=/largedata/albert/jaxns-evidence10-AC300-20260909
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Repeat for reference, lazy, blocks and combined, and each saved problem.
env PYTHONPATH="$PHANTOM_WORKTREE/src:$PHANTOM_WORKTREE" taskset -c 68 \
  conda run --no-capture-output -n jaxns_py python benchmarks/phantom_seeding/selector.py \
  --state "$PHANTOM_ARCHIVE/0.05/g10/seed-00/state.pkl" \
  --variant combined --block-size 256 --repeats 11 --output /tmp/g10-selector.json

# Actual frozen source for complete-goal reference timing.
env PYTHONPATH="$PHANTOM_REFERENCE/src:$PHANTOM_REFERENCE:$PHANTOM_WORKTREE" taskset -c 68 \
  conda run --no-capture-output -n jaxns_py python benchmarks/phantom_seeding/resume_goal.py \
  --state "$PHANTOM_ARCHIVE/0.05/g10/seed-00/state.pkl" \
  --variant reference --output /tmp/g10-goal-reference.json
# Use worktree PYTHONPATH and --variant combined for the matching candidate.

env PYTHONPATH="$PHANTOM_WORKTREE/src:$PHANTOM_WORKTREE" taskset -c 68 \
  conda run --no-capture-output -n jaxns_py python benchmarks/phantom_seeding/run_case.py \
  --case g10 --variant combined --uncertainty 0.05 --save-state --output /tmp/g10-complete

# Focused index suite; other recorded groups use the same environment.
env PYTHONPATH="$PHANTOM_WORKTREE/src:$PHANTOM_WORKTREE" taskset -c 64 \
  conda run --no-capture-output -n jaxns_py pytest -q \
  cicd/tests/test_phantom_index.py cicd/tests/test_phantom_seeds.py
```

Other pytest groups were `test_core.py`, `test_distributed_core.py`, and
`test_state.py test_phantom_results.py`, split into fresh processes as
described above. Run `ruff check` on the changed files and
`flake8 --max-line-length=100` on the new index, new tests and benchmark files.
`verify_archive.py` checks the complete saved runs at their documented local
paths. `collect_measurements.py` rechecks selector bytes and whole-run hashes
and regenerates the compact evidence file from `results/`.

## Integration onto develop (2026-09-11)

The PR is based on develop `772cc2b`. It brings in full A and the exact
optimization; develop previously discarded phantom coordinates. The measured
5.53x comparison above is against the frozen full-A reference, not against
classic-only develop. Collecting phantoms now retains their coordinates and
makes them available as seeds; public posterior samples remain classic-only.

The port preserves develop's additive allocation and its default increments.
It does not bring over the experiment branch's weighted-allocation API or its
older multiplicative uniform schedule. The benchmark discovery pass uses an
additive increment of 300 on develop and the original multiplier of one on
the frozen reference, so both use the same root-count sequence.

[Integration evidence](develop-integration.json) records new G10, CG10 and
SS10 three-goal runs with a checkpoint round trip. Every scientific array
fingerprint, continuation key, counter and accuracy metric matches the recorded
frozen-reference run exactly. A new complete SS10 goal from archived seed 0
at 0.02 also matches all scientific fields, keys, likelihood counts and fresh
accuracy metrics. Its three warm runs took 30.58–30.77 s (median 30.63 s);
this integration check has no new contemporaneous reference timing and does
not replace the balanced 30-seed comparison.

All 186 integration regression cases passed: 84 phantom-index/seed/result/state
cases, 54 core cases in three bounded compilation processes, and 48 distributed
core/allocation cases. Ruff passes all touched Python files. Flake8 with the
100-column convention passes new files; existing touched files have 21 baseline
findings, down from 22, with none introduced. The performance-and-intent review
also made benchmark model/metric dependencies self-contained, preserved the
frozen model module names needed by saved pickles, and avoided a full byte-string
copy during fresh-run hashing. No precision or array-ownership contract changed.

Exactness is demonstrated against full A in the exercised paths. This PR
preserves A's measured problem-dependent recovery; it does not claim a new
independence guarantee or solve the remaining O(CPD) retained-coordinate memory.
Related issue #292 describes an earlier bounded-reservoir proposal; this PR
implements the subsequently agreed exact full-population design in Ticket 0021.
