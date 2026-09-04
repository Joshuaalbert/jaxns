# Phantom-seed reservoir benchmark

This benchmark measures whether letting retained Markov-chain states
participate in stationary seed selection improves exploration without an
unacceptable efficiency or memory cost. It uses maintained standard-problem
builders and the existing issue-246 mode-mass diagnostic rather than copying
scientific model definitions.

The two arms have identical models, seeds, allocation, isotropic direction
kernel, `5D` slice transitions, and `D` retained phantom slots per classic
sample. Both arms collect those phantoms. The only changed option is
`phantom_seeding`. Final evidence is always sampled from the classic race model
with 2,048 Monte Carlo shrinkage draws, so phantom conditioning cannot be
mistaken for an exploration improvement.

The default suite has 30 paired seeds for the maintained Gaussian, curved
Gaussian, spike--slab, and curved spike--slab problems. Run it from the
repository root:

```bash
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/jaxns-mplconfig \
  conda run -n jaxns_py python benchmarks/issue_292/run_suite.py \
  --output-dir /tmp/issue-292-results \
  --source-id issue-292-candidate
```

Each static arm runs in its own process and executes one unreported cold seed
before measurement. Five-seed AB/BA blocks alternate arm order within each
problem to control machine drift. `warm_wall_s` excludes compilation. Isolated
lowering, compilation, HLO size, and XLA temporary-memory measurements come
from `profile_hotpath.py` so cold cost is not pseudoreplicated across the five
scientific seeds in a timing block.
The benchmark records each run's classic-MC log-evidence error, uncertainty,
and z-score; component-zero and minor-mode mass; practical mode loss;
likelihood evaluations; classic samples; goal loops; wall time; resident state
bytes; bounded-pool occupancy; and eligible pool counts at the 90th, 99th, and
final observed birth-contour percentiles. Resident state bytes are the exact
sum of Pytree array-buffer sizes, not process RSS. Runs use opt-in unlimited
storage and fail rather than report evidence after a terminal stop. A
practical mode loss means the recovered posterior mass of the analytically
smaller component is below 10% of its true mass.

An exploratory fixed-source mixture can be selected without changing the
public API by passing `--phantom-seed-probability`. The retained comparison
should use the evidence-selected production value.

Profile the isolated compiled depth program for both feature paths with:

```bash
for arm in off on; do
  PYTHONPATH=src:. conda run -n jaxns_py python \
    benchmarks/issue_292/profile_hotpath.py \
    --phantom-seeding "$arm" \
    --output "/tmp/issue-292-hotpath-$arm.json"
done
```

Validate the pairing and create per-arm summaries plus 10,000-resample paired
bootstrap intervals with:

```bash
PYTHONPATH=src:. conda run -n jaxns_py python \
  benchmarks/issue_292/summarise.py /tmp/issue-292-results \
  --json /tmp/issue-292-summary.json \
  --markdown /tmp/issue-292-summary.md
```

The primary scientific comparison is the paired change in log-evidence RMSE.
For mode problems, mode-mass RMSE and practical loss rate show whether any gain
comes from preserving structure. Likelihood evaluations are the primary
hardware-independent efficiency measure. Warm wall time, classic-sample count,
goal-loop count, state size, and pool occupancy explain the implementation
cost and scaling mechanism.
