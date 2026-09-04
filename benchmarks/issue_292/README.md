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

Each static arm runs in its own process and executes one unreported warm-up
seed before measurement. `warm_wall_s` therefore excludes first compilation.
The suite alternates which arm runs first across problems to reduce systematic
machine-drift bias.
The benchmark records each run's classic-MC log-evidence error, uncertainty,
and z-score; component-zero and minor-mode mass; practical mode loss;
likelihood evaluations; classic samples; goal loops; wall time; resident state
bytes; and bounded-pool occupancy. Resident state bytes are the exact sum of
Pytree array-buffer sizes, not process RSS. A practical mode loss means the
recovered posterior mass of the analytically smaller component is below 10% of
its true mass.

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
