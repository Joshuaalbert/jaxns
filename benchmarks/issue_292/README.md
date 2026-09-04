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

## Selected policy

The retained design uses one representative per completed chain, planned
contour slots, capacity `d0`, and a fixed phantom-source probability of 0.1.
A point reservoir was rejected because chains that emit more correlated rows
would receive more seed probability. A value-independent global cluster
reservoir was also rejected: only 3--5 entries remained eligible near the
final contours in exploratory runs. Planned slots raised terminal eligibility
to roughly 87--95 clusters for replacement width 80.

Capacity and mixture sweeps used paired random seeds. Capacity `d0` was the
coverage knee: smaller pools degraded the basic-MVN efficiency, while `2d0`
added work without scientific benefit. Over 30 paired seeds, the selected 0.1
mixture changed basic-MVN log-evidence RMSE from 0.2522 to 0.2195 and CSS8 RMSE
from 0.1692 to 0.1612. CSS8 standardized-error variance changed from 1.179 to
1.046, mode-mass RMSE was effectively unchanged (0.0923 to 0.0934), and mean
likelihood evaluations changed by +1.38%. A 0.5 mixture improved the unimodal
case but worsened CSS8 RMSE to 0.1865, so it was rejected. The selected pool
added 35,520 resident bytes and 36,521 bytes to a committed checkpoint.

The resident/checkpoint scaling profile isolates the two seed-pool banks from
ordinary sample and phantom-likelihood storage. For the standard float32
homogeneous coordinates and float64 scientific arithmetic, an `R`-entry pool
occupies exactly `R * (8D + 84)` bytes. Production uses `R = d0 = 30D`, hence
the pool costs `240D^2 + 2520D` bytes and remains independent of the completed
sample count. Paired measurements at sample capacities 10,000 and 100,000
were identical:

| Dimension | Pool capacity | Resident increment | Checkpoint increment |
|---:|---:|---:|---:|
| 2 | 60 | 6,000 B | 6,959 B |
| 8 | 240 | 35,520 B | 36,479 B |
| 32 | 960 | 326,400 B | 327,405 B |
| 128 | 3,840 | 4,254,720 B | 4,255,761 B |

The checkpoint figures are for one generation. The checkpoint manager retains
two generations at steady state, so durable disk overhead can be approximately
twice those increments. Reproduce this profile with:

```bash
PYTHONPATH=src:. conda run -n jaxns_py python \
  benchmarks/issue_292/profile_memory.py --checkpoint \
  --output /tmp/issue-292-memory.jsonl
```

Four isolated AB/BA hot-path profiles at `D=8`, `d0=240`, and replacement
width 80 measured an 8.9% median depth-runtime increase for the enabled path.
HLO text grew by 23.5%, median compilation time by 28.8%, and scheduled state
by 2.69%; XLA temporary bytes decreased by 13.3%. The disabled path constructs no
pool, and an exact paired run reproduced its parent-revision samples, evidence,
likelihood evaluations, and ESS bit for bit.

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
