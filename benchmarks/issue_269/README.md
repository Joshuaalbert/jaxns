# Issue 269: allocation cadence and multimodal survival

This benchmark tests the hypothesis recorded in issue #269: a smaller initial
root allocation and finer uniform allocation increments may give an explicit
uncertainty-driven outer goal more opportunities to add stationary threads,
preserving or rediscovering a weak mode before the final uncertainty target is
reached.

The screen keeps the maintained 8D `spike_slab` problem, ellipsoidal direction
law, stopping rule, and hard 24,000-sample ceiling fixed.  It varies only the
initial root degree, allocation increment, and (for one control) static shell
width:

| Schedule | Root degree | `delta_K` | Shell width | Purpose |
| --- | ---: | ---: | ---: | --- |
| `baseline` | 240 | 80 | 80 | Maintained start and coarse growth |
| `smaller_root_coarse` | 160 | 80 | 80 | Isolate the smaller initial allocation |
| `smaller_root_medium_fixed` | 160 | 40 | 80 | Medium cadence at fixed compiled width |
| `smaller_root_fine_fixed` | 160 | 20 | 80 | Fine cadence at fixed compiled width |
| `smaller_root_medium_matched` | 160 | 40 | 40 | Practical medium-width execution |
| `smaller_root_fine_matched` | 160 | 20 | 20 | Practical fine-width execution |
| `aggressive` | 120 | 20 | 20 | Stress case below the 144-sample first-fit gate |

The fixed-width rows isolate scientific behavior because changing shell width
also changes compiled batching and random-key grouping.  The matched-width row
measures the practical cost of avoiding inactive sampler lanes.  The
aggressive row is deliberately not a default candidate: it tests whether
delaying the first ellipsoid fit overwhelms any benefit from finer allocation.

The ordinary `run()` goal completes one allocation round, so it cannot test
this hypothesis. The runner instead continues until the expectation register
reports `log_Z_uncert <= 0.18`, while retaining the maintained `dlogZ` depth
condition. The tuning pilot required 3 baseline rounds, 4 coarse smaller-root
rounds, 7 medium rounds, 13 fine rounds, and 15 aggressive rounds.

Each record contains final evidence, posterior mode mass, sampler work, goal
success, and per-goal append boundaries. The epoch records distinguish newly
allocated full-prior roots (`log_L_constraint == -inf`) from constrained
descendants, and report component counts and maximum likelihoods in stable
append order.

Run the local CPU screen from the repository root with a hard timeout and a
non-interactive Matplotlib backend:

```bash
timeout 1800s env MPLBACKEND=Agg JAX_PLATFORMS=cpu PYTHONPATH=src:. \
  conda run --no-capture-output -n jaxns_py \
  python benchmarks/issue_269/run_allocation_matrix.py \
  --seeds 30 --output /tmp/issue-269-screen.json
```

Interrupted runs can be continued with `--resume`.  Summarise a completed
matrix with:

```bash
conda run --no-capture-output -n jaxns_py \
  python benchmarks/issue_269/summarise.py \
  /tmp/issue-269-screen.json --output /tmp/issue-269-screen.md
```

The preregistered promotion gate is a paired-bootstrap estimate of at least a
20% reduction in posterior mode-mass RMS error whose interval excludes zero,
without more than a 20% regression in evidence RMS error or likelihood
evaluations per effective posterior sample.

The reviewed run will be retained in `results.json`; its interpretation and
decision will be recorded in `REPORT.md` after the corrected screen completes.
