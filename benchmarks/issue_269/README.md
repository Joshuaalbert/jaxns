# Issue 269 mode-survival sweep

This benchmark asks whether repeated lineage allocations eventually recover
the weak component of the maintained 8D `spike_slab` problem. It uses a fixed
initial root degree of 240, allocation increments of 80, shell size 80, and the
existing ellipsoidal slice sampler. Only the Python goal is tightened; the
compiled depth condition remains `dlogZ = log1p(1e-3)`.

The transition-location run for persistent failing seed 4 is reproduced with:

```bash
MPLBACKEND=Agg PYTHONPATH=src:. \
  conda run -n jaxns_py python \
  benchmarks/issue_269/run_goal_sweep.py \
  --case spike_slab \
  --seed 4 \
  --targets 0.02,0.016 \
  --max-samples 2000000 \
  --output benchmarks/issue_269/seed4_goal_sweep.json
```

The runner checkpoints after each uncertainty target and resumes the exact
same scientific state. It refuses to overwrite an existing output. Use a new
path to repeat a run.

See `REPORT.md` for the earlier ten-seed screen, the transition result, and its
limits. The checked-in `seed4_goal_sweep.json` is the raw transition artifact.
