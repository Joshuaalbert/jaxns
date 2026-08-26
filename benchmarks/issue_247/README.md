# Issue 247 accuracy and performance evidence

This directory contains the version-isolated benchmark used for the review
gate. `run_v2_standard.py` targets both the PyPI wheel and `main`; the two
sources are measured through different import roots. `run_current_standard.py`
measures this implementation and exercises its explicit classic-only and
phantom-conditioned MC evidence APIs.

## Predeclared comparison

- Cases: every case in `cicd/tests/test_ns_standard_problems.py`.
- Modes: phantom retention disabled and enabled.
- Seeds: `0` through `29`; the MC shrinkage key is deterministically folded
  from each run key and uses 1,000 draws.
- Common settings: v2's default 30 independent Markov chains per dimension,
  five isotropic perfect slice transitions per dimension,
  `dZ/Z = log1p(1e-3)`, float64, and CPU backend. Recording phantoms does not
  change the current race tree. The current core retains `D` early-chain
  states without treating them as depth-loop live points and vmaps 10
  replacement chains per dimension. The width was selected before the final
  matrix from an explicit
  10D sweep: S=100 took 20.9 s versus 215.6 s at the former S=150 because the
  vmapped rejection loop advances to the slowest lane.
- V2 is configured through `c=30D`, not `num_live_points`: the latter is
  divided by `k + 1` in v2 and would silently compare fewer independent chains
  when phantoms are enabled. V2 otherwise uses its released maximum-sample
  default.
- Scientific release gate: no non-finite runs and no failures of the unchanged
  `3 sigma` expectation and `2 sigma` MC truth checks in
  `cicd/tests/test_ns_standard_problems.py`. The independent 30-seed matrix is a
  descriptive calibration check: bias, RMSE, coverage, and every failure are
  reported without selecting or dropping runs.
- Runtime non-inferiority: compare warmed end-to-end run plus result conversion
  at the same stopping goal. A 10% median regression beyond run-to-run IQR is
  material. Likelihood evaluations, ESS/evaluation, compilation, warmed depth
  execution, HLO text size, compiler memory analysis, and process peak RSS are
  reported separately so cheap-likelihood dispatch overhead cannot hide
  scientific throughput.

The first seed in each process includes any call-site cache population not
represented by explicit lowering/compilation. Later seeds are warmed. Every
timed JAX value is synchronised with `jax.block_until_ready()`.

## Baseline isolation

The latest PyPI release was resolved with:

```bash
conda run -n jaxns_py pip index versions jaxns
conda run -n jaxns_py pip install --no-deps \
  --target /tmp/jaxns_v2_269 jaxns==2.6.9
```

`origin/main` was archived at commit `2f356d6` into
`/tmp/jaxns_main_2f356d6`. The package trees are byte-identical after excluding
wheel cache directories:

```bash
diff -qr --exclude=__pycache__ \
  /tmp/jaxns_main_2f356d6/src/jaxns /tmp/jaxns_v2_269/jaxns
```

Each JSON record includes the imported distribution version and absolute
`jaxns.__file__` path. A PyPI invocation is:

```bash
cd /tmp
PYTHONPATH=/tmp/jaxns_v2_269:/home/albert/git/jaxns \
MPLCONFIGDIR=/tmp/matplotlib-issue247 \
conda run -n jaxns_py python \
  /home/albert/git/jaxns/benchmarks/issue_247/run_v2_standard.py \
  --case spike_slab --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 \
  --mc-draws 1000
```

The `main` command substitutes
`PYTHONPATH=/tmp/jaxns_main_2f356d6/src:/home/albert/git/jaxns` and adds
`--implementation-label main --source-id 2f356d6`. The current invocation is:

```bash
conda run -n jaxns_py python \
  benchmarks/issue_247/run_current_standard.py \
  --case spike_slab \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 \
  --mc-draws 1000
```

Add `--phantoms` for the retained-phantom mode and wrap an individual command
with `/usr/bin/time -v` for isolated process peak RSS. Aggregate raw JSONL with:

```bash
conda run -n jaxns_py python benchmarks/issue_247/summarise.py \
  benchmarks/issue_247/results/final_v2/*.jsonl \
  benchmarks/issue_247/results/final_comparison/*.jsonl
```

Regenerate the representative Spike–Slab cornerplots and diagnostics with:

```bash
conda run -n jaxns_py python \
  benchmarks/issue_247/generate_diagnostics.py
```

The tracked [REPORT.md](REPORT.md) records the exact hardware/software
observations and measured residual risks after the final review pass.
