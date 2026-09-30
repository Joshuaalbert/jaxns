# Ellipsoidal-direction benchmark

This benchmark compares an explicitly fitted, frozen GMM direction kernel with
the isotropic reference on every maintained standard problem. GMM cells first
run isotropically until the user-selected expected-log-evidence uncertainty,
call `state.fit_gmm_directions(iso_prob=...)` once using every stored classic,
and explicitly resume. No sampler configuration or automatic staging selects
the fit point. Each matrix cell uses the same 30 seeds, model, root allocation
(`30 * dimension`), replacement width (`10 * dimension`), slice count
(`5 * dimension`), and final termination condition. Phantom-off evidence uses
classic shrinkage; phantom-on evidence uses phantom-conditioned Monte Carlo
shrinkage.

Run the full matrix from the repository root:

```bash
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/mplconfig \
  conda run -n jaxns_py python benchmarks/issue_246/run_suite.py \
  --output-dir /tmp/issue-246-results
```

Aggregate it with:

```bash
PYTHONPATH=src:. conda run -n jaxns_py python \
  benchmarks/issue_246/summarise.py /tmp/issue-246-results \
  --csv benchmarks/issue_246/results.csv \
  --markdown /tmp/issue-246-table.md
```

The committed `REPORT.md`, `results.csv`, `comparisons.csv`, and `raw/`
directory were produced with this protocol.

`n` is the number of independent, paired nested-sampling seeds in one table
row. `run_s` includes compilation for seed zero; `median_steady_run_s` excludes
seed zero. Accuracy is based on Monte Carlo shrinkage estimates. The
spike--slab rows additionally report posterior mode-mass RMSE because correct
relative mode weights are part of those problems' design. Likelihood
evaluations remain the hardware-independent primary efficiency measure.

The low-level cost of the former device-conditional placement can be compared
with a Python-boundary prototype using the exact GMM update payload:

```bash
PYTHONPATH=src:. conda run -n jaxns_py python \
  benchmarks/issue_246/compare_fit_placement.py
```
