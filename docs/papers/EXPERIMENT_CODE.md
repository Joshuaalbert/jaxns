# Paper experiment code

The scripts in this checkout cover several stages of the paper's experimental
development. The dated directories preserve the model choices and input
layouts of their respective cohorts.

## Reusable benchmark and entry points

- [`../../benchmarks/paper_phantom_conditioning/`](../../benchmarks/paper_phantom_conditioning/)
  contains the early 8D model definitions, checkpoint runner, phantom-prefix
  reduction, posterior diagnostics, summaries, calibration studies and plots.
  Its README describes the protocol and command-line usage.
- [`run_experiments.py`](run_experiments.py) orchestrates that early benchmark
  in separate sampling, reduction, verification, summary and figure phases.
  Use `--dry-run` to inspect the commands without starting experiments.
- [`plot_mean_preview.py`](plot_mean_preview.py) draws the illustrative
  Gaussian, curved-Gaussian and spike--slab problems without running a sampler.

## Dated cohort scripts

| Directory under `paper-results/` | Purpose |
| --- | --- |
| `classic-seed-mean23-20260920/` | Mean `(2, -3)` cohort validation, rendering and remote handoff |
| `cg10-uniform-mean23-20260920/` | Uniform-allocation comparison and remote handoff |
| `classic-seed-mean22-20260921/` | Alternative mean `(2, -2)` launch, reference calculation and comparison |
| `results-claims-20260922/` | Baseline and phantom-seed tables and figures, including historical mode-recovery comparisons |
| `results-claims-mean23-20260922/` | Completed 10D prefix comparisons and the D10/D20/D40 comparison |

The renderers consume the corresponding cohort records, summaries and
manifests. Those inputs, durable checkpoints, source archives and generated
figures are separate experiment artifacts and are not supplied by this code
change. Restore the matching cohort directory before running a renderer.
The scripts retain their original filenames and output schemas so that they
can be used with those records.

The dated launch, resume and packaging scripts are records of operations on
dorrie. They retain absolute `/largedata/albert/...` paths and require their
original execution environment. In particular, some import
`benchmarks.classic_seed_evidence` from the corresponding archived execution
source, rather than from `develop`. Restore that source and its recorded
environment to replay those operations. They are not generic local launchers.

The experimental phantom-seed sampler implementation is kept on its separate
branch. Publishing these scripts does not add phantom seeding to the library.

## Regression checks

```bash
conda run -n jaxns_py python -m pytest -q cicd/tests/test_paper_experiment.py
```

The tests guard classic posterior mode-mass calculation, reject misaligned
membership arrays and prefix-specific posterior diagnostics, and check that
the default runner constructs its sampler using the `develop` interface.
