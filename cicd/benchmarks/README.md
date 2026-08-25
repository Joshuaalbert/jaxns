# Maintained Benchmarks

Benchmarks measure production paths. Model construction, validation, JAX
compilation, and warm-up must be reported separately from timed steady-state
execution. Every result records JAX/JAXLIB versions, device, precision,
sampler budgets, termination condition, random seeds, and phantom setting.

The standard-problem accuracy and performance release gate is maintained in
`benchmarks/v2_v3/`. It compares exact v2 and v3 commits in separate pinned
environments under matched termination, root allocation, slice effort, seeds,
precision, and hardware. Its schema rejects missing or duplicate seeds,
non-finite runs, mixed commits, and unmatched scientific settings. Summaries
are per problem and report posterior and mode-weight accuracy as well as
evidence, calibration, likelihood evaluations, memory, and separated timing.

Prepare isolated environments and run both sides from the repository root:

```bash
workspace=/tmp/jaxns-v2-v3-release
bash benchmarks/v2_v3/prepare_environments.sh "${workspace}" v2 HEAD
bash benchmarks/v2_v3/run_release_matrix.sh v2 "${workspace}" "${workspace}/results"
bash benchmarks/v2_v3/run_release_matrix.sh v3 "${workspace}" "${workspace}/results"
conda run --prefix "${workspace}/env-v3" python \
  benchmarks/v2_v3/summarise.py --release-gate \
  "${workspace}"/results/*.jsonl
```

The initial 1,200-run v3.0 evidence, raw JSONL, report, and diagnostic plots are
retained under `benchmarks/issue_247/`. Benchmarks are reviewed evidence, not
timing assertions on shared CI runners.
