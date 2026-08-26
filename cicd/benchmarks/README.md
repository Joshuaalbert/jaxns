# Maintained Benchmarks

Benchmarks measure production paths. Model construction, validation, JAX
compilation, and warm-up must be reported separately from timed steady-state
execution. Every result records JAX/JAXLIB versions, device, precision,
sampler budgets, termination condition, random seeds, and phantom setting.

The current standard-problem accuracy and performance suite is maintained in
`benchmarks/issue_247/`. Its report is per problem and compares the latest v2
release with the current implementation under matched termination,
live/root-point, and slice settings, with phantom collection both off and on.
Release-quality summaries use approximately 30 independent seeds per problem.

Run the existing matrix from the repository root:

```bash
conda run -n jaxns_py bash benchmarks/issue_247/run_matrix.sh
```

Benchmarks are reviewed evidence, not timing assertions on shared CI runners.

The distributed-runtime decision evidence is recorded on issue 252. Its
maintained benchmark must report complete-chain throughput for scalar and
vmapped workers, asynchronous versus barrier scheduling under homogeneous and
uneven task latency, compile time, IPC round-trip cost, and per-process peak
memory. GPU batching is not claimed without GPU measurements.
