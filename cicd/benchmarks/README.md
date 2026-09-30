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

The original distributed-runtime decision evidence is recorded on issue 252.
The topology-aware scalar-thread evidence is maintained in
`benchmarks/issue_267/`: thirty paired accuracy seeds with phantoms off/on,
scientific work, compile/warm and steady wall time, and complete-chain scaling
across explicit CPU worker counts. The authenticated TCP lifecycle is covered
by a real process integration test. GPU batching is not claimed without GPU
measurements.
