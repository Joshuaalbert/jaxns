# Issue 267 distributed topology benchmark

This maintained comparison isolates the scheduling and transport change on an
analytic one-dimensional problem. Local and distributed runs use the same
model, run keys, root degree 6, allocation increment 6, three perfect-bracket
slice steps, `dlogZ=0.1`, and finite sample budget. The distributed setting has
three scalar CPU workers so worker-local batching cannot change the chain law.

The seed-zero compile/warm run is recorded but excluded from steady summaries.
Thirty measured seeds are run with phantom collection off and on:

```bash
PYTHONPATH=src conda run -n jaxns_py python \
  benchmarks/issue_267/run_comparison.py \
  --seeds 30 \
  --output benchmarks/issue_267/results.json
```

This cheap model is a transport-overhead stress case, not the workload for
which distributed execution is recommended. The unchanged-tolerance standard
problem suite remains the release accuracy gate.

Complete-chain pool throughput and the matched local-core `vmap` scaling are
reproduced with:

```bash
PYTHONPATH=src conda run -n jaxns_py python \
  benchmarks/issue_267/run_throughput.py \
  --tasks 300 \
  --repeats 7 \
  --output benchmarks/issue_267/throughput.json
```

See `REPORT.md` for the reviewed tables and interpretation.
