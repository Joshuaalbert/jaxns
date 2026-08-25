# Issue 255 packed-parameter evaluation

This directory contains the evidence used to decide whether JAXNS should
replace its named Bayesian U-space pytree with `jaxctx.ParameterPack`.
`REPORT.md` records the decision and review; `results/` retains every timing
sample rather than only the medians quoted in the report.

The benchmark distinguishes four representations because they do not have
the same semantics:

- `current_u_pytree`: the current named `CtxParams` U-space leaves.
- `transport_flat_u`: a flat input reconstructed into the current U pytree at
  an execution boundary.
- `hypothetical_packed_u`: a benchmark-only uniformly sampled U leaf, sliced
  through the same priors. This is the sampling API that jaxctx does not
  currently provide.
- `parameter_pack_public_n`: the released optimization API, supplied with
  out-of-band inverse-transformed N values solely for matched transform
  timings. It is not a Bayesian sampling implementation.

The documented commands put this worktree's `src/` first on `sys.path`, and
the runners fail if the imported JAXNS source is elsewhere. They also set
Matplotlib's backend to `Agg`; the commands repeat that setting explicitly so
test or dependency imports cannot open windows.

Run the semantic check with:

```bash
PYTHONPATH=src MPLBACKEND=Agg JAX_PLATFORMS=cpu \
timeout --signal=TERM 60s \
  conda run --no-capture-output -n jaxns_py python \
  benchmarks/issue_255/check_sampling_semantics.py
```

Run one compiler/runtime/serialization case with:

```bash
PYTHONPATH=src MPLBACKEND=Agg JAX_PLATFORMS=cpu \
timeout --signal=TERM 240s \
  conda run --no-capture-output -n jaxns_py python \
  benchmarks/issue_255/benchmark_parameter_pack.py \
  --case d64_l64_b1
```

The recorded matrix used `d64_l1_b1`, `d1024_l1_b1`, `d64_l64_b1`, and
`d96_l8_b16_hetero`. Each process records five cache-cleared traces, three
cache-cleared lower/compile samples, 51 synchronized compiled calls, and 101
pickle protocol-5 host round trips. The `repeat_*.jsonl` records are complete
independent reruns used to expose sensitivity on this shared CPU host.

Run the bounded end-to-end sampler comparison with:

```bash
PYTHONPATH=src MPLBACKEND=Agg JAX_PLATFORMS=cpu \
timeout --signal=TERM 180s \
  conda run --no-capture-output -n jaxns_py python \
  benchmarks/issue_255/benchmark_full_nested_sampler.py
```

That runner warms both variants, then alternates execution order by seed for
30 synchronized seeds. The comparison is intentionally bounded and cheap; it
exercises the sampler hot path but is not a production accuracy run.
