# Issue 270 Checkpoint Evidence

## Question

Does adding opt-in automatic full-state checkpointing change ordinary run
performance, and is one full checkpoint per hour a material cost?

## Method

`run_checkpoint.py` was run in separate Python processes against baseline
commit `7c34ca7` and this branch. Both used Python 3.12.9, JAX/JAXLIB 0.10.0,
x64, and one CPU device.

The disabled-path comparison used the public `NestedSampler.run_until_goal`
path with no checkpoint directory. The deterministic workload had eight
dimensions, root degree 16, replacement width 8, and physical capacity 256.
Compilation was warmed before 200 measured runs per version. Each paired seed
produced the same valid count, root degree, and summed likelihood signature.

The opt-in measurement saved an actual complete `State` enlarged to physical
capacity 1,000,000. Each save included Pytree pickle serialization, file
fsync, SHA-256, atomic state publication, directory fsync, atomic manifest
publication, and old-generation pruning. Five generations were measured on
the local filesystem.

## Results

| Measurement | Baseline | Branch | Branch / baseline |
|---|---:|---:|---:|
| Compile + first run | 1.74044 s | 1.74059 s | 1.00009 |
| Steady median, 200 runs | 3.335 ms | 3.275 ms | 0.9820 |
| Steady mean, 200 runs | 3.335 ms | 3.309 ms | 0.9923 |
| Scientific signatures | exact | exact | — |

There is no measured regression when `checkpoint_dir` is omitted. The JAX
depth program itself is unchanged; the disabled path adds only host-side
optional-argument routing at the outer boundary.

The 1,000,000-capacity state occupied 104,001,927 bytes. Full atomic saves took
0.5717–0.6172 seconds, with a 0.6014-second median. At one save per hour this
is 0.0167% of wall time.

## Interpretation

The evidence supports the selected design: checkpointing remains entirely
opt-in, the ordinary scientific path has no material performance change, and
even a roughly 104 MB full state has negligible amortized cost at the default
one-hour cadence. Filesystem behavior will differ on networked storage, so the
measured absolute save time should not be treated as an NFS or parallel-filesystem
claim; cadence remains user-configurable for those environments.

Raw summary values are retained in `results.json`; the benchmark script emits
full per-run timings when reproduced.
