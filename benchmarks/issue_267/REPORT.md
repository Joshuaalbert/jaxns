# Issue 267 evidence report

Environment: JAX/JAXLIB 0.10.0, CPU, x64 enabled. Raw per-seed and per-repeat
records are in `results.json` and `throughput.json`; the commands are described
in `README.md`.

## Accuracy and scientific cost

Thirty paired seeds used the analytic truth `log Z = -0.1333792237`, root
degree 6, allocation increment 6, three perfect-bracket slices, and matched
`dlogZ=0.1` termination. `±` is the standard error of measured bias.

| Phantoms | Runner | Bias log Z | RMS log Z | SD log Z | Mean likelihood evaluations | Mean retained phantoms |
|---|---|---:|---:|---:|---:|---:|
| off | local | +0.00166 ± 0.00981 | 0.05283 | 0.05371 | 116.30 | 0.00 |
| off | 3 async workers | +0.00220 ± 0.00996 | 0.05366 | 0.05453 | 124.73 | 0.00 |
| on | local | +0.00166 ± 0.00981 | 0.05283 | 0.05371 | 116.30 | 28.53 |
| on | 3 async workers | +0.00220 ± 0.00996 | 0.05366 | 0.05453 | 124.73 | 29.20 |

The distributed bias difference is 0.00054, far below the approximately 0.010
standard error, and its RMS differs by 0.00082. Conservative in-flight lineage
reservations cost 7.25% more likelihood evaluations, improving on the roughly
10% overhead measured for the first local-only distributed implementation.

Scheduler credit was also selected from 30-seed comparisons rather than by
assuming more speculative work would improve throughput. One task per live
lane gave 124.73 mean evaluations, bias +0.00220, RMS 0.05366, and 46.18 ms
mean wall time. Adding one pool-wide spare task gave 140.13 evaluations, bias
+0.00491, RMS 0.05501, and 51.12 ms; adding one spare per lane gave 162.60
evaluations, bias +0.00757, RMS 0.05605, and 52.74 ms. Both prefetch variants
were dominated on this diagnostic, so the implementation fills all measured
worker lanes immediately but does not create a speculative shell beyond them.

Phantom collection changes neither classic log evidence nor likelihood count:
the maximum paired difference over all 30 seeds is exactly zero for both local
and distributed runners. This check caught and motivated stable task-ID commit
order: accepting whichever network result arrived first had allowed phantom
payload latency to choose a different classic trajectory.

## Wall time and complete-chain throughput

The analytic model is intentionally too cheap for process distribution.
Median steady wall time was 3.47 ms local (IQR 3.28--3.75) versus 45.92 ms
distributed (43.33--47.79) with phantoms off, and 3.50 ms (3.34--3.82) versus
46.64 ms (45.10--48.91) with phantoms on. The distributed path is about 13.3x
slower here. This is a fixed transport/Python overhead stress case and supports
keeping distributed execution opt-in for expensive likelihoods. Moving each
planning window to the host once before scalar serialization reduced this
overhead from the earlier 17x review measurement.

A separate real perfect-bracketing slice benchmark ran 300 logical threads per
repeat after compilation, for seven repeats. Each sample used ten slice
transitions; no no-op sampler replaced scientific work. The distributed and
local measurements below were collected together on the same host.

| Scalar CPU workers | Median samples/s | IQR samples/s | Mean samples/s | Median speedup |
|---:|---:|---:|---:|---:|
| 1 | 556.4 | 552.2--557.7 | 559.6 | 1.00x |
| 2 | 805.0 | 785.8--819.0 | 792.7 | 1.45x |
| 3 | 645.1 | 607.6--679.4 | 660.2 | 1.16x |

Two processes improve throughput materially on this host. A third contends for
the same CPU and regresses from the two-worker optimum, which is evidence for
explicit per-node worker configuration rather than automatically launching one
process per apparent device or core. No GPU was available, so no GPU batching
or scaling claim is made.

The same sampler calls were then executed in the standard core's execution
shape: `sample_request` vmapped at a fixed width inside one compiled JAX loop.
Every width received the same scalar random-key stream and performed exactly
3,000 logical likelihood evaluations per repeat.

| Local vmap width | Median samples/s | IQR samples/s | Mean samples/s | Median speedup |
|---:|---:|---:|---:|---:|
| 1 | 31,592 | 31,377--33,236 | 31,760 | 1.00x |
| 2 | 50,782 | 48,917--55,800 | 53,777 | 1.61x |
| 3 | 80,477 | 75,608--83,631 | 82,203 | 2.55x |
| 4 | 62,089 | 56,808--70,269 | 63,030 | 1.97x |
| 6 | 100,359 | 99,912--117,859 | 106,388 | 3.18x |
| 10 | 91,057 | 85,955--91,565 | 89,430 | 2.88x |
| 12 | 93,346 | 85,467--111,978 | 100,904 | 2.95x |

Width six had the highest measured median. It was 3.18x faster than the local
scalar execution shape and 124.7x faster than the best distributed result.
This is a sampler-execution-layer comparison, not an end-to-end nested-sampler
speedup: allocation, sorting, and result construction are intentionally absent.
The existing end-to-end cheap-model comparison above shows the smaller but
still material process/transport penalty.

The constraint in this throughput control admits the entire one-dimensional
prior, so all chains take exactly the same ten likelihood evaluations. It is
therefore the best case for `vmap` and deliberately contains no slowest-lane
penalty. The measured 3.18x local batching gain is sufficient to reject a
scalar-only distributed protocol: removing worker-local batching would remove
a useful execution shape. Variable-rejection evidence should instead guide a
user's choice of worker `batch_size`, for which scalar execution remains
available as size one.

Worker-reported process RSS high-water marks were approximately 354 MiB for one
worker, 355/363 MiB for two, and 356/354/360 MiB for three. These are measured
per-process RSS, not unique physical-memory measurements because shared pages
may be counted in more than one process. This linear worker-state cost is
another reason not to start more processes than measured throughput justifies.

The real lifecycle integration additionally runs a complete nested-sampling
depth over authenticated loopback TCP, dynamically adds the remote node after
the coordinator is ready, and drains/removes it without changing scientific
ownership. Loopback is protocol evidence, not a physical inter-machine latency
claim.
