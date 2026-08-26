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

A separate real perfect-bracketing slice benchmark queued 300 scalar logical
threads per repeat after compilation, for seven repeats. Each sample used ten
slice transitions; no no-op sampler replaced scientific work.

| Scalar CPU workers | Median samples/s | IQR samples/s | Mean samples/s | Median speedup |
|---:|---:|---:|---:|---:|
| 1 | 577.1 | 559.3--583.8 | 571.6 | 1.00x |
| 2 | 840.3 | 805.9--874.5 | 847.3 | 1.46x |
| 3 | 723.2 | 700.1--768.3 | 734.0 | 1.25x |

Two processes improve throughput materially on this host. A third contends for
the same CPU and regresses from the two-worker optimum, which is evidence for
explicit per-node worker configuration rather than automatically launching one
process per apparent device or core. No GPU was available, so no GPU batching
or scaling claim is made.

Worker-reported process RSS high-water marks were 353 MiB for one worker,
353/353 MiB for two, and 356/356/359 MiB for three. The arithmetic process sums
were 353, 706, and 1,071 MiB respectively; these are measured per-process RSS,
not unique physical-memory measurements because shared pages may be counted in
more than one process. This linear worker-state cost is another reason not to
start more processes than measured throughput justifies.

The real lifecycle integration additionally runs a complete nested-sampling
depth over authenticated loopback TCP, dynamically adds the remote node after
the coordinator is ready, and drains/removes it without changing scientific
ownership. Loopback is protocol evidence, not a physical inter-machine latency
claim.
