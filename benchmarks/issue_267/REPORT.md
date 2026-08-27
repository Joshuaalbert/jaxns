# Issue 267 evidence report

Environment: JAX/JAXLIB 0.10.0, CPU, x64 enabled. Raw per-seed and per-repeat
records are in `results.json`, `standard_spike_slab.json`,
`standard_spike_slab_phantoms.json`, `standard_worker_likelihood_smoke.json`,
and `throughput.json`; the commands are described in `README.md`.

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
and distributed runners in this scalar-worker control. This check originally
motivated stable task-ID commit order: accepting whichever network result
arrived first had allowed phantom payload latency to choose a different classic
trajectory.

## Representative ellipsoidal standard problem

The transport control above is intentionally cheap and one-dimensional. The
representative comparison uses the maintained 8D spike--slab problem with 240
root lineages, allocation increment 80, 40 perfect-bracketing slice
transitions, four GMM components, the resolved 144-effective-sample initial
fit gate, ten warm-refinement iterations, population size 1,024, and a 1%
isotropic safety probability. GMM fitting and contour-triggered refits happen
inside the timed scientific run. The local runner uses its measured width-80
replacement `vmap`; the distributed runner uses the best smoke-tested topology
on this CPU, one worker with `batch_size = 3`.

Here `n` is the number of independent measured run seeds; compile/warm seed
`-1` is excluded. Final evidence values use 1,000 Monte Carlo shrinkage draws.
Phantoms-off rows use classic conditioning, while phantoms-on rows use phantom
conditioning only in this final calculation. Both runners use the same
expectation-based depth/goal termination. The mode-mass diagnostic is computed
from the ordinary user-facing expectation result, so it measures whether the
sampler retained the spike and slab with appropriate classic posterior weight;
it is not presented as a phantom-conditioned posterior diagnostic.

| Problem | Runner | n | Expectation bias | SE bias | Expectation RMS | MC bias | MC RMS | Mean MC z | SD MC z | MC coverage | Mode-mass bias | Mode-mass RMS | Core run s median [IQR] | End-to-end s median [IQR] | Likelihood evals median [IQR] | ESS median [IQR] | Likelihood evals / ESS median [IQR] | GMM updates median [IQR] | Isotropic directions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spike_slab (phantoms off) | local | 30 | +0.02926 | 0.04627 | 0.25086 | +0.02999 | 0.24976 | +0.13 | 1.11 | 90.0% | +0.05599 | 0.17614 | 1.809 [1.766, 1.853] | 3.461 [3.393, 3.564] | 641,324 [585,972, 661,084] | 863.2 [850.4, 881.2] | 732.7 [684.9, 758.5] | 3.0 [2.0, 3.0] | 0.639 |
| spike_slab (phantoms off) | distributed | 30 | -0.03875 | 0.04830 | 0.26299 | -0.03826 | 0.26404 | -0.15 | 1.15 | 90.0% | -0.02501 | 0.12752 | 29.299 [29.027, 29.678] | 30.857 [30.627, 31.245] | 586,207 [577,688, 629,428] | 869.7 [860.3, 881.9] | 679.1 [662.5, 744.9] | 3.0 [3.0, 3.0] | 0.639 |
| spike_slab (phantoms on) | local | 30 | +0.02926 | 0.04627 | 0.25086 | +0.01336 | 0.25385 | +0.09 | 1.69 | 83.3% | +0.05599 | 0.17614 | 1.812 [1.758, 1.911] | 7.680 [7.480, 7.770] | 641,324 [585,972, 661,084] | 863.2 [850.4, 881.2] | 732.7 [684.9, 758.5] | 3.0 [2.0, 3.0] | 0.639 |
| spike_slab (phantoms on) | distributed | 30 | -0.03875 | 0.04830 | 0.26299 | -0.03146 | 0.26444 | -0.20 | 1.73 | 80.0% | -0.02501 | 0.12752 | 29.452 [29.246, 29.733] | 35.043 [34.763, 35.418] | 586,207 [577,688, 629,428] | 869.7 [860.3, 881.9] | 679.1 [662.5, 744.9] | 3.0 [3.0, 3.0] | 0.639 |

The paired off/on classic trajectories are exactly identical in all 60
runner/seed pairs for classic log evidence, sample count, likelihood count,
mode mass, GMM updates, direction count, and isotropic-direction count. Thus
phantom payload transport no longer changes the scientific sampling path.

After moving root likelihood evaluation out of the scientific process, the
current protocol replayed spike--slab seed 0 as a focused regression. Every
scientific field for both local and distributed runners is bitwise identical
to its corresponding record in the 30-seed artifact: log evidence, MC evidence,
sample and likelihood counts, ESS, mode mass, GMM updates, direction counts,
and isotropic counts. The current distributed core time was 29.676 s versus
29.802 s in the earlier record, within ordinary run-to-run timing variation.
This smoke is recorded in `standard_worker_likelihood_smoke.json`; the 30-seed
calibration statistics therefore remain applicable to the unchanged
scientific trajectories.

The local/distributed expectation-bias difference is 0.0680, approximately
one combined standard error (0.0669), and expectation RMS differs by 0.0121.
This does not resolve an accuracy regression at 30 seeds. The distributed
trajectory uses about 7% fewer likelihood evaluations per posterior ESS and
has lower mode-mass RMS, but it is 16.2 times slower in core wall time on this
fast CPU likelihood. Those efficiency differences describe different valid
allocation trajectories and are not claimed as a topology-invariant speedup.

Phantom conditioning does not improve aggregate evidence RMS in this
configuration, while its narrower conditional shrinkage uncertainty increases
z-score dispersion to 1.69 locally and 1.73 distributed. On the local runs,
phantom evidence error correlates 0.76 with mode-mass error. Splitting the runs
at median absolute mode error gives phantom z SD 0.94 in the better-sampled
half and 1.85 in the worse-sampled half; phantom RMS improves from 0.225 to
0.195 in the former but worsens from 0.272 to 0.301 in the latter. This is
evidence that phantom shrinkage can be calibrated conditional on adequate mode
representation, but does not model constrained-sampler mode-weight error. The
current spike--slab setting therefore does not support a claim of calibrated
end-to-end phantom uncertainty. Follow-up calibration and API decisions are
tracked in #269.

The isotropic-direction column includes both the explicit 1% safety draws and
the required fallback when no fitted component has a sampled peak above the
current strict contour. It is therefore expected to exceed 1%; reporting it
prevents a nominally ellipsoidal benchmark from hiding that much of a run used
the safe kernel.

The representative vmapped worker exposed a stricter version of the same
invariant. Worker lanes already complete together, but the supervisor formerly
sent their result payloads as separate messages. Retained phantom bytes could
therefore change whether the next refill arrived within the worker's bounded
batch-fill interval, changing later `vmap` groups despite stable task-ID commit
order. Protocol version 5 returns and replays a complete worker assignment as
one protocol group, and distinguishes root likelihood operations from complete
constrained-chain operations. The scientific core commits its contiguous
stable-ID prefix before refill. Scalar task identity, acknowledgements, and
exact retry remain unchanged. The phantoms-off/on matrix reported here is a paired
trajectory check for that transport invariant.

## Scheduler and GMM interaction

An equal-work scheduler control used 300 already-registered ten-slice tasks,
two workers with `batch_size = 3`, and seven repeats. Task IDs, random keys,
contours, seeds, numerical result digests, and likelihood-evaluation counts
were identical within every comparison. With zero synthetic planner latency,
prequeueing the immutable ready manifest reduced median time from 0.291 s to
0.236 s (23.7%), while a Python I/O thread regressed to 0.319 s. At 0.5 ms of
planner work per commit the threaded refill was best (0.284 s versus 0.368 s,
22.9% faster), and at 2 ms it was 0.649 s versus 0.804 s (19.3% faster). Thus a
thread has marginal value only when there is planner work to overlap; the
existing asynchronous coordinator already handles transport concurrency.

The end-to-end growth candidate cannot hold scientific work fixed because
every returned edge is a valid race-tree observation and must be retained. On
the representative spike--slab smoke seed, queuing every visible allocation
head advanced the contour frontier continuously: GMM updates rose from 3 to
66, likelihood evaluations from 584,165 to 706,605, and wall time from 53.2 s
to 72.0 s. It was therefore not selected from a one-seed smoke as a release
policy. This does not prove it has no value for a seconds-per-likelihood
cluster workload; #267 remains open for repeated physical-node Pareto evidence
over wall time, likelihood work, evidence calibration, and posterior mode
weights.

## Wall time and complete-chain throughput

The analytic model is intentionally too cheap for process distribution.
Median steady wall time was 2.63 ms local (IQR 2.56--2.74) versus 40.50 ms
distributed (39.10--42.35) with phantoms off, and 2.66 ms (2.55--2.76) versus
42.70 ms (39.88--45.26) with phantoms on. The distributed path is about
15--16x slower here. This is a fixed transport/Python overhead stress case and
supports keeping distributed execution opt-in for expensive likelihoods.
Keeping one registration from worker-only root evaluation through constrained
sampling and moving each planning window to the host once before scalar
serialization avoid additional fixed overhead.

A separate real perfect-bracketing slice benchmark ran 300 logical threads per
repeat after compilation, for seven repeats. Each sample used ten slice
transitions; no no-op sampler replaced scientific work. The distributed and
local measurements below were collected together on the same host.

| Scalar CPU workers | Median samples/s | IQR samples/s | Mean samples/s | Median speedup |
|---:|---:|---:|---:|---:|
| 1 | 682.3 | 645.7--687.8 | 665.1 | 1.00x |
| 2 | 970.8 | 917.4--1,001.9 | 957.2 | 1.42x |
| 3 | 905.5 | 888.2--913.3 | 905.1 | 1.33x |

Two processes improve median throughput by about 42% on this host. A third is
about 7% slower than two at the median and consumes another process's memory;
their IQRs overlap. This is evidence for explicit
per-node worker configuration rather than automatically launching one process
per apparent device or core. No GPU was available, so no GPU batching or
scaling claim is made.

The same sampler calls were then executed in the standard core's execution
shape: `sample_request` vmapped at a fixed width inside one compiled JAX loop.
Every width received the same scalar random-key stream and performed exactly
3,000 logical likelihood evaluations per repeat.

The final sampler-interface review compared the pre-refactor PR commit with
the explicit `AbstractSampler` contract added to remove concrete-type
inspection. At width 12, both lowered HLOs contained 1,170 lines and seven
while operations, and every one of seven repeats performed exactly 3,000
likelihood evaluations. The pre-refactor median was 151,773 samples/s and the
explicit-interface median was 183,188 samples/s; compile-and-warm measurements
were 0.869 s and 0.892 s respectively. These sub-millisecond repeats are too
variable to claim a speedup, but they exclude a resolved execution regression.
The review initially found a larger 1,190-line HLO because isotropic direction
counters had accidentally become observable; restoring their existing zero
diagnostic contract returned the HLO operation structure to the reference.

| Local vmap width | Median samples/s | IQR samples/s | Mean samples/s | Median speedup |
|---:|---:|---:|---:|---:|
| 1 | 66,904 | 62,253--67,480 | 62,630 | 1.00x |
| 2 | 57,827 | 57,718--100,635 | 77,346 | 0.86x |
| 3 | 93,639 | 81,842--130,074 | 105,270 | 1.40x |
| 4 | 121,429 | 104,834--124,762 | 114,092 | 1.81x |
| 6 | 157,596 | 156,036--159,874 | 157,205 | 2.36x |
| 10 | 147,520 | 143,292--148,446 | 142,446 | 2.21x |
| 12 | 177,964 | 139,555--181,942 | 162,815 | 2.66x |

Width 12 had the highest measured median. It was 2.66x faster than the local
scalar execution shape and 183.3x faster than the best distributed result.
This is a sampler-execution-layer comparison, not an end-to-end nested-sampler
speedup: allocation, sorting, and result construction are intentionally absent.
The existing end-to-end cheap-model comparison above shows the smaller but
still material process/transport penalty.

The constraint in this throughput control admits the entire one-dimensional
prior, so all chains take exactly the same ten likelihood evaluations. It is
therefore the best case for `vmap` and deliberately contains no slowest-lane
penalty. The measured 2.66x local batching gain is sufficient to reject a
scalar-only distributed protocol: removing worker-local batching would remove
a useful execution shape. Variable-rejection evidence should instead guide a
user's choice of worker `batch_size`, for which scalar execution remains
available as size one.

Worker-reported process RSS high-water marks were approximately 354 MiB for one
worker, 355/356 MiB for two, and 353/355/355 MiB for three. These are measured
per-process RSS, not unique physical-memory measurements because shared pages
may be counted in more than one process. This linear worker-state cost is
another reason not to start more processes than measured throughput justifies.

The same run measures the GMM direction-state compatibility change selected in
this PR. Giving each task byte-distinct fit counters while holding every key,
seed, contour, and direction-defining array fixed took a median 0.850 s
(IQR 0.833--0.862) when the full sampler state split batching, versus 0.679 s
(0.670--0.709) when observational counters were excluded. That is 20.2% lower
wall time with exactly 3,000 likelihood evaluations and matching numerical
outputs in every repeat.

The real lifecycle integration additionally runs a complete nested-sampling
depth over trusted-network loopback TCP, dynamically adds a remote node, kills
and automatically replaces its worker under a fresh lease, and drains/removes
it without changing scientific ownership. It also removes the only node during
an active run, leaves the scientific process starved for five times its 0.1 s
coordinator-health timeout, then rejoins the node and completes the same run
without a distributed exception or a changed checkpoint. Coordinator restart
requests are retained by exact worker instance and replayed on the node's next
outbound supervisor heartbeat. Loopback is protocol evidence, not a physical
inter-machine latency claim.
