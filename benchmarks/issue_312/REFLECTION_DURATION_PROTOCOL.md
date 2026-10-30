# Reflection resolution versus duration: frozen protocol

2026-10-27; frozen before new held-out or evaluation outcomes. Exact baseline
295893708805ff668178fc443cbda63faf2ac75d. Benchmark/test/report scope only.
No new calibration, core changes, manuscript edits or GitHub publication.

## Scientific comparison

Use the unchanged run_ridge.prior_model(1e-5,0.3), 18D unit cube, with real
float64 unit coordinates and the existing TFP float32(0.05) moderate scale.
The mathematical reference logZ=0 and existing independent NumPy likelihood
reference/tolerance5e-5 stay unchanged. All arms use fresh ARM runs.

Three arms, each uniform and evidence_improving allocation:
- fixed: original ReflectionSampler, h=.003, B=10, nominal duration .03.
- short: original FrozenContourReflectionSampler, old h(parent), B=10.
- duration: same old h(parent) is desired resolution; B=ceil(.03/h), then
  actual nominal h=.03/B. Both parameters are frozen for the whole chain.
  For parent peak-gap>49, including -infinity parent, explicitly B10/h.003.
  Keep old minimum-gap clamp. Apply existing independent U(.8,1.2) jitter
  AFTER integer rounding, giving total trajectory time .03*jitter.

Reuse schedule SHA256
21242cb5554c0b7c23c4750d1c8aa579e3d9856ff670a0f9c18133b1771a28d6.
This is not newly calibrated to70% acceptance. No seed/position/velocity/outcome
selection of B/h, rescue, early path stop, retries, or retuning. Report ceil
quantization explicitly: h is never larger than desired h; B*h=.03 up to
floating-point rounding. Expected B10--152 (interior13--152). At gaps
.1/5/15/40/49 the old schedule gives B152/37/22/14/13; early B10.
Report full node/interior rounding range and do not alter it after evaluation.
The original pilot cost is inherited overhead, not newly executed work.

Nine cycles, one ordinary isotropic perfect-bracket slice per cycle; Gaussian
unnormalized velocity refresh and independent trajectory jitter unchanged.
Retain eight complete-cycle endpoints excluding the final child, including
rejected repeats. Phantoms have neither seed access nor posterior weights.
Roots150, replacement width150, delta_K150, depth log1p(.001), unlimited growth.

## Seeds, budgets and reductions

Seeds30--59 in every arm/allocation:180 fresh runs,360 snapshots. Matched initial
prior hashes across all6 cells for each seed. Evaluation key=PRNGKey(seed).
First complete batch crossing2M and4M logical model calls;4M primary,2M secondary.
No ESS/uncertainty stopping. Original allocation_observer semantics unchanged;
physical pauses are not scientific epochs. Count all gradient calls once in
model-call totals, and retain value-only work separately. Per-child logical
gradients=B(parent)*9; roots have zero gradients. Audit stored per-child B/h,
gradients against exact replay and an independent NumPy schedule calculation.
Heterogeneous vmapped loops may execute masked hardware work; logical counts
are not hardware instruction counts, so report wall time separately.

Evidence backend numpy_allocation_evidence unchanged:4096 draws/mode/snapshot,
batch64, C_min20, shared cluster Gamma weights, existing Kish gate. Retain
classic moments, classic MC and phantom MC estimator conventions separately.
Evidence key3122713 folded by seed,budget,allocation index(uniform0,evidence1),
arm index(fixed0,short1,duration2),condition flag; uint32 words seed NumPy RNG.
Save every draw/input. Independent evidence RNG cannot affect sampling.

100000 paired bootstrap seed resamples, NumPy RNG3122714, indices shape
(100000,30); use same indices for all contrasts. Primary duration-minus-short
and duration-minus-fixed per allocation and estimator; secondary short-minus-
fixed. Retain both budgets and all phantom-minus-classic MC contrasts.
Pointwise95% percentile intervals, no multiplicity-adjusted or winner claim.
All seeds included; no selecting acceptance/settings using evaluation results.

## Correctness and held-out gates

Before evaluation: coordinator reviews protocol, source and validation; freeze
executed source/protocol/old-schedule hashes. Existing reflection, observer,
phantom evidence and frozen-short tests pass unchanged. Added tests cover
constant-B10 exact control/RNG parity, old short control, ceil/tails/clamp,
heterogeneous-count vmap, actual float64/CPU, gradients/value-call accounting,
strict support/rejected repeats, and reversal with the same fixed h/B/jitter
including B152. Preserve existing position2e-9/velocity2e-8 reversal tolerances;
unexpected failures must be diagnosed and discussed before evaluation.
Phantom off/on must preserve every other classic/continuation leaf for both
allocations, including RNG and scheduling. Test checkpoint/growth continuation
with heterogeneous contours/counts. No silent tolerance relaxation.

Fresh exact independent ensembles8192 at gaps.1,5,15,40,100,1000,infinity,
all3 arms,21 cases. NumPy SeedSequence[3122711,case,stream] with stream0
initial and stream1 reference; JAX3122712 fold case, split single-path/chain
then8192 keys. Same initial points/keys across arms. One complete path measures
acceptance; separately run full9-cycle chains. Save raw initial/reference,
proposed and accepted endpoints, rejection decisions, keys, B,h,actual jittered
h and duration, every cycle endpoint, gradients/evaluations. Early arms must
be exactly equal including RNG/counts. Rejections remain self-transitions.

At gaps<50 use the same exact15-ball/residual draws; all19 marginal/radial
PIT KS statistics must satisfy maxKS<5/sqrt8192. At finite broad contours use
correct cube-truncated exact reference and all coordinate/residual/radius
2-sample KS, maxKS<5*sqrt(2/8192). Whole prior uses coordinate uniform PITs.
Reference normalization/precision and support unchanged. Report single-path
Wilson95% acceptance, full-chain acceptance, absolute movement and ESJD per
actual gradient; acceptance is descriptive and never a retuning gate.

Both allocations need safe duration smoke runs with complete exact replay
before evaluation. Non-evaluation smoke seeds3122718/3122719, snapshots30k/60k.
Replay from saved keys/seeds must exactly reproduce child coordinates,
likelihoods, calls and all8 phantom likelihoods; gradients=B(parent)*9.
Replay work/time is separately reported overhead, never part of sampling.

## Outcomes, provenance and execution

Report bias, across-seed error SD/RMSE, bootstrap SE/intervals, quoted SD,
Gaussian95% and MC percentile coverage, per-seed evidence MCSE, calls/overshoot,
gradients/value-only work, timings, epochs and remaining-tail diagnostic.
Posterior SD, CDF sup error and W1 for x1,x16,x17,x5; Kish ESS is weight ESS.
Movement by the existing parent-gap bins(0,5],(5,15],(15,40],(40,100],
(100,1000],>1000: acceptance,B,h,T,absolute RMS,endpoint correlation. Require
>=30 pairs for per-bin movement. ESJD per gradient is sum squared displacement
/ sum actual gradients, not RMS divided by a fixed90 or normalized by a
collapsed cloud. Report aggregate sums and denominators for reproducibility.

CPU ONLY: JAX_PLATFORMS=cpu, CUDA_VISIBLE_DEVICES='', JAX_ENABLE_X64=1;
OMP/OPENBLAS/MKL threads1. Pin before JAX import; assert source/backend/actual
unit dtype. One CPU per worker, start4, maximum24 after measured memory/runtime.
ReserveCPU70 for coordinator audits; tests/reporting on other idle cores.
Preserve unrelated processes; maintain>=64GiB available RAM and>=20GiB disk.
Use robust detached controller, immutable completion markers and atomic arrays/
state/JSON with SHA256 sidecars. No overwriting completed runs. Preserve full
final State, keys/parent/seed/batch IDs, sources, manifests, inputs/draws and
logs. Failures are diagnosed for the exact case without protocol changes.

Worktree:/largedata/albert/git/jaxns-reflection-duration-20261027.
Outputs:/largedata/albert/jaxns-reflection-duration-20261027.
Status:/tmp/jaxns-reflection-duration-status-20261027.md.
Coordinator independently audits and publishes; deliver local source/report
commit and checksummed incremental2958937..HEAD bundle. Figures PDF/PNG paths,
no image base64. Prior experiment is preserved untouched.
