# Frozen contour step tuning on the stiff ridge

Protocol dated 2026-10-27, before calibration or evaluation. Baseline is exactly
`4e9cc2927b2d4ab4c50529af00201a4fd9960fe8`. This benchmark-only experiment
implements the coordinator's handoff and review-01. No production changes.

## Target and controlled comparison

Reuse `run_ridge.prior_model(1e-5, 0.3)`: normalized 18D unit-cube model with
15 Gaussian residuals (x0 and x2:16), three flat coordinates (x1,x16,x17),
and true log evidence effectively zero. Independently check its normalization
and NumPy values. The unit-Jacobian shear gives, while the ball stays inside
the cube,

V(g) = 1e-5 * 0.05^14 * pi^(15/2) * (2g)^(15/2) / Gamma(17/2).

Thus log V = constant + (15/2) log g, and one compression e-fold multiplies
g by exp(-2/15), not exp(-1/15). This is a toy-specific analytic contour
coordinate, not a general compression estimator. The schedule uses the actual
model's fixed peak minus the PARENT contour; never the seed value or position.

Fresh ARM controls: fixed h=0.003 versus one frozen pilot schedule, each with
uniform and evidence_improving allocation; all seeds 0--29 in all four cells.
Use roots=150, replacement_width=150, delta_K=150, default depth
log1p(1e-3), unlimited growth. Nine cycles of one isotropic perfect-bracket
slice and ten midpoint Hug reflections; true gradients, independent Gaussian
unnormalized velocity and U(0.8,1.2) time jitter per trajectory. No rescue,
path-dependent stopping, retry-until-accept, or other kernel tuning.

## Calibration (independent of evaluation)

Forty contours g_k=49 exp(-2 k/15), k=0,...,39. At each contour independently,
start h=0.003 and make exactly 64 Robbins--Monro updates. Each update uses
256 independent exact constrained draws and ONE complete B=10 reflection
trajectory per draw, with fresh velocity and jitter and no slice step.
Endpoint strict-support acceptance is the Bernoulli observation. Set
eta_t=0.8/(1+t/8)^0.6, t=0,...,63, then
log h_{t+1}=clip(log h_t + eta_t*(mean_acceptance-0.7), log(1e-6), log(0.03)).
Freeze each node to exp(mean(log h_t for t=48,...,63)), before seeing any
held-out validation or evaluation outcome. Fixed budget: 655,360 paths,
6,553,600 gradient calls and 655,360 endpoint value calls; independent-draw
construction/initial values are accounted separately. No data-dependent stop.

Use log-linear interpolation of h in log g between nodes. Clamp below the
smallest positive node to its h, including g<=0 (no valid constrained seed
exists at/above the peak). Above g=49 use common fixed h=0.003, including
parent contour -infinity. A discontinuity at the early-region boundary is
allowed: h is fixed at a parent contour for the entire chain and trajectory.
Non-finite contours other than -infinity are errors at external boundaries.
The schedule is NOT adapted online. Calibration arrays and schedule hashes
are frozen before launching evaluation nested runs. Pilot arrays never enter
evidence or posterior inference. Disclose total pilot work and amortization.

RNG namespaces: pilot NumPy SeedSequence [3122701,k,t]; JAX fold-in root
3122702 by k and t, then split into 256 independent paths. Validation uses
NumPy [3122703,case,stream] and JAX root3122704. Evaluation remains PRNGKey(seed).
No calibration choices use any evaluation result.

## Correctness and held-out checks

Run unchanged reflection reversal/volume/stationarity, allocation observer,
and phantom evidence reference tests. Add frozen-schedule tests for tails,
constant-schedule exact RNG/output parity, strict support, reversibility,
self-transitions, counts, true f64 model tracing, and phantom off/on classic
scientific-state plus continuation-key parity. CPU backend asserted.

After freezing calibration, use 8192 fresh independent seeds per contour at
5,15,40 for both arms. Separately measure single-path acceptance with Wilson
95% intervals, and validate full nine-cycle chain stationarity. Record every
one of the 19 exact marginal/radial PIT KS statistics; fixed acceptance gate
max KS <5/sqrt(8192). Acceptance near70% is descriptive, not a stationarity
proof or a basis for retuning. Record movements and correlations in cube units.
Compare common early h at gaps100,1000 and the whole prior (-infinity contour)
on fresh exact draws. For early gaps use rejection from a valid enclosing ball
or box, never untruncated ball draws; validate support. Compare to a separate
exact sample using two-sample KS on all coordinates, Gaussian residuals and
radius, with bound5*sqrt(2/8192). Whole prior uses exact coordinate uniform PITs.
Early acceptance is explicitly outside the tuned regime. No tests are dropped
if they fail; diagnose scientific correctness before evaluation dispatch.

## Evaluation, diagnostics, and evidence

Save snapshots at first COMPLETE batch crossing 2M and4M total logical model
calls, counting roots and each value-and-gradient once. Primary comparison4M;
no ESS/uncertainty stopping. Reuse `allocation_observer` without changes: the
production driver owns allocation and growth; physical pauses do not create
epochs. Match initial prior hashes across all four arms for each seed.

Retain eight complete-cycle endpoints (rejected repeats included), excluding
the final child. No midpoint phantoms, no phantom seeds. Save classic arrays,
seed/parent/batch identities and per-chain RNG keys, evidence inputs, continuation
state at final budget, source manifests, counts, progress, and checksums.
Measure actual path acceptance by deterministic post-run diagnostic replay of
saved chain keys/seeds, separately accounting replay time/model/gradient work.
Replay must reproduce recorded classic endpoints and phantom likelihoods; no
replay result enters the continuing tree. This extra offline work is disclosed.

Evidence uses the existing `numpy_allocation_evidence` backend unchanged,
4096 draws/snapshot/mode, batches64, C_min20, shared cluster weights and the
production Kish gate. Independent RNG root3122705 folded by seed, budget,
allocation index (uniform=0,evidence=1), arm (fixed=0,tuned=1), mode(0/1);
use its uint32 words as NumPy SeedSequence entropy. Archive complete draws,
gates, inputs and MCSE. Retain both classic moment and classic MC estimators
so a convention change cannot masquerade as phantom conditioning.

Audit every archive independently: hashes/source, dtype, cube/support,
NumPy likelihood tolerance5e-5 (existing scalar-constant rounding tolerance),
root/call/gradient counts, first budget crossing and overshoot, matched priors,
seed eligibility, moments/weight ESS, and phantom cluster identities/support.
Report logZ bias, error SD, RMSE, bootstrap SE/interval, mean quoted SD,
95% Gaussian coverage using1.959963984540054 and MC percentile coverage.
Use 100000 paired seed bootstrap draws, RNG3122706, all30 seeds. Primary
contrasts tuned-minus-fixed separately per allocation and estimator; retain
both budgets and phantom-minus-classic contrasts without choosing a winner.

Report x1,x16,x17,x5 posterior SD, weighted CDF sup error and Wasserstein
error, classic Kish weight ESS (not mixing ESS), and actual seed-to-child RMS,
pair correlations, h and path acceptance in parent gap bins (0,5],(5,15],
(15,40],(40,100],(100,1000],>1000. Require >=30 pairs for per-bin movement
summaries. Report absolute displacement per gradient and any normalized
versions against TRUE marginal scales, not collapsed empirical widths.
Report calls/overshoot, gradients, elapsed/compilation timings, completed
epochs and the peak-based remaining-evidence diagnostic of the baseline.

## Execution and source freeze

CPU ONLY: JAX_PLATFORMS=cpu, CUDA_VISIBLE_DEVICES='', x64; enforce actual
JAXCTX base_dtype float64 before tracing. Assert backend, model/sample dtype,
and imported source. Conda jaxns_py, one pinned CPU per worker; BLAS/OpenMP
threads1. Initial tests/pilot use cores2--5 with measured near-zero activity;
unrelated unpinned process stays untouched. Start evaluation with four workers;
increase to at most24 disjoint CPUs only after measuring memory/runtime/load.
Keep >=64GiB MemAvailable and >=20GiB free on output filesystem. Never change
current workers to improve utilization. Scientific protocol is independent of
process concurrency. Every write is atomic where publication matters; refuse
to overwrite completed archives. No manuscript, GitHub, or production edits.

Outputs: `/largedata/albert/jaxns-reflection-tuning-20261027/`; worktree:
`/largedata/albert/git/jaxns-reflection-tuning-20261027`. Freeze protocol,
calibration and executed source hashes before evaluation. Coordinator status:
`/tmp/jaxns-reflection-tuning-status-20261027.md`. Scientific deviations require
an explicit recorded amendment before the affected work, never silent changes.
