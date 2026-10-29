# Online reflection resolution and duration — proposed protocol

2026-10-29. Review version 3, incorporating coordinator reviews01--03 before training
or evaluation (versions1--2 are archived separately). Baseline: 1d84f157496f5b7a20659fdc5da9ca41fd8cec09. The coordinator
must approve the final source and validation before large evaluation. Changes
to this proposal are recorded before affected work, not selected from results.

## Scientific scope and controls

Reuse the unchanged 18D run_ridge.prior_model(1e-5, 0.3), normalized likelihood,
float64 cube and original TFP moderate scale. Initial roots 150, replacement
width 150, delta_K 150, unlimited sample growth, depth log1p(0.001). Both
uniform and evidence_improving allocations retain their existing semantics.

Every child has nine cycles, one ordinary full-chord isotropic slice per cycle,
Gaussian velocity refresh, independent Uniform(0.8, 1.2) trajectory jitter,
strict endpoint rejection and retained self-transitions. Exactly eight complete
cycle phantoms exclude the final child. Phantoms supply evidence conditioning
only; no coordinates enter seed selection or controller feedback.

Three fresh arms: fixed (h=0.003, B=10, nominal T=0.03); online (controller below,
starting at that same setting); frozen (independently trained contour policy,
no evaluation feedback). Frozen retains the same 90/10 exploration mixture as
online, isolating policy adaptation from randomized parameter choice. None
uses whitening, a surrogate, exact posterior coordinates, evidence truth,
acceptance targets, additional scouting likelihoods or repeated feedback chains.

The online arm is experimental adaptive MCMC. Freezing a selected setting for
one chain establishes only the conditional fixed-kernel construction, not
stationarity of history-dependent adaptation. Frozen comparisons and exact
ensemble checks do not prove that the online race tree is unbiased.

## Controller coordinate and chronological update clock

Use only the classic race tree available BEFORE dispatch. Reuse build_block_state
and classic_dirichlet_concentrations. For each actual planned parent contour
lambda, compute c(lambda) = -sum_{L_g <= lambda} log(alpha_gt / alpha_total).
This is minus log expected classic volume, not an oracle likelihood-gap map.
Also retain incoming K of its parent block (the sentinel uses root degree).
The exact planned parent values, c and K are computed without model calls.

The policy has 65 fixed compression knots c=0,2,...,128, storing log desired h
and log nominal T. The observation stratum is floor(c/2), clamped to 0..63;
report out-of-range frequency. Compare perturbations only within the SAME
two-e-fold stratum and unchanged local policy generation. No pooling across
different allocations, seeds, controller generations or widely separated
contours. The two-e-fold width is a prespecified compromise on exploration
counts; report each stratum's observed contour/compression range.

Each new, completed, nonroot chain contributes exposure log1p(1/K_parent)
to its stratum. This is an expected-shrinkage INFORMATION-EXPOSURE clock:
it is approximately one compression e-fold per K new chains at that contour.
It is not a claim that revisiting a contour advances the deepest frontier.
A revisit adds genuinely new completed-chain observations once, using its
pre-dispatch c/K. Backward contour movement and allocation-round changes do
not independently trigger updates. Initial and later roots contribute no
feedback/exposure. Record deepest observed c separately.

A stratum becomes eligible only after exposure >=1, at least 64 central
observations, and at least eight observations for EACH sign of a parameter's
perturbation. Each parameter can update independently if its own pair passes.
No catch-up updates from the same observations. Retain both cumulative and
pending exposure, attempted/eligible/successful updates and per-stratum
suppression reasons. Require a smoke/preflight to exercise real updates; if
none occur, report the feasibility problem before training/evaluation. Missing-count cases remain
unchanged and are reported. The observer adds a whole completed batch before
computing changes; all changes affect only the NEXT batch. A change resets
pending statistics in every stratum whose interpolated predictor changes;
raw observations and all old sufficient statistics remain archived.

## Parameter choice and objective

For every child, draw one independent parameter-choice uniform from
fold_in(chain_key, 3122901), leaving the original chain key unchanged. Probabilities:
0.90 central; 0.025 h/1.5; 0.025 h*1.5; 0.025 T/1.5; 0.025 T*1.5.
The draw and the frozen controller determine parameters before observing seed
coordinates, velocity or outcomes. The whole nine-cycle chain uses that choice.
Only the existing independent per-cycle trajectory jitter changes realised time.

Interpolate log h/log T linearly between learned compression anchors. With no
anchors use the fixed setting. Outside their range extend the nearest slope,
clipped to +/-log(1.25) per compression e-fold, for at most two e-folds, then
hold constant. With one anchor hold constant. Evaluate that predictor on the
fixed 65-knot grid; JAX interpolates that fixed-shape table. Record interpolation,
extrapolation, distance clamp and all parameter-bound events per child.

For setting a retain n_a, J_a=sum ||U_child-U_seed||^2 over ALL 18 cube coordinates,
and W_a=sum actual total model calls (gradients counted once). Q_a=J_a/W_a,
never mean of individual J/W. The numerator is whole-chain movement including
ordinary slice transitions; rejected reflection paths are not dropped. The
controller has no per-cycle acceptance or reflection-only movement feedback.

For each eligible dimension p in {log h, log T}, update its anchor at the
stratum centre c=2j+1 by
  delta_p = log(1.25) * (Q_plus-Q_minus)/(Q_plus+Q_minus).
If both Q values are zero, delta is zero. The old value is the frozen predictor
at that centre. Store one anchor per stratum; interpolate the anchors to rebuild
the fixed table. All eligible changes in a batch use the SAME old table and
statistics. This bounds a single multiplicative update by 1.25, uses no target
acceptance, and leaves clipping/candidate collisions visible in the ledger.

Bound desired h to [0.001,0.03] and nominal T to [0.003,0.1]. Enforce T<=256*h
by shortening requested T only when necessary, recorded explicitly. Choose
B=ceil(T/h), minimum 1, maximum 256, and effective h=T/B. Treat ratios within
8 float64 eps * max(1,abs(T/h)) of an integer as that integer before ceil, to
avoid an accidental extra bounce from log/exp roundoff. Report quantization
and the cap separately. The initial central/control pair is represented by
exact constants B=10 and h=0.003. Jitter follows rounding. All roots have
B/h/T/gradients zero and one value call, including subsequently allocated roots.

The desired-parameter domain was narrowed before training/evaluation because
an independent probe found tiny-step/many-reflection round-trip failure at
B256,T.00256,h1e-5 (including accepted paths). Source b8dffd9 and failed/raw
probes remain archived. At the revised bounds B<=100, the256cap is inactive
and nominal T>=.003. Ceil rounding can reduce effective h toward.00075 near
T=.003,desired h=.001; jitter can reduce it further. Desired h is NOT a lower
bound on effective or jittered h. No integrator or tolerance changed.

Coordinator's revised4480-path,35-setting,140-cell probe passed the original
position2e-9/velocity2e-8 limits, including rounding boundaries: production
self-reversal max velocity8.34e-13 and equivalent analytic-field NumPy/JAX
phase parity8.12e-12. This supports the probed range, not every possible
floating-point path. An intermediate diagnostic compared TFP standardized
arithmetic with centered manual residual arithmetic; its different vector
fields must not be confused with equivalent-field phase parity. Exact scripts,
raw arrays and receipts for every stage are archived under coordinator-audits.

## Training, evaluation and inference

Training: four independent online runs per allocation, seeds 3122910..3122913,
4M logical calls each, before any evaluation is launched or inspected. Freeze
one ACTUAL likelihood-contour policy per allocation using only these training
runs. For each final training race, reconstruct its classic c(lambda) profile;
interpolate likelihood at c=0,2,...,128, clamping to that tree's finite observed
likelihood range. The common grid is the sorted unique union of those 4*65
likelihoods, with both finite extreme likelihoods included (at most 268 knots).
For every common lambda, evaluate each final training policy at that tree's
own c(lambda), then average its two log parameters equally across all four
runs. Include unchanged regions. This is a geometric mean, not best-run
selection. The frozen sampler uses linear interpolation in actual lambda and
constant endpoint tails, solely from this immutable table; no evaluation
population compression, peak, seed, or posterior information enters lookup.
Evaluation c/K remain diagnostics only. Frozen-versus-online is a diagnostic
comparison with different training histories, not a pure adaptation-bias test. Retain all training runs, work, wall
cost, histories and masks. Charge roughly 32M total training calls separately;
no fresh scout/calibration ensembles contribute controller feedback.

Evaluation: seeds 60..89 matched across all three arms and both allocations,
180 runs, snapshots at first COMPLETE batch crossing 2M and 4M logical model
calls. 4M is primary. Record actual crossing and overshoot, not interpolated
budgets. Initial roots and underlying chain streams match until parameter
choice intentionally changes transitions. No uncertainty/ESS stopping.

Unchanged numpy_allocation_evidence reduction: 4096 draws per snapshot/mode,
batches 64, C_min=20, shared cluster gamma weights, existing plateau rules.
Retain classic moments, classic MC and phantom MC separately. Evidence key
3122902 folded by seed, budget, allocation index (uniform=0, evidence=1), arm
(fixed=0, online=1, frozen=2) and condition flag; uint32 words seed NumPy RNG.
Use 100000 paired bootstrap seed resamples, NumPy RNG3122903, same index matrix
for all contrasts. Primary online-minus-fixed and frozen-minus-fixed per
allocation and estimator; online-minus-frozen diagnostic. Retain both budgets
and all phantom-minus-classic contrasts. Pointwise intervals, no winner or
multiplicity-adjusted claim. Include all seeds, failures and caps.

Report bias, error SD, RMSE/SE, reported SD, Gaussian and MC-percentile 95%
coverage, paired RMSE intervals and per-run MCSE. Classic posterior diagnostics
remain x1/x16/x17/x5 SD, exact CDF sup error and W1, plus Kish weight ESS.
Record movement and correlations (coordinates AND squared coordinates) in
compression and prior parent-gap bins. Keep raw jump sums and actual gradient,
value-only and total-work denominators. Post-run replay gives per-cycle
acceptance/movement at separately charged validation cost; it never feeds back.

## Integration boundary and deterministic continuation

All changes are benchmark/test files. No src/jaxns, allocation, scheduler,
evidence, defaults, public API, model params or phantom schema changes.

Extend the existing benchmark run_with_observer with one explicit optional
before-batch sampler supplier. It receives State and the replayed CoreWorkBatch,
prepares exact parent compression data and returns an immutable benchmark
sampler. Reuse the same planned work in the after-batch observer; check it
against returned constraints/row count. The default absent hook retains the
old control path. Production still owns scheduling, seed publication, capacity
growth, continuation keys, depth/goal boundaries and stop counters.

A small typed prepared-context Pytree carries sorted planned parent contours
and exact c/K arrays of width150; a typed policy carries fixed [65,2] log
parameters and scalar version. They are dynamic leaves, not static tuples,
model arguments, phantom fields or closure constants. The scalar sampler looks
up its requested contour and independently chooses its perturbation from its
original chain key, then calls existing ReflectionSampler.run_chain. Reuse
existing phantom assembly and return contracts; no production replay for
feedback. The observer's host learner owns raw sums and updates.

Full benchmark checkpoint = production State + typed controller (policy,
anchors, pending n/J/W/exposure/generations and version) + committed ledger
position + budgets/counters + source/protocol hashes. Atomically publish both
as one checkpoint with checksum. State alone is not a resumable online run.
Retain per-child original keys/seed/parent indices, pre-dispatch c/K, choice,
controller version, requested/effective h/T/B and actual counts. Checkpoint
restore uses these exact values, never refits from future/final samples.

Shared runner/reduction functions should be extended at explicit experiment
configuration seams, not copied into another scientific implementation. If
this cannot be done within benchmark ownership, stop and report the boundary.

## Gates, execution and storage

Before training, freeze protocol/source and pass focused tests and both
allocation smokes. Large evaluation additionally requires the frozen training
policy, final held-out validation and coordinator EVALUATION_RELEASE binding
all hashes. No release file means no evaluation dispatch. Smokes use seeds
3122920/3122921 and budgets30k/60k; a longer non-evaluation smoke up to300k is
allowed solely to exercise controller updates, recording all extra work.

Required gates: original fixed sampler exact child/phantom/count/RNG parity
with frozen central policy; phantom-off/on classic/continuation parity;
heterogeneous B/h phase reversal at existing tolerances (position2e-9,
velocity2e-8); exact-support/rejection self-transitions; predictor bounds,
ceil/caps/extrapolation and zero movement; ratio-of-sums; chronological updates,
revisit clock, no future-data use; initial AND newly allocated root accounting;
checkpoint/resume and buffer-growth parity; same-shape value changes do not
retrace/recompile. Inspect representative lowered code/memory and separate
compile, controller, transfer, sampling, writes and replay timing.

Reuse exact constrained draws and unchanged stationarity thresholds: independent
8192-member initial/reference ensembles, gaps0.1,5,15,40,100,1000,infinity;
PIT maxKS<5/sqrt8192 inside the cube, early two-sample maxKS<5*sqrt(2/8192).
Validate fixed and independently frozen learned policies (both allocations)
including the exploratory mixture; online uses these same frozen-kernel laws,
without claiming an ensemble test proves adaptive-history stationarity.
NumPy SeedSequence[3122904,case,stream], JAX3122905 folded by case with separate
single-path/full-chain keys. Replay every smoke child/phantom/count under its
actual policy version before review. Never relax existing tolerances.

CPU only: pin one CPU before JAX import; JAX_PLATFORMS=cpu,
CUDA_VISIBLE_DEVICES='', JAX_ENABLE_X64=1; OMP/OPENBLAS/MKL threads1. Verify
imported source and actual dtype/backend. CPU70 reserved for coordinator.
Start at four workers; at most24 after memory/time measurement. Keep >=64GiB
MemAvailable and >=20GiB filesystem headroom, preserve unrelated jobs. Current
/largedata free space is about95GiB. Budget <=15GiB total new experiment data;
forecast from smokes before training/evaluation, stop admission if forecast
cannot fit. Retain complete states and raw arrays on dorrie, atomic checkpoints,
checksums/manifests/progress and exact source/audit scripts. Send only compact
reports/figures/provenance/bundle; no large transfer archive by default.

Worktree: /largedata/albert/git/jaxns-reflection-online-20261029
Output: /largedata/albert/jaxns-reflection-online-20261029
Status: /tmp/jaxns-reflection-online-status-20261029.md
Coordinator owns independent review/publication. No GitHub writes or push.
