# Online reflection tuning on the issue-312 toy

The user has authorised the next experiment: learn step size and steps per
cycle online from ordinary sampling, keep the number of cycles fixed, and
avoid a large scouting operation every shrinkage e-fold. Continue on the
user-designated CPU-only screen 58428.jaxns. The coordinator owns the final
scientific review and publication.

## Ownership and workspace

- Create a NEW isolated worktree and branch from exact published commit
  1d84f157496f5b7a20659fdc5da9ca41fd8cec09. Fetch the experimental branch if
  needed. Suggested worktree /largedata/albert/git/jaxns-reflection-online-20261029,
  branch experiment/reflection-online-cpu-20261029, output
  /largedata/albert/jaxns-reflection-online-20261029.
- Read AGENTS.md, LEARNINGS.md, relevant full design docs, the previous tuning
  and duration protocols/reports and benchmark call paths before changing code.
- Benchmark-only experiment. No production core, evidence, scheduling,
  allocation, defaults or public API changes. Do not use model params or
  phantom fields as an untyped backchannel. Explain any boundary problem
  before implementing across it. Reuse existing scientific code and evidence
  reduction rather than copying it.
- Apply performance-and-intent-review-flow to dynamic sampler data,
  compilation, counters and code clarity. Preserve old outputs/worktrees.
- Update /tmp/jaxns-reflection-online-status-20261029.md at milestones with
  exact source/output paths. Do not push or write GitHub yourself.

## Intended experiment

18-D curved ridge, sigma_stiff=1e-5, curvature=.3, same normalized likelihood,
actual float64 cube, 150 initial roots, replacement_width=150, delta_K=150.
Keep 9 cycles, one ordinary isotropic slice per cycle, isotropic Gaussian
velocities, independent U(.8,1.2) h jitter, exactly 8 complete-cycle phantoms,
strict endpoint acceptance and rejection self-transitions unchanged. No
whitening, learned curvature, local endpoint stopping, or retry-until-success.

Learn h and nominal duration T separately, with B=ceil(T/h) (document the
rounding convention and effective h). For the control h=.003, B=10, T=.03.
Choose any exploratory setting before the chain starts, independent of its
position/velocity conditional on the frozen controller; keep the setting
fixed for that entire chain. Updating from production history is explicitly
experimental adaptive MCMC, NOT proved stationary just by freezing a chain.

Use ordinary completed sampling data. Ideally learn from whole-chain actual
seed-to-child squared displacement and the actual total work already present
in the observer. That avoids changing production return schemas or replaying
likelihoods for online feedback. Per-cycle acceptance and movement can be
computed in the post-run audit and must be accounted as validation overhead,
not used as if free feedback in the sampler. Do NOT repeat a chain to get
feedback during production. Include zero reflection movement on rejections;
whole-chain movement may include ordinary slice transitions and should be
labelled accordingly. Tune using all unit-cube coordinates, not oracle flat
coordinate identities or exact truth/evidence.

Suggested controller: around 90% current setting and 10% randomized nearby h
or T settings; bounded multiplicative perturbations and updates, enough
observations before moving, smooth contour-indexed log h/log T estimates,
interpolation plus bounded extrapolation. Pool only comparable contours and
retain raw per-setting jump/work sums. Optimize ratio of total squared jumps
to total work, not mean of per-chain ratios or a prescribed acceptance rate.
No extra scouting likelihood calls. Update using approximately one e-fold of
new compression information, not allocation-round or raw batch number. Actual
parent contours can be revisited under evidence allocation: state the precise
rule for those visits. Keep controller version and parameters for every child.

First give the coordinator a concrete small protocol and integration design
for review BEFORE large runs. You may implement and run focused correctness
tests/smokes while it is reviewed. Suggest a simple controller, avoid a large
new framework. Any dynamic arrays must stay fixed-shape pytree leaves so
changes in values do not cause recompilation per update.

## Comparison and validation

Plan fixed / online / independently frozen learned schedule arms, both uniform
and evidence_improving, matched 2M and 4M logical-model-call snapshots. Keep
allocation/stopping meaning unchanged and audit budget crossings. Suggest
30 fresh matched evaluation seeds 60..89. For the frozen arm, use a small
independent training cohort (suggest 4 seeds per allocation, outside all past
and evaluation seeds), aggregate/freeze the contour policy before inspecting
evaluation outcomes; charge/report training work separately. The frozen arm
is a diagnostic comparator, not by itself a proof that online adaptation is
unbiased. Report actual interpolation/extrapolation and clipping frequency.

Record evidence classic moments, classic MC and phantom MC (4096 draws),
RMSE, bias, coverage, paired seed bootstrap intervals; posterior x1/x16/x17/x5
widths/CDF errors; per-coordinate movement/correlation incl squared coordinates;
gradient vs value-only calls, complete work, wall/compilation/controller costs,
and parameter histories. No acceptance-rate success criterion. Include every
seed, failures/caps and all experimental overhead honestly.

Tests: fixed-controller exact parity with original fixed kernel (RNG, children,
phantoms, counts); frozen parameter heterogeneous B/h reversal and stationary
ensemble checks (existing exact constrained toy draws); predictor bounds and
extrapolation; strict chronological update/no future-data use; roots always
zero gradients including newly allocated roots; all source/controller state
required for deterministic replay/resume. Diagnose compile cache behavior
with same-shape different controller values. Reuse numerical tolerances.

Freeze protocol/source before training/evaluation; smoke both allocations and
replay every returned child/phantom/count under its actual controller version.
Coordinator review before evaluation release. Independent root audit will
inspect source and a subset/all raw summaries. Retain exact raw seed indices,
keys, per-child B/h/T, controller snapshots/history and work ledgers so the
root can verify without guessing. Keep shared random stream for fixed/online
baseline comparison until parameter choice intentionally differs.

CPU only, pinned one CPU per worker with BLAS one thread, start conservatively
and expand up to 24 after memory/time measurement; reserve CPU70 for coordinator
audits. Check current jobs/resources first. Local coordinator disk is short:
keep heavy raw outputs on dorrie, send only compact reports/figures/source
bundle/receipts (no giant archives). Bound disk growth and preserve prior data.

Final handoff: clean experiment commits, incremental Git bundle from exact
published baseline, succinct report/summary/plots and raw manifest. Track under
existing issue312/PR314. Coordinator handles integration, GitHub and user reply.
