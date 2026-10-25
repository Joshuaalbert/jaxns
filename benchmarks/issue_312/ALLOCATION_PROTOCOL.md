# Allocation comparison: prespecified protocol

Compare `uniform`, `evidence_improving`, and `posterior_improving` for each of
the unchanged exact scaled-Hessian and reflection kernels. Use seeds 0–29
for every cell: 180 runs. Start all six cells from the same 150 prior draws
for a given seed, verified by hashes. The target is the 18D curved ridge with
`sigma_stiff=1e-5`, curvature 0.3, and effectively zero log evidence.

Keep float64 cube coordinates, replacement width 150, `delta_K=150`, no
phantoms, automatic buffer growth and the default per-epoch depth condition
`dlogZ=log1p(1e-3)`. Hessian uses 90 MH-corrected contour-scaled slice steps.
Reflection uses nine cycles of one isotropic slice and ten midpoint
reflections, with step size 0.003. Change allocation only.

Save snapshots at the first completed replacement batch reaching 2,000,000
and 4,000,000 cumulative logical model calls, including roots. The primary
comparison is at four million; two million shows development with work.
All strategies use this same budget rule, without uncertainty or ESS stopping.
The finite budget may interrupt a depth epoch. Report actual call totals,
the last batch's cost, completed-depth counts and peak gaps. Do not describe
a budget snapshot as a completed scientific goal or assume it is converged.
Overshoot is at most one complete batch. This preserves each complete chain
and the fixed replacement width; it does not cut off costly chains selectively.

Record actual seed indices by replaying the deterministic seed planner after
each accepted batch, using exactly the original state and random key. This
adds no likelihood calls. Check the replayed contour and valid prefix against
accepted rows and independently test seed replay against actual generated
children. The observer uses existing bounded depth continuations; production
allocation, sampling, state schemas and goal handling are unchanged.

Primary metrics: evidence RMSE and bias across all seeds, quoted uncertainty
and coverage, posterior marginal widths and Wasserstein distances for x1,
x16, x17 and x5, and Kish weight ESS. Record gradient work separately.
Compute actual seed-to-child RMS displacement and correlation by fixed parent
peak-gap intervals: (0,5], (5,15], (15,40], (40,100], (100,1000], and >1000
nats. This controls for contours becoming narrower; pair correlation is not
an integrated autocorrelation time or a corrected ESS. Summarise results
across runs rather than treating all dependent children as independent data.

Validate all archives, source hashes, matched initial populations, root and
gradient accounting, strict parent support, independent NumPy likelihoods,
weighted moments and budget crossing. Use a fixed paired bootstrap over seed
indices for uncertainty in RMSE differences. Preserve all 30 seeds per cell;
do not stop a cohort after a favourable partial result.
