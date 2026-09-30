# Analysis-only plateau correction

The A+C300 sampling source remains aa79a0d, with no changes to its complete
src tree or problem definitions. The original sparse evidence reducer stopped
on SS10 seed16 because seven classic samples share a likelihood block at
logL=-139.98022055646703. They have identical coordinates, including x5=8
at the prior boundary, and individual classic posterior weights about8.34e-55.
No samples, likelihoods, posterior weights, or sampling trajectories are changed.

The core already supports plateau blocks. Its strict probability is
(G_gt + weighted_B)/(G_gt + G_eq + G_lt + weighted_A).
Thus G_eq + G_lt can be drawn as Gamma(alpha_eq+alpha_lt,1), equal to
Gamma(m+1,1) for a plateau. The phantom equality and open-interval counts
sum to A-B and cancel from this marginal. Singleton complements remain
Exponential(1); the otherwise-unused equality key supplies plateau complements.
This is the exact evidence marginal of the existing three-category model.
The reducer does not compute phantom-conditioned posterior sample weights.

Four tests pass: dense three-category core comparisons on mixed singleton and
plateau blocks with exact phantom equalities at two Kish thresholds; an
analytic Beta(4,8) sevenfold-plateau moment check; and the existing continuous
sparse-prefix regression. A separate 64-draw, ten-prefix comparison against
the frozen pre-correction module confirms bitwise identical singleton draws
and gates. Ruff and Flake8 (100 columns) pass for the touched source/tests.

Analysis records retain the sampling commit and additionally identify the
analysis commit, reducer/runner checksums, tied-block count and largest block.
The analysis runner verifies the complete sampler source tree and exact
model/reference checksums against each saved manifest before loading it.
Completed original analyses were necessarily singleton-only (the old reducer
rejected all others), so they are retained alongside corrected analyses.
The initial failure and retry attempts remain in the dispatch audit.
