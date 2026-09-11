# Posterior recovery candidates, 10D, seed 0

This isolated branch combines A's complete phantom seed population with the
weighted-allocation API previously tested in C-prime. The R240 tables remain
separate. The historical model source is pinned at
https://github.com/Joshuaalbert/jaxns-cosmology/tree/67e45dac6d9b273c4f4d957433f53228bf9ea7e0/jaxns_cosmology/models.
The four source files in reference_models are unmodified provenance fixtures.

All new runs use d0=300, width=100, 100 isotropic slice transitions without step
out, 99 retained phantom states available for seed selection, depth remainder
threshold log(1.001), seed 0, CPU execution with JAX x64 enabled, and one pinned
CPU each. The existing model API represents unit-prior coordinates in float32;
likelihoods and integration use float64. Numerical thread pools are limited to
one. At most four candidate runs are needed (below the user's ceiling of 60).

The first complete goal is uniform with delta_K=1. Later goals use delta_K=300:
ceil(0.1 * evidence_gap + 0.9 * posterior_gap), where each component gap is
already rounded, and the mixture is not peak-renormalised. Stop Eggbox at classic
Kish ESS >=5000, Rosenbrock >=3000, and Rastrigin >=15000. Every goal is recorded
and checkpointed with its random stream. ESS and corner plots use classic
expected posterior weights only. Phantom seeds may change the resulting classic
tree; phantoms themselves do not enter the posterior or Kish ESS.

Eggbox's positive fifth-power expression is log L, not L. Rosenbrock follows
the actual squared-x curved code expression. Rastrigin is separable, allowing
independent 1D quadrature for its evidence and marginal density.

SS10 uses A plus pure evidence-improving allocation after the same first goal,
and stops at classic expected sigma(log Z)<0.05. Its finite-box evidence is the
sum of component Gaussian CDF masses divided by 12^10. Two normalised component
densities are added without a factor of one half. The requested prose and
repository disagree on mean vectors and whether 0.08/0.8 denote covariance
diagonals or standard deviations. The user selected the repository definition on 2026-09-09: means shifted
only in the first two coordinates and covariance diagonals 0.08/0.8. The
completed SS10 run uses that definition; alternative calculations are not runs.

The comparison is exploratory and uses one seed. Kish ESS is a concentration
measure of the classic weights, not a guarantee of mode discovery or an
independence-adjusted ESS. The previous paper used different stopping criteria;
likelihood-call comparisons are descriptive, not matched-accuracy speedups.
