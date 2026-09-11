# A+C evidence experiments at D=10 (2026-09-09)

The active paper cases are G10, CG10, and the linked repository SS10.
G10 uses N(0,I) prior, mean (3,0,...,0), covariance with diagonal 1 and
off-diagonal 0.99. CG10 applies the component-centered, unit-Jacobian shear
x2 = z2 + 0.4*((z1-3)^2-1). SS10 uses U[-4,8]^10 and the sum of two
normalized Gaussian densities: means (6,6,0,...,0) and (2.5,2.5,0,...,0),
covariances 0.08I and 0.8I. REFERENCES.json records deterministic log Z.

Baseline: 30 seeds (0--29) for each case; d0=300, shell width 100,
100 isotropic no-step-out slices, 99 retained phantoms, all valid phantom
U points eligible for seeding, no bounded reservoir. First full goal uses
uniform allocation with delta_K=1; subsequent goals use evidence-improving
allocation with delta_K=300. Depth dlogZ=log(1.001). Stop at classic
expected sigma(log Z)<0.05. Posterior weights always use classics only.

SS10 extension: resume the same 30 full-capacity trees and stored random
streams to 0.02. If spike-mass recovery fails, resume to 0.01, and then
0.005 as the final attempt. Recovery is descriptively defined before these
runs as spike-mass RMSE <=0.05 and maximum absolute mass error <=0.15,
relative to the exact bounded mass 0.5000077444258325. The criterion is not
a formal hypothesis test. Stop decisions depend on posterior recovery,
not on obtaining a preferred phantom result. Publish all completed stages.
The uncertainty target may be achieved while recovery fails.

Mass uses classic posterior weights times exact component responsibilities.
For each fixed tree compute 2048 paired shrinkage draws at phantom prefix
sizes 0,10,...,90, groups of 10, gate C_min=20. The prefix changes only
evidence conditioning; it does not change the classic posterior mass.
Evidence comparisons report empirical RMSE, bias, mean reported uncertainty,
coverage, and paired RMSE change. Bootstrap whole seeds (10,000 resamples,
seed 20260909) to quantify Monte Carlo uncertainty across runs. Preserve
the seed-level values, draws, states, manifests and worker logs.

One distinct pinned CPU per worker, numerical threads=1, CPU backend with
JAX x64 enabled; the inherited jaxctx unit-prior dtype remains float32.
At most 60 workers. Conservative memory reservations reduce concurrency
for tighter stages: 8/24/64/192 GiB per worker for targets
0.05/0.02/0.01/0.005, against a 480 GiB reservation budget. These are
scheduling bounds, not changes to the sampler. Record actual peak RSS,
likelihood calls, goal iterations, cumulative and incremental wall time.

Checkpoint every five goals and at completion. Preserve full state capacity
for resumption: a focused test found that trimming before resumption changes
the trajectory despite preserving PRNG keys. Pickle/reload without trimming
matches uninterrupted continuation exactly. Trim only for analysis.
The final checkpoint and state are hardlinks to avoid duplicate disk usage.

Validation before launch: eight focused tests passed, including model/prior
agreement with SciPy, conjugate G10 evidence, adaptive and Hermite CG10
quadrature, SS10 identity with the previous repository model, CLI smoke,
full phantom checkpoint continuation, and the existing core resume test.
Ruff and Flake8 (100-column limit) passed for the new harness and tests.
No production sampler source is modified by this experiment branch.
