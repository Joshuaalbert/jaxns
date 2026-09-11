# Posterior recovery candidates: 10D, seed 0

Three completed ESS-target runs on dorrie use branch `paper/posterior-candidates10`, frozen
source `b44d3b7bdfaa870315a7f09a2abe63c22a765bb3`, forked from A+C240 at
`1d7db867a998dc30928179c40013ef4e52f3379b`. The original eight-dimensional R240 tables have since been superseded by
[the 30-seed A+C300 evidence cohort](../evidence10-AC300-20260909/README.md).
The three ESS-target experiments below remain in the paper. [Protocol](PROTOCOL.md) and [machine-readable results](SUMMARY.json).

Each run uses 300 initial roots, width 100, 100 isotropic slice transitions,
99 retained phantom points per chain in A's seed pool, and seed 0. The first
uniform goal is followed by a 10% evidence / 90% posterior allocation-gap
mixture. Classic expected posterior weights determine Kish ESS and every plot.
Runs were concurrent on one pinned CPU each (cores 0, 1, and 2).

| Problem | Target Kish ESS | Achieved ESS | Goals | Likelihood calls | Classic log Z ± shrinkage SD | Reference log Z |
|---|---:|---:|---:|---:|---:|---:|
| Eggbox 10d | 5,000 | 5,493 | 3 | 5,924,166 | 209.9162 ± 0.2728 | 210.11148649 |
| Rosenbrock 10d | 3,000 | 3,681 | 1 | 9,098,530 | -44.2477 ± 0.3447 | -43.10835521 |
| Rastrigin 10d | 15,000 | 16,487 | 7 | 22,428,321 | -38.4240 ± 0.2232 | -38.10866469 |

**Rosenbrock stops in the first uniform goal.** It uses A's phantom seed pool,
but its requested ESS threshold is met before mixed allocation is exercised.

All three reach their ESS thresholds with visible posterior errors:

- Eggbox retains uneven marginal mode weights. Means range from 14.79 to 20.53
  against 5π = 15.708. Its marginal reference SD is 9.39842.
- Rosenbrock substantially underrepresents later-coordinate tails: x10 mean
  0.0527 and SD 0.1267 versus reference 0.5576 and 0.8753.
- Rastrigin retains asymmetry and mode-weight errors. Its x9 mean is −0.8577,
  versus zero; its x1 mean is +0.4074. Each exact marginal SD is 0.70642.

Kish ESS describes concentration of represented classic weights. These results
show that meeting that target does not establish posterior recovery. This is
one seed, with no paired A-only or alternative-allocation controls on these
new targets; the run does not isolate the source of the errors.

## Plots

Full classic-weighted corners, also included in the manuscript:

- [Eggbox corner](figures/eggbox10_corner.pdf), [marginals](figures/eggbox10_marginals.png)
- [Rosenbrock corner](figures/rosenbrock10_corner.pdf), [marginals](figures/rosenbrock10_marginals.png)
- [Rastrigin corner](figures/rastrigin10_corner.pdf), [marginals](figures/rastrigin10_marginals.png)

Orange curves are deterministic one-dimensional marginal references. Pair
panels use classic-weighted histograms; 68% and 95% contour levels use 0.6-bin
smoothing. No phantom posterior samples or resampled pseudo-independent
posterior samples are plotted. The exact plotting ranges are in SUMMARY.json.

## Relation to the earlier comparison

The model adapters match the pinned functions from
[jaxns-cosmology](https://github.com/Joshuaalbert/jaxns-cosmology/tree/67e45dac6d9b273c4f4d957433f53228bf9ea7e0/jaxns_cosmology/models),
checked by executing their original likelihood function bodies against the new
adapters. The earlier
[comparison paper, Tables 1–3](https://arxiv.org/html/2409.18464v2)
reported JAXNS ESS 1.6k / 3.9M likelihood calls for Rosenbrock, and 54k / 104M
calls for Rastrigin. Its Eggbox table has no JAXNS result. Those settings and
ESS values differ from this run, so these counts are not matched-accuracy
speedups. The new runs still show the posterior-recovery difficulties that
motivated these candidates.

## Independent numerical references

[reference.py](reference.py) evaluates Eggbox through a positive-coefficient
power series and Rosenbrock through forward/backward integration of its chain
kernel. Eggbox's final 800- and 1200-term estimates agree within 1e-10 in log Z,
with independent 1D/2D quadrature checks of the series. The initial 600-term
pilot differs by 3.3e-10; its failed stricter check is preserved in validation.
Rosenbrock's 1024- and 2048-node quadratures agree within 1e-8 for log Z, all
means and all SDs. Rastrigin uses factorised adaptive quadrature in the case
module. [REFERENCE.json](REFERENCE.json) records all refinement levels.
The references were not used in the stopping decisions.

## Validation and retained states

64 focused tests pass: 22 adapter, ESS, and weighted-allocation checks plus 42
allocation, all-phantom, and resume regressions. Ruff passes for the four
modified production modules and new run/model/test files. New files pass
Flake8; the five findings in production depth.py are identical to the A+C
parent, as recorded in the two raw Flake8 logs. Initial adapter-test failures
are preserved along with their corrected passing results.

Full states and latest goal checkpoints remain under
`/largedata/albert/jaxns-posterior10-20260908/{case}/seed-00/`.
The report retains manifests, classic posterior arrays, per-goal records,
references, figures, and source/state checksums in AUDIT.json. Large state
pickles are retained on dorrie rather than committed to Git.
The paper source and new figures are updated; the full manuscript PDF cannot
be rebuilt in this checkout because TeX, sn-jnl.cls, cite.bib, and the existing
censor.drawio.pdf figure are absent.

## Repository SS10 completed: A+C, evidence stopping

The user selected the historical repository definition on 2026-09-09.
Both components are 10D, with means (6,6,0,…,0) and (2.5,2.5,0,…,0), and
covariances 0.08 I and 0.8 I. The prior is uniform on [-4,8]^10, and the two
normalised Gaussian densities are added with coefficient one each.
The exact bounded log Z is -24.1559348061, with spike fraction
50.000774%.
[Exact reference and marginal checks](SS10_REFERENCE.json).

The run uses A+C: all 99 retained phantom coordinates per chain are eligible
seeds; allocation after the first uniform goal is pure evidence-improving.
Seed 0, d0=300, width=100, 100 slice transitions, depth threshold log(1.001),
and one pinned core (CPU 3). It stopped at the first goal with classic expected
sigma(log Z)<0.05. Source commit `259d436bde03f502a93019817c3d749fd2f04207`
has identical sampler, model, and runner code to the three earlier candidates.

| Quantity | Result |
|---|---:|
| Completed goals | 18 |
| Likelihood calls | 16,573,991 |
| Classic expected log Z | -24.744189 |
| Classic expected sigma(log Z) | 0.049639 |
| Classic expected log Z error | -0.588254 |
| Classic Kish ESS | 15,633 |
| Recovered spike responsibility mass | 0.265438% |
| Reference spike mass | 50.000774% |

[SS10 full corner](figures/ss10_corner.pdf),
[compact marginals](figures/ss10_marginals.png).
The spike is severely underrepresented. Spike-dominated classic and retained
phantom samples appear in goal 1, so the run did visit it early; the final tree
contains 548 spike-dominated classic points.
[Append-order diagnostics](runs/ss10/DISCOVERY.json) use the explicit criterion
that the spike density exceeds the slab density. These counts do not measure
posterior mass or independence.

The paired prefix analysis uses 2048 shrinkage draws, prefix sizes 0,10,…,90,
cluster gate C_min=20, and dimension 10, with no extra likelihood evaluations:

| Prefix | Mean log Z | Shrinkage SD | Error from reference |
|---|---:|---:|---:|
| 0 | -24.744946 | 0.049394 | -0.589011 |
| 90 | -24.858271 | 0.013065 | -0.702336 |

The phantom estimate lies near the slab-only log Z (−24.84909748). Its smaller
reported uncertainty does not repair the missing component mass. This is one
seed and therefore not an empirical RMSE or calibration experiment. The
posterior uses classic weights once, independent of phantom evidence prefix.
[All prefix results](runs/ss10/ANALYSIS.json) and
[retained evidence draws](runs/ss10/evidence_draws.npz).

The two repository-specific adapter/evidence tests pass again, in addition to
the earlier 64-test validation. Independent adaptive quadrature verifies SS10
marginal normalisation, means, and variances. Full state and checkpoint remain
in `/largedata/albert/jaxns-posterior10-20260908/ss10/seed-00/`.

To reproduce SS10 from the experiment worktree, run
`python -m benchmarks.paper_reproduction.run_posterior --case ss10
--ss10-definition repository --output RUN_ROOT/ss10/seed-00`
with the documented CPU/thread environment and one pinned core. Then run
`python -m benchmarks.paper_reproduction.analyse_ss10 --root RUN_ROOT`
and `plot.py --root RUN_ROOT --cases ss10`.

To reproduce the three completed runs from the frozen worktree, use the
`benchmarks.paper_reproduction.run_posterior` module with `--case`, `--output`,
the single-core/thread environment in the manifest, and `taskset -c CPU`.
Then run `reference.py --output RUN_ROOT/references` and
`plot.py --root RUN_ROOT`. The model's Rastrigin reference is in each CORE.json.
