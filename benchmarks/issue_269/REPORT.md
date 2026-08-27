# Issue 269 large-lineage evidence

Environment: JAX/JAXLIB 0.10.0, CPU, x64 enabled, Python 3.12.9, and
Matplotlib `Agg`. The maintained 8D `spike_slab` truth is
`log Z = -18.3958906841`; its weak component has analytic posterior mass
0.1162586879. A run classifies the mode as dead when measured mass is below the
fixed preregistered threshold of 0.001.

## Screen and persistent seed

The first ten paired seeds had the same two mode deaths, seeds 4 and 9, at all
four uncertainty goals tested: 0.12, 0.10, 0.08, and 0.06. Those targets used
about 960, 1,280, 2,000, and 3,520--3,600 lineages respectively. Smaller
uncertainty therefore did not change the 2/10 failure rate at ordinary lineage
counts.

Seed 4 was then followed to much lower uncertainty. Runs at the same seed and
configuration are deterministic prefixes. The 0.02 checkpoint was also
reproduced exactly in two independently started sweeps.

| Target log-Z uncertainty | Rounds | Final lineages | Samples | Weak-mode mass | Status |
|---:|---:|---:|---:|---:|---|
| 0.12 | 10 | 960 | 23,613 | 2.714e-15 | dead |
| 0.10 | 14 | 1,280 | 31,445 | 4.245e-11 | dead |
| 0.08 | 23 | 2,000 | 49,180 | 4.297e-9 | dead |
| 0.06 | 43 | 3,600 | 88,403 | 4.183e-7 | dead |
| 0.04 | 98 | 8,000 | 195,717 | 3.446e-5 | dead |
| 0.03 | 175 | 14,160 | 346,074 | 2.146e-4 | dead |
| 0.025 | 253 | 20,400 | 498,095 | 3.842e-4 | dead |
| 0.020 | 396 | 31,840 | 777,218 | 6.042e-4 | dead |
| 0.016 | 620 | 49,760 | 1,213,904 | 1.050e-3 | survives |

At target 0.016 the goal register reached 0.01598997 without hitting the
2,000,000-sample ceiling. The best weak-component log likelihood was -1.7003,
the evidence error was -0.12175, and cumulative wall time was 9,814 seconds
(2.73 hours). The larger sample ceiling was necessary: the earlier 500,000 and
1,000,000 ceilings could locate the transition only from below.

## Decision

This targeted run supports the lineage-count explanation: a mode that remained
dead through tens of thousands of lineages crossed the fixed survival threshold
at roughly 50,000 lineages. It does not establish calibrated mode weight. The
recovered mass is only 0.903% of the analytic weak-mode mass, an underweighting
factor of about 111. Nor does one deliberately selected failing seed estimate a
population failure rate.

The fine schedule took 620 allocation rounds and 2.73 CPU hours for this one
seed, so it is evidence about mechanism and scale, not a suitable production
policy. Before changing allocation, the next experiment should reach roughly
50,000 initial or coarsely allocated lineages directly, verify scientific
parity with this transition, and then repeat enough independent seeds to
measure survival and evidence calibration.
