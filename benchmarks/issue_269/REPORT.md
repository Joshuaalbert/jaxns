# Issue 269 allocation-cadence result

## Decision

Do not promote a smaller initial root allocation or a finer allocation
increment for mode-death mitigation. The common uncertainty goal did force
multiple completed allocation rounds, but none of the six candidate schedules
passed the preregistered mode-survival gate.

The result supports a narrower mechanism: the weak mode is usually lost when
the smaller initial root population fails to establish a viable lineage.
Adding many later full-prior roots rarely rediscovers a high-likelihood weak
mode, even when the goal requires 13--16 allocation rounds.

## Reviewed run

The screen used 30 paired measured seeds per schedule on the analytic 8D
`spike_slab`. It kept the constrained sampler, uniform allocation law,
`dlogZ=log1p(1e-3)` compiled depth condition, and 24,000-sample hard ceiling
fixed. The Python outer goal continued until expected
`log_Z_uncert <= 0.18` after a completed depth epoch.

All 210 measured runs reached the uncertainty goal without hitting the hard
sample ceiling. The reviewed CPU/x64 run used JAX/JAXlib 0.10.0,
`MPLBACKEND=Agg`, and worktree commit
`f6c5b1eb80ba25d162eef463767120241cbb5d3f`. The benchmark records and validates
the imported worktree source path to prevent the conda environment's editable
install from silently selecting another checkout.

| Schedule | Goal rounds (range) | Final roots | Mode loss | Mode RMS | Evidence RMS | Evaluations / ESS | Median run seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 3--4 | 400 | 30.0% | 0.1762 | 0.2307 | 714 | 3.74 |
| `smaller_root_coarse` | 4 | 400 | 50.0% | 0.1415 | 0.2289 | 658 | 4.07 |
| `smaller_root_medium_fixed` | 7--8 | 400 | 53.3% | 0.1426 | 0.2072 | 652 | 6.32 |
| `smaller_root_fine_fixed` | 13--14 | 400 | 50.0% | 0.1402 | 0.2251 | 656 | 10.0 |
| `smaller_root_medium_matched` | 7 | 400 | 56.7% | 0.1522 | 0.2320 | 654 | 3.92 |
| `smaller_root_fine_matched` | 13--14 | 400 | 43.3% | 0.1556 | 0.2848 | 655 | 4.91 |
| `aggressive` | 15--16 | 400 | 53.3% | 0.1646 | 0.2559 | 652 | 4.98 |

Mode loss means inferred weak-mode mass below `1e-3`. Although three fixed
shell candidates have approximately 19--21% point improvements in mode-mass
RMS, every paired 95% bootstrap interval includes zero. Their discrete mode
loss is also 20.0--23.3 percentage points worse than baseline. The
`smaller_root_medium_matched` loss regression resolves at 26.7 percentage
points worse, with a paired interval of 3.3--50.0 points worse. No row meets
the requirement for a resolved 20% mode-RMS improvement with improved mode
survival.

## Allocation-epoch evidence

A weak-mode peak is counted as discovered when an epoch contains a
component-zero sample with `log_L > -5`. Successful runs reach approximately
`-1` at that peak, while failed runs remain tens of log units lower, so the
threshold only classifies the visibly separated regimes.

| Schedule | Miss after first depth epoch | Never discovers peak | Late peak discovery | Final mass loss |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 9/30 | 9/30 | 0/30 | 9/30 |
| `smaller_root_coarse` | 16/30 | 15/30 | 1/30 | 15/30 |
| `smaller_root_medium_fixed` | 16/30 | 16/30 | 0/30 | 16/30 |
| `smaller_root_fine_fixed` | 16/30 | 16/30 | 0/30 | 15/30 |
| `smaller_root_medium_matched` | 19/30 | 18/30 | 1/30 | 17/30 |
| `smaller_root_fine_matched` | 16/30 | 15/30 | 1/30 | 13/30 |
| `aggressive` | 19/30 | 18/30 | 1/30 | 16/30 |

Thus the smaller schedules create more allocation opportunities, but they
start with 16--19 weak-peak misses rather than nine for baseline and produce
at most one late peak discovery per schedule. The smaller initial population
dominates the cadence benefit in this configuration.

## Correctness finding during the screen

The first uncertainty-goal rerun exposed a separate boundary error: after
sample-buffer growth, the Python goal was evaluated while the same compiled
depth epoch was unfinished. That allowed physical initial capacity to affect
the scientific stopping point. The core now bypasses custom goal evaluation
until an initial or completed depth boundary, with a regression test comparing
grown and preallocated continuations. The raw matrix above was rerun only
after that fix; the pre-fix records are not used as evidence.

## Promotion gate

No candidate advances to the retained-phantom confirmation. Matching the shell
width does reduce device work from inactive sampler lanes, but it does not
repair mode survival and is not independently eligible under the preregistered
scientific-first decision rule.
