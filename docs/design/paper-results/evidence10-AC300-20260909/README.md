# A+C300 evidence experiments and SS10 recovery

The later [0.015 and cancelled posterior-ESS follow-up](../ss10-followup-20260911/README.md)
is archived separately. The 120-cell snapshot below remains the complete paired
phantom-analysis dataset at 0.05/0.02.

The paper uses G10, CG10, and the linked repository SS10, with 30 seeds
(0–29) per problem, d0=300, and classic expected log-evidence uncertainty below
0.05, plus matched SS10 continuations to 0.02. [Comparison](COMPARISON.md),
[protocol and amendments](PROTOCOL.md),
[summary](SUMMARY.json), and [audit](AUDIT.json).

**Baseline complete: 90 runs and paired prefix analyses. The remaining SS10
ladder was cancelled by the user on 2026-09-10.** All 30 matched 0.02 runs
and analyses also completed. Both completed stops are installed in the paper,
comparison, recovery figure, and this 120-cell audited snapshot.
Seven 0.01 checkpoints are preserved, but no 0.01 run completed and 0.005
never started. See [operational status](OPERATIONS.md) and
[cancellation record](CANCELLED.json). Recovery means mass RMSE ≤0.05 and
maximum absolute mass error ≤0.15; phantom benefit is assessed separately.

At the baseline, classic / Ph90 evidence RMSE is 0.0473 / 0.0239 for G10,
0.0328 / 0.0286 for CG10, and 0.6256 / 0.6250 for SS10. Only the G10 RMSE
improvement is resolved by the paired bootstrap interval. SS10 mean classic
spike mass is 21.48%, against 50.0008%; 18 of 30 seeds assign it less than 10%.
Ph90 lowers SS10 mean reported uncertainty from 0.0491 to 0.0139 while its
empirical error stays large. All prefixes share one classic posterior per tree.

At 0.02, classic / Ph90 evidence RMSE is 0.5268 / 0.5226, and mean reported
SD is 0.0199 / 0.0059. Spike-mass RMSE improves from 0.3975 to 0.3227, but
neither stop meets the recovery criterion. Mean cumulative likelihood calls
increase 6.73-fold, from 28.742 million to 193.296 million.

The [paired bias contrasts](SS10_CONTRASTS.json) show slightly smaller absolute
cohort bias with Ph90 at both stops, with both 95% intervals including zero:
[-0.0242, +0.0176] at 0.05 and [-0.0094, +0.0069] at 0.02. The paired RMSE
contrasts are also unresolved. This supports no resolved additional bias or
RMSE in these cohorts, not proven equivalence or a general no-harm claim.
Classic / Ph90 interval coverage is 3/30 / 1/30 at 0.05 and 2/30 / 0/30 at
0.02: conditional precision increases while unresolved component-mass error
already affects classic inference. Prefix comparisons hold A and C fixed.

## Reproduction and provenance

Sampling branch `paper/evidence10-AC300` is frozen at
`aa79a0dc395c0d64b85cbb88d0e3e732decd7361`; the A+C core descends from
`a18124bf5635145c96b00410774463424b67d35c`. Analysis branch
`paper/evidence10-plateau-analysis` is frozen at
`5667381608eced7e3fcb754f32cc6c59db185735`. Its full `src` tree and model sources
are identical to the sampling branch. The analysis-only amendment handles
an observed tied block using the core's existing plateau law. It preserves
singleton draws bitwise. See [analysis validation](source/analysis/validation/README.md).

`records/<stop>/<problem>/seed-XX/` contains audited manifests, core summaries,
analysis records, all 2048 paired evidence draws per prefix, and goal progress.
`source/` preserves both experiment amendments, focused tests, and validation.
`scripts/` contains the launch, continuation, analysis, plotting, archiving,
and paper-generation scripts; source paths and commits are recorded in
[SOURCES.json](SOURCES.json) and [ANALYSIS_SOURCES.json](ANALYSIS_SOURCES.json).
The earlier failed sparse reduction and successful retry are preserved in
[RETRIES.json](RETRIES.json). No failed or inconvenient seed is excluded.

Full-capacity states, checkpoints, classic posterior arrays, and worker logs
remain on dorrie under `/largedata/albert/jaxns-evidence10-AC300-20260909`.
Verified state copies may live under
`/ceph_dorrie/albert/jaxns-evidence10-AC300-20260909`; the original state paths
remain usable through symlinks. `STATE_ARCHIVE.json` records raw-pickle hashes.
[SNAPSHOT.json](SNAPSHOT.json) identifies the completed cohorts in this checkout.

To reproduce the revised manuscript and its paired-bias table from this
installed snapshot, run from the repository root:

```bash
conda run -n jaxns_py python docs/design/paper-results/evidence10-AC300-20260909/scripts/evidence10-paper.py --completed-only --report-dir docs/design/paper-results/evidence10-AC300-20260909 --output /tmp/evidence10-paper.tex
```

The generator uses the installed `scripts/evidence10-template.tex` from `8bbe91f` as its template and the
updated protocol fragment alongside the script. Review its output before
replacing a manuscript with later edits. It reads completed summaries and
writes reporting artifacts only. The raw output-root manuscript is a historical
draft; the installed generator includes the cancellation and paired-bias
interpretation. Do not rerun the old launchers or automatic finalizer.

[Snapshot validation](SNAPSHOT_VALIDATION.json) verifies all 600 installed
record hashes, recomputes evidence statistics and paired contrasts from saved
draws, checks matched continuation records, and matches all four full-prefix
tables against the manuscript. Reproduce it with
`conda run -n jaxns_py python docs/design/paper-results/evidence10-AC300-20260909/scripts/evidence10-validate-snapshot.py`.
Ruff and Flake8 (100-column reporting convention) pass for the changed scripts.

One CPU is pinned per worker, with numerical thread counts set to one and at
most 60 concurrent workers. Tighter runs reduce concurrency according to
measured memory requirements. Sampling/model source is held fixed throughout.

The full LaTeX build is unavailable in this checkout: no TeX compiler,
`sn-jnl.cls`, or `cite.bib` is installed, and the pre-existing
`images/censor.drawio.pdf` is missing. The changed figures were rendered and
visually reviewed; local labels, references, braces, and environments are checked.
