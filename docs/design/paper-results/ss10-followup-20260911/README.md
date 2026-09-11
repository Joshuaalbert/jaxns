# SS10 completed 0.015 and cancelled posterior-ESS follow-up

The 30 matched A+C continuations from 0.02 to 0.015 all completed. Mean classic
spike mass rises from 0.27593 to 0.31265 (truth 0.5000077444); spike-mass RMSE
falls from 0.32273 to 0.27969. None of the three completed uncertainty stops
meets the prespecified RMSE <=0.05 / maximum error <=0.15 recovery rule.
See [summary](SUMMARY.json), [table](CLASSIC_TABLE.tex), and
[figure](ss10_followup.pdf).

These evidence diagnostics use the deterministic classic expectation and its
reported uncertainty. Normal-approximation coverage is 2/30 at 0.02 and 1/30
at 0.015. They are not the Monte Carlo means and central intervals used in
[the earlier paired phantom tables](../evidence10-AC300-20260909/README.md).
No new phantom-prefix evidence analysis was run at 0.015. Posterior mass always
uses classic weights, independently of evidence prefix.

A subsequent pure posterior-improving allocation run targeted twice each
seed's fixed original 0.015 Kish ESS, retaining all-phantom seeding. The user
cancelled it after seeds 7,8,9,14,16,23,28 completed: five active runs were
interrupted and eighteen queued runs cancelled. The completed subset doubled
ESS but changed spike mass by only 0.00058–0.01068. Scheduling and completion
select this subset; it does not estimate a 30-seed population effect or prove
that further posterior allocation could never recover structure.
[Cancellation record](POSTERIOR_STOPPED.json).

## Provenance and reproduction

Original input sampler: aa79a0dc395c0d64b85cbb88d0e3e732decd7361.
Efficient sampler: 650ae05, with unchanged production source through both runs.
The 0.015 runner was frozen at 82cd8d3; the posterior runner at 6b9ed19.
Their respective source snapshots are under `source/`. The 0.015 continuation
report and original compact provenance are retained in
[CONTINUATION_REPORT.md](CONTINUATION_REPORT.md) and
[CONTINUATION_REPORT.json](CONTINUATION_REPORT.json).

`records/0.015/seed-XX` contains all 30 completed CORE, MANIFEST and goal-progress
records. `records/posterior-ess2/seed-XX` contains the seven completed records.
[RECORD_SHA256.json](RECORD_SHA256.json) identifies all installed raw records.
Full states and interrupted checkpoints stay in the original dorrie worktree:
`/largedata/albert/git/jaxns-efficient-phantom-A-20260910/benchmarks/phantom_seeding/results/`.
No state or pickle is included in this paper update. No jobs remain authorized.

Rebuild statistics and the standalone figure (does not launch jobs):

```bash
conda run -n jaxns_py python docs/design/paper-results/ss10-followup-20260911/report.py
```

The script verifies all completed stopping conditions, fixed ESS targets,
matched cumulative calls/goals, and installed record hashes. It recomputes
classic summaries and paired RMSE-change intervals without loading full states.
Mass-RMSE SEs use 10,000 resamples with seed 20260909, matching the earlier
paper table; paired changes use 20,000 resamples with seed 292, matching the
continuation report. The manuscript embeds CLASSIC_TABLE.tex verbatim.

The earlier manuscript generator reproduces only the historical 0.05/0.02
snapshot; do not use it to overwrite the revised paper. Build the manuscript
from docs/design/paper.tex when the journal class and bibliography are available.

Validation: all 600 records and prefix tables in the earlier 120-cell snapshot
pass the installed audit. The new 37 completed records, fixed targets,
continuation accounting, and classic table pass the follow-up report checks.
LaTeX braces, environments, labels and references balance. A full typeset build
is unavailable: this checkout has no TeX compiler, journal class or cite.bib;
the pre-existing missing censorship figure has a placeholder. Archived logs,
patches and historical tables are retained verbatim, including their original
whitespace, to preserve recorded hashes.
