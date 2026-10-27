# Audited reflection duration experiment

The [report](../REFLECTION_DURATION_REPORT.md) compares three reflection kernels
on the same stiff curved toy, with 30 fresh matched seeds per allocation/kernel
cell. The [protocol](../REFLECTION_DURATION_PROTOCOL.md) was frozen before
validation and evaluation. All 180 CPU runs and 360 snapshots are included.

- `SUMMARY.json` preserves every estimator, budget, paired bootstrap contrast,
  posterior diagnostic and movement statistic from the completed experiment.
- `ALL_RECORDS.csv` contains one row for each of the 360 snapshots.
- `SCHEDULE.json` is the inherited contour schedule; there was no new tuning.
- `SOURCE_FREEZE.json` and `EVALUATION_RELEASE.json` bind the executed source,
  protocol, schedule and validation that preceded evaluation.
- `coordinator-audits/` contains independent raw-record, held-out, accounting,
  numerical and correction audits. `audit-sources.tar.gz` preserves the exact
  audit sources and their original hashes without reformatting them.
- `DIAGNOSTIC_CORRECTION.json` records a reporting correction to six derived
  midpoint-count/duration fields. Original records remain on dorrie. Independent
  comparison verified that all other fields and 540 raw inputs were unchanged.
- `logs/` records 29 combined tests, two additional diagnostic regression tests,
  Ruff and Flake8 checks. The movement renderer was also exercised separately.
  `coordinator-publication-checks.log` distinguishes the remote Ruff rules from
  additional local rules that report unchanged pre-existing style diagnostics.
- PNG/PDF pairs contain evidence RMSE, posterior CDFs and chain movement.
- `SHA256SUMS` verifies the curated copies in this directory.

Executed sampling/reduction source is commit
`724d5e9a94d2db417573cd8cc473224e82a60a13`. Reporting and diagnostic correction
were committed as `8de275df47f007e20b935bc0e8e5bb3c7debda4c`. The published
movement figures were subsequently rendered from the unchanged audited summary
with shorter labels; the report adds captions and links. Original rendering
hashes in `SUMMARY_READY.json` and `POSTPROCESS_SOURCE.json` describe the remote
handoff. No sampling, evidence reduction or chain replay was repeated for these
presentation changes.

Full resumable states, coordinates, evidence inputs/draws, chain keys and
diagnostics, source archives, original derived records and failed preliminary
attempts remain at
`dorrie:/largedata/albert/jaxns-reflection-duration-20261027/`.
See the [reproduction notes](../REFLECTION_DURATION_REPRODUCTION.md) for the
exact environment and record schema. `HANDOFF_FINAL.json` and
`PACKAGE_ARTIFACTS.json` describe the complete remote archives; these large
archives are retained outside Git.
