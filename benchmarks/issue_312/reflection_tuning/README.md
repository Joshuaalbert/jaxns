# Audited reflection step-size experiment

See the [report](../REFLECTION_TUNING_REPORT.md) and prespecified
[protocol](../REFLECTION_TUNING_PROTOCOL.md) for the scientific comparison.
These are exact copies of the completed CPU experiment's compact results;
the coordinator independently audited calibration, held-out transitions,
all 120 nested runs, and all 240 snapshots before publication.

- `SUMMARY.json` contains both budgets, every estimator, paired bootstrap
  contrasts, posterior errors, movement diagnostics and execution costs.
- `SCHEDULE.json` is the schedule frozen before evaluation, including the
  per-contour calibration history. It targets 70% endpoint acceptance.
- `NORMALIZATION.json` records the likelihood normalization and float64 audit.
- PNG/PDF pairs show evidence error, posterior CDFs and actual chain movement.
- `coordinator-audits/` contains three independent audit receipts and the
  exact audit scripts in `audit-sources.tar.gz`, with their original hashes.
  These scripts read the full raw output tree on dorrie and use independent
  NumPy/SciPy calculations, without importing the experiment's helpers.
- `final-combined-checks.log` records the 24 passing CPU tests.
- `PROVENANCE_FINAL.json` and `HANDOFF_FINAL.json` preserve the remote agent's
  handoff before the coordinator imported and published the work. Their
  report hash refers to the original report at commit `5390065`; the versioned
  report now also links these artifacts and embeds the reviewed figures.
- `SHA256SUMS` verifies the copies in this directory.

The complete raw output remains at
`dorrie:/largedata/albert/jaxns-reflection-tuning-20261027/`.
The 208 MiB `compact-records-20261027.tar.gz` there contains all per-run
manifests, records, evidence draws and chain diagnostics. The 6 MiB
`review-provenance-20261027.tar.gz` contains source, logs and validation
receipts. Their checksums and content inventories are recorded in the
handoff. Full classic coordinates, states, evidence inputs, pilot arrays
and held-out arrays remain separately in the same output tree.

Scientific source is commit `da352c53ab98ecee12932abdbb2dd83ed670a689`.
Only reporting and publication artifacts changed afterwards; no numerical
experiment was rerun or retuned from its evaluation results.
