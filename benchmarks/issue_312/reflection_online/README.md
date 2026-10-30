# Audited online reflection experiment

The [report](../REFLECTION_ONLINE_REPORT.md) includes every prespecified arm, allocation, estimator and budget. See the [protocol](../REFLECTION_ONLINE_PROTOCOL.md) and [reproduction notes](../REFLECTION_ONLINE_REPRODUCTION.md).

There are 180 CPU runs: 30 independent evaluation seed groups paired across three arms and two allocations, with 360 snapshots. Four independent training seeds are paired across allocations and disjoint from evaluation. This is not 180 independent observations or eight mutually independent training seeds.

- SUMMARY.json and ALL_RECORDS.csv preserve all estimates, costs, diagnostics and 100,000-resample paired contrasts. Every seed is included. Online-versus-fixed primary intervals include zero; the frozen evidence-allocation comparison improves in this cohort, without repairing narrow flat marginals or establishing adaptive validity.
- Source, protocol, training/evaluation releases, frozen actual-likelihood policies and validation are preserved. SCIENTIFIC_COMPLETION.json binds all 180 completed reductions and confirms clean source47a9384 before any post-run edits.
- AGGREGATE_AUDIT.json independently checks raw draws, classic posterior quantities, initial-population pairing and all 48 contrasts without importing the report/reducer implementation. Coordinator sources/receipts verify every chronological history, training, policy construction and held-out law. FIRST100 is explicitly partial; EVALUATION_FINAL_LEDGER_AUDIT covers all180.
- diagnostic-repair/ preserves the failed reduction and exact retry receipt. Two undefined squared-coordinate correlations became null; all raw data stayed unchanged. The exact executed wrapper is in audit-sources.tar.gz, not a reusable serializer implementation. A direct typed correlation helper/regression was added only after all sampling and reduction.
- All45 final tests passed. PERFORMANCE_INTENT_REVIEW.md distinguishes measured costs, compiler plans and inferred limitations. Full-process compile time is not subtracted from the differently scoped core timer.
- Five PNG/PDF pairs show evidence, posterior CDFs, gap/compression movement and policy histories. PRESENTATION_REVIEW.json verifies that the frozen-axis display correction changed no numerical summary fields. W1 retains the original 8,192-midpoint-quantile approximation.
- SHA256SUMS verifies every curated file. Exact audit scripts are archived unchanged with their source digests. No full states, posterior coordinate arrays or raw evidence inputs are committed to Git.

All complete raw outputs, checkpoints, failed/preliminary attempts, frozen source archive and compact handoff packages remain under dorrie:/largedata/albert/jaxns-reflection-online-20261029/. HANDOFF_FINAL.json records package paths and SHA256s. The workstation coordinator owns publication; no remote GitHub writes, push or paper edits occurred.
