# Current paper results: SS8/CSS8 revision 2026-09-08

COMPARISON.md compares R/A/C/Cprime at240 on all four problems. SS8/CSS8 are fresh 30-seed cohorts; unchanged G8/CG8 measurements are explicitly indexed in G8_CG8_INPUTS.json. summary.json and contrasts.json retain full-precision per-seed errors and bootstrap intervals. AUDIT.json confirms240 new cells,240 reused G8/CG8 cells,480 new phases, and maximum60 workers.

paper.tex and R240_tables.tex use R240 only. Mode error is reported once per classic posterior, independently of phantom-prefix evidence conditioning. evidence_problems.pdf is the updated geometry figure. MANUSCRIPT_VALIDATION.json records source checks and the missing TeX build dependencies.

SOURCES.json in validation/ freezes the exact experiment commits. All raw State pickles, classic posteriors, and2048x10 evidence draws are retained at /largedata/albert/jaxns-ss8-css8-20260908/{R240,A240,C240,Cprime240}/{spike_slab,curved_spike_slab8}/seed-00..29. Original check logs, dispatcher events, and collector scripts are preserved in validation/. No worker source changes occurred during the batch.

Old SS8/CSS8 cells were retired by directory rename, preserving their contents. RETIRED_PATHS.json maps every old path to /largedata/albert/jaxns-superseded-SS8-CSS8-pre-20260908. Archived old comparisons must not be interpreted as measurements of the revised models.

The batch uses the jaxns_py conda environment, CPU float64, one pinned core per worker and at most60 concurrent workers. The captured dispatcher is the exact execution script. Use the frozen commits for exact-source reruns; the later documentation commits also correct the launcher's old CSS8 case name without changing the measured algorithm.
