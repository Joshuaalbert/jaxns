# SS10 ladder: stopped at the user's request

On **2026-09-10 at 12:06:44 UTC**, the user cancelled the remaining ladder
because its estimated runtime was too long. All seven 0.01 sampling workers,
the dispatcher, archiver, report builder, and finalizer were terminated.
No targeted process remained active; no forced kill was needed.

The G10/CG10/SS10 0.05 cohorts and the SS10 0.02 cohort each have all 30
sampling results and analyses. No 0.01 run reached its goal; seeds 0–6 have
preserved full-capacity checkpoints. The 0.005 stage never started.
Checkpoints cover their last periodic saves, not every logged goal.

[CANCELLED.json](CANCELLED.json) records the stopped processes, checkpoint
paths/timestamps, completed counts, and previous queue state. The authoritative
output root is `/largedata/albert/jaxns-evidence10-AC300-20260909`; its
`STATUS.json` is cancelled. The completed 0.02 report is under that root's
`report/`. The manuscript and installed 120-cell snapshot now include both
completed stops, with revised interpretation and paired-bias contrasts.

Do not restart any dispatcher or helper without a new user instruction.
The implementation/design discussion continues in
[the revised phantom-seeding proposal](../../efficient-phantom-seeding.md).
The notes below describe historical operation, not currently active jobs.

## Historical operational handoff

Snapshot: 2026-09-09, after baseline commit `22d6bf3`.

The 90 baseline trees and analyses are complete. The 0.02 stage is running
30 matched SS10 continuations, ten at a time; later stages are conditional
on the descriptive spike-recovery gate in PROTOCOL.md. The sampler and model
sources remain frozen. No additional seed-pool or allocation changes are made.

All live outputs are under:
`/largedata/albert/jaxns-evidence10-AC300-20260909`.

- `STATUS.json`: current stage, active and pending phases, failures, reservations.
- `dispatch.log`: every start, finish, affinity check, and stage decision.
- `DECISIONS.json`: completed stage masses and the continuation decision.
- `FINISHED.json`: appears only after the requested ladder completes or recovers.
- `ARCHIVE_FINISHED.json`: verified full-capacity states archived to Ceph.
- `REPORT_FINISHED.json`: complete cohorts independently audited and plotted.
- `PUBLISHED.json`: final report and paper installed into this checkout.
- `PUBLISH_READY.json`: final report ready, but intervening paper edits were
  detected and preserved instead of overwritten.
- `FAILED.json`: a new worker failure; do not claim completion if present.
  The old reducer failure is separately preserved as `FAILED-first-dispatch.json`.

Active scripts are the durable copies in `report/scripts/`; running processes
were launched from corresponding `/tmp` copies. Do not rerun the original
recovery dispatcher: its initial recovery block is specific to the recorded
singleton-reducer failure. Use saved full-capacity checkpoints if recovery
from a new failure becomes necessary; trimming changes subsequent sampling.

The finalizer waits for the full audited report and validates local LaTeX
structure after installation. It checks hashes of the baseline paper, report
index, report README, and changed manuscript figures before writing. If those
have changed, it leaves the generated report under `report/` for review.
It does not create a new commit or alter the frozen experiment branches.

The long-running tools sessions at this snapshot are dispatcher 73508,
archiver 96424, report builder 93985, and finalizer 75021. The raw process
logs are `dispatch.log`, `archive.log`, `postprocess-recovered.log`, and
`finalize.log`. Every compute worker is pinned to one CPU; helper scripts
use CPUs 64–66. At most 60 compute workers are permitted.

## Coordinator replacement at 16:14 UTC

The global 44 GiB reservation was too conservative: completed 0.02 cores peak
around 15 GiB. `evidence10-live-dispatch.py` now estimates per-tree reservations
from projected full state capacity. It adopted the existing workers without
restarting them, retired only the old coordinator process, and raised active
sampling concurrency to 22 immediately. Three sampling jobs remained queued
at replacement, with five cores complete. The limit remains 60 distinct cores,
subject to a 512 GiB memory reservation budget and a 48 GiB available-memory
margin. See PROTOCOL.md for the estimate and its measured calibration.

Current dispatcher tools session: 92891. `ACTIVE_SCHEDULER.json` records its
PID and adopted workers. Other helpers and output paths are unchanged. The
old dispatcher session 73508 is intentionally retired. The new dispatcher
includes a one-time adoption block; do not launch another copy against live
workers. Its events continue in the same dispatch.log for the existing audit.

## Calibrated future pools at 23:08 UTC

Dispatcher session 73608 (`evidence10-calibrated-dispatch.py`) now owns the
same live workers. Session 92891 and its coordinator PID 4058031 were retired;
no sampling or analysis worker was restarted. The active coordinator PID is
recorded in ACTIVE_SCHEDULER.json (4092075 at replacement).

The 0.02 scheduling rule is unchanged. Before 0.01 and 0.005, the dispatcher
fits phase-specific memory models from completed matched stages and records
MEMORY_CALIBRATION_<target>.json. See MEMORY_FORECAST.md and the frozen
107-observation calibration/test snapshot for current estimates. Predicted
reservations are uncapped: RESOURCE_WAIT.json records any eventual phase that
cannot fit, rather than launching it with an underestimated reservation.
The final installer includes all MEMORY_* report artifacts automatically.
