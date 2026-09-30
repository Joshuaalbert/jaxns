"""Publish reports as each complete cohort arrives, then build the final paper."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path("/largedata/albert/jaxns-evidence10-AC300-20260909")
SCRIPTS = OUT / "report/scripts"
assert len(os.sched_getaffinity(0)) == 1
env = os.environ.copy()
env.update(
    {
        name: "1"
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
)
env["MPLBACKEND"] = "Agg"
seen = []
while True:
    if (OUT / "FAILED.json").exists():
        raise RuntimeError("Sampling failed; no complete final report is claimed")
    cohorts = [
        (stage, case)
        for stage in ("0.05", "0.02", "0.01", "0.005")
        for case in (("g10", "cg10", "ss10") if stage == "0.05" else ("ss10",))
        if len(list((OUT / stage / case).glob("seed-*/ANALYSIS.json"))) == 30
    ]
    finished = (OUT / "FINISHED.json").exists()
    archived = (OUT / "ARCHIVE_FINISHED.json").exists()
    if cohorts != seen or (finished and archived):
        final = finished and archived
        cmd = [sys.executable, str(SCRIPTS / "evidence10-report.py"), "--root", str(OUT)]
        if not final:
            cmd.append("--completed-only")
        subprocess.run(cmd, env=env, check=True)
        subprocess.run(
            [sys.executable, str(SCRIPTS / "evidence10-plot.py"), "--root", str(OUT)],
            env=env,
            check=True,
        )
        seen = cohorts
        print(json.dumps(dict(completed_cohorts=cohorts, final=final)), flush=True)
        if all(("0.05", case) in cohorts for case in ("g10", "cg10", "ss10")):
            cmd = [sys.executable, str(SCRIPTS / "evidence10-paper.py")]
            if not final:
                cmd.append("--completed-only")
            subprocess.run(cmd, env=env, check=True)
        if final:
            (OUT / "REPORT_FINISHED.json").write_text(
                json.dumps(
                    dict(complete=True, report=str(OUT / "report"), cohorts=cohorts), indent=2
                )
            )
            break
    time.sleep(10)
