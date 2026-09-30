"""Move completed, no-longer-needed local states to verified Ceph copies."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

OUT = Path("/largedata/albert/jaxns-evidence10-AC300-20260909")
DEST = Path("/ceph_dorrie/albert/jaxns-evidence10-AC300-20260909")
assert len(os.sched_getaffinity(0)) == 1
DEST.mkdir(parents=True, exist_ok=True)
while True:
    finished = (OUT / "FINISHED.json").exists()
    candidates = []
    for stage in ("0.05", "0.02", "0.01", "0.005"):
        for analysis in sorted((OUT / stage).glob("*/seed-*/ANALYSIS.json")):
            cell = analysis.parent
            state = cell / "state.pkl"
            if state.is_symlink() or not state.exists():
                continue
            next_stage = {"0.05": "0.02", "0.02": "0.01", "0.01": "0.005"}.get(stage)
            later_complete = (
                next_stage
                and (OUT / next_stage / cell.parent.name / cell.name / "CORE.json").exists()
            )
            local_pressure = shutil.disk_usage(OUT).free < 300 * 1024**3
            if (
                local_pressure
                or cell.parent.name != "ss10"
                or stage == "0.005"
                or later_complete
                or finished
            ):
                candidates.append(cell)
    for cell in candidates:
        started = time.perf_counter()
        state = cell / "state.pkl"
        target = DEST / cell.relative_to(OUT) / "state.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".copying")
        assert not target.exists() and not temporary.exists()
        with state.open("rb") as stream:
            expected = hashlib.file_digest(stream, "sha256").hexdigest()
        with state.open("rb") as source, temporary.open("xb") as dest:
            shutil.copyfileobj(source, dest, length=16 * 1024**2)
            dest.flush()
            os.fsync(dest.fileno())
        with temporary.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected != actual:
            raise RuntimeError(f"Archive checksum mismatch: {cell}")
        temporary.replace(target)
        record = dict(
            source=str(state),
            target=str(target),
            bytes=target.stat().st_size,
            sha256=expected,
            verified_after_copy=True,
            seconds=time.perf_counter() - started,
            affinity=sorted(os.sched_getaffinity(0)),
        )
        with (cell / "STATE_ARCHIVE.json").open("x") as stream:
            json.dump(record, stream, indent=2)
        # Replace each hardlink only after a complete independently verified
        # copy exists. The original paths continue to load the identical pickle.
        for name in ("state.pkl", "checkpoint.pkl"):
            link = cell / (name + ".archive-link")
            link.symlink_to(target)
            link.replace(cell / name)
        print(json.dumps(record), flush=True)
    if finished:
        (OUT / "ARCHIVE_FINISHED.json").write_text(
            json.dumps(
                dict(
                    complete=True,
                    destination=str(DEST),
                    states=len(list(OUT.glob("*/*/seed-*/STATE_ARCHIVE.json"))),
                ),
                indent=2,
            )
        )
        break
    if (OUT / "FAILED.json").exists():
        raise RuntimeError("Sampling failed; preserve remaining local states for diagnosis")
    time.sleep(30)
