"""Use completed matched-stage measurements to size future SS10 worker pools."""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evidence10_memory import calibrate, project_tree

SAMPLING = Path("/largedata/albert/git/jaxns-paper-evidence10-AC300")
ANALYSIS = Path("/largedata/albert/git/jaxns-paper-evidence10-analysis")
OUT = Path("/largedata/albert/jaxns-evidence10-AC300-20260909")
CPUS = sorted(os.sched_getaffinity(0))[:60]
BUDGET_GIB = 512
TRUTH_MASS = 0.5000077444258325
OLD_PID = 4058031
assert len(CPUS) == 60
source = json.loads((OUT / "SOURCES.json").read_text())
analysis_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ANALYSIS, text=True,
).strip()
for root in (SAMPLING, ANALYSIS):
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root)
    assert subprocess.check_output(
        ["git", "rev-parse", "HEAD:src"], cwd=root, text=True,
    ).strip() == source["src_tree"]
assert subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=SAMPLING, text=True,
).strip() == source["commit"]
assert not (OUT / "FAILED.json").exists()
assert "T" in Path(f"/proc/{OLD_PID}/stat").read_text().split()[2]
retries = json.loads((OUT / "RETRIES.json").read_text())
assert all(row["resolved"] for row in retries)
decisions = json.loads((OUT / "DECISIONS.json").read_text())
assert [row["stage"] for row in decisions] == ["0.05"]
finished_cells = 90
max_active = 60


def event(kind, **record):
    print(json.dumps(dict(event=kind, time=time.time(), **record)), flush=True)


def memory_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable is unavailable")


class AdoptedProcess:
    """Poll an existing worker without changing its process or random state."""

    def __init__(self, pid, cell, phase):
        self.pid = pid
        self.record = cell / f"{phase.upper()}.json"
        proc = Path(f"/proc/{pid}/stat")
        self.start_ticks = proc.read_text().split()[21] if proc.exists() else None

    def poll(self):
        proc = Path(f"/proc/{self.pid}/stat")
        if proc.exists():
            fields = proc.read_text().split()
            if fields[21] == self.start_ticks and fields[2] not in ("Z", "X"):
                return None
        # The worker writes its complete phase record only after success.
        return 0 if self.record.exists() else 1


ledger = {}
started = None
for line in (OUT / "dispatch.log").read_text().splitlines():
    if not line.startswith("{"):
        continue
    row = json.loads(line)
    started = row["time"] if started is None else min(started, row["time"])
    if row["event"] == "start":
        ledger[row["cpu"]] = row
    elif row["event"] == "finish":
        assert ledger.pop(row["cpu"])["pid"] == row["pid"]
adopted = {}
for cpu, row in ledger.items():
    assert row["stage"] == "0.02"
    case, seed, phase = row["job"]
    cell = OUT / "0.02" / case / f"seed-{seed:02d}"
    adopted[cpu] = (
        AdoptedProcess(row["pid"], cell, phase),
        Path(row["log"]).open("a"), tuple(row["job"]),
    )
(OUT / "ACTIVE_SCHEDULER.json").write_text(json.dumps(dict(
    pid=os.getpid(), script=__file__, replaced_pid=OLD_PID,
    adopted_workers=[row[0].pid for row in adopted.values()],
    budget_gib=BUDGET_GIB, max_workers=60,
), indent=2))
event("scheduler_adoption", old_pid=OLD_PID, pid=os.getpid(), workers=len(adopted))
for stage in ("0.02", "0.01", "0.005"):
    previous = {"0.02": "0.05", "0.01": "0.02", "0.005": "0.01"}.get(stage)
    cases = ("ss10", "g10", "cg10") if stage == "0.05" else ("ss10",)

    fits = None
    exponents = None
    if stage != "0.02":
        fits, exponents, calibration = calibrate(OUT, previous)
        assert len(exponents) == 30
        calibration["next_stage"] = stage
        calibration_path = OUT / "report" / f"MEMORY_CALIBRATION_{stage.replace('.', 'p')}.json"
        calibration_path.write_text(json.dumps(calibration, indent=2) + "\n")
        event("memory_calibration", stage=stage, path=str(calibration_path),
              fits=calibration["fits"], sample_exponents=exponents)

    def reservation_for(job):
        case, seed, phase = job
        cell = OUT / stage / case / f"seed-{seed:02d}"
        core_path = cell / "CORE.json"
        if core_path.exists():
            record = json.loads(core_path.read_text())
            capacity = record["state_capacity"]
            predicted_samples = record["classic_samples"]
        else:
            record = json.loads(
                (OUT / previous / case / f"seed-{seed:02d}" / "CORE.json").read_text()
            )
            if fits is None:
                predicted_samples = (
                    record["classic_samples"]
                    * (record["classic_log_Z_uncert"] / float(stage)) ** 2 * 1.15
                )
            else:
                _, predicted_samples = project_tree(record, float(stage), exponents[seed])
            progress_path = cell / "progress.jsonl"
            if progress_path.exists():
                lines = progress_path.read_text().splitlines()
                if lines:
                    try:
                        current = json.loads(lines[-1])
                    except json.JSONDecodeError:
                        current = json.loads(lines[-2])
                    if fits is None:
                        current_prediction = (
                            current["classic_samples"]
                            * (current["classic_log_Z_uncert"] / float(stage)) ** 2 * 1.15
                        )
                    else:
                        current["state_capacity"] = record["state_capacity"]
                        _, current_prediction = project_tree(
                            current, float(stage), exponents[seed],
                        )
                    predicted_samples = max(predicted_samples, current_prediction)
            capacity = record["state_capacity"]
            while capacity < predicted_samples:
                capacity *= 2
        if fits is not None:
            return fits[phase].reserve(capacity, predicted_samples)
        # Keep the existing 0.02 scheduling rule for the adopted workers.
        # The intercept and slope bound the observed 0.05/0.02 core RSS.
        # Account for power-of-two buffers and retain 20% per-job headroom.
        predicted_gib = 1.2 * (3.0 + 0.0000138 * capacity)
        return min(BUDGET_GIB, max(8, math.ceil(predicted_gib)))

    pending = [
        (case, seed, phase)
        for phase in ("core", "analysis")
        for seed in range(30)
        for case in cases
        if not (OUT / stage / case / f"seed-{seed:02d}" / f"{phase.upper()}.json").exists()
    ]
    active = adopted if stage == "0.02" else {}
    active_jobs = {tuple(row[2]) for row in active.values()}
    pending = [job for job in pending if tuple(job) not in active_jobs]
    stage_failures = []
    last_audit = 0.0
    last_resource_wait = 0.0
    while pending or active:
        for cpu, (process, log, job) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            case, seed, phase = job
            cell = OUT / stage / case / f"seed-{seed:02d}"
            success = code == 0 and (cell / f"{phase.upper()}.json").exists()
            event(
                "finish",
                stage=stage,
                cpu=cpu,
                pid=process.pid,
                job=job,
                exit_code=code,
                success=success,
            )
            del active[cpu]
            if not success:
                stage_failures.append(dict(stage=stage, job=job, exit_code=code))
            for retry in retries:
                if retry["stage"] == stage and retry["job"] == list(job) and success:
                    retry["resolved"] = True
                    retry["successful_pid"] = process.pid
                    (OUT / "RETRIES.json").write_text(json.dumps(retries, indent=2))
        if stage_failures:
            pending.clear()
        reservations = {
            tuple(job): reservation_for(job)
            for job in pending + [row[2] for row in active.values()]
        }
        occupied = {tuple(row[2][:2]) for row in active.values()}
        for cpu in CPUS:
            if cpu in active or stage_failures:
                continue
            reserved = sum(reservations[row[2]] for row in active.values())
            available = memory_available_gib()
            ready = None
            for index, job in enumerate(pending):
                case, seed, phase = job
                if tuple(job[:2]) in occupied:
                    continue
                cell = OUT / stage / case / f"seed-{seed:02d}"
                if phase == "analysis" and not (cell / "CORE.json").exists():
                    continue
                reservation = reservations[job]
                if reserved + reservation > BUDGET_GIB or available < reservation + 48:
                    continue
                ready = index
                break
            if ready is None:
                continue
            if shutil.disk_usage(OUT).free < 50 * 1024**3:
                event("low_disk", free_bytes=shutil.disk_usage(OUT).free)
                break
            job = pending.pop(ready)
            case, seed, phase = job
            root = SAMPLING if phase == "core" else ANALYSIS
            cell = OUT / stage / case / f"seed-{seed:02d}"
            cell.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(
                {
                    name: "1"
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "BLIS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "JAX_NUM_THREADS",
                        "TF_NUM_INTRAOP_THREADS",
                        "TF_NUM_INTEROP_THREADS",
                    )
                }
            )
            env.update(
                JAX_PLATFORMS="cpu",
                JAX_ENABLE_X64="true",
                PYTHONUNBUFFERED="1",
                MPLBACKEND="Agg",
                PYTHONPATH=f"{root / 'src'}:{root}",
                XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
            )
            cmd = [
                "taskset",
                "-c",
                str(cpu),
                sys.executable,
                "-m",
                "benchmarks.paper_reproduction.run_evidence10",
                "--case",
                case,
                "--seed",
                str(seed),
                "--phase",
                phase,
                "--goal-log-z-uncert",
                stage,
                "--output",
                str(cell),
            ]
            if previous and phase == "core":
                cmd.extend(["--resume-from", str(OUT / previous / case / f"seed-{seed:02d}")])
            log_path = cell / f"{phase}.log"
            if log_path.exists():
                log_path = cell / f"{phase}.retry-01.log"
            log = log_path.open("x")
            process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            os.sched_setaffinity(process.pid, {cpu})
            active[cpu] = (process, log, job)
            occupied.add(tuple(job[:2]))
            event(
                "start",
                stage=stage,
                cpu=cpu,
                pid=process.pid,
                job=job,
                root=str(root),
                log=str(log_path),
                reserved_gib=reservation,
            )
        max_active = max(max_active, len(active))
        if time.time() - last_audit > 30:
            affinity = []
            for cpu, (process, _, job) in active.items():
                try:
                    actual = sorted(os.sched_getaffinity(process.pid))
                except ProcessLookupError:
                    continue
                if actual != [cpu]:
                    raise RuntimeError(f"Affinity changed: {process.pid}: {actual}")
                affinity.append(dict(pid=process.pid, cpu=cpu, job=job))
            event(
                "affinity_audit", stage=stage, workers=affinity,
                reserved_gib=sum(reservations[row[2]] for row in active.values()),
                available_gib=memory_available_gib(),
            )
            last_audit = time.time()
        (OUT / "STATUS.json").write_text(
            json.dumps(
                dict(
                    stage=stage,
                    active=len(active),
                    pending=len(pending),
                    failures=stage_failures,
                    elapsed_seconds=time.time() - started,
                    max_active=max_active,
                    reserved_gib=sum(reservations[row[2]] for row in active.values()),
                    active_jobs=[row[2] for row in active.values()],
                ),
                indent=2,
            )
        )
        if not active and pending:
            wait = dict(
                stage=stage, budget_gib=BUDGET_GIB, available_gib=memory_available_gib(),
                jobs=[dict(job=job, required_gib=reservations[job]) for job in pending],
                reason="Pending phases exceed the memory/disk currently available",
            )
            (OUT / "RESOURCE_WAIT.json").write_text(json.dumps(wait, indent=2) + "\n")
            if time.time() - last_resource_wait > 300:
                event("resource_wait", **wait)
                last_resource_wait = time.time()
            time.sleep(30)
        elif active:
            (OUT / "RESOURCE_WAIT.json").unlink(missing_ok=True)
            time.sleep(2)
    if stage_failures:
        (OUT / "FAILED.json").write_text(json.dumps(stage_failures, indent=2))
        raise RuntimeError(f"Worker failures: {stage_failures}")
    for case in cases:
        assert len(list((OUT / stage / case).glob("seed-*/ANALYSIS.json"))) == 30
    finished_cells += 30 * len(cases)
    masses = [
        json.loads((OUT / stage / "ss10" / f"seed-{seed:02d}" / "ANALYSIS.json").read_text())[
            "classic_spike_mass"
        ]
        for seed in range(30)
    ]
    errors = [mass - TRUTH_MASS for mass in masses]
    rmse = (sum(error**2 for error in errors) / 30) ** 0.5
    max_error = max(abs(error) for error in errors)
    recovered = rmse <= 0.05 and max_error <= 0.15
    decision = dict(
        stage=stage,
        masses=masses,
        mass_rmse=rmse,
        max_mass_error=max_error,
        recovered=recovered,
        continue_ladder=stage == "0.05" or (not recovered and stage != "0.005"),
    )
    decisions.append(decision)
    (OUT / "DECISIONS.json").write_text(json.dumps(decisions, indent=2))
    event("stage_complete", **decision)
    if not decision["continue_ladder"]:
        break
assert all(retry["resolved"] for retry in retries)
(OUT / "FINISHED.json").write_text(
    json.dumps(
        dict(
            complete=True,
            cells=finished_cells,
            stages=[d["stage"] for d in decisions],
            max_parallel=max_active,
            cpus=CPUS,
            elapsed_seconds=time.time() - started,
            ss10_recovered=decisions[-1]["recovered"],
        ),
        indent=2,
    )
)
event("finished", cells=finished_cells)
