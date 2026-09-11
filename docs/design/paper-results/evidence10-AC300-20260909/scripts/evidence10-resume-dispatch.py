"""Finish saved-tree analyses and continue the frozen matched SS10 ladder."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SAMPLING = Path("/largedata/albert/git/jaxns-paper-evidence10-AC300")
ANALYSIS = Path("/largedata/albert/git/jaxns-paper-evidence10-analysis")
OUT = Path("/largedata/albert/jaxns-evidence10-AC300-20260909")
CPUS = sorted(os.sched_getaffinity(0))[:60]
BUDGET_GIB = 480
TRUTH_MASS = 0.5000077444258325
assert len(CPUS) == 60
source = json.loads((OUT / "SOURCES.json").read_text())
analysis_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ANALYSIS, text=True
).strip()
for root in (SAMPLING, ANALYSIS):
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root)
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD:src"], cwd=root, text=True).strip()
        == source["src_tree"]
    )
assert (
    subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SAMPLING, text=True).strip()
    == source["commit"]
)
assert json.loads((OUT / "STATUS.json").read_text())["active"] == 0
failures = json.loads((OUT / "FAILED.json").read_text())
for failure in failures:
    case, seed, phase = failure["job"]
    assert failure["stage"] == "0.05" and phase == "analysis"
    log = OUT / "0.05" / case / f"seed-{seed:02d}" / "analysis.log"
    assert "The sparse paper sweep requires singleton blocks." in log.read_text()
for case in ("g10", "cg10", "ss10"):
    assert len(list((OUT / "0.05" / case).glob("seed-*/CORE.json"))) == 30
analysis_source = dict(
    root=str(ANALYSIS),
    commit=analysis_commit,
    src_tree=source["src_tree"],
    files={
        name: hashlib.sha256(
            (ANALYSIS / "benchmarks/paper_reproduction" / name).read_bytes()
        ).hexdigest()
        for name in ("prefix_sweep.py", "run_evidence10.py")
    },
)
with (OUT / "ANALYSIS_SOURCES.json").open("x") as stream:
    json.dump({analysis_commit: analysis_source}, stream, indent=2)
retries = [
    dict(
        failure,
        resolved=False,
        analysis_commit=analysis_commit,
        reason="Extend sparse reduction to the existing plateau law",
    )
    for failure in failures
]
(OUT / "RETRIES.json").write_text(json.dumps(retries, indent=2))
(OUT / "FAILED.json").rename(OUT / "FAILED-first-dispatch.json")
started = min(
    json.loads(line)["time"]
    for line in (OUT / "dispatch.log").read_text().splitlines()
    if line.startswith("{")
)
decisions = []
finished_cells = 0
max_active = 60  # Verified maximum in the preserved first-dispatch log.


def event(kind, **record):
    print(json.dumps(dict(event=kind, time=time.time(), **record)), flush=True)


event("recovery_start", sampling_commit=source["commit"], analysis_commit=analysis_commit)
for stage in ("0.05", "0.02", "0.01", "0.005"):
    previous = {"0.02": "0.05", "0.01": "0.02", "0.005": "0.01"}.get(stage)
    cases = ("ss10", "g10", "cg10") if stage == "0.05" else ("ss10",)
    reservation = 8
    if previous:
        records = [
            json.loads(path.read_text())
            for phase in ("CORE", "ANALYSIS")
            for path in (OUT / previous / "ss10").glob(f"seed-*/{phase}.json")
        ]
        observed_peak = max(row["peak_rss_bytes"] for row in records) / 1024**3
        projected = observed_peak * (float(previous) / float(stage)) ** 2 * 1.15
        reservation = min(BUDGET_GIB, 4 * math.ceil(projected / 4))
        event(
            "memory_reservation",
            stage=stage,
            prior_peak_gib=observed_peak,
            projected_gib=projected,
            reserved_gib=reservation,
            max_workers=min(60, BUDGET_GIB // reservation),
        )
    pending = [
        (case, seed, phase)
        for phase in ("core", "analysis")
        for seed in range(30)
        for case in cases
        if not (OUT / stage / case / f"seed-{seed:02d}" / f"{phase.upper()}.json").exists()
    ]
    active = {}
    stage_failures = []
    last_audit = 0.0
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
        occupied = {tuple(row[2][:2]) for row in active.values()}
        for cpu in CPUS:
            if cpu in active or stage_failures:
                continue
            if (len(active) + 1) * reservation > BUDGET_GIB:
                break
            ready = None
            for index, job in enumerate(pending):
                case, seed, phase = job
                if tuple(job[:2]) in occupied:
                    continue
                cell = OUT / stage / case / f"seed-{seed:02d}"
                if phase == "analysis" and not (cell / "CORE.json").exists():
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
            event("affinity_audit", stage=stage, workers=affinity)
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
                    reserved_gib=len(active) * reservation,
                    active_jobs=[row[2] for row in active.values()],
                ),
                indent=2,
            )
        )
        if not active and pending:
            raise RuntimeError("Pending work cannot be dispatched; inspect disk/dependencies")
        if active:
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
