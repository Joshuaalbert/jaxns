"""Run 30 pinned SS10 continuations with a budget for projected final memory.

Longer projected jobs enter first. Reservations include future buffer growth,
checkpointing and analysis; instantaneous free RAM alone cannot set concurrency.
The final summary compares classic accuracy with the archived 0.02 baseline.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--forecast", type=Path, required=True)
parser.add_argument("--memory-gib", type=float, default=400.)
parser.add_argument("--max-workers", type=int, default=60)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
if args.memory_gib < 80 or not 1 <= args.max_workers <= 60:
    parser.error("Reserve at least 80 GiB and use between 1 and 60 workers")
root = Path(__file__).resolve().parents[2]
archive = Path("/largedata/albert/jaxns-evidence10-AC300-20260909/0.02/ss10")
forecast = json.loads(args.forecast.read_text())
assert sorted(row["seed"] for row in forecast) == list(range(30))
args.output.mkdir(parents=True, exist_ok=True)
source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
runner = Path(__file__).with_name("continue_ss10.py")
runner_hash = hashlib.sha256(runner.read_bytes()).hexdigest()
# Give all currently large trees another doubling of headroom. Seed 18 had a
# transient allocation stall, so its forecast receives the same larger reserve.
large_seeds = [3, 8, 10, 14, 16, 18, 21, 23, 26, 29]
manifest = dict(
    target=.015, initial_target=.02, seeds=list(range(30)), source_commit=source_commit,
    sampler_root=str(root), archive=str(archive), runner_sha256=runner_hash,
    dispatcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    memory_budget_gib=args.memory_gib, max_workers=args.max_workers,
    reservation_gib={str(seed): 80. if seed in large_seeds else 44. for seed in range(30)},
    forecast=forecast, cpus=list(range(30)), final_analysis="classic evidence and spike mass",
    phantom_prefix_analysis=False,
)
if args.resume:
    assert json.loads((args.output / "MANIFEST.json").read_text()) == manifest
else:
    with (args.output / "MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
pending = sorted(forecast, key=lambda row: row["hours"], reverse=True)
active = []
finished = []
failed = []
started = time.time()
while pending or active:
    for job in list(active):
        code = job["process"].poll()
        if code is None:
            continue
        job["stream"].close()
        active.remove(job)
        if code == 0 and (args.output / f"seed-{job['seed']:02d}/CORE.json").exists():
            finished.append(job["seed"])
        else:
            failed.append(dict(seed=job["seed"], exit_code=code))
        print(json.dumps(dict(event="worker_exit", seed=job["seed"], exit_code=code)), flush=True)
    available_gib = next(
        int(line.split()[1]) / 2**20 for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    reserved = sum(job["memory_gib"] for job in active)
    admission_available = available_gib
    for prediction in list(pending):
        seed = prediction["seed"]
        cell = args.output / f"seed-{seed:02d}"
        if (cell / "CORE.json").exists():
            finished.append(seed)
            pending.remove(prediction)
            continue
        need = manifest["reservation_gib"][str(seed)]
        if (len(active) >= args.max_workers or reserved + need > args.memory_gib
                or admission_available < need + 48.
                or shutil.disk_usage(args.output).free < 100 * 2**30):
            continue
        assert hashlib.sha256(runner.read_bytes()).hexdigest() == runner_hash
        cell.mkdir(exist_ok=True)
        command = [
            "taskset", "-c", str(seed), "conda", "run", "--no-capture-output", "-n", "jaxns_py",
            "python", str(runner), "--seed", str(seed), "--target", ".015",
            "--input", str(archive / f"seed-{seed:02d}"), "--output", str(cell),
        ]
        if (cell / "MANIFEST.json").exists():
            command.append("--resume")
        env = dict(os.environ, PYTHONPATH=f"{root}/src:{root}", JAX_PLATFORMS="cpu",
                   JAX_ENABLE_X64="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        stream = (cell / "core.log").open("a")
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        active.append(dict(seed=seed, process=process, stream=stream, memory_gib=need))
        pending.remove(prediction)
        reserved += need
        admission_available -= need
        print(json.dumps(dict(event="start", seed=seed, pid=process.pid,
                              cpu=seed, memory_gib=need)), flush=True)
    live = []
    for job in active:
        progress = args.output / f"seed-{job['seed']:02d}/progress.jsonl"
        latest = None
        if progress.exists():
            # A writer can be midway through its last line. Only read complete lines.
            lines = progress.read_text().splitlines(keepends=True)
            complete_lines = [line for line in lines if line.endswith("\n")]
            if complete_lines:
                latest = json.loads(complete_lines[-1])
        live.append(dict(seed=job["seed"], pid=job["process"].pid, cpu=job["seed"],
                         memory_gib=job["memory_gib"], latest=latest))
    status = dict(elapsed_seconds=time.time() - started, started_unix=started,
                  pending=[row["seed"] for row in pending], active=live,
                  finished=sorted(finished), failed=failed, available_gib=available_gib,
                  reserved_gib=reserved, free_disk_gib=shutil.disk_usage(args.output).free / 2**30)
    temporary = args.output / "STATUS.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(args.output / "STATUS.json")
    if pending or active:
        time.sleep(5)
if failed:
    raise RuntimeError(f"Failed workers; inspect their logs and checkpoints: {failed}")
records = []
for seed in range(30):
    result = json.loads((args.output / f"seed-{seed:02d}/CORE.json").read_text())
    baseline = json.loads((archive / f"seed-{seed:02d}/CORE.json").read_text())
    baseline_analysis = json.loads((archive / f"seed-{seed:02d}/ANALYSIS.json").read_text())
    assert result["complete"] and result["classic_log_Z_uncert"] < .015
    records.append(dict(seed=seed, result=result, baseline=dict(
        log_Z=baseline["classic_log_Z"], log_Z_uncert=baseline["classic_log_Z_uncert"],
        spike_mass=baseline_analysis["classic_spike_mass"],
    )))
summary = dict(complete=True, seeds=30, elapsed_seconds=time.time() - started, targets={})
for target in ("0.02", "0.015"):
    values = [row["baseline"] if target == "0.02" else row["result"]["accuracy"]
              for row in records]
    truth = records[0]["result"]["accuracy"]
    error = np.array([row["log_Z"] - truth["true_log_Z"] for row in values])
    spike_error = np.array([row["spike_mass"] - truth["true_spike_mass"] for row in values])
    sigma = np.array([row["log_Z_uncert"] for row in values])
    summary["targets"][target] = dict(
        classic_log_Z_bias=float(error.mean()),
        classic_log_Z_rmse=float(np.sqrt((error**2).mean())),
        spike_mass_rmse=float(np.sqrt((spike_error**2).mean())),
        mean_spike_mass=float(np.mean([row["spike_mass"] for row in values])),
        classic_95pct_coverage=float(np.mean(np.abs(error) <= 1.96 * sigma)),
    )
summary["records"] = records
(args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(dict(event="complete", targets=summary["targets"])), flush=True)
