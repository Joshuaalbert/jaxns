"""Run paired SS10 allocation goals from 30 immutable completed 0.02 trees.

A pair stays on one CPU and alternates which implementation runs first across
seeds. Memory reservations use the archived capacity, with headroom for full
input/output states, compilation, and accuracy reduction. No archive is edited.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--memory-gib", type=float, default=400.0)
parser.add_argument("--first-seed", type=int, default=0)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
reference = Path("/largedata/albert/git/jaxns-paper-evidence10-AC300")
archive = Path("/largedata/albert/jaxns-evidence10-AC300-20260909/0.02/ss10")
args.output.mkdir(parents=True, exist_ok=True)
manifest = {
    "mode": "one allocation goal beyond each completed 0.02 tree",
    "seeds": list(range(30)),
    "archive": str(archive),
    "candidate_root": str(root),
    "reference_root": str(reference),
    "candidate_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    "warm_repeats": 3,
    "cpus": list(range(30)),
    "memory_budget_gib": args.memory_gib,
    "memory_reservation_gib_by_capacity": {"857600": 24.0, "1715200": 44.0},
    "variant_order": "reference first for even seeds, combined first for odd seeds",
    "harness_sha256": {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in (
            "resume_goal.py",
            "accuracy.py",
            "state_fingerprints.py",
            "cohort_goals.py",
        )
    },
}
(args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
pending = []
for seed in range(args.first_seed, 30):
    metadata = json.loads((archive / f"seed-{seed:02d}/CORE.json").read_text())
    capacity = metadata["state_capacity"]
    assert capacity in (857600, 1715200)
    pending.append(
        {
            "seed": seed,
            "memory_gib": 24.0 if capacity == 857600 else 44.0,
            "variants": ["reference", "combined"]
            if seed % 2 == 0
            else ["combined", "reference"],
        }
    )
active = []
finished = []
failed = []
started = time.time()
while pending or active:
    available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    reserved = sum(job["memory_gib"] for job in active)
    for job in list(pending):
        if (
            reserved + job["memory_gib"] > args.memory_gib
            or available_kib * 1024 < (job["memory_gib"] + 48.0) * 2**30
        ):
            continue
        pending.remove(job)
        active.append(job)
        reserved += job["memory_gib"]
        job["process"] = None
    for job in list(active):
        if job["process"] is not None:
            code = job["process"].poll()
            if code is None:
                continue
            job["stream"].close()
            if code != 0:
                failed.append(
                    {
                        "seed": job["seed"],
                        "variant": job["running_variant"],
                        "exit_code": code,
                    }
                )
                active.remove(job)
                continue
            job["process"] = None
        cell = args.output / f"seed-{job['seed']:02d}"
        cell.mkdir(exist_ok=True)
        while job["variants"] and (cell / (job["variants"][0] + ".json")).exists():
            job["variants"].pop(0)
        if not job["variants"]:
            finished.append(job["seed"])
            active.remove(job)
            continue
        variant = job["variants"][0]
        source = reference if variant == "reference" else root
        env = dict(
            os.environ,
            PYTHONPATH=f"{source}/src:{source}:{root}",
            JAX_PLATFORMS="cpu",
            JAX_ENABLE_X64="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        job["stream"] = (cell / f"{variant}.log").open("w")
        job["running_variant"] = variant
        job["process"] = subprocess.Popen(
            [
                "taskset",
                "-c",
                str(job["seed"]),
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "jaxns_py",
                "python",
                str(root / "benchmarks/phantom_seeding/resume_goal.py"),
                "--state",
                str(archive / f"seed-{job['seed']:02d}/state.pkl"),
                "--variant",
                variant,
                "--ss10-accuracy",
                "--output",
                str(cell / f"{variant}.json"),
            ],
            cwd=root,
            env=env,
            stdout=job["stream"],
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(
            json.dumps(
                {
                    "event": "start",
                    "seed": job["seed"],
                    "variant": variant,
                    "pid": job["process"].pid,
                    "memory_gib": job["memory_gib"],
                }
            ),
            flush=True,
        )
    status = {
        "elapsed_seconds": time.time() - started,
        "pending": [job["seed"] for job in pending],
        "active": [
            {
                "seed": job["seed"],
                "variant": job["running_variant"],
                "pid": job["process"].pid,
                "memory_gib": job["memory_gib"],
            }
            for job in active
        ],
        "finished": sorted(finished),
        "failed": failed,
        "available_gib": available_kib / 2**20,
    }
    temp = args.output / "STATUS.tmp"
    temp.write_text(json.dumps(status, indent=2) + "\n")
    temp.replace(args.output / "STATUS.json")
    if pending or active:
        time.sleep(5)
if failed:
    raise RuntimeError(f"Failed workers: {failed}")
print(json.dumps({"event": "complete", "seeds": sorted(finished)}), flush=True)
