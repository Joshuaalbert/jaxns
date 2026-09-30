"""Preview future pools from the current completed and in-progress SS10 trees."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics

from evidence10_memory import calibrate, project_tree

ROOT = Path("/largedata/albert/jaxns-evidence10-AC300-20260909")
fits, exponents, calibration = calibrate(ROOT, "0.02")
records = {}
incomplete = []
for seed in range(30):
    cell = ROOT / "0.02/ss10" / f"seed-{seed:02d}"
    if (cell / "CORE.json").exists():
        records[seed] = json.loads((cell / "CORE.json").read_text())
    else:
        incomplete.append(seed)
        prior = json.loads((ROOT / "0.05/ss10" / cell.name / "CORE.json").read_text())
        current = json.loads((cell / "progress.jsonl").read_text().splitlines()[-1])
        exponent = math.log(
            current["classic_samples"] / prior["classic_samples"]
        ) / math.log(prior["classic_log_Z_uncert"] / current["classic_log_Z_uncert"])
        exponents[seed] = max(2.0, exponent)
        current["state_capacity"] = prior["state_capacity"]
        capacity, samples = project_tree(current, 0.02, exponents[seed])
        records[seed] = dict(
            classic_samples=samples,
            state_capacity=capacity,
            classic_log_Z_uncert=0.02,
        )
forecasts = {}
for target in (".01", ".005"):
    rows = []
    for seed, previous in records.items():
        capacity, samples = project_tree(previous, float(target), exponents[seed])
        rows.append(
            dict(
                seed=seed,
                predicted_capacity=capacity,
                predicted_samples=samples,
                core_gib=fits["core"].reserve(capacity, samples),
                analysis_gib=fits["analysis"].reserve(capacity, samples),
            )
        )
        records[seed] = dict(
            classic_samples=samples,
            state_capacity=capacity,
            classic_log_Z_uncert=float(target),
        )
    selected = []
    reserved = 0
    for row in rows:
        if reserved + row["core_gib"] <= 512 and len(selected) < 60:
            selected.append(row["seed"])
            reserved += row["core_gib"]
    forecasts[target] = dict(
        rows=rows,
        first_core_pool=selected,
        first_core_pool_gib=reserved,
        oversized_analysis_seeds=[
            row["seed"] for row in rows if row["analysis_gib"] > 512
        ],
    )
result = dict(
    provisional=True,
    unfinished_0p02_seeds=incomplete,
    fits={phase: asdict(fit) for phase, fit in fits.items()},
    sample_exponent_by_seed=exponents,
    forecasts=forecasts,
)
(ROOT / "report/MEMORY_PREVIEW.json").write_text(json.dumps(result, indent=2) + "\n")
for target, f in forecasts.items():
    print(
        target,
        "first pool",
        len(f["first_core_pool"]),
        "reserved_GiB",
        f["first_core_pool_gib"],
    )
    for phase in ("core", "analysis"):
        values = [row[phase + "_gib"] for row in f["rows"]]
        print(
            phase,
            "min/median/max GiB",
            min(values),
            statistics.median(values),
            max(values),
        )
    print("oversized analyses", f["oversized_analysis_seeds"])
