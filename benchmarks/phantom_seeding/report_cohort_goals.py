"""Collect paired allocation-goal timings, state equality and SS10 accuracy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--partial", action="store_true")
args = parser.parse_args()
pairs = []
records = []
for seed in range(30):
    cell = args.input / f"seed-{seed:02d}"
    if not all(
        (cell / f"{variant}.json").exists() for variant in ("reference", "combined")
    ):
        continue
    old = json.loads((cell / "reference.json").read_text())
    new = json.loads((cell / "combined.json").read_text())
    for field in (
        "fingerprints",
        "classics",
        "capacity",
        "calls",
        "new_calls",
        "goals",
        "random_key",
        "goal_key",
        "input_classics",
        "input_capacity",
        "state_path",
    ):
        assert old[field] == new[field], (seed, field)
    for field in old["accuracy"]:
        if field != "analysis_seconds":
            np.testing.assert_array_equal(
                old["accuracy"][field],
                new["accuracy"][field],
                err_msg=f"Seed {seed}, accuracy {field}",
            )
    pairs.append(
        {
            "seed": seed,
            "input_capacity": old["input_capacity"],
            "input_classics": old["input_classics"],
            "classics": old["classics"],
            "new_calls": old["new_calls"],
            "goals": old["goals"],
            "new_classics": old["classics"] - old["input_classics"],
            "old_seconds": old["median_seconds"],
            "new_seconds": new["median_seconds"],
            "speedup": old["median_seconds"] / new["median_seconds"],
            "old_warm_rss_bytes": old["warm_peak_rss_bytes"],
            "new_warm_rss_bytes": new["warm_peak_rss_bytes"],
            "memory_ratio": new["warm_peak_rss_bytes"] / old["warm_peak_rss_bytes"],
            "log_Z": old["accuracy"]["log_Z"],
            "log_Z_uncert": old["accuracy"]["log_Z_uncert"],
            "log_Z_error": old["accuracy"]["log_Z_error"],
            "spike_mass": old["accuracy"]["spike_mass"],
            "spike_mass_error": old["accuracy"]["spike_mass_error"],
            "scientific_fields_and_accuracy_exact": True,
        }
    )
    records.append({"seed": seed, "reference": old, "combined": new})
if not args.partial:
    assert len(pairs) == 30, f"Only {len(pairs)}/30 pairs complete"
if not pairs:
    print("No complete pairs yet")
    raise SystemExit(0)
summary = {"completed_pairs": len(pairs), "all_fields_exact": True}
for field in (
    "old_seconds",
    "new_seconds",
    "speedup",
    "old_warm_rss_bytes",
    "new_warm_rss_bytes",
    "memory_ratio",
    "new_calls",
    "spike_mass",
    "log_Z_uncert",
):
    values = np.asarray([pair[field] for pair in pairs])
    summary[field] = {
        "median": float(np.median(values)),
        "quartiles": np.quantile(values, [0.25, 0.75]).tolist(),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }
summary["capacity_groups"] = []
for capacity in sorted({pair["input_capacity"] for pair in pairs}):
    group = [pair for pair in pairs if pair["input_capacity"] == capacity]
    summary["capacity_groups"].append(
        {
            "capacity": capacity,
            "seeds": len(group),
            "old_median_seconds": float(
                np.median([pair["old_seconds"] for pair in group])
            ),
            "new_median_seconds": float(
                np.median([pair["new_seconds"] for pair in group])
            ),
            "old_median_warm_rss_bytes": float(
                np.median(
                    [pair["old_warm_rss_bytes"] for pair in group],
                )
            ),
            "new_median_warm_rss_bytes": float(
                np.median(
                    [pair["new_warm_rss_bytes"] for pair in group],
                )
            ),
            "geometric_mean_speedup": float(
                np.exp(
                    np.mean(
                        np.log(
                            [pair["speedup"] for pair in group],
                        )
                    )
                )
            ),
        }
    )
for field in ("log_Z_error", "spike_mass_error"):
    error = np.asarray([pair[field] for pair in pairs])
    summary[field] = {
        "bias": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }
error = np.asarray([pair["log_Z_error"] for pair in pairs])
sigma = np.asarray([pair["log_Z_uncert"] for pair in pairs])
summary["classic_95pct_coverage"] = float(np.mean(np.abs(error) <= 1.96 * sigma))
summary["below_002_uncertainty"] = int(np.count_nonzero(sigma <= 0.02))
summary["mean_spike_mass"] = float(np.mean([pair["spike_mass"] for pair in pairs]))
# Resample seed pairs together; repetitions within one seed are not independent
# scientific replications. The confidence interval describes these CPU timings.
speedups = np.asarray([pair["speedup"] for pair in pairs])
selection = np.random.default_rng(292).integers(0, len(pairs), size=(20000, len(pairs)))
bootstrap = np.exp(np.log(speedups[selection]).mean(axis=1))
summary["total_old_goal_seconds"] = sum(pair["old_seconds"] for pair in pairs)
summary["total_new_goal_seconds"] = sum(pair["new_seconds"] for pair in pairs)
summary["total_work_speedup"] = (
    summary["total_old_goal_seconds"] / summary["total_new_goal_seconds"]
)
for field in ("log_Z_error", "spike_mass_error"):
    errors = np.asarray([pair[field] for pair in pairs])[selection]
    summary[field]["bias_bootstrap_95pct"] = np.quantile(
        errors.mean(axis=1),
        [0.025, 0.975],
    ).tolist()
    summary[field]["rmse_bootstrap_95pct"] = np.quantile(
        np.sqrt(np.mean(errors**2, axis=1)),
        [0.025, 0.975],
    ).tolist()
summary["geometric_mean_speedup"] = float(np.exp(np.log(speedups).mean()))
summary["geometric_mean_speedup_bootstrap_95pct"] = np.quantile(
    bootstrap, [0.025, 0.975]
).tolist()
archive = Path("/largedata/albert/jaxns-evidence10-AC300-20260909/0.02/ss10")
archived_accuracy = [
    json.loads((archive / f"seed-{seed:02d}/ANALYSIS.json").read_text())
    for seed in range(30)
]
control = {
    variant: json.loads(
        (args.input / "scale005-seed-00" / f"{variant}.json").read_text()
    )
    for variant in ("reference", "combined")
}
assert control["reference"]["fingerprints"] == control["combined"]["fingerprints"]
source_manifests = {}
for record in records:
    for variant in ("reference", "combined"):
        hashes = record[variant].pop("source_sha256")
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        source_manifests[digest] = hashes
        record[variant]["source_sha256_manifest"] = digest
output = {
    "scaling_control_005_seed0": control,
    "source_manifests": source_manifests,
    "machine": json.loads((args.input / "MACHINE.json").read_text()),
    "interpretation": (
        "One further goal iteration from each completed 0.02 tree, not fresh full runs"
    ),
    "complete": len(pairs) == 30,
    "summary": summary,
    "pairs": pairs,
    "records": records,
    "archived_input_accuracy": archived_accuracy,
    "manifest": json.loads((args.input / "MANIFEST.json").read_text()),
    "memory_note": "Use warm_peak_rss_bytes. Per-run marker resets mean the legacy "
    "total_peak_rss_bytes "
    "field is not a lifetime process maximum in this cohort.",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
lines = [
    "# SS10 0.02: paired complete goal iterations",
    "",
    (
        f"Completed pairs: {len(pairs)}/30. Every completed pair has identical scientific fields, "
        "keys, likelihood counts and classic posterior/evidence accuracy."
    ),
    "",
    (
        "Each seed starts from its archived completed 0.02 tree and advances exactly one "
        "additional "
        "goal iteration (goal_loop_iter increases by one, including any internal "
        "allocation rounds). "
        "This is not timing from initialization to the uncertainty target. "
        "Both implementations run serially on the same pinned CPU, with balanced run order and "
        "one cold plus three warm repetitions. Old states use frozen aa79a0d; candidate sampler "
        "code is unchanged from 650ae05."
    ),
    "",
    (
        "| Capacity | Seeds | Median old s | Median new s | Paired geometric speedup | "
        "Median old warm GiB | Median new warm GiB |"
    ),
    "|---|---:|---:|---:|---:|---:|---:|",
]
for group in summary["capacity_groups"]:
    lines.append(
        f"| {group['capacity']:,} | {group['seeds']} | {group['old_median_seconds']:.2f} | "
        f"{group['new_median_seconds']:.2f} | {group['geometric_mean_speedup']:.2f}x | "
        f"{group['old_median_warm_rss_bytes'] / 2**30:.2f} | "
        f"{group['new_median_warm_rss_bytes'] / 2**30:.2f} |",
    )
lines.extend(
    [
        "",
        (
            "| Seed | Capacity | Old s | New s | Speedup | Old warm GiB | New warm GiB | "
            "Spike mass, both |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
)
for pair in pairs:
    lines.append(
        f"| {pair['seed']} | {pair['input_capacity']:,} | {pair['old_seconds']:.2f} | "
        f"{pair['new_seconds']:.2f} | {pair['speedup']:.2f}x | "
        f"{pair['old_warm_rss_bytes'] / 2**30:.2f} | {pair['new_warm_rss_bytes'] / 2**30:.2f} | "
        f"{pair['spike_mass']:.6f} |",
    )
ci = summary["geometric_mean_speedup_bootstrap_95pct"]
lines.extend(
    [
        "",
        (
            f"Geometric mean speedup: {summary['geometric_mean_speedup']:.3f}x; "
            f"paired-seed bootstrap 95% interval [{ci[0]:.3f}, {ci[1]:.3f}]. "
            "This interval describes variation across the measured pairs; shared-host load changed "
            "as workers finished and is not covered as an independent systematic uncertainty."
        ),
        "",
        (
            f"Classic log-Z bias: {summary['log_Z_error']['bias']:.6f}; "
            f"RMSE: {summary['log_Z_error']['rmse']:.6f}. "
            f"Spike-mass RMSE: {summary['spike_mass_error']['rmse']:.6f}. "
            f"Nominal classic 95% log-Z interval coverage: {summary['classic_95pct_coverage']:.3f}."
        ),
        "",
        (
            f"All {summary['below_002_uncertainty']} completed outputs have reported classic "
            "sigma(log Z) <= 0.02. Reaching this stopping threshold does not "
            "imply correct mode mass. "
            f"Mean spike mass is {summary['mean_spike_mass']:.6f}."
        ),
        "",
        (
            "Truth is log Z = -24.15593480605331 and spike mass = 0.5000077444258325. "
            "Posterior accuracy uses classic rows and soft component responsibilities. "
            "Phantoms remain seeds/evidence observations, not posterior rows. "
            "Accuracy here is after the added goal. The raw JSON also preserves all 30 "
            "archived input-tree phantom-prefix analyses; these are not new analyses of the "
            "output trees."
        ),
        "",
        (
            "Memory is measured resident high-water RSS for each warm goal using the worker's own "
            "Linux marker, reset before each timed call. Persistent compilation/allocator buffers "
            "remain resident and count. The input snapshot stays reusable throughout "
            "the benchmark; "
            "this is a different buffer lifetime from a fresh complete run. "
            "Fingerprint and posterior-analysis work occurs after each "
            "timed call. Use warm_peak_rss_bytes; the legacy total_peak_rss_bytes field is not a "
            "lifetime process maximum after marker resets. Cold time and one-time legacy index "
            "conversion time is recorded separately. No claim of bounded "
            "full-history storage is made."
        ),
    ]
)
seed0 = next(pair for pair in pairs if pair["seed"] == 0)
control_speedup = (
    control["reference"]["median_seconds"] / control["combined"]["median_seconds"]
)
control_reference_gib = control["reference"]["warm_peak_rss_bytes"] / 2**30
control_combined_gib = control["combined"]["warm_peak_rss_bytes"] / 2**30
seed0_old_gib = seed0["old_warm_rss_bytes"] / 2**30
seed0_new_gib = seed0["new_warm_rss_bytes"] / 2**30
lines.extend(
    [
        "",
        "Matched seed-0 scaling control using the same harness:",
        "",
        (
            "| Saved target | Old warm s | New warm s | Speedup | Old warm GiB | New warm GiB | "
            "Added likelihood calls, both |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| 0.05 | {control['reference']['median_seconds']:.2f} | "
            f"{control['combined']['median_seconds']:.2f} | "
            f"{control_speedup:.2f}x | "
            f"{control_reference_gib:.2f} | "
            f"{control_combined_gib:.2f} | "
            f"{control['reference']['new_calls']:,} |"
        ),
        (
            f"| 0.02 | {seed0['old_seconds']:.2f} | {seed0['new_seconds']:.2f} | "
            f"{seed0['speedup']:.2f}x | {seed0_old_gib:.2f} | "
            f"{seed0_new_gib:.2f} | {seed0['new_calls']:,} |"
        ),
        "",
        (
            "The deeper goal also adds more samples and likelihood evaluations. "
            "The raw time growth "
            "therefore mixes increased goal work with increased history size; it is not a pure "
            "asymptotic scaling measurement."
        ),
        "",
        "![Paired goal timing, memory and common accuracy](cohort-ss10-002.png)",
    ]
)
args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
