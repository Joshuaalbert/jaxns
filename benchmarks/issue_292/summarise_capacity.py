"""Summarise exploratory issue-292 capacity/mixture JSONL files."""

import argparse
import json
import math
import statistics
from pathlib import Path


def _mean(records, field):
    return statistics.fmean(record[field] for record in records)


def _rmse(records, field):
    return math.sqrt(statistics.fmean(
        record[field] ** 2 for record in records
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    grouped = {}
    for path in sorted(args.directory.glob("*.jsonl")):
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if not records:
            continue
        first = records[0]
        key = (
            first["case"],
            first["pool_capacity_multiplier"],
            first["phantom_seed_probability"],
        )
        grouped[key] = records

    print(
        "case cap p n logZ_bias logZ_rmse z_var mode_rmse evals "
        "wall_s eligible90 eligible99 eligible_final state_mib ckpt_mib"
    )
    for (case, capacity, probability), records in sorted(grouped.items()):
        z_values = [
            record["log_Z_error"] / record["log_Z_uncert"]
            for record in records
        ]
        mode_rmse = (
            "--"
            if records[0]["mode_mass_error"] is None
            else f"{_rmse(records, 'mode_mass_error'):.5f}"
        )
        print(
            case,
            f"{capacity:g}",
            f"{probability:g}",
            len(records),
            f"{_mean(records, 'log_Z_error'):.5f}",
            f"{_rmse(records, 'log_Z_error'):.5f}",
            f"{statistics.variance(z_values):.4f}",
            mode_rmse,
            f"{_mean(records, 'likelihood_evaluations'):.1f}",
            f"{statistics.median(r['warm_wall_s'] for r in records):.4f}",
            f"{_mean(records, 'pool_eligible_p90'):.1f}",
            f"{_mean(records, 'pool_eligible_p99'):.1f}",
            f"{_mean(records, 'pool_eligible_final'):.1f}",
            f"{_mean(records, 'state_bytes') / 2 ** 20:.4f}",
            f"{_mean(records, 'checkpoint_array_bytes') / 2 ** 20:.4f}",
        )


if __name__ == "__main__":
    main()
