"""Run the matched issue-246 benchmark matrix in isolated processes."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tests.test_ns_standard_problems import (
    STANDARD_PROBLEM_CASES_BY_NAME,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cases",
        default=",".join(STANDARD_PROBLEM_CASES_BY_NAME),
    )
    parser.add_argument(
        "--directions",
        default="isotropic,ellipsoidal",
    )
    parser.add_argument(
        "--phantom-settings",
        default="off,on",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in range(30)),
    )
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--min-effective-samples", type=int)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=1024)
    parser.add_argument("--prob-isotropic", type=float, default=1e-2)
    parser.add_argument("--mc-draws", type=int, default=1000)
    parser.add_argument("--source-id", default="working-tree")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cases = args.cases.split(",")
    directions = args.directions.split(",")
    phantom_settings = args.phantom_settings.split(",")
    unknown_cases = set(cases) - set(STANDARD_PROBLEM_CASES_BY_NAME)
    if unknown_cases:
        raise ValueError(f"Unknown standard problems: {sorted(unknown_cases)}")
    if set(directions) - {"isotropic", "ellipsoidal"}:
        raise ValueError("directions must contain isotropic or ellipsoidal.")
    if set(phantom_settings) - {"off", "on"}:
        raise ValueError("phantom-settings must contain off or on.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_standard.py")
    for case in cases:
        for direction in directions:
            for phantom_setting in phantom_settings:
                output = args.output_dir / (
                    f"{case}--{direction}--phantoms-{phantom_setting}.jsonl"
                )
                command = [
                    sys.executable,
                    str(runner),
                    "--case",
                    case,
                    "--direction",
                    direction,
                    "--seeds",
                    args.seeds,
                    "--components",
                    str(args.components),
                    "--iterations",
                    str(args.iterations),
                    "--population-size",
                    str(args.population_size),
                    "--prob-isotropic",
                    str(args.prob_isotropic),
                    "--mc-draws",
                    str(args.mc_draws),
                    "--source-id",
                    args.source_id,
                    "--output",
                    str(output),
                ]
                if args.min_effective_samples is not None:
                    command.extend([
                        "--min-effective-samples",
                        str(args.min_effective_samples),
                    ])
                if phantom_setting == "on":
                    command.append("--phantoms")

                print(
                    f"Running {case}: {direction}, phantoms {phantom_setting}",
                    flush=True,
                )
                # Each matrix cell gets a fresh process so compilation and
                # peak-memory measurements cannot leak across configurations.
                subprocess.run(
                    command,
                    check=True,
                    env=os.environ.copy(),
                    stdout=(subprocess.DEVNULL if args.quiet else None),
                )


if __name__ == "__main__":
    main()
