"""Run issue-292 standard problems in isolated paired-arm processes."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from cicd.tests.test_ns_standard_problems import (
    STANDARD_PROBLEM_CASES_BY_NAME,
)

DEFAULT_CASES = (
    "basic_mvn",
    "weak_curved_mvn8",
    "spike_slab",
    "weak_curved_spike_slab8",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in range(30)),
    )
    parser.add_argument("--warmup-seed", type=int, default=292_000)
    parser.add_argument("--mc-draws", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument(
        "--phantom-seed-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument("--source-id", default="working-tree")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cases = args.cases.split(",")
    unknown_cases = set(cases) - set(STANDARD_PROBLEM_CASES_BY_NAME)
    if unknown_cases:
        raise ValueError(f"Unknown standard problems: {sorted(unknown_cases)}")
    if args.block_size <= 0:
        raise ValueError("block-size must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runner = Path(__file__).with_name("run_standard.py")
    seeds = [int(value) for value in args.seeds.split(",")]
    for case_index, case in enumerate(cases):
        for block_index, block_start in enumerate(
            range(0, len(seeds), args.block_size)
        ):
            block = seeds[block_start:block_start + args.block_size]
            # AB/BA blocks control machine drift within every problem while
            # retaining isolated static specialisations and one warm-up per
            # timing block.
            arm_order = (
                ("off", "on")
                if (case_index + block_index) % 2 == 0
                else ("on", "off")
            )
            for phantom_seeding in arm_order:
                output = args.output_dir / (
                    f"{case}--block-{block_index:02d}--"
                    f"phantom-seeding-{phantom_seeding}.jsonl"
                )
                command = [
                    sys.executable,
                    str(runner),
                    "--case",
                    case,
                    "--phantom-seeding",
                    phantom_seeding,
                    "--seeds",
                    ",".join(str(seed) for seed in block),
                    "--warmup-seed",
                    str(args.warmup_seed + block_index),
                    "--mc-draws",
                    str(args.mc_draws),
                    "--phantom-seed-probability",
                    str(args.phantom_seed_probability),
                    "--source-id",
                    args.source_id,
                    "--output",
                    str(output),
                ]
                print(
                    f"Running {case} block {block_index}: phantom seeding "
                    f"{phantom_seeding}",
                    flush=True,
                )
                # Separate processes keep executable caches and memory
                # accounting from leaking between static feature paths.
                subprocess.run(
                    command,
                    check=True,
                    env=os.environ.copy(),
                    stdout=(subprocess.DEVNULL if args.quiet else None),
                )


if __name__ == "__main__":
    main()
