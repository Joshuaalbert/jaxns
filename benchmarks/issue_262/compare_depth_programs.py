"""Compare complete compiled depth programs for all direction policies."""

import argparse
import json
from pathlib import Path

import jax

from benchmarks.issue_246.run_standard import _depth_program, _environment
from jaxns.constrained_sampler import (
    EllipsoidalDirection,
    UniDimSliceSampler,
)
from jaxns.core import NestedSampler
from tests.test_ns_standard_problems import (
    STANDARD_PROBLEM_CASES_BY_NAME,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="basic_mvn")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.touch(exist_ok=False)

    for policy in ("isotropic", "ellipsoidal", "streaming"):
        model, _ = STANDARD_PROBLEM_CASES_BY_NAME[args.case].build_case()
        dimension = int(model.U_ndims())
        direction = None
        if policy != "isotropic":
            direction = EllipsoidalDirection(
                update="streaming" if policy == "streaming" else "warm",
            )
        sampler = UniDimSliceSampler(
            model=model,
            num_slices=5 * dimension,
            direction=direction,
        )
        nested_sampler = NestedSampler(
            model=model,
            root_allocation_degree=30 * dimension,
            shell_size=10 * dimension,
            max_samples=3000 * dimension,
            sampler=sampler,
        )
        jax.clear_caches()
        record = {
            "case": args.case,
            "direction": policy,
            "dimension": dimension,
            "root_degree": 30 * dimension,
            "replacement_width": 10 * dimension,
            "num_slices": 5 * dimension,
            "environment": _environment(),
            **_depth_program(nested_sampler),
        }
        with args.output.open("a") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
