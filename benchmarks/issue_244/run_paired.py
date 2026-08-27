"""Pair current and continuation execution in one process."""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path

import jax

from cicd.tests.test_ns_standard_problems import (
    STANDARD_PROBLEM_CASES_BY_NAME,
)
from jaxns.constrained_sampler import (
    ConstrainedSampleBatch,
    ConstrainedSampleRequest,
    UniDimSliceSampler,
    _sample_complete_chains,
)
from jaxns.core import NestedSampler


@dataclasses.dataclass(slots=True, frozen=True)
class CompleteChainSampler(UniDimSliceSampler):
    """Benchmark-only owner of the complete-chain reference strategy."""

    def get_samples(
            self,
            request: ConstrainedSampleRequest,
            *,
            args=(),
            params=None,
    ) -> ConstrainedSampleBatch:
        return _sample_complete_chains(
            self,
            request,
            args=args,
            params=params,
        )


CompleteChainSampler.register_pytree()


def _run(sampler: NestedSampler, key) -> tuple[object, float]:
    started = time.perf_counter()
    state = sampler.run(key)
    jax.block_until_ready(state)
    return state, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="basic_mvn")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = STANDARD_PROBLEM_CASES_BY_NAME[args.case]
    output = {"case": args.case, "records": []}
    for phantoms in (False, True):
        model, truth = case.build_case()
        dimension = int(model.U_ndims())
        num_slices = 5 * dimension
        phantom_count = dimension if phantoms else 0
        burn_in = num_slices - 1 - phantom_count
        candidate = NestedSampler(
            model=model,
            collect_phantom_samples=phantoms,
        )
        current = NestedSampler(
            model=model,
            collect_phantom_samples=phantoms,
            sampler=CompleteChainSampler(
                model=model,
                num_slices=num_slices,
                collect_phantom_samples=phantoms,
                phantom_burn_in=max(0, burn_in),
            ),
        )
        for seed in range(args.seeds):
            key = jax.random.PRNGKey(seed)
            # Alternate order after compilation so thermal or background drift
            # cannot systematically favour one execution strategy.
            if seed % 2 == 0:
                candidate_state, candidate_s = _run(candidate, key)
                current_state, current_s = _run(current, key)
            else:
                current_state, current_s = _run(current, key)
                candidate_state, candidate_s = _run(candidate, key)
            candidate_result = candidate_state.to_result().trim()
            current_result = current_state.to_result().trim()
            output["records"].append({
                "phantoms": phantoms,
                "seed": seed,
                "truth_log_Z": float(truth),
                "current_log_Z": float(current_result.log_Z_mean),
                "candidate_log_Z": float(candidate_result.log_Z_mean),
                "current_evaluations": int(
                    current_result.total_num_likelihood_evaluations
                ),
                "candidate_evaluations": int(
                    candidate_result.total_num_likelihood_evaluations
                ),
                "current_s": current_s,
                "candidate_s": candidate_s,
            })
            args.output.write_text(
                json.dumps(output, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    ratios = [
        record["candidate_s"] / record["current_s"]
        for record in output["records"]
        if record["seed"] != 0
    ]
    output["median_wall_ratio"] = statistics.median(ratios)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
