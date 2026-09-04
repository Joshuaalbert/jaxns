"""Profile resident and checkpoint scaling of the phantom seed pool.

The scientific sample capacity and retained phantom-likelihood width are held
identical between arms. The enabled state is made by adding only the bounded
pool to the disabled state, so the reported delta isolates phantom seeding
from phantom evidence conditioning and ordinary sample storage.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import jax
from jax import numpy as jnp

from benchmarks.issue_292.capacity_exploration import (
    CHECKPOINT_MODEL,
    _checkpoint_bytes,
)
from benchmarks.issue_292.run_standard import _tree_nbytes
from jaxns.algorithm.initialisation import _build_init_state
from jaxns.mixed_precision import mp_policy
from jaxns.sampling.seeding import PhantomSeedPool


def _integer_list(value: str) -> tuple[int, ...]:
    """Parse one comma-separated positive-integer benchmark axis."""
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of positive integers."
        )
    return values


def _initial_state(
        dimension: int,
        sample_capacity: int,
        root_degree: int,
):
    """Construct the exact initial array layout without likelihood work."""
    # CHECKPOINT_MODEL is the same picklable static payload in both arms. The
    # benchmark varies only array shapes and subtracts paired state sizes, so
    # model serialization cannot enter the reported pool increment.
    U_samples = jnp.zeros(
        (root_degree, dimension),
        # Unit-hypercube coordinates use the standard problem's float32 prior
        # dtype; likelihood and contour arithmetic remain float64.
        dtype=jnp.float32,
    )  # [d0, D]
    log_likelihoods = jnp.zeros(
        (root_degree,),
        dtype=mp_policy.measure_dtype,
    )  # [d0]
    num_evaluations = jnp.ones(
        (root_degree,),
        dtype=mp_policy.count_dtype,
    )  # [d0]
    return _build_init_state(
        CHECKPOINT_MODEL,
        (),
        None,
        U_samples,
        log_likelihoods,
        num_evaluations,
        sample_capacity=sample_capacity,
        # Both arms retain the same likelihood-only phantom prefix. This is
        # the issue-292 scientific benchmark policy and is not pool overhead.
        num_phantom=dimension,
        phantom_seed_capacity=0,
    )


def _record(
        dimension: int,
        sample_capacity: int,
        root_degree_multiplier: int,
        measure_checkpoint: bool,
) -> dict[str, int | None]:
    """Measure one exact same-state feature toggle."""
    root_degree = root_degree_multiplier * dimension
    disabled = _initial_state(
        dimension,
        sample_capacity,
        root_degree,
    )
    # Production capacity is one root population. Each bank owns one U point
    # and scalar eligibility metadata per slot; active and staging make two
    # banks whose size is independent of the scientific sample capacity.
    pool = PhantomSeedPool.empty(
        root_degree,
        disabled.U_supremum,  # [D]
    )
    enabled = dataclasses.replace(disabled, phantom_seed_pool=pool)
    jax.block_until_ready((disabled, enabled))

    disabled_bytes = _tree_nbytes(disabled)
    enabled_bytes = _tree_nbytes(enabled)
    pool_bytes = _tree_nbytes(pool)
    expected_pool_bytes = 2 * root_degree * (
        dimension * disabled.U_supremum.dtype.itemsize
        + 4 * jnp.dtype(mp_policy.measure_dtype).itemsize
        + jnp.dtype(mp_policy.index_dtype).itemsize
        + 2 * jnp.dtype(mp_policy.bool_dtype).itemsize
    )
    if pool_bytes != expected_pool_bytes:
        raise RuntimeError(
            "Measured pool leaves do not match the documented bank layout."
        )
    checkpoint_disabled_bytes = None
    checkpoint_enabled_bytes = None
    checkpoint_increment_bytes = None
    if measure_checkpoint:
        checkpoint_disabled_bytes = _checkpoint_bytes(disabled)
        checkpoint_enabled_bytes = _checkpoint_bytes(enabled)
        checkpoint_increment_bytes = (
            checkpoint_enabled_bytes - checkpoint_disabled_bytes
        )

    return {
        "dimension": dimension,
        "sample_capacity": sample_capacity,
        "retained_phantoms_per_cluster": dimension,
        "root_degree": root_degree,
        "pool_capacity": root_degree,
        "expected_pool_array_bytes": expected_pool_bytes,
        "disabled_state_array_bytes": disabled_bytes,
        "enabled_state_array_bytes": enabled_bytes,
        "pool_array_bytes": pool_bytes,
        "resident_increment_bytes": enabled_bytes - disabled_bytes,
        "checkpoint_disabled_bytes": checkpoint_disabled_bytes,
        "checkpoint_enabled_bytes": checkpoint_enabled_bytes,
        "checkpoint_increment_bytes": checkpoint_increment_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dimensions",
        type=_integer_list,
        default=(2, 8, 32, 128),
    )
    parser.add_argument(
        "--sample-capacities",
        type=_integer_list,
        default=(10_000, 100_000),
    )
    parser.add_argument("--root-degree-multiplier", type=int, default=30)
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Also serialize both full states for every cell.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.root_degree_multiplier <= 0:
        raise ValueError("root-degree-multiplier must be positive.")

    records = []
    for dimension in args.dimensions:
        for sample_capacity in args.sample_capacities:
            root_degree = args.root_degree_multiplier * dimension
            if sample_capacity < root_degree:
                raise ValueError(
                    "Every sample capacity must be at least the root degree."
                )
            record = _record(
                dimension,
                sample_capacity,
                args.root_degree_multiplier,
                args.checkpoint,
            )
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    args.output.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
