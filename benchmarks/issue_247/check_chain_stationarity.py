"""Check retained-chain stationarity over repeated constrained transitions.

This diagnostic deliberately starts from exact rejection samples at one fixed
contour. Reusing each generation's returned classic samples as the next seeds
detects drift in the slice kernel separately from race-tree scheduling.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import jax
import numpy as np
from jax import numpy as jnp

from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.samples import SeedPoint
from tests.test_ns_standard_problems import STANDARD_PROBLEM_CASES_BY_NAME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="spike_slab10")
    parser.add_argument("--chains", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--constraint-quantile", type=float, default=0.99)
    parser.add_argument("--reference-draws", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=247)
    args = parser.parse_args()

    model, _ = STANDARD_PROBLEM_CASES_BY_NAME[args.case].build_case()
    ndims = int(model.U_ndims())
    key = jax.random.PRNGKey(args.seed)
    reference_key, chain_key = jax.random.split(key)
    reference_u = jax.vmap(model.sample_U)(
        jax.random.split(reference_key, args.reference_draws)
    )
    reference_log_l = jax.vmap(
        lambda u: model.log_likelihood(u, allow_nan=False)
    )(reference_u)
    reference_log_l_np = np.asarray(reference_log_l)
    constraint = float(np.quantile(reference_log_l_np, args.constraint_quantile))
    constrained_idx = np.flatnonzero(reference_log_l_np > constraint)
    if constrained_idx.size < args.chains:
        raise ValueError(
            f"Only {constrained_idx.size} exact constrained seeds are available."
        )
    initial_idx = constrained_idx[:args.chains]
    seed_u = jax.tree.map(lambda value: value[initial_idx], reference_u)
    seed_log_l = reference_log_l[initial_idx]

    constrained_reference = reference_log_l_np[constrained_idx]
    check_quantiles = np.quantile(
        constrained_reference,
        np.asarray([0.25, 0.5, 0.75]),
    )
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=5 * ndims,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=4 * ndims - 1,
    )

    def sample_one(sample_key, u0, log_l0):
        return sampler.get_sample(
            sample_key,
            jnp.asarray(constraint, dtype=jnp.float64),
            SeedPoint(U0=u0, log_L0=log_l0),
        )

    expected_survival = np.asarray([
        np.mean(constrained_reference > threshold)
        for threshold in check_quantiles
    ])

    def positive_first_coordinate_fraction(u_samples):
        x_samples = jax.vmap(model.transform_to_X)(u_samples)
        first_leaf = jax.tree.leaves(x_samples)[0]
        return float(np.mean(np.asarray(first_leaf[..., 0]) > 0.0))

    expected_positive_fraction = positive_first_coordinate_fraction(seed_u)
    for generation in range(args.generations):
        chain_key, generation_key = jax.random.split(chain_key)
        keys = jax.random.split(generation_key, args.chains)
        next_u, next_log_l, _, phantoms = jax.vmap(sample_one)(
            keys,
            seed_u,
            seed_log_l,
        )
        next_log_l_np = np.asarray(next_log_l)
        phantom_log_l_np = np.asarray(phantoms.log_L).reshape((-1,))
        phantom_u_flat = jax.tree.map(
            lambda value: value.reshape((-1,) + value.shape[2:]),
            phantoms.U_samples,
        )
        record = {
            "generation": generation + 1,
            "case": args.case,
            "chains": args.chains,
            "constraint": constraint,
            "expected_survival": expected_survival.tolist(),
            "expected_positive_first_coordinate_fraction": (
                expected_positive_fraction
            ),
            "classic_positive_first_coordinate_fraction": (
                positive_first_coordinate_fraction(next_u)
            ),
            "phantom_positive_first_coordinate_fraction": (
                positive_first_coordinate_fraction(phantom_u_flat)
            ),
            "classic_survival": [
                float(np.mean(next_log_l_np > threshold))
                for threshold in check_quantiles
            ],
            "phantom_survival": [
                float(np.mean(phantom_log_l_np > threshold))
                for threshold in check_quantiles
            ],
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        seed_u = next_u
        seed_log_l = next_log_l


if __name__ == "__main__":
    main()
