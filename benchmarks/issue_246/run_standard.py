"""Matched accuracy and efficiency runner for ellipsoidal directions."""

import argparse
import dataclasses
import importlib.metadata
import json
import os
import platform
import resource
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.special import logsumexp

import jaxns
from cicd.tests.test_ns_standard_problems import (
    STANDARD_PROBLEM_CASES_BY_NAME,
)
from jaxns.constrained_sampler import (
    EllipsoidalDirection,
    UniDimSliceSampler,
)
from jaxns.core import NestedSampler, _run_depth

MODE_PROBLEMS = {
    "spike_slab": {
        "prior_variance": np.full(8, 4.0),
        "means": np.stack([
            np.concatenate([np.full(4, 3.5), np.zeros(4)]),
            np.concatenate([np.full(4, -3.0), np.full(4, 1.5)]),
        ]),
        "variances": np.stack([
            np.concatenate([np.full(4, 0.05), np.full(4, 0.4)]),
            np.concatenate([np.full(4, 0.6), np.full(4, 0.08)]),
        ]),
        "masses": np.asarray([0.25, 0.75]),
    },
    "spike_slab10": {
        "prior_variance": np.full(10, 9.0),
        "means": np.stack([
            np.concatenate([np.full(3, 4.0), np.zeros(7)]),
            np.concatenate([
                np.full(3, -3.5),
                np.full(3, 2.0),
                np.zeros(4),
            ]),
        ]),
        "variances": np.stack([
            np.concatenate([np.full(3, 0.03), np.full(7, 0.7)]),
            np.concatenate([
                np.full(3, 0.5),
                np.full(3, 0.06),
                np.full(4, 0.9),
            ]),
        ]),
        "masses": np.asarray([0.4, 0.6]),
    },
    "weak_curved_spike_slab8": {
        "prior_variance": np.asarray([4.5, 3.2, 2.8, 2.6, 2.5, 2.3, 2.1, 2.0]),
        "means": np.asarray([
            [2.6, -0.8, 1.2, 0.0, 0.4, -0.3, 0.9, -0.6],
            [-2.1, 1.0, -0.9, 0.6, -0.5, 0.7, -1.1, 0.4],
        ]),
        "variances": np.asarray([
            [0.18, 0.35, 0.26, 0.42, 0.38, 0.34, 0.29, 0.31],
            [0.45, 0.16, 0.37, 0.3, 0.28, 0.24, 0.2, 0.33],
        ]),
        "masses": np.asarray([0.55, 0.45]),
        "beta": 0.14,
    },
    "weak_curved_spike_slab10": {
        "prior_variance": np.asarray([
            5.0, 3.8, 3.2, 2.9, 2.7, 2.6, 2.4, 2.2, 2.0, 1.8,
        ]),
        "means": np.asarray([
            [2.8, -0.9, 1.0, 0.5, -0.2, 0.7, 0.4, -0.5, 0.8, -0.3],
            [-2.5, 1.2, -1.1, -0.4, 0.6, -0.8, -0.3, 0.9, -0.7, 0.5],
        ]),
        "variances": np.asarray([
            [0.14, 0.33, 0.22, 0.36, 0.31, 0.29, 0.35, 0.27, 0.25, 0.3],
            [0.41, 0.12, 0.3, 0.28, 0.26, 0.2, 0.24, 0.18, 0.23, 0.34],
        ]),
        "masses": np.asarray([0.48, 0.52]),
        "beta": 0.12,
    },
}


def _environment() -> dict:
    return {
        "jaxns_distribution_version": importlib.metadata.version("jaxns"),
        "jaxns_module": os.path.realpath(jaxns.__file__),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "x64": bool(jax.config.jax_enable_x64),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _normal_log_density(x, mean, variance):
    return -0.5 * np.sum(
        np.log(2.0 * np.pi * variance) + np.square(x - mean) / variance,
        axis=-1,
    )


def _mode_mass(case_name: str, results) -> tuple[float | None, float | None]:
    """Return inferred and analytic posterior mass of component zero."""
    if case_name not in MODE_PROBLEMS:
        return None, None
    config = MODE_PROBLEMS[case_name]
    samples = np.asarray(jax.tree.leaves(results.X_samples)[0])
    if "beta" in config:
        samples = samples.copy()
        samples[:, 1] -= config["beta"] * (
            np.square(samples[:, 0]) - config["prior_variance"][0]
        )
    component_log_density = np.stack([
        np.log(mass) + _normal_log_density(samples, mean, variance)
        for mass, mean, variance in zip(
            config["masses"],
            config["means"],
            config["variances"],
            strict=True,
        )
    ], axis=1)
    assignments = np.argmax(component_log_density, axis=1)
    posterior_weights = np.exp(np.asarray(results.log_dp))
    estimated = float(np.sum(posterior_weights[assignments == 0]))

    component_log_evidence = np.asarray([
        np.log(mass)
        + _normal_log_density(
            mean,
            np.zeros_like(mean),
            config["prior_variance"] + variance,
        )
        for mass, mean, variance in zip(
            config["masses"],
            config["means"],
            config["variances"],
            strict=True,
        )
    ])
    truth = float(np.exp(
        component_log_evidence[0] - logsumexp(component_log_evidence)
    ))
    return estimated, truth


def _depth_program(ns: NestedSampler) -> dict:
    state = dataclasses.replace(
        ns.initialise(jax.random.PRNGKey(0)),
        goal_loop_iter=jnp.asarray(1, jnp.int32),
    )
    start = time.perf_counter()
    lowered = _run_depth.lower(
        state,
        ns.sampler,
        ns.depth_condition,
        shell_size=int(ns.shell_size),
        allocation_target=ns.allocation_target,
        root_degree=int(ns.root_allocation_degree),
        delta_K=int(ns.delta_K),
        max_samples=ns.max_samples,
    )
    lower_s = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - start
    execution = []
    for _ in range(3):
        start = time.perf_counter()
        output = compiled(state, ns.sampler, ns.depth_condition)
        jax.block_until_ready(output)
        execution.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    return {
        "lower_s": lower_s,
        "compile_s": compile_s,
        "depth_s": execution,
        "hlo_bytes": len(lowered.as_text().encode()),
        "temporary_bytes": (
            None if memory is None else int(memory.temp_size_in_bytes)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--direction",
        choices=("isotropic", "ellipsoidal"),
        required=True,
    )
    parser.add_argument("--phantoms", action="store_true")
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
    parser.add_argument("--measure-depth-program", action="store_true")
    parser.add_argument("--source-id", default="working-tree")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.touch(exist_ok=False)

    model, truth = STANDARD_PROBLEM_CASES_BY_NAME[args.case].build_case()
    dimension = int(model.U_ndims())
    num_slices = 5 * dimension
    root_degree = 30 * dimension
    shell_size = min(root_degree, 10 * dimension)
    retained_phantoms = dimension if args.phantoms else 0
    direction = None
    if args.direction == "ellipsoidal":
        direction = EllipsoidalDirection(
            num_components=args.components,
            min_effective_samples=args.min_effective_samples,
            num_iterations=args.iterations,
            population_size=args.population_size,
            prob_isotropic=args.prob_isotropic,
        )
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=num_slices,
        collect_phantom_samples=args.phantoms,
        max_phantom_samples=(
            retained_phantoms if args.phantoms else None
        ),
        direction=direction,
    )
    nested_sampler = NestedSampler(
        model=model,
        root_allocation_degree=root_degree,
        shell_size=shell_size,
        max_samples=100 * root_degree,
        collect_phantom_samples=args.phantoms,
        sampler=sampler,
    )
    program = (
        _depth_program(nested_sampler)
        if args.measure_depth_program
        else {}
    )
    base = {
        "case": args.case,
        "direction": args.direction,
        "phantoms": args.phantoms,
        "source_id": args.source_id,
        "truth_log_Z": float(truth),
        "dimension": dimension,
        "root_degree": root_degree,
        "replacement_width": shell_size,
        "num_slices": num_slices,
        "dlogZ": float(nested_sampler.depth_condition.dlogZ),
        "num_components": None if direction is None else direction.num_components,
        "min_effective_samples": (
            None if direction is None else direction.min_effective_samples
        ),
        "num_iterations": None if direction is None else direction.num_iterations,
        "population_size": None if direction is None else direction.population_size,
        "prob_isotropic": None if direction is None else direction.prob_isotropic,
        "environment": _environment(),
        **program,
    }
    conditioning = "phantom" if args.phantoms else "classic"
    for seed in [int(value) for value in args.seeds.split(",")]:
        key = jax.random.PRNGKey(seed)
        start = time.perf_counter()
        state = nested_sampler.run(key)
        jax.block_until_ready(state)
        run_s = time.perf_counter() - start

        start = time.perf_counter()
        results = state.to_result().trim()
        jax.block_until_ready(results)
        result_s = time.perf_counter() - start

        start = time.perf_counter()
        evidence = results.sample_evidence_mc(
            num_samples=args.mc_draws,
            conditioning=conditioning,
            key=jax.random.fold_in(key, 1),
        )
        jax.block_until_ready(evidence)
        mc_s = time.perf_counter() - start
        mode_mass, mode_mass_truth = _mode_mass(args.case, results)
        sampler_data = state.sampler_data
        record = dict(base)
        record.update({
            "seed": seed,
            "run_s": run_s,
            "result_s": result_s,
            "mc_s": mc_s,
            "log_Z_mean": float(results.log_Z_mean),
            "log_Z_uncert": float(results.log_Z_uncert),
            "log_Z_error": float(results.log_Z_mean - truth),
            "mc_log_Z_mean": float(evidence.log_Z_mean),
            "mc_log_Z_std": float(evidence.log_Z_uncert),
            "mc_log_Z_error": float(evidence.log_Z_mean - truth),
            "mc_z_score": float(
                (evidence.log_Z_mean - truth) / evidence.log_Z_uncert
            ),
            "classic_samples": int(results.total_num_samples),
            "phantom_samples": int(results.total_phantom_samples),
            "likelihood_evaluations": int(
                results.total_num_likelihood_evaluations
            ),
            "ess": float(results.ess),
            "mode_mass": mode_mass,
            "mode_mass_truth": mode_mass_truth,
            "mode_mass_error": (
                None if mode_mass is None else mode_mass - mode_mass_truth
            ),
            "fit_updates": (
                0 if sampler_data is None else int(sampler_data.num_updates)
            ),
            "directions": (
                0 if sampler_data is None else int(sampler_data.num_directions)
            ),
            "isotropic_directions": (
                0 if sampler_data is None else int(sampler_data.num_isotropic)
            ),
            "isotropic_fraction": (
                1.0
                if sampler_data is None
                else None
                if int(sampler_data.num_directions) == 0
                else float(
                    sampler_data.num_isotropic
                    / jnp.maximum(sampler_data.num_directions, 1)
                )
            ),
            "process_peak_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        })
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if args.output is not None:
            with args.output.open("a") as output:
                output.write(line + "\n")


if __name__ == "__main__":
    main()
