"""Accuracy and performance evidence for periodic random-chart sampling."""

# Importing the selected source tree must happen before importing JAXNS.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "suite",
        choices=(
            "all-false",
            "von-mises",
            "seam",
            "jones",
            "ridge-sweep",
            "jones-geometry",
        ),
    )
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


ARGS = _arguments()
SOURCE_ROOT = (
    ARGS.source_root
    if ARGS.source_root is not None
    else Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT / "src"))

import jax
import numpy as np
from jax import numpy as jnp
from jaxctx.priors.prior import Prior
from scipy.special import erf
from tensorflow_probability.substrates import jax as tfp

from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.results import NestedSamplerResults


tfpd = tfp.distributions
CONCENTRATION = 12.0
SEAM_CONCENTRATION = 100.0
RIDGE_CONCENTRATION = 60.0


def _ordinary_prior_model():
    value = Prior(
        tfpd.Uniform(low=0.0, high=1.0),
        name="value",
    ).realise()
    return -jnp.square(value)


def _angular_prior_model(centre):
    angle = Prior(
        tfpd.Uniform(low=-jnp.pi, high=jnp.pi),
        name="angle",
    ).realise(periodic=True)
    return CONCENTRATION * jnp.cos(angle - centre)


def _seam_periodic_prior_model(centre):
    angle = Prior(
        tfpd.Uniform(low=-jnp.pi, high=jnp.pi),
        name="angle",
    ).realise(periodic=True)
    return SEAM_CONCENTRATION * jnp.cos(angle - centre)


def _seam_hard_prior_model(centre):
    angle = Prior(
        tfpd.Uniform(low=-jnp.pi, high=jnp.pi),
        name="angle",
    ).realise()
    return SEAM_CONCENTRATION * jnp.cos(angle - centre)


def _ridge_log_likelihood(first, second, centre):
    return (
        RIDGE_CONCENTRATION * jnp.cos(first - second)
        + RIDGE_CONCENTRATION * jnp.cos(second - centre)
    )


def _ridge_hard_prior_model(centre):
    first = Prior(
        tfpd.Uniform(low=-jnp.pi, high=jnp.pi),
        name="first",
    ).realise()
    second = Prior(
        tfpd.Uniform(low=-jnp.pi, high=jnp.pi),
        name="second",
    ).realise()
    return _ridge_log_likelihood(first, second, centre)


def _ridge_periodic_prior_model(centre):
    first = Prior(
        tfpd.Uniform(low=-jnp.pi, high=jnp.pi),
        name="first",
    ).realise(periodic=True)
    second = Prior(
        tfpd.Uniform(low=-jnp.pi, high=jnp.pi),
        name="second",
    ).realise(periodic=True)
    return _ridge_log_likelihood(first, second, centre)


def _angle_moment_error(
        name: str,
        expected: complex,
) -> Callable[[NestedSamplerResults], float]:
    def calculate(result: NestedSamplerResults) -> float:
        trimmed = result.trim()
        # [N] classic posterior mass and physical angle in radians.
        weight = np.exp(np.asarray(trimmed.log_dp))
        angle = np.asarray(trimmed.X_samples[name])
        moment = np.sum(weight * np.exp(1j * angle))
        return float(np.abs(moment - expected))

    return calculate


def _ridge_moment_error(
        centre: float,
) -> Callable[[NestedSamplerResults], float]:
    ratio = float(
        jax.scipy.special.i1e(RIDGE_CONCENTRATION)
        / jax.scipy.special.i0e(RIDGE_CONCENTRATION)
    )

    def calculate(result: NestedSamplerResults) -> float:
        trimmed = result.trim()
        # [N] classic posterior mass and two physical angles in radians.
        weight = np.exp(np.asarray(trimmed.log_dp))
        first = np.asarray(trimmed.X_samples["first"])
        second = np.asarray(trimmed.X_samples["second"])
        moments = np.asarray([
            np.sum(weight * np.exp(1j * (first - second))),
            np.sum(weight * np.exp(1j * (second - centre))),
            np.sum(weight * np.exp(1j * (first - centre))),
        ])
        expected = np.asarray([ratio, ratio, ratio * ratio])
        return float(np.sqrt(np.mean(np.square(np.abs(moments - expected)))))

    return calculate


def _evidence_samples(result, conditioning: str, seed: int):
    shrinkage = result.sample_evidence_mc(
        # 256 draws make MC error in the ensemble mean negligible relative to
        # the between-run NS error while keeping the 30-seed benchmark cheap.
        num_samples=256,
        conditioning=conditioning,
        key=jax.random.PRNGKey(90_000 + seed),
    )
    values = np.asarray(shrinkage.log_Z_samples)
    return float(np.mean(values)), float(np.std(values))


def _run(
        label: str,
        sampler: NestedSampler,
        truth: float,
        seeds: int,
        conditioning: str,
        use_mc: bool = False,
        moment_error: Callable[[NestedSamplerResults], float] | None = None,
) -> list[dict]:
    # Clearing once per configuration makes the first record a cold
    # compile-plus-execute measurement. Later records use the same programs
    # and isolate steady-state end-to-end wall time.
    jax.clear_caches()
    records = []
    for seed in range(seeds):
        started = time.perf_counter()
        state = sampler.run(jax.random.PRNGKey(10_000 + seed))
        result = state.to_result()
        jax.block_until_ready(result.log_Z_mean)
        elapsed = time.perf_counter() - started
        if use_mc:
            estimate, uncertainty = _evidence_samples(
                result,
                conditioning,
                seed,
            )
        else:
            estimate = float(result.log_Z_mean)
            uncertainty = float(result.log_Z_uncert)
        record = {
            "label": label,
            "seed": seed,
            "cold": seed == 0,
            "wall_s": elapsed,
            "log_Z": estimate,
            "log_Z_uncert": uncertainty,
            "log_Z_error": estimate - truth,
            "likelihood_evaluations": int(
                result.total_num_likelihood_evaluations
            ),
            "classic_samples": int(result.total_num_samples),
            "phantom_samples": int(result.total_phantom_samples),
        }
        if moment_error is not None:
            record["circular_moment_error"] = moment_error(result)
        records.append(record)
    return records


def _all_false() -> list[dict]:
    model = Model(prior_model=_ordinary_prior_model)
    truth = float(np.log(0.5 * np.sqrt(np.pi) * erf(1.0)))
    sampler = NestedSampler(model=model)
    return _run(
        "ordinary",
        sampler,
        truth,
        ARGS.seeds,
        conditioning="classic",
    )


def _von_mises() -> list[dict]:
    model = Model(prior_model=_angular_prior_model)
    truth = float(
        jnp.log(jax.scipy.special.i0e(CONCENTRATION))
        + CONCENTRATION
    )
    records = []
    for centre in (-float(np.pi) + 0.03, 0.0, 1.2):
        sampler = NestedSampler(
            model=model,
            args=(jnp.asarray(centre),),
            root_allocation_degree=120,
        )
        records.extend(_run(
            f"von_mises_{centre:+.5f}",
            sampler,
            truth,
            ARGS.seeds,
            conditioning="classic",
            moment_error=_angle_moment_error(
                "angle",
                float(
                    jax.scipy.special.i1e(CONCENTRATION)
                    / jax.scipy.special.i0e(CONCENTRATION)
                ) * np.exp(1j * centre),
            ),
        ))
    return records


def _jones() -> list[dict]:
    from cicd.tests.test_ns_standard_problems import (
        _jones_scalar_model_case,
    )

    model, truth = _jones_scalar_model_case()
    records = []
    for collect in (False, True):
        sampler = NestedSampler(
            model=model,
            collect_phantom_samples=collect,
        )
        records.extend(_run(
            "jones_phantom" if collect else "jones_classic",
            sampler,
            float(truth),
            ARGS.seeds,
            conditioning="phantom" if collect else "classic",
            use_mc=True,
        ))
    return records


def _seam() -> list[dict]:
    truth = float(
        jnp.log(jax.scipy.special.i0e(SEAM_CONCENTRATION))
        + SEAM_CONCENTRATION
    )
    records = []
    for method, prior_model in (
        ("hard", _seam_hard_prior_model),
        ("random_chart", _seam_periodic_prior_model),
    ):
        model = Model(prior_model=prior_model)
        for placement, centre in (
            ("seam", -float(np.pi) + 0.01),
            ("interior", 0.0),
        ):
            sampler = NestedSampler(
                model=model,
                args=(jnp.asarray(centre),),
                root_allocation_degree=120,
            )
            records.extend(_run(
                f"{method}_{placement}",
                sampler,
                truth,
                ARGS.seeds,
                conditioning="classic",
                moment_error=_angle_moment_error(
                    "angle",
                    float(
                        jax.scipy.special.i1e(SEAM_CONCENTRATION)
                        / jax.scipy.special.i0e(SEAM_CONCENTRATION)
                    ) * np.exp(1j * centre),
                ),
            ))
    return records


def _ridge_sweep() -> list[dict]:
    truth = 2.0 * float(
        jnp.log(jax.scipy.special.i0e(RIDGE_CONCENTRATION))
        + RIDGE_CONCENTRATION
    )
    records = []
    for method, prior_model in (
        ("hard", _ridge_hard_prior_model),
        ("random_chart", _ridge_periodic_prior_model),
    ):
        model = Model(prior_model=prior_model)
        for placement, centre in (
            ("seam", -float(np.pi) + 0.01),
            ("interior", 0.0),
        ):
            for root_degree in (4, 8, 16, 32, 60):
                sampler = NestedSampler(
                    model=model,
                    args=(jnp.asarray(centre),),
                    root_allocation_degree=root_degree,
                )
                records.extend(_run(
                    f"ridge_{method}_{placement}_K{root_degree}",
                    sampler,
                    truth,
                    ARGS.seeds,
                    conditioning="classic",
                    moment_error=_ridge_moment_error(centre),
                ))
    return records


def _jones_geometry() -> list[dict]:
    from benchmarks.issue_275.jones_reference import reference_summary
    from cicd.tests.test_ns_standard_problems import (
        _jones_scalar_model_case,
    )

    truth, constant_moment = reference_summary(
        order=128,
        uncertainty_order=256,
    )
    records = []
    for method, periodic in (("hard", False), ("random_chart", True)):
        model, test_truth = _jones_scalar_model_case(periodic=periodic)
        np.testing.assert_allclose(float(test_truth), truth, atol=1e-10)
        for root_degree in (16, 32, 64, 120):
            sampler = NestedSampler(
                model=model,
                root_allocation_degree=root_degree,
            )
            records.extend(_run(
                f"jones_{method}_K{root_degree}",
                sampler,
                truth,
                ARGS.seeds,
                conditioning="classic",
                use_mc=True,
                moment_error=_angle_moment_error(
                    "constant",
                    constant_moment,
                ),
            ))
    return records


def main() -> None:
    if ARGS.suite == "all-false":
        records = _all_false()
    elif ARGS.suite == "von-mises":
        records = _von_mises()
    elif ARGS.suite == "seam":
        records = _seam()
    elif ARGS.suite == "jones":
        records = _jones()
    elif ARGS.suite == "ridge-sweep":
        records = _ridge_sweep()
    else:
        records = _jones_geometry()
    output = {
        "suite": ARGS.suite,
        "source_root": str(SOURCE_ROOT),
        "records": records,
    }
    if ARGS.output is not None:
        ARGS.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
