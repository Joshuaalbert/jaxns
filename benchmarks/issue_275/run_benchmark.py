"""Accuracy and performance evidence for periodic random-chart sampling."""

# Importing the selected source tree must happen before importing JAXNS.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "suite",
        choices=("all-false", "von-mises", "seam", "jones"),
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


tfpd = tfp.distributions
CONCENTRATION = 12.0
SEAM_CONCENTRATION = 100.0


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
        centre: float | None = None,
        concentration: float = CONCENTRATION,
        use_mc: bool = False,
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
        if centre is not None:
            trimmed = result.trim()
            # [N] normalised classic posterior weights and physical angles.
            weight = np.exp(np.asarray(trimmed.log_dp))
            angle = np.asarray(trimmed.X_samples["angle"])
            moment = np.sum(weight * np.exp(1j * angle))
            expected_length = float(
                jax.scipy.special.i1e(concentration)
                / jax.scipy.special.i0e(concentration)
            )
            expected_moment = expected_length * np.exp(1j * centre)
            record.update({
                "centre": centre,
                "circular_moment_error": float(
                    np.abs(moment - expected_moment)
                ),
                "circular_angle_error": float(
                    np.abs(np.angle(moment * np.exp(-1j * centre)))
                ),
            })
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
            centre=centre,
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
                centre=centre,
                concentration=SEAM_CONCENTRATION,
            ))
    return records


def main() -> None:
    if ARGS.suite == "all-false":
        records = _all_false()
    elif ARGS.suite == "von-mises":
        records = _von_mises()
    elif ARGS.suite == "seam":
        records = _seam()
    else:
        records = _jones()
    output = {
        "suite": ARGS.suite,
        "source_root": str(SOURCE_ROOT),
        "records": records,
    }
    if ARGS.output is not None:
        ARGS.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
