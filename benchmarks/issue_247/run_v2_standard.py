"""Reproducible released-v2/main baseline for issue 247.

Run this file from outside the checkout with the desired baseline package first
on ``PYTHONPATH``. The emitted module path is an explicit guard against an
accidental import of the editable current checkout.
"""

import argparse
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
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import erf, logsumexp
from jaxns.framework.bases import PriorModelGen
from jaxns.framework.model import Model
from jaxns.framework.prior import Prior
from jaxns.nested_samplers.common.types import TerminationCondition
from jaxns.public import NestedSampler
from tensorflow_probability.substrates import jax as tfp

import jaxns
from jaxns.utils import sample_evidence

tfpd = tfp.distributions
tfb = tfp.bijectors

STANDARD_TRUTH = {
    "basic": -0.2873361755422241,
    "basic2": -0.4054651081081643,
    "basic3": -3.4954280007708576,
    "plateau": 0.0,
    "basic_mvn": -24.60646255387843,
    "spike_slab": -18.395890684114224,
    "spike_slab10": -23.07211773465837,
    "weak_curved_mvn8": -12.097600655895445,
    "weak_curved_spike_slab8": -12.996345755265684,
    "weak_curved_spike_slab10": -16.144555366107006,
}


def _log_normal(x, mean, cov):
    factor = jnp.linalg.cholesky(cov)
    delta = solve_triangular(factor, x - mean, lower=True)
    return (
        -0.5 * x.size * jnp.log(2.0 * jnp.pi)
        - jnp.sum(jnp.log(jnp.diag(factor)))
        - 0.5 * delta @ delta
    )


def _mixture_evidence(prior_mean, prior_cov, means, covs, weights):
    component_log_z = jnp.asarray([
        _log_normal(mean, prior_mean, prior_cov + cov)
        for mean, cov in zip(means, covs, strict=True)
    ])
    return logsumexp(jnp.log(weights) + component_log_z)


def _curve(ndims, beta, sigma0):
    del ndims

    def forward(z):
        shift = beta * (z[..., 0] ** 2 - sigma0 ** 2)
        return jnp.concatenate(
            [z[..., :1], (z[..., 1] + shift)[..., None], z[..., 2:]],
            axis=-1,
        )

    def inverse(x):
        shift = beta * (x[..., 0] ** 2 - sigma0 ** 2)
        return jnp.concatenate(
            [x[..., :1], (x[..., 1] - shift)[..., None], x[..., 2:]],
            axis=-1,
        )

    return tfb.Inline(
        forward_fn=forward,
        inverse_fn=inverse,
        inverse_log_det_jacobian_fn=lambda _: jnp.asarray(0.0),
        forward_log_det_jacobian_fn=lambda _: jnp.asarray(0.0),
        forward_min_event_ndims=1,
        is_constant_jacobian=True,
    )


def _model(prior_distribution, log_likelihood):
    def prior_model() -> PriorModelGen:
        x = yield Prior(prior_distribution, name="x")
        return x

    return Model(prior_model=prior_model, log_likelihood=log_likelihood)


def build_case(name):
    if name == "basic":
        model = _model(
            tfpd.Uniform(0.0, 1.0),
            lambda x: -jnp.sum(x ** 2),
        )
        # Analytic value; the runner replaces this with the exact release
        # fixture reference so both implementations use identical truth data.
        truth = jnp.log(0.5 * jnp.sqrt(jnp.pi) * erf(jnp.asarray(1.0)))
        return model, truth, 1
    if name == "basic2":
        model = _model(
            tfpd.Uniform(0.0, 1.0),
            lambda x: jnp.log(1.0 - x ** 2),
        )
        return model, jnp.log(jnp.asarray(2.0 / 3.0)), 1
    if name == "basic3":
        def prior_model() -> PriorModelGen:
            x = yield Prior(tfpd.Uniform(0.0, 2.0), name="x")
            y = yield Prior(tfpd.Normal(2.0, x), name="y")
            return x + y

        model = Model(
            prior_model=prior_model,
            log_likelihood=lambda z: -z ** 2,
        )
        return model, jnp.asarray(STANDARD_TRUTH[name]), 2
    if name == "plateau":
        return _model(tfpd.Uniform(0.0, 1.0), lambda _: 0.0), jnp.asarray(0.0), 1
    if name == "basic_mvn":
        ndims = 8
        prior_mean = 6.0 * jnp.ones(ndims)
        prior_cov = jnp.asarray(np.eye(ndims) + 0.99 * (1.0 - np.eye(ndims)))
        likelihood_mean = jnp.zeros(ndims)
        likelihood_cov = jnp.eye(ndims)
        distribution = tfpd.MultivariateNormalTriL(
            prior_mean,
            jnp.linalg.cholesky(prior_cov),
        )
        likelihood = tfpd.MultivariateNormalTriL(
            likelihood_mean,
            jnp.linalg.cholesky(likelihood_cov),
        )
        truth = _log_normal(
            likelihood_mean,
            prior_mean,
            prior_cov + likelihood_cov,
        )
        return _model(distribution, likelihood.log_prob), truth, ndims

    if name in ("spike_slab", "spike_slab10"):
        if name == "spike_slab":
            ndims = 8
            prior_diag = 4.0 * jnp.ones(ndims)
            means = jnp.stack([
                jnp.concatenate([3.5 * jnp.ones(4), jnp.zeros(4)]),
                jnp.concatenate([-3.0 * jnp.ones(4), 1.5 * jnp.ones(4)]),
            ])
            covs = jnp.stack([
                jnp.diag(jnp.concatenate([0.05 * jnp.ones(4), 0.4 * jnp.ones(4)])),
                jnp.diag(jnp.concatenate([0.6 * jnp.ones(4), 0.08 * jnp.ones(4)])),
            ])
            weights = jnp.asarray([0.25, 0.75])
        else:
            ndims = 10
            prior_diag = 9.0 * jnp.ones(ndims)
            means = jnp.stack([
                jnp.concatenate([4.0 * jnp.ones(3), jnp.zeros(7)]),
                jnp.concatenate([-3.5 * jnp.ones(3), 2.0 * jnp.ones(3), jnp.zeros(4)]),
            ])
            covs = jnp.stack([
                jnp.diag(jnp.concatenate([0.03 * jnp.ones(3), 0.7 * jnp.ones(7)])),
                jnp.diag(jnp.concatenate([0.5 * jnp.ones(3), 0.06 * jnp.ones(3), 0.9 * jnp.ones(4)])),
            ])
            weights = jnp.asarray([0.4, 0.6])
        prior_mean = jnp.zeros(ndims)
        prior_cov = jnp.diag(prior_diag)
        distribution = tfpd.MultivariateNormalTriL(
            prior_mean,
            jnp.linalg.cholesky(prior_cov),
        )
        mixture = tfpd.MixtureSameFamily(
            tfpd.Categorical(probs=weights),
            tfpd.MultivariateNormalTriL(
                means,
                jnp.linalg.cholesky(covs),
            ),
        )
        truth = _mixture_evidence(
            prior_mean,
            prior_cov,
            means,
            covs,
            weights,
        )
        return _model(distribution, mixture.log_prob), truth, ndims

    if name == "weak_curved_mvn8":
        ndims = 8
        prior_diag = jnp.asarray([2.5, 1.7, 2.0, 1.5, 1.8, 1.6, 1.4, 1.9])
        likelihood_mean = jnp.asarray([2.0, -1.0, 0.8, -0.4, 0.2, -0.3, 1.2, -0.7])
        likelihood_diag = jnp.asarray([0.25, 0.45, 0.3, 0.35, 0.5, 0.4, 0.28, 0.32])
        prior_cov = jnp.diag(prior_diag)
        likelihood_cov = jnp.diag(likelihood_diag)
        curve = _curve(ndims, 0.18, jnp.sqrt(prior_diag[0]))
        distribution = tfpd.TransformedDistribution(
            tfpd.MultivariateNormalTriL(
                jnp.zeros(ndims),
                jnp.linalg.cholesky(prior_cov),
            ),
            curve,
        )
        likelihood = tfpd.MultivariateNormalTriL(
            likelihood_mean,
            jnp.linalg.cholesky(likelihood_cov),
        )
        return (
            _model(distribution, lambda x: likelihood.log_prob(curve.inverse(x))),
            _log_normal(likelihood_mean, jnp.zeros(ndims), prior_cov + likelihood_cov),
            ndims,
        )

    if name in ("weak_curved_spike_slab8", "weak_curved_spike_slab10"):
        if name.endswith("8"):
            ndims = 8
            beta = 0.14
            prior_diag = jnp.asarray([4.5, 3.2, 2.8, 2.6, 2.5, 2.3, 2.1, 2.0])
            means = jnp.stack([
                jnp.asarray([2.6, -0.8, 1.2, 0.0, 0.4, -0.3, 0.9, -0.6]),
                jnp.asarray([-2.1, 1.0, -0.9, 0.6, -0.5, 0.7, -1.1, 0.4]),
            ])
            covs = jnp.stack([
                jnp.diag(jnp.asarray([0.18, 0.35, 0.26, 0.42, 0.38, 0.34, 0.29, 0.31])),
                jnp.diag(jnp.asarray([0.45, 0.16, 0.37, 0.3, 0.28, 0.24, 0.2, 0.33])),
            ])
            weights = jnp.asarray([0.55, 0.45])
        else:
            ndims = 10
            beta = 0.12
            prior_diag = jnp.asarray([5.0, 3.8, 3.2, 2.9, 2.7, 2.6, 2.4, 2.2, 2.0, 1.8])
            means = jnp.stack([
                jnp.asarray([2.8, -0.9, 1.0, 0.5, -0.2, 0.7, 0.4, -0.5, 0.8, -0.3]),
                jnp.asarray([-2.5, 1.2, -1.1, -0.4, 0.6, -0.8, -0.3, 0.9, -0.7, 0.5]),
            ])
            covs = jnp.stack([
                jnp.diag(jnp.asarray([0.14, 0.33, 0.22, 0.36, 0.31, 0.29, 0.35, 0.27, 0.25, 0.3])),
                jnp.diag(jnp.asarray([0.41, 0.12, 0.3, 0.28, 0.26, 0.2, 0.24, 0.18, 0.23, 0.34])),
            ])
            weights = jnp.asarray([0.48, 0.52])
        prior_mean = jnp.zeros(ndims)
        prior_cov = jnp.diag(prior_diag)
        curve = _curve(ndims, beta, jnp.sqrt(prior_diag[0]))
        distribution = tfpd.TransformedDistribution(
            tfpd.MultivariateNormalTriL(
                prior_mean,
                jnp.linalg.cholesky(prior_cov),
            ),
            curve,
        )
        mixture = tfpd.MixtureSameFamily(
            tfpd.Categorical(probs=weights),
            tfpd.MultivariateNormalTriL(means, jnp.linalg.cholesky(covs)),
        )
        truth = _mixture_evidence(prior_mean, prior_cov, means, covs, weights)
        return _model(distribution, lambda x: mixture.log_prob(curve.inverse(x))), truth, ndims
    raise ValueError(f"Unknown standard problem: {name}")


def _environment():
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--implementation-label",
        choices=("v2-pypi", "main"),
        default="v2-pypi",
    )
    parser.add_argument("--source-id", default="jaxns==2.6.9")
    parser.add_argument("--phantoms", action="store_true")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in range(30)),
    )
    parser.add_argument("--root-multiplier", type=int)
    parser.add_argument("--mc-draws", type=int, default=1000)
    parser.add_argument("--output")
    args = parser.parse_args()
    output_path = None if args.output is None else Path(args.output)
    if output_path is not None:
        output_path.touch(exist_ok=False)

    model, _, ndims = build_case(args.case)
    truth = jnp.asarray(STANDARD_TRUTH[args.case])
    root_multiplier = args.root_multiplier
    if root_multiplier is None:
        root_multiplier = 30
    root_degree = root_multiplier * ndims
    num_slices = 5 * ndims
    num_phantom = ndims if args.phantoms else 0
    ns = NestedSampler(
        model=model,
        c=root_degree,
        num_slices=num_slices,
        k=num_phantom,
        init_efficiency_threshold=0.0,
        verbose=False,
    )
    term_cond = TerminationCondition(
        dlogZ=jnp.log1p(jnp.asarray(1e-3)),
        max_samples=jnp.asarray(ns.max_samples),
    )
    run = jax.jit(lambda key: ns(key, term_cond))
    example_key = jax.random.PRNGKey(0)

    lower_start = time.perf_counter()
    lowered = run.lower(example_key)
    lower_s = time.perf_counter() - lower_start
    compile_start = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - compile_start

    base = {
        "implementation": args.implementation_label,
        "source_id": args.source_id,
        "case": args.case,
        "phantoms": args.phantoms,
        "truth_log_Z": float(truth),
        "ndims": ndims,
        "root_degree": root_degree,
        "replacement_width": int(ns.nested_sampler.num_live_points),
        "num_slices": num_slices,
        "num_retained_phantoms": num_phantom,
        "dlogZ": float(jnp.log1p(jnp.asarray(1e-3))),
        "lower_s": lower_s,
        "compile_s": compile_s,
        "hlo_text_bytes": len(lowered.as_text().encode()),
        "environment": _environment(),
    }
    for seed in [int(value) for value in args.seeds.split(",")]:
        key = jax.random.PRNGKey(seed)
        run_start = time.perf_counter()
        reason, state = compiled(key)
        jax.block_until_ready((reason, state))
        run_s = time.perf_counter() - run_start

        result_start = time.perf_counter()
        results = ns.to_results(reason, state)
        jax.block_until_ready(results)
        result_s = time.perf_counter() - result_start

        mc_start = time.perf_counter()
        log_z_samples = sample_evidence(
            jax.random.fold_in(key, 1),
            results.num_live_points_per_sample,
            results.log_L_samples,
            S=args.mc_draws,
        )
        log_z_samples.block_until_ready()
        mc_s = time.perf_counter() - mc_start
        record = dict(base)
        record.update({
            "seed": seed,
            "run_s": run_s,
            "result_s": result_s,
            "mc_s": mc_s,
            "log_Z_mean": float(results.log_Z_mean),
            "log_Z_uncert": float(results.log_Z_uncert),
            "log_Z_error": float(results.log_Z_mean - truth),
            "mc_log_Z_mean": float(jnp.mean(log_z_samples)),
            "mc_log_Z_std": float(jnp.std(log_z_samples)),
            "mc_log_Z_error": float(jnp.mean(log_z_samples) - truth),
            "mc_z_score": float(
                (jnp.mean(log_z_samples) - truth)
                / jnp.maximum(jnp.std(log_z_samples), jnp.finfo(jnp.float64).tiny)
            ),
            "classic_samples": int(results.total_num_samples - results.total_phantom_samples),
            "phantom_samples": int(results.total_phantom_samples),
            "likelihood_evaluations": int(results.total_num_likelihood_evaluations),
            "ess": float(results.ESS),
            "process_peak_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        })
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if output_path is not None:
            with output_path.open("a") as output_file:
                output_file.write(line + "\n")


if __name__ == "__main__":
    main()
