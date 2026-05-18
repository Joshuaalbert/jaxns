"""Run difficult-problem benchmarks with the v3 local runtime."""

# ruff: noqa: E402

import argparse
import sys
from datetime import date
from pathlib import Path
from time import monotonic_ns

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxctx.priors.prior import Prior

from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.runtime import LoadBalancerClient
from jaxns.termination_condition import TerminationCondition

tfpd = tfp.distributions


DIFFICULT_DIMENSION = 10
DEFAULT_TARGET_NUM_LIVE_POINTS = 100 * DIFFICULT_DIMENSION
DEFAULT_MAX_SAMPLES = 2 * DEFAULT_TARGET_NUM_LIVE_POINTS
DEFAULT_SHELL_SIZE = DEFAULT_TARGET_NUM_LIVE_POINTS // 2


def _eggbox_prior_model():
    """Evaluate the eggbox prior model.

    Returns:
        Scalar log likelihood.
    """
    ndim = DIFFICULT_DIMENSION
    # z: [ndim]
    z = Prior(
        tfpd.Uniform(
            low=jnp.zeros(ndim),
            high=10. * jnp.pi * jnp.ones(ndim),
        ),
        name='z',
    ).realise()
    y = 1
    for i in range(ndim):
        y *= jnp.cos(z[i] / 2)
    y = jnp.power(2. + y, 5)
    return y


def build_eggbox_model(ndim: int) -> Model:
    """
    Builds the eggbox model.

    Args:
        ndim:  The number of dimensions of the eggbox function.

    Returns:
        model: The eggbox model.
    """

    if ndim != DIFFICULT_DIMENSION:
        raise ValueError(f"Only {DIFFICULT_DIMENSION}D benchmarks are supported.")
    model = Model(prior_model=_eggbox_prior_model)
    return model


def _rastrigin_prior_model():
    """Evaluate the Rastrigin prior model.

    Returns:
        Scalar log likelihood.
    """
    ndim = DIFFICULT_DIMENSION
    x_min = -5.12
    x_max = 5.12
    # z: [ndim]
    z = Prior(
        tfpd.Uniform(
            low=x_min * jnp.ones(ndim),
            high=x_max * jnp.ones(ndim),
        ),
        name='z',
    ).realise()
    a = jnp.asarray(10.)
    y = a * ndim
    for i in range(ndim):
        y += jnp.power(z[i], 2) - a * jnp.cos(2 * jnp.pi * z[i])
    return -y


def build_rastrigin_model(ndim: int) -> Model:
    """
    Builds the Rastrigin model.

    Args:
        ndim:  The number of dimensions of the rastrigin function.

    Returns:
        model: The rastrigin model.
    """

    if ndim != DIFFICULT_DIMENSION:
        raise ValueError(f"Only {DIFFICULT_DIMENSION}D benchmarks are supported.")
    model = Model(prior_model=_rastrigin_prior_model)
    return model


def _rosenbrock_prior_model():
    """Evaluate the Rosenbrock prior model.

    Returns:
        Scalar log likelihood.
    """
    ndim = DIFFICULT_DIMENSION
    # z: [ndim]
    z = Prior(
        tfpd.Uniform(
            low=-5 * jnp.ones(ndim),
            high=5 * jnp.ones(ndim),
        ),
        name='z',
    ).realise()
    y = 0.
    for i in range(ndim - 1):
        y += (
            100. * jnp.power(z[i + 1] - jnp.power(z[i], 2), 2)
            + jnp.power(1 - z[i], 2)
        )
    return -y


def build_rosenbrock_model(ndim: int) -> Model:
    """
    Builds the Rosenbrock model.

    Args:
        ndim: Number of input dimensions the function should take.

    Returns:
        model: The Rosenbrock model.
    """

    if ndim != DIFFICULT_DIMENSION:
        raise ValueError(f"Only {DIFFICULT_DIMENSION}D benchmarks are supported.")
    model = Model(prior_model=_rosenbrock_prior_model)

    return model


def _spikeslab_prior_model():
    """Evaluate the spike-slab prior model.

    Returns:
        Scalar log likelihood.
    """
    ndim = DIFFICULT_DIMENSION
    # z: [ndim]
    z = Prior(
        tfpd.Uniform(
            low=-4. * jnp.ones(ndim),
            high=8. * jnp.ones(ndim),
        ),
        name='z',
    ).realise()
    # mean_1, mean_2: [ndim]
    mean_1 = jnp.array([6., 6.])
    mean_2 = jnp.array([2.5, 2.5])
    for i in range(ndim - 2):
        mean_1 = jnp.append(mean_1, 0.)
        mean_2 = jnp.append(mean_2, 0.)
    # cov_1, cov_2: [ndim, ndim]
    cov_1 = 0.08 * jnp.eye(ndim)
    cov_2 = 0.8 * jnp.eye(ndim)
    gauss_1 = tfp.distributions.MultivariateNormalFullCovariance(
        loc=mean_1,
        covariance_matrix=cov_1,
    ).log_prob(z)
    gauss_2 = tfp.distributions.MultivariateNormalFullCovariance(
        loc=mean_2,
        covariance_matrix=cov_2,
    ).log_prob(z)
    y = jnp.logaddexp(gauss_1, gauss_2)
    return y


def build_spikeslab_model(ndim: int) -> Model:
    """
    Builds the SpokeSlab model.

    Args:
        ndim: Number of input dimensions the function should take.

    Returns:
        model: The SpokeSlab model.
    """

    if ndim != DIFFICULT_DIMENSION:
        raise ValueError(f"Only {DIFFICULT_DIMENSION}D benchmarks are supported.")
    model = Model(prior_model=_spikeslab_prior_model)

    return model


def all_models() -> dict[str, Model]:
    """
    Return all the models

    Returns:
        A dictionary of models
    """
    return dict(
        eggbox=build_eggbox_model(ndim=DIFFICULT_DIMENSION),
        rastrigin=build_rastrigin_model(ndim=DIFFICULT_DIMENSION),
        rosenbrock=build_rosenbrock_model(ndim=DIFFICULT_DIMENSION),
        spikeslab=build_spikeslab_model(ndim=DIFFICULT_DIMENSION),
    )


class Timer:
    """Measure one benchmark section.

    Attributes:
        elapsed_seconds: Wall-clock seconds observed inside the context.
    """

    def __enter__(self):
        self.t0 = monotonic_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_seconds = (monotonic_ns() - self.t0) / 1e9
        print(f"Time to execute: {self.elapsed_seconds} seconds.")


def main():
    """Run all difficult-problem benchmarks and write a dated report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-num-live-points", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shell-size", type=int, default=None)
    parser.add_argument("--num-slices", type=int, default=24)
    parser.add_argument("--phantom-burn-in", type=int, default=4)
    parser.add_argument("--allocation-target", default="uniform")
    parser.add_argument(
        "--worker-spec",
        action="append",
        dest="worker_specs",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / date.today().isoformat(),
    )
    args = parser.parse_args()

    target_num_live_points = (
        DEFAULT_TARGET_NUM_LIVE_POINTS
        if args.target_num_live_points is None
        else args.target_num_live_points
    )
    max_samples = (
        2 * target_num_live_points
        if args.max_samples is None
        else args.max_samples
    )
    shell_size = (
        max(1, target_num_live_points // 2)
        if args.shell_size is None
        else args.shell_size
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_specs = list(args.worker_specs or ["cpu:*:2"])
    records = []

    for model_name, model in all_models().items():
        print(f"Testing model {model_name}")
        model.sanity_check(
            jax.random.PRNGKey(0),
            num_samples=1000,
        )
        sampler = UniDimSliceSampler(
            model=model,
            num_slices=args.num_slices,
            phantom_burn_in=args.phantom_burn_in,
            collect_phantom_samples=True,
            direction_kernel="ellipsoidal",
        )
        with LoadBalancerClient(address="local") as lb:
            lb.add_workers(worker_specs)
            actual_worker_count = sum(
                int(getattr(compute_sector, "num_workers", 0))
                for compute_sector in getattr(lb, "compute_sectors", ())
            )
            ns: NestedSampler = lb.get_nested_sampler(
                model=model,
                collect_phantoms=True,
                sampler=sampler,
                target_num_live_points=target_num_live_points,
                max_samples=max_samples,
                shell_size=shell_size,
            )
            with Timer() as timer:
                state = ns.run_until_goal(
                    goal_cond=lambda state: False,
                    depth_cond=TerminationCondition(max_samples=ns.max_samples),
                    allocation_target=args.allocation_target,
                    key=jax.random.PRNGKey(42),
                )
                state.num_samples.block_until_ready()
        results = state.to_result().trim()
        results.plot_diagnostics(save_file=output_dir / f"{model_name}_diagnostics.png")
        results.plot_cornerplot(save_name=output_dir / f"{model_name}_cornerplot.png")
        results.summary(f_obj=output_dir / f"{model_name}_summary.txt")
        records.append({
            "model": model_name,
            "runtime_seconds": timer.elapsed_seconds,
            "total_samples": int(results.total_num_samples),
            "likelihood_evaluations": int(
                results.total_num_likelihood_evaluations
            ),
            "log_Z_mean": float(results.log_Z_mean),
            "log_Z_uncert": float(results.log_Z_uncert),
            "actual_worker_count": actual_worker_count,
        })

    report_path = output_dir / f"difficult_problems_{date.today().isoformat()}.md"
    lines = [
        "# Difficult Problems Benchmark Report",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## Configuration",
        "",
        f"- dimension: {DIFFICULT_DIMENSION}",
        f"- target_num_live_points: {target_num_live_points}",
        f"- live_points_per_dimension: {target_num_live_points / DIFFICULT_DIMENSION:.1f}",
        f"- max_samples: {max_samples}",
        f"- shell_size: {shell_size}",
        f"- num_slices: {args.num_slices}",
        f"- phantom_burn_in: {args.phantom_burn_in}",
        f"- allocation_target: {args.allocation_target}",
        f"- worker_specs: {', '.join(worker_specs)}",
        "",
        "## Results",
        "",
        "| model | runtime s | samples | likelihood evals | log Z | log Z uncert | workers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record['model']} | "
            f"{record['runtime_seconds']:.3f} | "
            f"{record['total_samples']} | "
            f"{record['likelihood_evaluations']} | "
            f"{record['log_Z_mean']:.6g} | "
            f"{record['log_Z_uncert']:.6g} | "
            f"{record['actual_worker_count']} |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote Markdown report to {report_path}")


if __name__ == '__main__':
    main()
