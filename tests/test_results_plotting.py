import dataclasses
import io

import jax
import matplotlib
import numpy as np
from jax import numpy as jnp
from jaxctx import CtxParams

from jaxns.results import NestedSamplerResults

matplotlib.use("Agg")


class _FakeCtxParams:
    def __init__(self, **items):
        self._items = items

    def iter_items(self):
        return list(self._items.items())

    def get_dotted(self, name):
        return self._items[name]


def _make_fake_results(num_samples: int = 64) -> NestedSamplerResults:
    x = np.linspace(0.0, 1.0, num_samples)
    y = np.stack([
        np.linspace(-1.0, 1.0, num_samples),
        np.linspace(2.0, 3.0, num_samples)
    ], axis=-1)

    x_j = jnp.asarray(x)
    y_j = jnp.asarray(y)
    x_samples = _FakeCtxParams(x=x_j, y=y_j)

    log_l = -0.5 * (x_j - 0.5) ** 2
    log_posterior_density = log_l - 0.1

    return NestedSamplerResults(
        log_Z_mean=jnp.asarray(0.0),
        log_Z_uncert=jnp.asarray(0.1),
        ess=jnp.asarray(32.0),
        H_mean=jnp.asarray(-0.7),
        total_num_samples=jnp.asarray(num_samples),
        total_phantom_samples=jnp.asarray(0),
        total_num_likelihood_evaluations=jnp.asarray(10 * num_samples),
        log_efficiency=jnp.log(jnp.asarray(0.5)),
        termination_reason=jnp.asarray(0),
        U_samples=x_samples,
        X_samples=x_samples,
        log_L_constraints=jnp.full((num_samples,), -jnp.inf),
        log_L_phantom=jnp.full((num_samples, 0), -jnp.inf),
        valid_phantom=jnp.zeros((num_samples,), dtype=jnp.bool_),
        log_L=log_l,
        log_dp=jnp.asarray(np.linspace(-2.0, -1.0, num_samples)),
        log_X_mean=-jnp.linspace(0.0, 1.0, num_samples),
        log_posterior_density=log_posterior_density,
        num_live_points_per_sample=jnp.full((num_samples,), 20),
        num_likelihood_evaluations_per_sample=jnp.full((num_samples,), 2),
        log_L_supremum=jnp.max(log_l),
        U_supremum=x_samples,
        X_supremum=x_samples,
        log_L_map=jnp.max(log_posterior_density),
        U_map=x_samples,
        X_map=x_samples,
    )


def _make_weighted_pytree_results() -> NestedSamplerResults:
    results = _make_fake_results(num_samples=4)
    x = jnp.asarray([-2.0, -0.5, 1.0, 3.0])
    y = jnp.asarray(
        [
            [1.0, 0.0],
            [2.0, -1.0],
            [0.5, 4.0],
            [-2.0, 3.0],
        ]
    )
    samples = CtxParams({"x": x, "y": y})
    point = CtxParams({"x": x[-1], "y": y[-1]})
    return dataclasses.replace(
        results,
        U_samples=samples,
        X_samples=samples,
        log_dp=jnp.log(jnp.asarray([0.1, 0.2, 0.3, 0.4])),
        U_supremum=point,
        X_supremum=point,
        U_map=point,
        X_map=point,
    )


def _scientific_array_snapshot(results: NestedSamplerResults) -> list[np.ndarray]:
    arrays = []
    for field in dataclasses.fields(results):
        value = getattr(results, field.name)
        if hasattr(value, "iter_items"):
            arrays.extend(
                np.array(item, copy=True)
                for _, item in value.iter_items()
            )
            continue
        arrays.extend(
            np.array(leaf, copy=True)
            for leaf in jax.tree.leaves(value)
            if hasattr(leaf, "dtype")
        )
    return arrays


def test_plot_diagnostics_writes_file(tmp_path):
    results = _make_fake_results()
    output_file = tmp_path / "diagnostics.png"

    results.plot_diagnostics(save_file=str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plot_cornerplot_writes_file(tmp_path):
    results = _make_fake_results()
    output_file = tmp_path / "cornerplot.png"

    results.plot_cornerplot(save_name=str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_integrate_fn_over_posterior_normalizes_and_preserves_pytree():
    results = _make_weighted_pytree_results()
    weights = np.asarray([0.1, 0.2, 0.3, 0.4])
    x = np.asarray(results.X_samples.get_dotted("x"))
    y = np.asarray(results.X_samples.get_dotted("y"))

    def posterior_quantity(sample):
        sample_x = sample.get_dotted("x")
        sample_y = sample.get_dotted("y")
        return {
            "signed": sample_x - 0.75,
            "nested": (
                jnp.square(sample_y),
                jnp.asarray([sample_x, sample_y[0]]),
            ),
        }

    expected = {
        "signed": np.sum(weights * (x - 0.75)),
        "nested": (
            np.sum(weights[:, None] * np.square(y), axis=0),
            np.asarray([
                np.sum(weights * x),
                np.sum(weights * y[:, 0]),
            ]),
        ),
    }
    unbatched = results.integrate_fn_over_posterior(posterior_quantity)
    batched = results.integrate_fn_over_posterior(
        posterior_quantity,
        batch_size=3,
    )
    normalisation = results.integrate_fn_over_posterior(
        lambda _: jnp.asarray(1.0),
        semi_positive=True,
    )

    assert jax.tree.structure(unbatched) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf, batched_leaf in zip(
            jax.tree.leaves(unbatched),
            jax.tree.leaves(expected),
            jax.tree.leaves(batched),
            strict=True,
    ):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-6)
        np.testing.assert_allclose(batched_leaf, expected_leaf, rtol=1e-6)
    np.testing.assert_allclose(normalisation, 1.0, rtol=1e-6)


def test_summary_and_plots_do_not_mutate_scientific_results(tmp_path):
    results = _make_weighted_pytree_results()
    before = _scientific_array_snapshot(results)

    summary_output = io.StringIO()
    results.summary(f_obj=summary_output)
    results.plot_diagnostics(save_file=tmp_path / "diagnostics.png")
    results.plot_cornerplot(save_name=tmp_path / "cornerplot.png")

    assert summary_output.getvalue()
    after = _scientific_array_snapshot(results)
    assert len(after) == len(before)
    for before_array, after_array in zip(before, after, strict=True):
        np.testing.assert_array_equal(after_array, before_array)
