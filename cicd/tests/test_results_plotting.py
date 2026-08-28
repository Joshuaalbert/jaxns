import dataclasses
import io
from types import SimpleNamespace

import matplotlib
import numpy as np
from jax import numpy as jnp
from jax import tree_util

from jaxns.results import NestedSamplerResults, _weighted_percentile

matplotlib.use("Agg")


@tree_util.register_pytree_node_class
class _FakeCtxParams:
    def __init__(self, **items):
        self._items = items

    def iter_items(self):
        return list(self._items.items())

    def get_dotted(self, name):
        return self._items[name]

    def tree_flatten(self):
        names = tuple(self._items)
        return tuple(self._items[name] for name in names), names

    @classmethod
    def tree_unflatten(cls, names, values):
        return cls(**dict(zip(names, values, strict=True)))


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
    max_like_idx = int(jnp.argmax(log_l))
    map_idx = int(jnp.argmax(log_posterior_density))
    x_supremum = _FakeCtxParams(
        x=x_j[max_like_idx],
        y=y_j[max_like_idx],
    )
    x_map = _FakeCtxParams(
        x=x_j[map_idx],
        y=y_j[map_idx],
    )

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
        U_supremum=x_supremum,
        X_supremum=x_supremum,
        log_L_map=jnp.max(log_posterior_density),
        U_map=x_map,
        X_map=x_map,
    )


def test_plot_diagnostics_writes_file(tmp_path):
    results = _make_fake_results()
    output_file = tmp_path / "diagnostics.png"

    results.plot_diagnostics(save_file=str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plot_diagnostics_orders_append_results_by_negative_log_x(monkeypatch):
    from matplotlib import pyplot as plt

    results = _make_fake_results(num_samples=4)
    negative_log_x = np.asarray([2.0, 0.25, 4.0, 1.0])
    live_points = np.asarray([22, 20, 24, 21])
    log_l = np.asarray([2.0, 0.0, 3.0, 1.0])
    posterior_mass = np.asarray([0.30, 0.05, 0.50, 0.15])
    likelihood_evaluations = np.asarray([4, 2, 5, 3])
    results = dataclasses.replace(
        results,
        log_X_mean=jnp.asarray(-negative_log_x),
        num_live_points_per_sample=jnp.asarray(live_points),
        log_L=jnp.asarray(log_l),
        log_dp=jnp.log(jnp.asarray(posterior_mass)),
        num_likelihood_evaluations_per_sample=jnp.asarray(
            likelihood_evaluations
        ),
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    results.plot_diagnostics()
    figure = plt.gcf()
    try:
        axes = figure.axes
        order = np.argsort(negative_log_x, kind="stable")
        expected_x = negative_log_x[order]
        for axis_index in (0, 1, 2, 3, 5):
            np.testing.assert_allclose(
                axes[axis_index].lines[0].get_xdata(),
                expected_x,
            )
        np.testing.assert_allclose(
            axes[4].collections[0].get_offsets()[:, 0],
            expected_x,
        )
        np.testing.assert_array_equal(
            axes[0].lines[0].get_ydata(),
            live_points[order],
        )
        np.testing.assert_allclose(
            axes[1].lines[0].get_ydata(),
            np.exp(log_l[order] - np.max(log_l)),
        )
        np.testing.assert_allclose(
            axes[2].lines[0].get_ydata(),
            posterior_mass[order],
        )
        # This specifically guards against accumulating in append order and
        # only sorting the already-invalid cumulative values for display.
        np.testing.assert_allclose(
            axes[3].lines[0].get_ydata(),
            np.cumsum(posterior_mass[order]),
        )
        np.testing.assert_allclose(
            axes[4].collections[0].get_offsets()[:, 1],
            1.0 / likelihood_evaluations[order],
        )
        np.testing.assert_allclose(
            axes[5].lines[0].get_ydata(),
            np.exp(-negative_log_x[order] + log_l[order]),
        )
    finally:
        plt.close(figure)


def test_plot_cornerplot_writes_file(tmp_path):
    results = _make_fake_results()
    output_file = tmp_path / "cornerplot.png"

    results.plot_cornerplot(save_name=str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plot_cornerplot_uses_authoritative_map_and_supremum(monkeypatch):
    from matplotlib import pyplot as plt

    results = dataclasses.replace(
        _make_fake_results(),
        X_supremum=_FakeCtxParams(
            x=jnp.asarray(0.2),
            y=jnp.asarray([2.2, 2.3]),
        ),
        X_map=_FakeCtxParams(
            x=jnp.asarray(0.8),
            y=jnp.asarray([2.7, 2.8]),
        ),
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    results.plot_cornerplot(variables=["x"])
    figure = plt.gcf()
    try:
        axis = figure.axes[0]
        np.testing.assert_allclose(axis.lines[0].get_xdata(), [0.2, 0.2])
        np.testing.assert_allclose(axis.lines[1].get_xdata(), [0.8, 0.8])
    finally:
        plt.close(figure)


def test_summary_does_not_compare_incommensurate_phantom_statistics():
    results = dataclasses.replace(
        _make_fake_results(),
        total_phantom_samples=jnp.asarray(128),
    )
    output = io.StringIO()

    results.summary(f_obj=output)

    summary = output.getvalue()
    assert "logZ (classic expected)=" in summary
    assert "posterior Kish ESS (classic expected weights)=" in summary
    assert "with phantom" not in summary


def test_plot_evidence_compares_explicit_conditionings_and_exact_value(
        monkeypatch,
        tmp_path,
):
    calls = []

    def _sample_evidence_mc(
            self,
            num_samples,
            *,
            conditioning,
            key,
            diagnostics,
    ):
        del self, key
        calls.append((num_samples, conditioning, diagnostics))
        offset = 0.0 if conditioning == "classic" else 0.2
        return SimpleNamespace(
            log_Z_samples=jnp.linspace(-0.5, 0.5, num_samples) + offset
        )

    monkeypatch.setattr(
        NestedSamplerResults,
        "sample_evidence_mc",
        _sample_evidence_mc,
    )
    output_file = tmp_path / "evidence.png"

    _make_fake_results().plot_evidence(
        num_samples=64,
        conditionings=("classic", "phantom"),
        exact_log_Z=0.1,
        save_name=output_file,
    )

    assert calls == [
        (64, "classic", False),
        (64, "phantom", False),
    ]
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_evidence_equivalent_live_points_is_not_posterior_ess(monkeypatch):
    calls = []

    def _sample_evidence_mc(
            self,
            num_samples,
            *,
            conditioning,
            key,
            batch_size,
            C_min,
            diagnostics,
    ):
        del self, key
        calls.append(
            (num_samples, conditioning, batch_size, C_min, diagnostics)
        )
        return SimpleNamespace(
            H_samples=jnp.asarray([2.0, 4.0, jnp.nan]),
            log_Z_samples=jnp.asarray([0.0, 2.0, 10.0]),
        )

    monkeypatch.setattr(
        NestedSamplerResults,
        "sample_evidence_mc",
        _sample_evidence_mc,
    )
    results = _make_fake_results()
    expected = 3.0 / 1.0

    explicit = results.evidence_equivalent_live_points(
        num_samples=3,
        conditioning="phantom",
        key=jnp.asarray([0, 1], dtype=jnp.uint32),
        batch_size=2,
        C_min=12,
    )

    np.testing.assert_allclose(explicit, expected)
    assert not hasattr(results, "ess_with_phantom")
    assert calls == [
        (3, "phantom", 2, 12, False),
    ]


def test_weighted_percentile_retains_dominant_boundary_sample_mass():
    samples = np.asarray([0.0, 10.0, 20.0])
    log_weights = np.log(np.asarray([0.8, 0.1, 0.1]))

    percentiles = _weighted_percentile(
        samples,
        log_weights,
        [50, 90],
    )

    # Midpoint interpolation places the samples at cumulative probabilities
    # 0.40, 0.85, and 0.95. The former implementation discarded the first
    # sample's 80% mass and incorrectly returned 10 and 18 here.
    np.testing.assert_allclose(percentiles, [20.0 / 9.0, 15.0])
