import matplotlib
import numpy as np
from jax import numpy as jnp

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
