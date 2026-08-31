import dataclasses

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.diagnostics.insertion import insert_index_diagnostic
from jaxns.diagnostics.plotting import _weighted_percentile
from jaxns.diagnostics.reference import (
    bruteforce_evidence,
    bruteforce_posterior_samples,
)
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree
from jaxns.random_utils import resample

tfpd = tfp.distributions


def test_resample():
    x = random.normal(key=random.PRNGKey(0), shape=(50,))
    logits = -jnp.ones(50)
    samples = {'x': x}
    assert jnp.all(resample(random.PRNGKey(0), samples, logits)['x'] == resample(random.PRNGKey(0), x, logits))


def test_weighted_percentile():
    # Test the weighted percentile function
    samples = np.asarray([1, 2, 3, 4, 5])
    log_weights = np.asarray([0, 0, 0, 0, 0])
    percentiles = [50]
    assert np.allclose(_weighted_percentile(samples, log_weights, percentiles), 3.0)


def test_bruteforce_utilities_preserve_model_unit_dtypes():
    def prior_model():
        x = Prior(
            tfpd.Uniform(
                low=jnp.asarray(0.0, dtype=jnp.float32),
                high=jnp.asarray(1.0, dtype=jnp.float32),
            ),
            name='x',
        ).realise()
        y = Prior(
            tfpd.Uniform(
                low=jnp.asarray(0.0, dtype=jnp.float64),
                high=jnp.asarray(1.0, dtype=jnp.float64),
            ),
            name='y',
        ).realise()
        return -jnp.square(x.astype(jnp.float64)) - jnp.square(y)

    model = Model(prior_model=prior_model)

    # jaxctx validates each realised U leaf against the corresponding prior.
    # Exercising mixed precision proves that grid flattening does not erase
    # those leaf-level dtype contracts.
    log_evidence = bruteforce_evidence(model=model, grid_res=4)
    _, weights = bruteforce_posterior_samples(model=model, grid_res=4)

    assert jnp.isfinite(log_evidence)
    assert jnp.all(jnp.isfinite(weights.log_abs_val))


def test_bruteforce_evidence_assigns_unit_volume_to_constant_likelihood():
    """A regular U-space grid must integrate the unit likelihood to one."""

    def prior_model():
        Prior(
            tfpd.Uniform(
                low=jnp.zeros(2),
                high=jnp.ones(2),
            ),
            name='x',
        ).realise()
        return jnp.asarray(0.0)

    log_evidence = bruteforce_evidence(
        model=Model(prior_model=prior_model),
        grid_res=4,
    )

    np.testing.assert_allclose(log_evidence, 0.0, atol=1e-8)


@dataclasses.dataclass(frozen=True, slots=True)
class MockPyTree171(PureDataclassPytree):
    x: jax.Array  # [N]


MockPyTree171.register_pytree()


def test_gh171(tmp_path):
    pytree = MockPyTree171(jnp.array([1., 2., 3.]))
    filename = str(tmp_path / "results.pkl")
    pytree.save(filename)
    loaded_pytree = MockPyTree171.load(filename)
    np.testing.assert_allclose(loaded_pytree.x, pytree.x)


@pytest.mark.parametrize('seed', [42, 45, 46, 47, 48, 49])
def test_insert_index_diagnostic_uniform(seed):
    np.random.seed(seed)
    indices = np.random.randint(0, 100, 10000)
    p_value = insert_index_diagnostic(indices, num_live_points=100)
    print('Should be big', p_value)
    assert p_value > 0.01


@pytest.mark.parametrize('seed', [42, 45, 46, 47, 48, 49])
def test_insert_index_diagnostic_nonuniform(seed):
    np.random.seed(seed)
    indices = np.random.normal(0, 100, 10000)
    indices -= np.min(indices)
    indices /= np.max(indices)
    indices *= 100
    indices = indices.astype(int)
    p_value = insert_index_diagnostic(indices, num_live_points=100)
    print('Should be small', p_value)
    assert p_value < 0.01
