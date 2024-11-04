from typing import NamedTuple

import jax
import numpy as np
import pytest
from jax import random, numpy as jnp

from jaxns.plotting import weighted_percentile
from jaxns.utils import resample, _bit_mask, save_pytree, load_pytree, insert_index_diagnostic


def test_resample():
    x = random.normal(key=random.PRNGKey(0), shape=(50,))
    logits = -jnp.ones(50)
    samples = {'x': x}
    assert jnp.all(resample(random.PRNGKey(0), samples, logits)['x'] == resample(random.PRNGKey(0), x, logits))


def test_bit_mask():
    assert _bit_mask(1, width=2) == [1, 0]
    assert _bit_mask(2, width=2) == [0, 1]
    assert _bit_mask(3, width=2) == [1, 1]


def test_weighted_percentile():
    # Test the weighted percentile function
    samples = np.asarray([1, 2, 3, 4, 5])
    log_weights = np.asarray([0, 0, 0, 0, 0])
    percentiles = [50]
    assert np.allclose(weighted_percentile(samples, log_weights, percentiles), 3.0)


class MockPyTree171(NamedTuple):
    x: jax.Array


def test_gh171(tmp_path):
    pytree = MockPyTree171(jnp.array([1., 2., 3.]))
    save_pytree(pytree, str(tmp_path / "results.json"))
    loaded_pytree = load_pytree(str(tmp_path / "results.json"))
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
