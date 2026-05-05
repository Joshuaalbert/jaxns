from typing import NamedTuple

import jax
import numpy as np
import pytest
from jax import random, numpy as jnp

from jaxns.results import _weighted_percentile
from jaxns.samples import PhantomSamples, Samples
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
    assert np.allclose(_weighted_percentile(samples, log_weights, percentiles), 3.0)


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


def test_samples_resize_preserves_constraints_and_shapes():
    """Samples.resize should preserve data and extend phantom buffers consistently."""

    samples = Samples(
        log_L_constraints=jnp.asarray([0.0, 1.0]),
        log_likelihoods=jnp.asarray([0.5, 1.5]),
        sample_ids=jnp.asarray([0, 1]),
        U_samples={'x': jnp.asarray([[1.0], [2.0]])},
        out_degree=jnp.asarray([1, 0]),
        num_likelihood_evaluations=jnp.asarray([3, 4]),
        phantom_samples=PhantomSamples(
            U_samples={'x': jnp.asarray([[[10.0]], [[20.0]]])},
            valid_mask=jnp.asarray([[True], [False]]),
            log_L=jnp.asarray([[0.25], [0.75]]),
        ),
    )

    resized = samples.resize(4)

    assert resized.log_L_constraints.shape == (4,)
    np.testing.assert_allclose(np.asarray(resized.log_L_constraints[:2]), np.asarray(samples.log_L_constraints))
    assert resized.log_likelihoods.shape == (4,)
    np.testing.assert_array_equal(np.asarray(resized.sample_ids), np.asarray([0, 1, 2, 3]))
    assert resized.U_samples['x'].shape == (4, 1)
    assert resized.phantom_samples.U_samples['x'].shape == (4, 1, 1)


def test_samples_sort_breaks_equal_likelihood_ties_with_sample_id():
    """Samples.sort should use sample_ids as the deterministic tie-breaker."""

    samples = Samples(
        log_L_constraints=jnp.asarray([0.0, 0.0, 0.0]),
        log_likelihoods=jnp.asarray([1.0, 1.0, 1.0]),
        sample_ids=jnp.asarray([2, 0, 1]),
        U_samples={'x': jnp.asarray([[2.0], [0.0], [1.0]])},
        out_degree=jnp.asarray([0, 0, 0]),
        num_likelihood_evaluations=jnp.asarray([1, 1, 1]),
        phantom_samples=PhantomSamples(
            U_samples=None,
            valid_mask=jnp.zeros((3, 0), dtype=bool),
            log_L=jnp.zeros((3, 0)),
        ),
    )

    sorted_samples = samples.sort()

    np.testing.assert_array_equal(np.asarray(sorted_samples.sample_ids), np.asarray([0, 1, 2]))
    np.testing.assert_allclose(np.asarray(sorted_samples.U_samples['x']).reshape((-1,)), np.asarray([0.0, 1.0, 2.0]))
