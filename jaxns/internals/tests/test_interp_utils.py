import jax.random
import numpy as np
import pytest
from jax import numpy as jnp

from jaxns.internals.interp_utils import get_interp_indices_and_weights, apply_interp, left_broadcast_multiply, \
    InterpolatedArray


def test_get_interp_indices_and_weights():
    xp = [0, 1, 2, 3]
    x = 1.5
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 1
    assert alpha0 == 0.5
    assert i1 == 2
    assert alpha1 == 0.5

    x = 0
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1
    assert i1 == 1
    assert alpha1 == 0

    x = 3
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 2
    assert alpha0 == 0
    assert i1 == 3
    assert alpha1 == 1

    x = -1
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == -1

    x = 4
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == 4

    x = 5
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == 5

    xp = [0., 0.]
    x = 0.
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1
    assert i1 == 1
    assert alpha1 == 0.

    xp = [0., 0.]
    x = -1
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 2.
    assert i1 == 1
    assert alpha1 == -1.

    xp = [0.]
    x = 0.5
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1.
    assert i1 == 0
    assert alpha1 == 0.

    # Vector ops
    xp = [0, 1, 2, 3]
    x = [1.5, 1.5]
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    np.testing.assert_array_equal(i0, jnp.asarray([1, 1]))
    np.testing.assert_array_equal(alpha0, jnp.asarray([0.5, 0.5]))
    np.testing.assert_array_equal(i1, jnp.asarray([2, 2]))
    np.testing.assert_array_equal(alpha1, jnp.asarray([0.5, 0.5]))

    xp = [0, 1, 2, 3]
    x = [1.5, 2.5]
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    np.testing.assert_array_equal(i0, jnp.asarray([1, 2]))
    np.testing.assert_array_equal(alpha0, jnp.asarray([0.5, 0.5]))
    np.testing.assert_array_equal(i1, jnp.asarray([2, 3]))
    np.testing.assert_array_equal(alpha1, jnp.asarray([0.5, 0.5]))

    # xp = [0, 1, 2, 3]
    # x = [-0.5, 3.5]
    # (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    # print(i0, alpha0, i1, alpha1)
    # np.testing.assert_array_equal(i0, jnp.asarray([1, 2]))
    # np.testing.assert_array_equal(alpha0, jnp.asarray([0.5, 0.5]))
    # np.testing.assert_array_equal(i1, jnp.asarray([2, 3]))
    # np.testing.assert_array_equal(alpha1, jnp.asarray([0.5, 0.5]))


@pytest.mark.parametrize('regular_grid', [True, False])
def test_apply_interp(regular_grid):
    xp = jnp.linspace(0., 1., 10)
    x = jnp.linspace(-0.1, 1.1, 10)
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=regular_grid)
    np.testing.assert_allclose(apply_interp(xp, i0, alpha0, i1, alpha1), x, atol=1e-6)

    x = jnp.linspace(-0.1, 1.1, 10)
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=regular_grid)
    assert apply_interp(jnp.zeros((4, 5, 10, 6)), i0, alpha0, i1, alpha1, axis=2).shape == (4, 5, 10, 6)

    x = 0.
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=regular_grid)
    assert apply_interp(jnp.zeros((4, 5, 10, 6)), i0, alpha0, i1, alpha1, axis=2).shape == (4, 5, 6)

    print(
        jax.jit(lambda x: apply_interp(x, i0, alpha0, i1, alpha1, axis=2)).lower(
            jnp.zeros((4, 5, 10, 6))).compile().cost_analysis()
    )
    # [{'bytes accessed': 1440.0, 'utilization1{}': 2.0, 'bytes accessed0{}': 960.0, 'bytes accessedout{}': 480.0, 'bytes accessed1{}': 960.0}]
    # [{'bytes accessed1{}': 960.0,  'utilization1{}': 2.0, 'bytes accessedout{}': 480.0, 'bytes accessed0{}': 960.0, 'bytes accessed': 1440.0}]


def test_regular_grid():
    # Inside bounds
    xp = jnp.linspace(0., 1., 10)
    fp = jax.random.normal(jax.random.PRNGKey(0), (10, 15))
    x = jnp.linspace(0., 1., 100)
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=False)
    f_no = apply_interp(fp, i0, alpha0, i1, alpha1)

    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=True)
    f_yes = apply_interp(fp, i0, alpha0, i1, alpha1)
    np.testing.assert_allclose(
        f_yes, f_no,
        atol=1e-6
    )

    # Outside bounds
    x = jnp.linspace(-0.1, 1.1, 100)
    xp = jnp.linspace(0., 1., 10)
    fp = jax.random.normal(jax.random.PRNGKey(0), (10, 15))
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=False)
    f_no = apply_interp(fp, i0, alpha0, i1, alpha1)

    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=True)
    f_yes = apply_interp(fp, i0, alpha0, i1, alpha1)
    np.testing.assert_allclose(
        f_yes, f_no,
        atol=1e-6
    )


def test_left_broadcast_multiply():
    assert np.all(left_broadcast_multiply(np.ones((2, 3)), np.ones((2,))) == np.ones((2, 3)))
    assert np.all(left_broadcast_multiply(np.ones((2, 3)), np.ones((2, 3))) == np.ones((2, 3)))
    assert np.all(
        left_broadcast_multiply(np.ones((1, 2, 3, 4, 5)), np.ones((3, 4)), axis=2) == np.ones((1, 2, 3, 4, 5)))
    assert np.all(
        left_broadcast_multiply(np.ones((1, 2, 3, 4, 5)), np.ones((3, 4)), axis=-3) == np.ones((1, 2, 3, 4, 5)))


@pytest.mark.parametrize('regular_grid', [True, False])
def test_interpolated_array(regular_grid: bool):
    # scalar time
    times = jnp.linspace(0, 10, 100)
    values = jnp.sin(times)
    interp = InterpolatedArray(times, values, regular_grid=regular_grid)
    assert interp(5.).shape == ()
    np.testing.assert_allclose(interp(5.), jnp.sin(5), atol=2e-3)

    # vector time
    assert interp(jnp.array([5., 6.])).shape == (2,)
    np.testing.assert_allclose(interp(jnp.array([5., 6.])), jnp.sin(jnp.array([5., 6.])), atol=2e-3)

    # Now with axis = 1
    times = jnp.linspace(0, 10, 100)
    values = jnp.stack([jnp.sin(times), jnp.cos(times)], axis=0)  # [2, 100]
    interp = InterpolatedArray(times, values, axis=1, regular_grid=regular_grid)
    assert interp(5.).shape == (2,)
    np.testing.assert_allclose(interp(5.), jnp.array([jnp.sin(5), jnp.cos(5)]), atol=2e-3)

    # Vector
    assert interp(jnp.array([5., 6., 7.])).shape == (2, 3)
    np.testing.assert_allclose(interp(jnp.array([5., 6., 7.])),
                               jnp.stack([jnp.sin(jnp.array([5., 6., 7.])), jnp.cos(jnp.array([5., 6., 7.]))],
                                         axis=0),
                               atol=2e-3)
