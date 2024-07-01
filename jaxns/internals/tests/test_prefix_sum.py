import time
from functools import partial
from typing import NamedTuple

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from jaxns.internals.prefix_sum import _slice_along_axis, _interleave, _compute_max_num_levels, scan_associative


def test__slice_along_axis():
    a = jnp.arange(5)
    np.testing.assert_allclose(
        _slice_along_axis(a, stop=-1),
        a[:-1]
    )


def test__interleave():
    # even
    a = jnp.asarray([0, 1, 2, 3])
    b = jnp.asarray([4, 5, 6, 7])
    expected_interleaved = jnp.asarray([0, 4, 1, 5, 2, 6, 3, 7])
    np.testing.assert_allclose(_interleave(a, b, axis=0), expected_interleaved)

    # odd
    a = jnp.asarray([0, 1, 2])
    b = jnp.asarray([3, 4, 5])
    expected_interleaved = jnp.asarray([0, 3, 1, 4, 2, 5])
    np.testing.assert_allclose(_interleave(a, b, axis=0), expected_interleaved)

    # multi-dim
    a = jnp.asarray(
        [
            [0, 1, 2],
            [0, 1, 2]
        ]
    )
    b = jnp.asarray(
        [
            [3, 4, 5],
            [3, 4, 5]
        ]
    )
    expected_interleaved_1 = jnp.asarray(
        [
            [0, 3, 1, 4, 2, 5],
            [0, 3, 1, 4, 2, 5]
        ]
    )
    expected_interleaved_0 = jnp.asarray(
        [
            [0, 1, 2],
            [3, 4, 5],
            [0, 1, 2],
            [3, 4, 5]
        ]
    )
    np.testing.assert_allclose(_interleave(a, b, axis=1), expected_interleaved_1)
    np.testing.assert_allclose(_interleave(a, b, axis=0), expected_interleaved_0)


def test_compute_max_num_levels():
    for batch_size in range(1, 1000):
        max_num_levels = _compute_max_num_levels(batch_size)
        print(batch_size, max_num_levels)
        assert batch_size < 2 ** (max_num_levels + 1)
        assert batch_size >= 2 ** (max_num_levels)


@pytest.mark.parametrize('num_elems', [10, 11])
@pytest.mark.parametrize('op', [jnp.add, jnp.multiply, jnp.minimum, jnp.maximum])
def test_scan_associative(num_elems, op):
    elems = jnp.arange(num_elems)
    results = scan_associative(
        op,
        elems,
        axis=0
    )
    np.testing.assert_allclose(
        results,
        tfp.math.scan_associative(
            op,
            elems
        )
    )


@pytest.mark.parametrize('num_elems', [10, 11])
def test_scan_associative_pytee(num_elems):
    class MockClass(NamedTuple):
        x: jax.Array
        y: jax.Array

    def op(x: MockClass, y: MockClass) -> MockClass:
        return MockClass(
            x=x.x + y.x,
            y=x.y + y.y
        )

    elems = MockClass(
        x=jnp.arange(num_elems),
        y=jnp.arange(num_elems)
    )
    results = scan_associative(
        op,
        elems,
        axis=0
    )
    np.testing.assert_allclose(
        results,
        tfp.math.scan_associative(
            op,
            elems
        )
    )


def test_scan_associative_rank_reducing():
    def per_elem_op(x) -> jax.Array:
        return jnp.sum(x)

    def associative_op(x, y):
        print(x.shape, y.shape)
        assert np.shape(x) == np.shape(x)
        assert np.shape(x) == ()
        return per_elem_op(x) + per_elem_op(y)

    xs = jnp.arange(10)
    _ = scan_associative(associative_op, xs)


@pytest.mark.parametrize('N', [32, 128, 512])
@pytest.mark.parametrize('M', [32, 128, 512])
def test_performance(N: int, M: int):
    import tensorflow_probability.substrates.jax as tfp
    inputs = jnp.ones((N, M, M))
    op = jnp.linalg.matmul

    fn_compiled = jax.jit(partial(tfp.math.scan_associative, fn=op)).lower(elems=inputs).compile()

    m = 3

    t0 = time.time()
    for _ in range(m):
        fn_compiled(elems=inputs).block_until_ready()
    t1 = time.time()
    tfp_runtime = (t1 - t0) / m

    fn_compiled = jax.jit(partial(scan_associative, fn=op)).lower(elems=inputs).compile()

    t0 = time.time()
    for _ in range(m):
        fn_compiled(elems=inputs).block_until_ready()
    t1 = time.time()
    internal_runtime = (t1 - t0) / m
    print(f"sequence length {N}, matrix mul {M}x{M} @ {M}x{M}:")
    print(f"\tRun time tfp.math.scan_associative {tfp_runtime}")
    print(f"\tRun time internal scan_associative {internal_runtime}")
    if internal_runtime < tfp_runtime:
        print(f"Internal faster by {100. * (1. - internal_runtime / tfp_runtime)} %")
    else:
        print(f"tfp.math.scan_associative faster by {100. * (1. - tfp_runtime / internal_runtime)} %")
