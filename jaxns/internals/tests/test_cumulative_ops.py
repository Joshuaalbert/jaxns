import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaxns.internals.cumulative_ops import cumulative_op_static, cumulative_op_dynamic, scan_associative_cumulative_op
from jaxns.internals.types import float_type, int_type


def test_cumulative_op_static():
    def op(accumulate, y):
        return accumulate + y

    init = jnp.asarray(0, float_type)
    xs = jnp.asarray([1, 2, 3], float_type)
    final_accumulate, result = cumulative_op_static(op=op, init=init, xs=xs)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([1, 3, 6], float_type))

    final_accumulate, result = cumulative_op_static(op=op, init=init, xs=xs, pre_op=True)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([0, 1, 3], float_type))


def test_scan_associative_cumulative_op_likelihoods():
    def log_likelihood(x) -> jax.Array:
        return jnp.sum(x)

    def add_log_probs(x, y):
        print(x, y)
        return log_likelihood(x) + log_likelihood(y)

    init = jnp.asarray(0, float_type)
    xs = jnp.arange(1, 11, dtype=float_type)
    final_accumulate, result = scan_associative_cumulative_op(op=add_log_probs, init=init, xs=xs)
    final_accumulate_expected, result_expected = cumulative_op_static(op=add_log_probs, init=init, xs=xs)
    # print(final_accumulate, final_accumulate_expected)
    # print(result, result_expected)
    assert final_accumulate == final_accumulate_expected
    np.testing.assert_allclose(result, result_expected)


@pytest.mark.parametrize("binary_op", [jnp.add, jnp.multiply, jnp.minimum, jnp.maximum])
def test_scan_associative_cumulative_op(binary_op):
    def op(accumulate, y):
        return binary_op(accumulate, y)

    init = jnp.asarray(1, float_type)
    xs = jnp.arange(1, 11, dtype=float_type)
    final_accumulate, result = scan_associative_cumulative_op(op=binary_op, init=init, xs=xs)
    final_accumulate_expected, result_expected = cumulative_op_static(op=op, init=init, xs=xs)
    assert final_accumulate == final_accumulate_expected
    np.testing.assert_allclose(result, result_expected)

    final_accumulate, result = scan_associative_cumulative_op(op=op, init=init, xs=xs, pre_op=True)
    final_accumulate_expected, result_expected = cumulative_op_static(op=op, init=init, xs=xs, pre_op=True)
    assert final_accumulate == final_accumulate_expected
    np.testing.assert_allclose(result, result_expected)


@pytest.mark.parametrize("binary_op", [jnp.subtract, jnp.true_divide])
def test_scan_associative_cumulative_not_associative_op(binary_op):
    def op(accumulate, y):
        return binary_op(accumulate, y)

    init = jnp.asarray(1, float_type)
    xs = jnp.arange(1, 11, dtype=float_type)
    final_accumulate, result = scan_associative_cumulative_op(op=binary_op, init=init, xs=xs)
    final_accumulate_expected, result_expected = cumulative_op_static(op=op, init=init, xs=xs)
    with pytest.raises(AssertionError):
        assert final_accumulate == final_accumulate_expected
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(result, result_expected)

    final_accumulate, result = scan_associative_cumulative_op(op=op, init=init, xs=xs, pre_op=True)
    final_accumulate_expected, result_expected = cumulative_op_static(op=op, init=init, xs=xs, pre_op=True)
    with pytest.raises(AssertionError):
        assert final_accumulate == final_accumulate_expected
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(result, result_expected)


def test_scan_associative_cumulative_op_with_pytrees():
    # Test with pytrees for xs and ys
    def op(accumulate, y):
        return jax.tree.map(lambda x, y: jnp.add(x, y), accumulate, y)

    init = {'a': jnp.asarray(0, float_type), 'b': jnp.asarray(0, float_type)}
    xs = {'a': jnp.asarray([1, 2, 3], float_type), 'b': jnp.asarray([4, 5, 6], float_type)}
    final_accumulate, result = scan_associative_cumulative_op(op=op, init=init, xs=xs)
    assert final_accumulate == {'a': 6, 'b': 15}
    assert all(result['a'] == jnp.asarray([1, 3, 6], float_type))
    assert all(result['b'] == jnp.asarray([4, 9, 15], float_type))

    final_accumulate, result = scan_associative_cumulative_op(op=op, init=init, xs=xs, pre_op=True)
    assert final_accumulate == {'a': 6, 'b': 15}
    assert all(result['a'] == jnp.asarray([0, 1, 3], float_type))
    assert all(result['b'] == jnp.asarray([0, 4, 9], float_type))


def test_cumulative_op_dynamic():
    def op(accumulate, y):
        return accumulate + y

    init = jnp.asarray(0, float_type)
    xs = jnp.asarray([1, 2, 3], float_type)
    stop_idx = jnp.asarray(3, int_type)
    final_accumulate, result = cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([1, 3, 6], float_type))

    final_accumulate, result = cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx, pre_op=True)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([0, 1, 3], float_type))

    stop_idx = jnp.asarray(2, int_type)
    final_accumulate, result = cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx)
    assert final_accumulate == 3
    assert all(result == jnp.asarray([1, 3, 0], float_type))

    final_accumulate, result = cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx, pre_op=True)
    assert final_accumulate == 3
    assert all(result == jnp.asarray([0, 1, 0], float_type))
