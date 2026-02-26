import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaxns.cumulative_ops import cumulative_op_static, cumulative_op_dynamic, scan_associative_cumulative_op, batch_reduce
from jaxns.cumulative_ops import scan_or_while_loop
from jaxns.mixed_precision import mp_policy


def test_scan_or_while_loop():
    def scan_fn(carry, x):
        return carry + x, carry + x

    carry_init = 0
    xs = jnp.arange(10)

    # Test with static length
    _, ys_scan = scan_or_while_loop(scan_fn, carry_init, xs, length=10)
    assert jnp.all(ys_scan == jnp.cumsum(xs))

    # Test with dynamic length
    length = jnp.array(10)
    _, ys_while = scan_or_while_loop(scan_fn, carry_init, xs, length=length)
    assert jnp.all(ys_while == jnp.cumsum(xs))

    # Test with dynamic length
    length = jnp.array(7)
    _, ys_while = scan_or_while_loop(scan_fn, carry_init, xs, length=length)
    assert jnp.all(ys_while == jnp.cumsum(xs).at[7:].set(0))


def test_cumulative_op_static():
    def op(accumulate, y):
        return accumulate + y

    init = jnp.asarray(0, mp_policy.measure_dtype)
    xs = jnp.asarray([1, 2, 3], mp_policy.measure_dtype)
    final_accumulate, result = cumulative_op_static(op=op, init=init, xs=xs)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([1, 3, 6], mp_policy.measure_dtype))

    final_accumulate, result = cumulative_op_static(op=op, init=init, xs=xs, pre_op=True)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([0, 1, 3], mp_policy.measure_dtype))


def test_scan_associative_cumulative_op_likelihoods():
    def log_likelihood(x) -> jax.Array:
        return jnp.sum(x)

    def add_log_probs(x, y):
        print(x, y)
        return log_likelihood(x) + log_likelihood(y)

    init = jnp.asarray(0, mp_policy.measure_dtype)
    xs = jnp.arange(1, 11, dtype=mp_policy.measure_dtype)
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

    init = jnp.asarray(1, mp_policy.measure_dtype)
    xs = jnp.arange(1, 11, dtype=mp_policy.measure_dtype)
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

    init = jnp.asarray(1, mp_policy.measure_dtype)
    xs = jnp.arange(1, 11, dtype=mp_policy.measure_dtype)
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

    init = {'a': jnp.asarray(0, mp_policy.measure_dtype), 'b': jnp.asarray(0, mp_policy.measure_dtype)}
    xs = {'a': jnp.asarray([1, 2, 3], mp_policy.measure_dtype), 'b': jnp.asarray([4, 5, 6], mp_policy.measure_dtype)}
    final_accumulate, result = scan_associative_cumulative_op(op=op, init=init, xs=xs)
    assert final_accumulate == {'a': 6, 'b': 15}
    assert all(result['a'] == jnp.asarray([1, 3, 6], mp_policy.measure_dtype))
    assert all(result['b'] == jnp.asarray([4, 9, 15], mp_policy.measure_dtype))

    final_accumulate, result = scan_associative_cumulative_op(op=op, init=init, xs=xs, pre_op=True)
    assert final_accumulate == {'a': 6, 'b': 15}
    assert all(result['a'] == jnp.asarray([0, 1, 3], mp_policy.measure_dtype))
    assert all(result['b'] == jnp.asarray([0, 4, 9], mp_policy.measure_dtype))


def test_cumulative_op_dynamic():
    def op(accumulate, y):
        return accumulate + y

    init = jnp.asarray(0, mp_policy.measure_dtype)
    xs = jnp.asarray([1, 2, 3], mp_policy.measure_dtype)
    stop_idx = jnp.asarray(3, mp_policy.count_dtype)
    final_accumulate, result = cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([1, 3, 6], mp_policy.measure_dtype))

    final_accumulate, result = cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx, pre_op=True)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([0, 1, 3], mp_policy.measure_dtype))

    stop_idx = jnp.asarray(2, mp_policy.count_dtype)
    final_accumulate, result = cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx)
    assert final_accumulate == 3
    assert all(result == jnp.asarray([1, 3, 0], mp_policy.measure_dtype))

    final_accumulate, result = cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx, pre_op=True)
    assert final_accumulate == 3
    assert all(result == jnp.asarray([0, 1, 0], mp_policy.measure_dtype))


def test_batch_reduce():
    import numpy as np

    def f(x):
        return jnp.mean(x ** 2)

    xs = jnp.arange(10).reshape(10, 1)

    # Test without batching
    result = batch_reduce(f, xs, reduce_fn=jnp.sum, batch_size=None)
    expected = np.sum([f(x) for x in xs], axis=0)
    np.testing.assert_allclose(result, expected)

    # Test with batching
    result = batch_reduce(f, xs, reduce_fn=jnp.sum, batch_size=3)
    np.testing.assert_allclose(result, expected)

    # Test with remainder
    result = batch_reduce(f, xs, reduce_fn=jnp.sum, batch_size=4)
    np.testing.assert_allclose(result, expected)


def test_batch_reduce_vec_kernel():
    def f_vec(x):
        return x ** 2

    xs = jnp.arange(10).reshape(10, 1)
    # Test without batching
    result = batch_reduce(f_vec, xs, reduce_fn=jnp.sum, batch_size=None, vectorised_kernel=True)
    expected = jnp.sum(f_vec(xs), axis=0)
    np.testing.assert_allclose(result, expected)

    # Test with batching
    result = batch_reduce(f_vec, xs, reduce_fn=jnp.sum, batch_size=3, vectorised_kernel=True)
    expected = jnp.sum(f_vec(xs), axis=0)
    np.testing.assert_allclose(result, expected)

    # Test with remainder
    result = batch_reduce(f_vec, xs, reduce_fn=jnp.sum, batch_size=4, vectorised_kernel=True)
    expected = jnp.sum(f_vec(xs), axis=0)
    np.testing.assert_allclose(result, expected)
