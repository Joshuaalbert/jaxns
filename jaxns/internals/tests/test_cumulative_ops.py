import jax
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


def test_scan_associative_cumulative_op():
    def op(accumulate, y):
        return accumulate + y

    init = jnp.asarray(0, float_type)
    xs = jnp.asarray([1, 2, 3], float_type)
    final_accumulate, result = scan_associative_cumulative_op(op=op, init=init, xs=xs)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([1, 3, 6], float_type))

    final_accumulate, result = scan_associative_cumulative_op(op=op, init=init, xs=xs, pre_op=True)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([0, 1, 3], float_type))

    # Test with pytrees for xs and ys
    def op(accumulate, y):
        return jax.tree.map(lambda x, y: x + y, accumulate, y)

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
