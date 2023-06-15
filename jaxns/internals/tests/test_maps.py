import jax
from jax import numpy as jnp

from jaxns.internals.maps import replace_index, chunked_pmap, prepare_func_args, get_index


def test_replace_index():
    operand = jnp.asarray([0, 1, 2, 3, 4])
    update = jnp.asarray([5, 5])
    start_idx = 0
    expect = jnp.asarray([5, 5, 2, 3, 4])
    assert jnp.all(replace_index(operand, update, start_idx) == expect)

    operand = jnp.asarray([0, 1, 2, 3, 4])
    update = jnp.asarray(5)
    start_idx = 0
    expect = jnp.asarray([5, 1, 2, 3, 4])
    assert jnp.all(replace_index(operand, update, start_idx) == expect)

    operand = jnp.asarray([0, 1, 2, 3, 4])
    update = jnp.asarray([5, 5])
    start_idx = 4
    expect = jnp.asarray([0, 1, 2, 5, 5])
    assert jnp.all(replace_index(operand, update, start_idx) == expect)

    operand = jnp.asarray([0, 1, 2, 3, 4])
    update = jnp.asarray(5)
    start_idx = 4
    expect = jnp.asarray([0, 1, 2, 3, 5])
    assert jnp.all(replace_index(operand, update, start_idx) == expect)


def test_chunked_pmap():
    def f(x, y):
        return x * y

    chunked_f = chunked_pmap(f, 1)
    x = jnp.arange(3)
    assert chunked_f(x, y=x).shape == x.shape
    assert jnp.all(chunked_f(x, y=x) == x ** 2)

    num_dev = len(jax.devices())
    chunked_f = chunked_pmap(f, num_dev)
    x = jnp.arange(num_dev)
    assert chunked_f(x, y=x).shape == x.shape
    assert jnp.all(chunked_f(x, y=x) == x ** 2)

    x = jnp.arange(3)
    assert chunked_f(x, y=x).shape == x.shape
    assert jnp.all(chunked_f(x, y=x) == x ** 2)


def test_prepare_func_args():
    prepare_func_args(lambda a: a)(a=1, b=2)

    def f(a, b=1):
        return a + b

    g = prepare_func_args(f)
    kwargs = dict(a=1, b=2, c=3)
    assert g(**kwargs) == f(kwargs['a'], b=kwargs['b'])
    kwargs = dict(a=1, c=3)
    assert g(**kwargs) == f(kwargs['a'])

    def f(a, b=2, *, c, d=4):
        return a + b + c + d

    g = prepare_func_args(f)
    kwargs = dict(a=5, b=6, c=7, d=8)
    assert g(**kwargs) == f(kwargs['a'], b=kwargs['b'], c=kwargs['c'], d=kwargs['d'])
    kwargs = dict(a=9, c=11)
    assert g(**kwargs) == f(kwargs['a'], c=kwargs['c'])


def test_get_index():
    operand = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    start_index = 0
    length = 1
    expect = jnp.asarray([[1, 2, 3]])
    assert jnp.allclose(get_index(operand, start_index, length), expect)

    operand = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    start_index = 0
    length = 2
    expect = operand
    assert jnp.allclose(get_index(operand, start_index, length), expect)

    operand = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    start_index = 1
    length = 2
    expect = operand
    assert jnp.allclose(get_index(operand, start_index, length), expect)

    operand = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    start_index = 1
    length = 1
    expect = jnp.asarray([[4, 5, 6]])
    assert jnp.allclose(get_index(operand, start_index, length), expect)
