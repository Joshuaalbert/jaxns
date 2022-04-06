import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
# Clear any cached backends so new CPU backend will pick up the env var.
from jax.lib import xla_bridge
xla_bridge.get_backend.cache_clear()
from jax import numpy as jnp

from jaxns.internals.maps import replace_index, chunked_pmap, prepare_func_args, get_index


def test_replace_index():
    operand = jnp.asarray([0,1,2,3,4])
    update = jnp.asarray([5, 5])
    start_idx = 0
    expect = jnp.asarray([5,5,2,3,4])
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
        return x*y
    chunked_f = chunked_pmap(f, 1)
    x = jnp.arange(3)
    assert chunked_f(x, y=x).shape == x.shape
    assert jnp.all(chunked_f(x, y=x) == x**2)



    chunked_f = chunked_pmap(f, 2)
    x = jnp.arange(2)
    assert chunked_f(x, y=x).shape == x.shape
    assert jnp.all(chunked_f(x, y=x) == x ** 2)

    x = jnp.arange(3)
    assert chunked_f(x, y=x).shape == x.shape
    assert jnp.all(chunked_f(x, y=x) == x ** 2)


def test_prepare_func_args():
    import inspect

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
    operand = jnp.asarray([[1,2,3],[4,5,6]])
    start_index = 0
    length = 1
    expect = jnp.asarray([[1,2,3]])
    assert jnp.allclose(get_index(operand,start_index,length), expect)

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
    expect = jnp.asarray([[4,5,6]])
    assert jnp.allclose(get_index(operand, start_index, length), expect)