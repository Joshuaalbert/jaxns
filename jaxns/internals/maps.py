import inspect
import logging
from typing import TypeVar

from jax import tree_map, pmap, numpy as jnp, lax, tree_util

from jaxns.internals.types import int_type

logger = logging.getLogger(__name__)


def replace_index(operand, update, start_index):
    """
    Replaces an index or slice with an update.
    If update is too big to respect start_index then start_index is shifted, which will give non-intuitive results.
    """
    if len(operand.shape) != len(update.shape):
        update = update[None]
    start_index = jnp.asarray(start_index, int_type)
    start_indices = [start_index] + [jnp.asarray(0, start_index.dtype)] * (len(update.shape) - 1)
    return lax.dynamic_update_slice(operand, update.astype(operand.dtype), start_indices)

def test_replace_index():
    # Test simple case
    operand = jnp.arange(10)
    update = jnp.arange(5)
    start_index = 3
    expected = jnp.array([0, 1, 2, 0, 1, 2, 3, 4, 8, 9])
    actual = replace_index(operand, update, start_index)
    assert jnp.allclose(actual, expected)
    # Test edge case
    operand = jnp.arange(10)
    update = jnp.arange(5)
    start_index = 8
    # expected = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
    expected = jnp.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]) # Shifted to fit!
    actual = replace_index(operand, update, start_index)
    assert jnp.allclose(actual, expected)

def get_index(operand, start_index, length):
    return lax.dynamic_slice(operand,
                             [start_index] + [jnp.asarray(0, int_type)] * (len(operand.shape) - 1),
                             (length,) + operand.shape[1:])


def prepare_func_args(f):
    """
    Takes a callable(a,b,...,z=Z) and prepares it into ``callable(**kwargs)``, such that only
    a,b,...,z are taken from ``**kwargs`` and the rest ignored.

    This allows ``f(**kwarg)`` to work even if ``f()`` is missing some keys from kwargs.

    Args:
        f: ``callable(a,b,...,z=Z)``

    Returns:
        ``callable(**kwargs)`` where ``**kwargs`` are the filtered for args of the original function.
    """
    if hasattr(f, "__old_name__"):
        raise ValueError(f"function {f.__old_name__} has already done prepare_func_args.")

    (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations) = \
        inspect.getfullargspec(f)

    # TODO: this gets displayed each time we prepare a function. Using a cache would be cleaner for user.
    if varargs is not None:
        logger.warning(f"Function {f.__name__} has *varargs parameter ({varargs}), and is being dropped.")
    if varkw is not None:
        logger.warning(f"Function {f.__name__} has **varkw parameter ({varkw}), and is being dropped.")

    expected_keys = set(args + kwonlyargs)

    if defaults is None:  # no defaults
        num_defaults = 0
        defaults = ()
    else:
        num_defaults = len(defaults)

    num_non_default = len(args) - num_defaults

    def _f(**kwargs):
        # Replace any defaults with kwargs
        args_with_values = dict(zip(args[num_non_default:], defaults))
        if kwonlydefaults is not None:
            args_with_values.update(kwonlydefaults)
        args_with_values.update(kwargs)
        args_with_values = dict(filter(lambda item: item[0] in expected_keys, args_with_values.items()))
        for key in expected_keys:
            if key not in args_with_values:
                raise KeyError(f"{f.__name__} is missing argument {key}")

        return f(**args_with_values)

    _f.__doc__ = f.__doc__
    _f.__old_name__ = f.__name__
    return _f


F = TypeVar('F')


def chunked_pmap(f: F, chunksize, *, batch_size=None) -> F:
    def _f(*args, batch_size=batch_size, **kwargs):
        def queue(*args, **kwargs):
            """
            Distributes the computation in queues which are computed with scan.
            Args:
                *args:
            """

            def body(state, X):
                (args, kwargs) = X
                return state, f(*args, **kwargs)

            _, result = lax.scan(body, (), (args, kwargs))
            return result

        if chunksize > 1:
            if batch_size is None:
                batch_size = args[0].shape[0] if len(args) > 0 else None
            assert batch_size is not None, "Couldn't get batch_size, please provide explicitly"
            remainder = batch_size % chunksize
            extra = (chunksize - remainder) % chunksize
            args = tree_map(lambda arg: _pad_extra(arg, chunksize), args)
            kwargs = tree_map(lambda arg: _pad_extra(arg, chunksize), kwargs)
            result = pmap(queue)(*args, **kwargs)
            result = tree_map(lambda arg: jnp.reshape(arg, (-1,) + arg.shape[2:]), result)
            if extra > 0:
                result = tree_map(lambda x: x[:-extra], result)
        else:
            result = queue(*args, **kwargs)
        return result

    _f.__doc__ = f.__doc__
    _f.__annotations__ = f.__annotations__
    return _f


def _pad_extra(arg, chunksize):
    N = arg.shape[0]
    remainder = N % chunksize
    if (remainder != 0) and (N > chunksize):
        # only pad if not a zero remainder
        extra = (chunksize - remainder) % chunksize
        arg = jnp.concatenate([arg] + [arg[0:1]] * extra, axis=0)
        N = N + extra
    else:
        extra = 0
    T = N // chunksize
    arg = jnp.reshape(arg, (chunksize, N // chunksize) + arg.shape[1:])
    return arg


def prepad(a, chunksize: int):
    return tree_map(lambda arg: _pad_extra(arg, chunksize), a)


T = TypeVar('T')


def remove_chunk_dim(py_tree: T) -> T:
    """
    Remove the chunk dimension from a pytree

    Args:
        py_tree: pytree to remove chunk dimension from

    Returns:
        pytree with chunk dimension removed
    """
    leaves = tree_util.tree_leaves(py_tree)

    # Check consistency
    for leaf in leaves:
        if len(leaf.shape) < 1:
            raise ValueError(f"Expected all leaves to have at least one dimension, got {leaf.shape}")
        if leaf.shape[0] != leaves[0].shape[0]:
            raise ValueError(
                f"Expected all leaves to have the same batch dimension, got {leaf.shape[0]} != {leaves[0].shape[0]}"
            )

    def _remove_chunk_dim(a):
        shape = list(a.shape)
        if len(shape) == 1:
            return a[0]
        shape = [shape[0] * shape[1]] + shape[2:]
        return jnp.reshape(a, shape)

    return tree_map(_remove_chunk_dim, py_tree)


def add_chunk_dim(py_tree: T, chunk_size: int) -> T:
    """
    Add a chunk dimension to a pytree

    Args:
        py_tree: pytree to add chunk dimension to
        chunk_size: size of chunk dimension

    Returns:
        pytree with chunk dimension added
    """

    def _add_chunk_dim(a):
        shape = list(a.shape)
        shape = [chunk_size, shape[0] // chunk_size] + shape[1:]
        return jnp.reshape(a, shape)

    return tree_map(_add_chunk_dim, py_tree)
