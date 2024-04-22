import inspect
import warnings
from typing import TypeVar, Callable, Optional

import jax
from jax import pmap, numpy as jnp, lax

from jaxns.internals.types import int_type


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
        warnings.warn(f"Function {f.__name__} has *varargs parameter ({varargs}), and is being dropped.")
    if varkw is not None:
        warnings.warn(f"Function {f.__name__} has **varkw parameter ({varkw}), and is being dropped.")

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
FV = TypeVar('FV')


def chunked_pmap(f: Callable[..., FV], chunk_size: Optional[int] = None, unroll: int = 1) -> Callable[..., FV]:
    """
    A version of pmap which chunks the input into smaller pieces to avoid memory issues.

    Args:
        f: callable
        chunk_size: the size of the chunks. Default is len(devices())
        unroll: the number of times to unroll the computation

    Returns:
        a chunked version of f
    """
    if chunk_size is None:
        chunk_size = len(jax.devices())

    def _f(*args, **kwargs):
        def queue(*args, **kwargs):
            """
            Distributes the computation in queues which are computed with scan.
            Args:
                *args:
            """

            def body(state, X):
                (args, kwargs) = X
                return state, f(*args, **kwargs)

            _, result = lax.scan(body, (), (args, kwargs), unroll=unroll)
            return result

        if chunk_size > 1:
            # Get from first leaf
            if len(args) > 0:
                batch_size = jax.tree.leaves(args)[0].shape[0]
            else:
                batch_size = jax.tree.leaves(kwargs)[0].shape[0]
            remainder = batch_size % chunk_size
            extra = (chunk_size - remainder) % chunk_size
            if extra > 0:
                (args, kwargs) = jax.tree.map(lambda x: _pad_extra(x, chunk_size), (args, kwargs))
            (args, kwargs) = jax.tree.map(
                lambda x: jnp.reshape(x, (chunk_size, x.shape[0] // chunk_size) + x.shape[1:]),
                (args, kwargs)
            )
            result = pmap(queue)(*args, **kwargs)  # [chunksize, batch_size // chunksize, ...]
            result = jax.tree.map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), result)
            if extra > 0:
                result = jax.tree.map(lambda x: x[:-extra], result)
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
    # arg = jnp.reshape(arg, (chunksize, N // chunksize) + arg.shape[1:])
    return arg


def prepad(a, chunksize: int):
    return jax.tree.map(lambda arg: _pad_extra(arg, chunksize), a)


T = TypeVar('T')


def remove_chunk_dim(py_tree: T) -> T:
    """
    Remove the chunk dimension from a pytree

    Args:
        py_tree: pytree to remove chunk dimension from

    Returns:
        pytree with chunk dimension removed
    """
    leaves = jax.tree.leaves(py_tree)

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

    return jax.tree.map(_remove_chunk_dim, py_tree)


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

    return jax.tree.map(_add_chunk_dim, py_tree)


def chunked_vmap(f, chunk_size: Optional[int] = None, unroll: int = 1):
    """
    A version of vmap which chunks the input into smaller pieces to avoid memory issues.

    Args:
        f: the function to be mapped
        chunk_size: the size of the chunks. Default is len(devices())
        unroll: the number of times to unroll the computation

    Returns:

    """
    if chunk_size is None:
        chunk_size = len(jax.devices())

    def _f(*args, **kwargs):
        def queue(*args, **kwargs):
            """
            Distributes the computation in queues which are computed with scan.
            Args:
                *args:
            """

            def body(state, X):
                (args, kwargs) = X
                return state, f(*args, **kwargs)

            _, result = lax.scan(f=body, init=(), xs=(args, kwargs), unroll=unroll)
            return result

        if chunk_size > 1:
            # Get from first leaf
            if len(args) > 0:
                batch_size = jax.tree.leaves(args)[0].shape[0]
            else:
                batch_size = jax.tree.leaves(kwargs)[0].shape[0]
            remainder = batch_size % chunk_size
            extra = (chunk_size - remainder) % chunk_size
            if extra > 0:
                (args, kwargs) = jax.tree.map(lambda x: _pad_extra(x, chunk_size), (args, kwargs))
            (args, kwargs) = jax.tree.map(
                lambda x: jnp.reshape(x, (chunk_size, x.shape[0] // chunk_size) + x.shape[1:]),
                (args, kwargs)
            )
            result = jax.vmap(queue)(*args, **kwargs)  # [chunksize, batch_size // chunksize, ...]
            result = jax.tree.map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), result)
            if extra > 0:
                result = jax.tree.map(lambda x: x[:-extra], result)
        else:
            result = queue(*args, **kwargs)
        return result

    _f.__doc__ = f.__doc__
    _f.__annotations__ = f.__annotations__
    return _f
