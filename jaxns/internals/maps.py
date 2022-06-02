import inspect
from jax import tree_map, pmap, numpy as jnp
from jax.lax import scan
from jax.lax import dynamic_update_slice, dynamic_slice
import logging

from jaxns.internals.types import int_type

logger = logging.getLogger(__name__)


def replace_index(operand, update, start_index):
    """
    Replaces an index or slice with an update.
    If update is too big to respect start_index then start_index is shifted, which will give non-intuitive results.
    """
    if len(operand.shape) != len(update.shape):
        update = update[None]
    start_indices = [start_index] + [jnp.asarray(0, start_index.dtype)] * (len(update.shape) - 1)
    return dynamic_update_slice(operand, update.astype(operand.dtype), start_indices)


def get_index(operand, start_index, length):
    return dynamic_slice(operand,
                         [start_index] + [jnp.asarray(0, int_type)] * (len(operand.shape) - 1),
                         (length,) + operand.shape[1:])


def dict_multimap(f, d, *args):
    """
    Map function across key, value pairs in dicts.

    Args:
        f: callable(d, *args)
        d: dict
        *args: more dicts

    Returns: dict with same keys as d, with values result of `f`.
    """
    if not isinstance(d, dict):
        return f(d, *args)
    mapped_results = dict()
    for key in d.keys():
        mapped_results[key] = f(d[key], *[arg[key] for arg in args])
    return mapped_results


def prepare_func_args(f):
    """
    Takes a callable(a,b,...,z=Z) and prepares it into callable(**kwargs), such that only
    a,b,...,z are taken from **kwargs and the rest ignored.

    This allows f(**kwarg) to work even if f() is missing some keys from kwargs.

    Args:
        f: callable(a,b,...,z=Z)

    Returns:
        callable(**kwargs) where **kwargs are the filtered for args of the original function.
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


def chunked_pmap(f, chunksize, *, batch_size=None):
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

            _, result = scan(body, (), (args, kwargs))
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
