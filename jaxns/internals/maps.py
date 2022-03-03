import inspect
import logging

logger = logging.getLogger(__name__)


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
    (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations) = \
        inspect.getfullargspec(f)


    if varargs is not None:
        logger.warning(f"Function {f.__name__} has *varargs parameter, and is being dropped.")
    if varkw is not None:
        logger.warning(f"Function {f.__name__} has **varkw parameter, and is being dropped.")

    expected_keys = set(args + kwonlyargs)


    if defaults is None: # no defaults
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
                raise KeyError(f"Missing argument {key}")

        return f(**args_with_values)
    _f.__doc__ = f.__doc__
    return _f