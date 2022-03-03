import inspect


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

    if defaults is None:  # already f is callable(**kwargs)
        return f

    def _f(**x):
        _args = []
        for i, arg in enumerate(args):
            if arg in x.keys():
                _args.append(x[arg])
            else:
                has_defaults = i >= (len(args) - len(defaults))
                if has_defaults:
                    j = i - (len(args) - len(defaults))
                    _args.append(defaults[j])
                else:
                    raise ValueError(f"Value for {arg} missing from inputs {list(x.keys())}, and defaults.")
        return f(*_args)

    return _f