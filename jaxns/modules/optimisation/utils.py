from typing import NamedTuple

from jax import numpy as jnp, tree_map
import logging
import numpy as np

from jaxns.internals.maps import dict_multimap
from jaxns.modules.optimisation.types import GlobalOptimiserResults

logger = logging.getLogger(__name__)

def _bit_mask(int_mask, width=8):
    """
    Convert an integer mask into a bit-mask. I.e. convert an integer into list of left-starting bits.

    Examples:

    1 -> [1,0,0,0,0,0,0,0]
    2 -> [0,1,0,0,0,0,0,0]
    3 -> [1,1,0,0,0,0,0,0]

    Args:
        int_mask: int
        width: number of output bits

    Returns:
        List of bits from left
    """
    return list(map(int, '{:0{size}b}'.format(int_mask, size=width)))[::-1]


def summary(results: GlobalOptimiserResults) -> str:
    """
    Gives a summary of the results of a nested sampling run.

    Args:
        results: NestedSamplerResults
    """
    main_s = []

    def _print(s):
        print(s)
        main_s.append(s)

    def _round(v, uncert_v):
        try:
            sig_figs = -int("{:e}".format(uncert_v).split('e')[1]) + 1
            return round(float(v), sig_figs)
        except:
            return float(v)

    _print("--------")
    termination_bit_mask = _bit_mask(results.termination_reason, width=6)
    _print("Termination Conditions:")
    for bit, condition in zip(termination_bit_mask, ['Small enough likelihood change with patience',
                                                     'Small enough likelihood change',
                                                     'Target likelihood contour reached',
                                                     "Used max num steps",
                                                     "Used max num likelihood evaluations",
                                                     "On a plateau"]):
        if bit == 1:
            _print(condition)
    _print("--------")
    _print("# likelihood evals: {}".format(results.total_num_likelihood_evaluations))
    _print("# samples: {}".format(results.total_num_samples))
    _print("# likelihood evals / sample: {:.1f}".format(
        results.total_num_likelihood_evaluations / results.total_num_samples))
    _print("--------")
    _print(f"Maximum logL={_round(results.log_L_max, results.log_L_max)}")

    samples = results.samples
    max_like_points = results.sample_L_max
    for name in samples.keys():
        _samples = samples[name].reshape((samples[name].shape[0], -1))
        _max_like_points = max_like_points[name].reshape((-1,))
        ndims = _samples.shape[1]
        _print("--------")
        _print("{}: max(L) est.".format(
            name if ndims == 1 else "{}[#]".format(name), ))
        for dim in range(ndims):
            _uncert = jnp.std(_samples[:, dim])
            _max_like_point = _max_like_points[dim]
            # two sig-figs based on uncert
            sig_figs = -int("{:e}".format(_uncert).split('e')[1]) + 1

            def _round(ar):
                return round(float(ar), sig_figs)

            _uncert = _round(_uncert)
            _print("{}: {}".format(
                name if ndims == 1 else "{}[{}]".format(name, dim),
                _round(_max_like_point)
            ))
    _print("--------")
    return "\n".join(main_s)

def _isinstance_namedtuple(obj) -> bool:
    return (
            isinstance(obj, tuple) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
    )


def save_pytree(pytree: NamedTuple, save_file: str):
    """
    Saves results of nested sampler in a npz file.

    Args:
        results: NestedSamplerResults
        save_file: str, filename
    """
    pytree_np = tree_map(lambda v: np.asarray(v) if v is not None else None, pytree)

    def _pytree_asdict(pytree):
        _data_dict = pytree._asdict()
        data_dict = {}
        for k, v in _data_dict.items():
            if _isinstance_namedtuple(v):
                data_dict[k] = _pytree_asdict(v)
            elif isinstance(v, (dict, np.ndarray, None.__class__)):
                data_dict[k] = v
            else:
                raise ValueError("key, value pair {}, {} unknown".format(k, v))
        return data_dict

    data_dict = _pytree_asdict(pytree_np)
    np.savez(save_file, **data_dict)


def save_results(results: GlobalOptimiserResults, save_file: str):
    """
    Saves results of nested sampler in a npz file.

    Args:
        results: NestedSamplerResults
        save_file: str, filename
    """
    save_pytree(results, save_file)


def load_pytree(save_file: str):
    """
    Loads saved nested sampler results from a npz file.

    Args:
        save_file: str

    Returns:
        NestedSamplerResults
    """
    _data_dict = np.load(save_file, allow_pickle=True)
    data_dict = {}
    for k, v in _data_dict.items():
        if v.size == 1:
            if v.item() is None:
                data_dict[k] = None
            else:
                data_dict[k] = dict_multimap(lambda v: jnp.asarray(v), v.item())
        else:
            data_dict[k] = jnp.asarray(v)

    return data_dict


def load_results(save_file: str) -> GlobalOptimiserResults:
    """
    Loads saved nested sampler results from a npz file.

    Args:
        save_file: str

    Returns:
        NestedSamplerResults
    """
    data_dict = load_pytree(save_file)
    return GlobalOptimiserResults(**data_dict)