import io
import logging
from typing import NamedTuple, TextIO, Union, Optional, Tuple, TypeVar, Callable

import numpy as np
from jax import numpy as jnp, tree_map, vmap, random, jit, lax

from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.maps import prepare_func_args
from jaxns.model.bases import BaseAbstractModel
from jaxns.random import resample_indicies
from jaxns.types import NestedSamplerResults, float_type, XType, UType
from jaxns.types import PRNGKey
from jaxns.warnings import deprecated

logger = logging.getLogger('jaxns')

__all__ = [
    'resample',
    'marginalise_static_from_U',
    'marginalise_dynamic_from_U',
    'marginalise_static',
    'marginalise_dynamic',
    'maximum_a_posteriori_point',
    'evaluate_map_estimate',
    'summary',
    'analytic_posterior_samples',
    'sample_evidence',
    'bruteforce_posterior_samples',
    'bruteforce_evidence',
    'save_pytree',
    'save_results',
    'load_pytree',
    'load_results'
]


def resample(key: PRNGKey, samples: XType, log_weights: jnp.ndarray, S: int = None, replace: bool = False) -> XType:
    """
    Resample the weighted samples into uniformly weighted samples.

    Args:
        key: PRNGKey
        samples: samples from nested sampled results
        log_weights: log-posterior weight
        S: number of samples to generate. Will use Kish's estimate of ESS if None.
        replace: whether to sample with replacement

    Returns:
        equally weighted samples
    """
    idx = resample_indicies(key, log_weights, S=S, replace=replace)
    return tree_map(lambda s: s[idx, ...], samples)


_V = TypeVar('_V')


def evaluate_map_estimate_from_U(results: NestedSamplerResults, model: BaseAbstractModel, fun: Callable[..., _V]) -> _V:
    """
    Marginalises function over posterior samples, where ESS is static.

    Args:
        results: results from run
        fun (:code:`callable(**kwargs)`): function to marginalise

    Returns:
        estimate at MAP sample point
    """
    map_sample_U = maximum_a_posteriori_point_U(results=results)
    V = model.prepare_input(U=map_sample_U)
    return fun(*V)


def marginalise_static_from_U(key: PRNGKey, U_samples: UType, model: BaseAbstractModel, log_weights: jnp.ndarray,
                              ESS: int,
                              fun: Callable[..., _V]) -> _V:
    """
    Marginalises function over posterior samples, where ESS is static.

    Args:
        key: PRNG key
        U_samples: array of U samples
        model: model
        log_weights: log weights from nested sampling
        ESS: static effective sample size
        fun (:code:`callable(**kwargs)`): function to marginalise

    Returns:
        expectation over resampled samples
    """
    U_samples = resample(key, U_samples, log_weights, S=ESS, replace=True)

    def eval(U):
        V = model.prepare_input(U=U)
        return fun(*V)

    marginalised = tree_map(lambda marg: jnp.nanmean(marg, axis=0), vmap(eval)(U_samples))
    return marginalised


def marginalise_dynamic_from_U(key: PRNGKey, U_samples: UType, model: BaseAbstractModel, log_weights: jnp.ndarray,
                               ESS: jnp.ndarray,
                               fun: Callable[..., _V]) -> _V:
    """
    Marginalises function over posterior samples, where ESS can be dynamic.

    Args:
        key: PRNG key
        U_samples: array of U samples
        model: model
        log_weights: log weights from nested sampling
        ESS: dynamic effective sample size
        fun (:code:`callable(**kwargs)`): function to marginalise

    Returns:
        expectation of `func` over resampled samples.
    """
    ESS = jnp.asarray(ESS)

    def eval(U):
        V = model.prepare_input(U=U)
        return fun(*V)

    def body(state):
        (key, i, count, marginalised) = state
        key, resample_key = random.split(key, 2)
        _samples = resample(resample_key, U_samples, log_weights, S=1)
        _sample = tree_map(lambda v: v[0], _samples)
        update = eval(_sample)
        count = tree_map(lambda y, c: jnp.where(jnp.any(jnp.isnan(y)), c, c + jnp.asarray(1, c.dtype)),
                         update, count)
        marginalised = tree_map(lambda x, y: jnp.where(jnp.isnan(y), x, x + y.astype(x.dtype)),
                                marginalised, update)
        return (key, i + jnp.ones_like(i), count, marginalised)

    test_output = fun(**tree_map(lambda v: v[0], U_samples))
    count = tree_map(lambda x: jnp.asarray(0, x.dtype), test_output)
    init_marginalised = tree_map(lambda x: jnp.zeros_like(x), test_output)
    (_, _, count, marginalised) = lax.while_loop(lambda state: state[1] < ESS,
                                             body,
                                             (key, jnp.array(0, ESS.dtype), count, init_marginalised))
    marginalised = tree_map(lambda x, c: x / c, marginalised, count)
    return marginalised


def marginalise_static(key: PRNGKey, samples: XType, log_weights: jnp.ndarray, ESS: int, fun: Callable[..., _V]) -> _V:
    """
    Marginalises function over posterior samples, where ESS is static.

    Args:
        key: PRNG key
        samples (dict): dict of batched array of nested sampling samples
        log_weights: log weights from nested sampling
        ESS: static effective sample size
        fun (:code:`callable(**kwargs)`): function to marginalise

    Returns:
        expectation over resampled samples
    """
    fun = prepare_func_args(fun)
    samples = resample(key, samples, log_weights, S=ESS, replace=True)
    marginalised = tree_map(lambda marg: jnp.nanmean(marg, axis=0), vmap(fun)(**samples))
    return marginalised


def marginalise_dynamic(key: PRNGKey, samples: XType, log_weights: jnp.ndarray, ESS: jnp.ndarray,
                        fun: Callable[..., _V]) -> _V:
    """
    Marginalises function over posterior samples, where ESS can be dynamic.

    Args:
        key: PRNG key
        samples (dict): dict of batched array of nested sampling samples
        log_weights: log weights from nested sampling
        ESS: dynamic effective sample size
        fun (:code:`callable(**kwargs)`): function to marginalise

    Returns:
        expectation of `func` over resampled samples.
    """
    fun = prepare_func_args(fun)
    ESS = jnp.asarray(ESS)

    def body(state):
        (key, i, count, marginalised) = state
        key, resample_key = random.split(key, 2)
        _samples = resample(resample_key, samples, log_weights, S=1)
        _sample = tree_map(lambda v: v[0], _samples)
        update = fun(**_sample)
        count = tree_map(lambda y, c: jnp.where(jnp.any(jnp.isnan(y)), c, c + jnp.asarray(1, c.dtype)),
                         update, count)
        marginalised = tree_map(lambda x, y: jnp.where(jnp.isnan(y), x, x + y.astype(x.dtype)),
                                marginalised, update)
        return (key, i + jnp.ones_like(i), count, marginalised)

    test_output = fun(**tree_map(lambda v: v[0], samples))
    count = tree_map(lambda x: jnp.asarray(0, x.dtype), test_output)
    init_marginalised = tree_map(lambda x: jnp.zeros_like(x), test_output)
    (_, _, count, marginalised) = lax.while_loop(lambda state: state[1] < ESS,
                                             body,
                                             (key, jnp.array(0, ESS.dtype), count, init_marginalised))
    marginalised = tree_map(lambda x, c: x / c, marginalised, count)
    return marginalised


def maximum_a_posteriori_point(results: NestedSamplerResults) -> XType:
    """
    Get the MAP point of a nested sampling result.
    Does this by choosing the point with largest L(x) p(x).

    Args:
        results (NestedSamplerResult): Nested sampler result

    Returns:
        dict of samples at MAP-point.
    """

    map_idx = jnp.argmax(results.log_posterior_density)
    map_points = tree_map(lambda x: x[map_idx], results.samples)
    return map_points


def maximum_a_posteriori_point_U(results: NestedSamplerResults) -> UType:
    """
    Get the MAP point of a nested sampling result.
    Does this by choosing the point with largest L(x) p(x).

    Args:
        results (NestedSamplerResult): Nested sampler result

    Returns:
        dict of samples at MAP-point.
    """

    map_idx = jnp.argmax(results.log_posterior_density)
    map_points = tree_map(lambda x: x[map_idx], results.U_samples)
    return map_points


def evaluate_map_estimate(results: NestedSamplerResults, fun: Callable[..., _V]) -> _V:
    """
    Marginalises function over posterior samples, where ESS is static.

    Args:
        results: results from run
        fun (:code:`callable(**kwargs)`): function to marginalise

    Returns:
        estimate at MAP sample point
    """
    fun = prepare_func_args(fun)
    map_sample = maximum_a_posteriori_point(results=results)
    return fun(**map_sample)


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


def summary(results: NestedSamplerResults, f_obj: Optional[Union[str, TextIO]] = None):
    """
    Gives a summary of the results of a nested sampling run.

    Args:
        results (NestedSamplerResults): Nested sampler result
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
    termination_bit_mask = _bit_mask(results.termination_reason, width=8)
    _print("Termination Conditions:")
    for bit, condition in zip(termination_bit_mask, ['Reached max samples',
                                                     'Evidence uncertainty low enough',
                                                     'Small remaining evidence',
                                                     'Reached ESS',
                                                     "Used max num likelihood evaluations",
                                                     'Likelihood contour reached',
                                                     'Sampler efficiency too low',
                                                     'All live-points are on a single plateau (potential numerical errors, consider 64-bit)']):
        if bit == 1:
            _print(condition)
    _print("--------")
    _print("# likelihood evals: {}".format(results.total_num_likelihood_evaluations))
    _print("# samples: {}".format(results.total_num_samples))
    _print("# slices: {:.1f}".format(results.total_num_slices))
    _print(
        "# slices / acceptance: {:.1f}".format(
            jnp.nanmean(jnp.where(results.num_slices > 0, results.num_slices, jnp.nan))))
    _print("# likelihood evals / sample: {:.1f}".format(
        results.total_num_likelihood_evaluations / results.total_num_samples))
    likelihood_evals_per_slice = jnp.sum(
        jnp.where(results.num_slices > 0, results.num_likelihood_evaluations_per_sample, 0.)) / results.total_num_slices
    _print("# likelihood evals / slice: {:.1f}".format(likelihood_evals_per_slice))
    _print("--------")
    _print("logZ={} +- {}".format(_round(results.log_Z_mean, results.log_Z_uncert),
                                  _round(results.log_Z_uncert, results.log_Z_uncert)))
    # _print("H={} +- {}".format(
    #     _round(results.H_mean, results.H_uncert), _round(results.H_uncert, results.H_uncert)))
    _print("H={}".format(
        _round(results.H_mean, results.H_mean)))
    _print("ESS={}".format(int(results.ESS)))
    max_like_idx = jnp.argmax(results.log_L_samples)
    max_like_points = tree_map(lambda x: x[max_like_idx], results.samples)
    samples = resample(random.PRNGKey(23426), results.samples, results.log_dp_mean, S=max(10, int(results.ESS)),
                       replace=True)

    max_map_idx = jnp.argmax(results.log_posterior_density)
    map_points = tree_map(lambda x: x[max_map_idx], results.samples)

    for name in samples.keys():
        _samples = samples[name].reshape((samples[name].shape[0], -1))
        _max_like_points = max_like_points[name].reshape((-1,))
        _map_points = map_points[name].reshape((-1,))
        ndims = _samples.shape[1]
        _print("--------")
        _print("{}: mean +- std.dev. | 10%ile / 50%ile / 90%ile | MAP est. | max(L) est.".format(
            name if ndims == 1 else "{}[#]".format(name), ))
        for dim in range(ndims):
            _uncert = jnp.std(_samples[:, dim])
            _max_like_point = _max_like_points[dim]
            _map_point = _map_points[dim]
            # two sig-figs based on uncert
            sig_figs = -int("{:e}".format(_uncert).split('e')[1]) + 1

            def _round(ar):
                return round(float(ar), sig_figs)

            _uncert = _round(_uncert)
            _print("{}: {} +- {} | {} / {} / {} | {} | {}".format(
                name if ndims == 1 else "{}[{}]".format(name, dim),
                _round(jnp.mean(_samples[:, dim])), _uncert,
                *[_round(a) for a in jnp.percentile(_samples[:, dim], jnp.asarray([10, 50, 90]))],
                _round(_map_point),
                _round(_max_like_point)
            ))
    _print("--------")
    if f_obj is not None:
        out = "\n".join(main_s)
        if isinstance(f_obj, str):
            with open(f_obj, 'w') as f:
                f.write(out)
        elif isinstance(f_obj, io.TextIOBase):
            f_obj.write(out)
        else:
            raise TypeError(f"Invalid f_obj: {type(f_obj)}")


def sample_evidence(key, num_live_points_per_sample, log_L_samples, S: int = 100) -> jnp.ndarray:
    """
    Sample the evidence distribution, but stochastically simulating the shrinkage distribution.

    Note: this produces approximate samples, since there is also an uncertainty in the placement of the contours during
    shrinkage. Incorporating this stochasticity into the simulation would require running an entire
    nested sampling many times.

    Args:
        key: PRNGKey
        num_live_points_per_sample: the number of live points for each sample
        log_L_samples: the log-L of samples
        S: The number of samples to produce

    Returns:
        samples of log(Z)
    """
    n_i = num_live_points_per_sample
    L = LogSpace(jnp.asarray([-jnp.inf], log_L_samples.dtype)).concatenate(
        LogSpace(log_L_samples))
    L_mid = (L[:-1] + L[1:]) * LogSpace(jnp.log(0.5))

    def evidence_chain(key):
        # T ~ Beta(n[i],1) <==> T ~ Kumaraswamy(n[i],1)
        log_T = jnp.log(random.uniform(key, n_i.shape, dtype=L_mid.dtype)) / n_i
        # log_T = jnp.where(n_i == 0., -jnp.inf, log_T)
        # log_T = jnp.log(random.beta(key, state.sample_collection.num_live_points, 1.))
        T = LogSpace(log_T)
        X = LogSpace(jnp.asarray([0.], log_L_samples.dtype)).concatenate(T).cumprod()
        dX = (X[:-1] - X[1:]).abs()
        dZ = dX * L_mid
        dZ = LogSpace(jnp.where(n_i == 0., -jnp.inf, dZ.log_abs_val))
        # dZ = LogSpace(jnp.where(jnp.isnan(dZ.log_abs_val), -jnp.inf, dZ.log_abs_val))
        Z = dZ.sum()
        # ESS = Z.square() / dZ.square().sum()
        return Z.log_abs_val

    log_Z_chains = vmap(evidence_chain)(random.split(key, S))
    Z_chains = LogSpace(log_Z_chains)
    return Z_chains.log_abs_val


def bruteforce_posterior_samples(model: BaseAbstractModel, S: int = 60) -> Tuple[XType, jnp.ndarray]:
    """
    Compute the posterior with brute-force over a regular grid.

    Args:
        model: model
        S: resolution of grid

    Returns:
        samples, and log-weight
    """
    u_vec = jnp.linspace(jnp.finfo(float_type).eps, 1. - jnp.finfo(float_type).eps, S)
    du = u_vec[1] - u_vec[0]
    args = jnp.stack([x.flatten() for x in jnp.meshgrid(*[u_vec] * model.U_ndims, indexing='ij')], axis=-1)
    samples = jit(vmap(model.transform))(args)
    log_L = jit(vmap(model.forward))(args)
    dZ = LogSpace(log_L) * LogSpace(jnp.log(du)) ** model.U_ndims
    return samples, dZ.log_abs_val


def bruteforce_evidence(model: BaseAbstractModel, S: int = 60):
    """
    Compute the evidence with brute-force over a regular grid.

    Args:
        model: model
        S: resolution of grid

    Returns:
        log(Z)
    """

    u_vec = jnp.linspace(jnp.finfo(float_type).eps, 1. - jnp.finfo(float_type).eps, S)
    du = u_vec[1] - u_vec[0]
    args = jnp.stack([x.flatten() for x in jnp.meshgrid(*[u_vec] * model.U_ndims, indexing='ij')], axis=-1)
    Z_true = (LogSpace(jit(vmap(model.forward))(args)).nansum() * LogSpace(
        jnp.log(du)) ** model.U_ndims)
    return Z_true.log_abs_val


@deprecated(bruteforce_posterior_samples)
def analytic_posterior_samples(model: BaseAbstractModel, S: int = 60):
    """
    Compute the evidence with brute-force over a regular grid.

    Args:
        model: model
        S: resolution of grid

    Returns:
        log(Z)
    """
    logger.warning(f"")

    u_vec = jnp.linspace(jnp.finfo(float_type).eps, 1. - jnp.finfo(float_type).eps, S)
    du = u_vec[1] - u_vec[0]
    args = jnp.stack([x.flatten() for x in jnp.meshgrid(*[u_vec] * model.U_ndims, indexing='ij')], axis=-1)
    samples = jit(vmap(model.transform))(args)
    log_L = jit(vmap(model.forward))(args)
    dZ = LogSpace(log_L) * LogSpace(jnp.log(du)) ** model.U_ndims
    return samples, dZ.log_abs_val


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
        pytree: Nested sampler result
        save_file: filename
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


def save_results(results: NestedSamplerResults, save_file: str):
    """
    Saves results of nested sampler in a npz file.

    Args:
        results (NestedSamplerResults): Nested sampler result
        save_file (str): filename
    """
    save_pytree(results, save_file)


def load_pytree(save_file: str):
    """
    Loads saved nested sampler results from a npz file.

    Args:
        save_file (str): filename

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
                data_dict[k] = tree_map(lambda v: jnp.asarray(v), v.item())
        else:
            data_dict[k] = jnp.asarray(v)

    return data_dict


def load_results(save_file: str) -> NestedSamplerResults:
    """
    Loads saved nested sampler results from a npz file.

    Args:
        save_file (str): filename

    Returns:
        NestedSamplerResults
    """
    data_dict = load_pytree(save_file)
    return NestedSamplerResults(**data_dict)
