from typing import NamedTuple

from jax import random, vmap, numpy as jnp, tree_map, jit
from jax.lax import while_loop
from jax.scipy.special import logsumexp
import logging
import numpy as np

from jaxns.internals.maps import dict_multimap, prepare_func_args
from jaxns.internals.log_semiring import cumulative_logsumexp, LogSpace
from jaxns.prior_transforms import PriorChain
from jaxns.internals.types import NestedSamplerResults, ThreadStats, float_type

logger = logging.getLogger(__name__)


def resample(key, samples, log_weights, S=None, replace=False):
    """
    resample the samples with weights which are interpreted as log_probabilities.
    Args:
        samples:
        weights:

    Returns: S samples of equal weight

    """
    if S is None:
        # ESS = (sum w)^2 / sum w^2

        S = int(jnp.exp(2. * logsumexp(log_weights) - logsumexp(2. * log_weights)))

    if not replace:
        # use cumulative_logsumexp because some log_weights could be really small
        log_p_cuml = cumulative_logsumexp(log_weights)
        log_r = log_p_cuml[-1] + jnp.log(1. - random.uniform(key, (S,)))
        idx = jnp.searchsorted(log_p_cuml, log_r)
    else:
        g = -random.gumbel(key, shape=log_weights.shape) - log_weights
        idx = jnp.argsort(g)[:S]
    return dict_multimap(lambda s: s[idx, ...], samples)


def marginalise_static(key, samples, log_weights, ESS, fun):
    """
    Marginalises function over posterior samples, where ESS is static.

    Args:
        key: PRNG key
        samples: dict of batched array of nested sampling samples
        log_weights: log weights from nested sampling
        ESS: static effective sample size
        fun: callable(**kwargs) to marginalise.

    Returns: expectation over resampled samples.
    """
    fun = prepare_func_args(fun)
    samples = resample(key, samples, log_weights, S=ESS, replace=True)
    marginalised = tree_map(lambda marg: jnp.nanmean(marg, axis=0), vmap(lambda d: fun(**d))(samples))
    return marginalised


def marginalise_dynamic(key, samples, log_weights, ESS, fun):
    """
    Marginalises function over posterior samples, where ESS can be dynamic.

    Args:
        key: PRNG key
        samples: dict of batched array of nested sampling samples
        log_weights: log weights from nested sampling
        ESS: dynamic effective sample size
        fun: callable(**kwargs) to marginalise.

    Returns: expectation over resampled samples.
    """
    fun = prepare_func_args(fun)

    def body(state):
        (key, i, marginalised) = state
        key, resample_key = random.split(key, 2)
        _samples = resample(resample_key, samples, log_weights, S=1)
        _sample = tree_map(lambda v: v[0], _samples)
        marginalised = tree_map(lambda x, y: x + y, marginalised, fun(**_sample))
        return (key, i + jnp.ones_like(i), marginalised)

    test_output = fun(**tree_map(lambda v: v[0], samples))
    (_, count, marginalised) = while_loop(lambda state: state[1] < ESS,
                                          body,
                                          (key, jnp.array(0.), tree_map(lambda x: jnp.zeros_like(x), test_output)))
    marginalised = tree_map(lambda x: x / count, marginalised)
    return marginalised


def estimate_map(samples, ESS=None):
    """
    Estimates MAP-point using a histogram of equally weighted samples.

    Args:
        samples: dict, equally weighted samples
        ESS: int

    Returns: dict of samples at MAP-point.
    """

    def _get_map(samples):
        shape = samples.shape[1:]
        samples = samples.reshape([samples.shape[0], -1]).T

        def _single_dim(samples):
            lower, upper = jnp.percentile(samples, [1, 99])
            if ESS is not None:
                bins = jnp.linspace(lower, upper, int(np.sqrt(ESS)))
            else:
                bins = jnp.linspace(lower, upper, int(jnp.sqrt(samples.size)))
            hist, _ = jnp.histogram(samples, bins=bins)
            centers = 0.5 * (bins[1:] + bins[:-1])
            return centers[jnp.argmax(hist)]

        return vmap(_single_dim)(samples).reshape(shape)

    return tree_map(_get_map, samples)


def maximum_a_posteriori_point(results: NestedSamplerResults):
    """
    Get the MAP point of a nested sampling result.
    Does this by choosing the point with largest L(x) p(x).

    Args:
        results: NestedSamplerResult

    Returns: dict of samples at MAP-point.
    """

    map_idx = jnp.argmax(results.log_dp_mean)
    map_points = tree_map(lambda x: x[map_idx], results.samples)
    return map_points


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


def summary(results: NestedSamplerResults) -> str:
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
    termination_bit_mask = _bit_mask(results.termination_reason, width=7)
    _print("Termination Conditions:")
    for bit, condition in zip(termination_bit_mask, ['Reached max samples',
                                                     'Evidence uncertainty low enough',
                                                     'Small remaining evidence',
                                                     'Reached ESS',
                                                     "Used max num steps",
                                                     "Used max num likelihood evaluations",
                                                     "All samples on plateau"]):
        if bit == 1:
            _print(condition)
    _print("--------")
    _print("# likelihood evals: {}".format(results.total_num_likelihood_evaluations))
    _print("# samples: {}".format(results.total_num_samples))
    _print("# likelihood evals / sample: {:.1f}".format(
        results.total_num_likelihood_evaluations / results.total_num_samples))
    _print("--------")
    _print("logZ={} +- {}".format(_round(results.log_Z_mean, results.log_Z_uncert),
                                  _round(results.log_Z_uncert, results.log_Z_uncert)))
    # _print("H={} +- {}".format(
    #     _round(results.H_mean, results.H_uncert), _round(results.H_uncert, results.H_uncert)))
    _print("H={}".format(
        _round(results.H_mean, results.H_mean)))
    _print("ESS={}".format(int(results.ESS)))

    max_like_idx = jnp.argmax(results.log_L_samples[:results.total_num_samples])
    max_like_points = tree_map(lambda x: x[max_like_idx], results.samples)
    samples = resample(random.PRNGKey(23426), results.samples, results.log_dp_mean, S=int(results.ESS), replace=True)

    map_points = maximum_a_posteriori_point(results)

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
    return "\n".join(main_s)


def evidence_posterior_samples(key, num_live_points_per_sample, log_L_samples, S: int = 100):
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


def analytic_log_evidence(prior_chain: PriorChain, log_likelihood, S: int = 60):
    """
    Compute the evidence with brute-force over a regular grid.

    Args:
        prior_chain: PriorChain of model
        log_likelihood: callable(**samples)
        S: int, resolution of grid

    Returns:
        log(Z)
    """
    log_likelihood = prepare_func_args(log_likelihood)
    if not prior_chain.built:
        prior_chain.build()

    u_vec = jnp.linspace(jnp.finfo(float_type).eps, 1. - jnp.finfo(float_type).eps, S)
    du = u_vec[1] - u_vec[0]
    args = jnp.stack([x.flatten() for x in jnp.meshgrid(*[u_vec] * prior_chain.U_ndims, indexing='ij')], axis=-1)
    Z_true = (LogSpace(jit(vmap(lambda arg: log_likelihood(**prior_chain(arg))))(args)).nansum() * LogSpace(
        jnp.log(du)) ** prior_chain.U_ndims)
    return Z_true.log_abs_val


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


def save_results(results: NestedSamplerResults, save_file: str):
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


def load_results(save_file: str) -> NestedSamplerResults:
    """
    Loads saved nested sampler results from a npz file.

    Args:
        save_file: str

    Returns:
        NestedSamplerResults
    """
    data_dict = load_pytree(save_file)
    thread_stats = ThreadStats(**data_dict.pop('thread_stats', None))
    return NestedSamplerResults(**data_dict, thread_stats=thread_stats)