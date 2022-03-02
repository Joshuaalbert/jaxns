from jax import random, vmap, numpy as jnp, tree_map, local_device_count, devices as get_devices, pmap, jit, device_get, \
    tree_multimap
from timeit import default_timer
from jax.lax import scan, while_loop
from jax.scipy.special import logsumexp
import logging
import numpy as np
import inspect

from jaxns.log_math import cumulative_logsumexp, LogSpace
from jaxns.types import NestedSamplerResults

logger = logging.getLogger(__name__)


def random_ortho_matrix(key, n):
    """
    Samples a random orthonormal num_parent,num_parent matrix from Stiefels manifold.
    From https://stackoverflow.com/a/38430739

    Args:
        key: PRNG seed
        n: Size of matrix, draws from O(num_options) group.

    Returns: random [num_options,num_options] matrix with determinant = +-1
    """
    H = random.normal(key, shape=(n, n))
    Q, R = jnp.linalg.qr(H)
    Q = Q @ jnp.diag(jnp.sign(jnp.diag(R)))
    return Q


def test_random_ortho_matrix():
    M = random_ortho_matrix(random.PRNGKey(42), 5)
    print(M.T @ M, M @ M.T)
    print(jnp.linalg.norm(M, axis=0),
          jnp.linalg.norm(M, axis=1))


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


def broadcast_shapes(shape1, shape2):
    """
    Broadcasts two shapes together.

    Args:
        shape1: tuple of int
        shape2: tuple of int

    Returns: tuple of int with resulting shape.
    """
    if isinstance(shape1, int):
        shape1 = (shape1,)
    if isinstance(shape2, int):
        shape2 = (shape2,)

    def left_pad_shape(shape, l):
        return tuple([1] * l + list(shape))

    l = max(len(shape1), len(shape2))
    shape1 = left_pad_shape(shape1, l - len(shape1))
    shape2 = left_pad_shape(shape2, l - len(shape2))
    out_shape = []
    for s1, s2 in zip(shape1, shape2):
        m = max(s1, s2)
        if ((s1 != m) and (s1 != 1)) or ((s2 != m) and (s2 != 1)):
            raise ValueError("Trying to broadcast {} with {}".format(shape1, shape2))
        out_shape.append(m)
    return tuple(out_shape)


def iterative_topological_sort(graph, start=None):
    """
    Get Depth-first topology.

    :param graph: dependency dict (like a dask)
        {'a':['b','c'],
        'c':['b'],
        'b':[]}
    :param start: str
        the node you want to search from.
        This is equivalent to the node you want to compute.
    :return: list of str
        The order get from `start` to all ancestors in DFS.
    """
    seen = set()
    stack = []  # path variable is gone, stack and order are new
    order = []  # order will be in reverse order at first
    if start is None:
        start = list(graph.keys())
    if not isinstance(start, (list, tuple)):
        start = [start]
    q = start
    while q:
        v = q.pop()
        if not isinstance(v, str):
            raise ValueError("Key {} is not a str".format(v))
        if v not in seen:
            seen.add(v)  # no need to append to path any more
            if v not in graph.keys():
                graph[v] = []
            q.extend(graph[v])

            while stack and v not in graph[stack[-1]]:  # new stuff here!
                order.append(stack.pop())
            stack.append(v)

    return stack + order[::-1]  # new return value!


def left_broadcast_mul(x, y):
    """
    Aligns on left dim and multiplies.
    Args:
        x: [D]
        y: [D,b0,...bN]

    Returns:
        [D,b0,...,bN]
    """
    return jnp.reshape(x, (-1,) + tuple([1] * (len(y.shape) - 1))) * y


def tuple_prod(t):
    """
    Product of shape tuple

    Args:
        t: tuple

    Returns:
        int
    """
    if len(t) == 0:
        return 1
    res = t[0]
    for a in t[1:]:
        res *= a
    return res


def msqrt(A):
    """
    Computes the matrix square-root using SVD, which is robust to poorly conditioned covariance matrices.
    Computes, M such that M @ M.T = A

    Args:
        A: [N,N] Square matrix to take square root of.

    Returns: [N,N] matrix.
    """
    U, s, Vh = jnp.linalg.svd(A)
    L = U * jnp.sqrt(s)
    return L


def resample(key, samples, log_weights, S=None):
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

    # use cumulative_logsumexp because some log_weights could be really small
    log_p_cuml = cumulative_logsumexp(log_weights)
    log_r = log_p_cuml[-1] + jnp.log(1. - random.uniform(key, (S,)))
    idx = jnp.searchsorted(log_p_cuml, log_r)
    return dict_multimap(lambda s: s[idx, ...], samples)


def normal_to_lognormal(mu, std):
    """
    Convert normal parameters to log-normal parameters.
    Args:
        mu:
        var:

    Returns:

    """
    var = std ** 2
    ln_mu = 2. * jnp.log(mu) - 0.5 * jnp.log(var)
    ln_var = jnp.log(var) - 2. * jnp.log(mu)
    return ln_mu, jnp.sqrt(ln_var)


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
    samples = resample(key, samples, log_weights, S=ESS)
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


def _debug_chunked_pmap(f, *args, chunksize=None):
    # TODO: remove dask, use jaxns chunked_pmap

    T = args[0].shape[0]
    # lazy import in case dask not installed
    from datetime import datetime
    import os
    from dask.threaded import get

    devices = get_devices()

    def build_pmap_body(dev_idx):
        fun = jit(f, device=devices[dev_idx])
        log = os.path.join(os.getcwd(), "chunk{:02d}.log".format(dev_idx))

        def pmap_body(*args):
            result = []
            with open(log, 'a') as f:
                for i in range(T):
                    item = jnp.ravel_multi_index((dev_idx, i), (chunksize, T))
                    logger.info("Starting item: {}".format(item))
                    f.write('{} {}'.format(datetime.now().isoformat(), "Starting item: {}".format(item)))
                    result.append(fun(*[a[i, ...] for a in args]))
                    tree_map(lambda a: a.block_until_ready(), result[-1])
                    logger.info("Done item: {}".format(item))
                    f.write('{} {}'.format(datetime.now().isoformat(), "Done item: {}".format(item)))
            result = tree_multimap(lambda *result: jnp.stack(result, axis=0), *result)
            return result

        return pmap_body

    # if jit_is_disabled():
    #     num_devices = local_device_count()
    #     dsk = {str(device): (build_pmap_body(device),) + tuple([arg[device] for arg in args]) for device in
    #            range(num_devices)}
    #     result_keys = [str(device) for device in range(num_devices)]
    #     result = get(dsk, result_keys, num_workers=num_devices)
    #     result = device_get(result)
    #     result = tree_multimap(lambda *result: jnp.stack(result, axis=0), *result)
    # else:
    num_devices = local_device_count()
    dsk = {str(device): (build_pmap_body(device),) + tuple([arg[device] for arg in args]) for device in
           range(num_devices)}
    result_keys = [str(device) for device in range(num_devices)]
    result = get(dsk, result_keys, num_workers=num_devices)
    result = device_get(result)
    result = tree_multimap(lambda *result: jnp.stack(result, axis=0), *result)
    return result


def chunked_pmap(f, *args, chunksize=None, use_vmap=False, per_device_unroll=False, debug=False, batch_size=None):
    """
    Calls pmap on chunks of moderate work to be distributed over devices.
    Automatically handle non-dividing chunksizes, by adding filler elements.

    Args:
        f: jittable, callable
        *args: ndarray arguments to map down first dimension
        chunksize: optional chunk size, should be <= local_device_count(). None is local_device_count.

    Returns: pytree mapped result.
    """
    if chunksize is None:
        chunksize = local_device_count()
    if chunksize > local_device_count():
        raise ValueError(f"blocksize should be <= {local_device_count()}.")
    if batch_size is None:
        N = args[0].shape[0]
    else:
        N = batch_size
    remainder = N % chunksize
    if (remainder != 0) and (N > chunksize):
        # only pad if not a zero remainder
        extra = chunksize - remainder
        if N >= chunksize:
            args = tree_map(lambda arg: jnp.concatenate([arg, arg[:extra]], axis=0), args)
        else:
            args = tree_map(lambda arg: jnp.concatenate([arg] + [arg[-1:]] * extra, axis=0), args)
        N = batch_size + extra
    args = tree_map(lambda arg: jnp.reshape(arg, (chunksize, N // chunksize) + arg.shape[1:]), args)
    T = N // chunksize
    logger.info(f"Distributing {N} over {chunksize} devices in queues of length {T}.")
    t0 = default_timer()

    def pmap_body(*args):
        """
        Distributes the computation in queues which are computed with scan.
        Args:
            *args: 
        """

        def body(state, args):
            return state, f(*args)

        _, result = scan(body, (), args, unroll=1)
        return result

    if debug:
        result = _debug_chunked_pmap(f, *args, chunksize=chunksize)
    elif use_vmap:
        result = vmap(pmap_body)(*args)
    elif per_device_unroll:
        devices = get_devices()
        if len(devices) < chunksize:
            raise ValueError(f"Not enough devices {len(devices)} for blocksize {chunksize}")
        result = []
        for i in range(chunksize):
            dev = devices[i]
            _func = jit(pmap_body, device=dev)
            _args = tree_map(lambda x: x[i], args)
            result.append(_func(*_args))
        # result = [device_get(r) for r in result]
        result = tree_multimap(lambda *x: jnp.stack(x, axis=0), *result)
    else:
        result = pmap(pmap_body)(*args)
    result = tree_map(lambda arg: jnp.reshape(arg, (-1,) + arg.shape[2:]), result)
    if remainder != 0:
        # only slice if not a zero remainder
        result = tree_map(lambda x: x[:-extra], result)
    dt = default_timer() - t0
    logger.info(f"Time to run: {dt} s, rate: {N / dt} / s, normalised rate: {N / dt / chunksize} / s / device")
    return result


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


def maximum_a_posteriori_point(results):
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
                                                     'Absolute evidence error low enough',
                                                     'Likelihood contour reached',
                                                     'Small remaining evidence',
                                                     'Likelihood peak reached',
                                                     'Reached ESS',
                                                     "Reached max num threads"]):
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
    samples = resample(random.PRNGKey(23426), results.samples, results.log_dp_mean, S=int(results.ESS))

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


def squared_norm(x1, x2):
    # r2_ij = sum_k (x_ik - x_jk)^2
    #       = sum_k x_ik^2 - 2 x_jk x_ik + x_jk^2
    #       = sum_k x_ik^2 + x_jk^2 - 2 X X^T
    # r2_ij = sum_k (x_ik - y_jk)^2
    #       = sum_k x_ik^2 - 2 y_jk x_ik + y_jk^2
    #       = sum_k x_ik^2 + y_jk^2 - 2 X Y^T
    x1 = x1
    x2 = x2
    r2 = jnp.sum(jnp.square(x1), axis=1)[:, None] + jnp.sum(jnp.square(x2), axis=1)[None, :]
    r2 = r2 - 2. * (x1 @ x2.T)
    return r2


def latin_hypercube(key, num_samples, num_dim, cube_scale):
    """
    Sample from the latin-hypercube defined as the continuous analog of the discrete latin-hypercube.
    That is, if you partition each dimension into `num_samples` equal volume intervals then there is (conditionally)
    exactly one point in each interval. We guarantee that uniformity by randomly assigning the permutation of each dimension.
    The degree of randomness is controlled by `cube_scale`. A value of 0 places the sample at the center of the grid point,
    and a value of 1 places the value randomly inside the grid-cell.

    Args:
        key: PRNG key
        num_samples: number of samples in total to draw
        num_dim: number of dimensions in each sample
        cube_scale: The scale of randomness, in (0,1).

    Returns:
        latin-hypercube samples of shape [num_samples, num_dim]
    """
    key1, key2 = random.split(key, 2)
    cube_scale = jnp.clip(cube_scale, 0., 1.)
    samples = vmap(lambda key: random.permutation(key, num_samples))(random.split(key2, num_dim)).T
    samples += random.uniform(key1, shape=samples.shape, minval=0.5 - cube_scale / 2., maxval=0.5 + cube_scale / 2.)
    samples /= num_samples
    return samples


def density_estimation(xstar, x, alpha=1. / 3., order=1):
    assert len(x.shape) == 2

    N = x.shape[0]
    m = int(pow(N, 1. - alpha))
    s = N // m
    N = m * s
    x = x[:N]

    def single_density(xstar):
        dist = jnp.linalg.norm(x - xstar, ord=order, axis=-1)  # N
        dist = jnp.reshape(dist, (s, m))  # s,m
        min_dist = jnp.min(dist, axis=0)  # m
        avg_dist = jnp.mean(min_dist)  # scalar
        return 0.5 / ((1. + s) * avg_dist)

    return vmap(single_density)(xstar)


def test_density_estimation():
    np.random.seed(42)
    x = jnp.asarray(np.random.standard_gamma(1., 100))[:, None]
    xstar = jnp.linspace(0., 20., 1000)[:, None]
    import pylab as plt

    plt.plot(xstar, density_estimation(xstar, x))
    plt.hist(np.random.standard_gamma(1., 10000), bins=np.linspace(0, 20, 100), density=True, alpha=0.5)
    plt.hist(x[:, 0], bins=np.linspace(0., 20, 100), density=True, alpha=0.5)
    plt.show()


def prepare_func_args(f):
    """
    Takes a callable(a,b,...,z=Z) and prepares it into callable(**kwargs), such that only
    a,b,...,z are taken fro **kwargs and the rest ignored.

    Args:
        f: callable(a,b,...,z=Z)

    Returns:
        callable(**kwargs)
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


def linear_to_log_stats(log_f_mean, *, log_f2_mean=None, log_f_var=None):
    """
    Converts normal to log-normal stats.
    Args:
        log_f_mean: log(E(f))
        log_f2_mean: log(E(f**2))
        log_f_var: log(Var(f))
    Returns:
        E(log(f))
        Var(log(f))
    """
    f_mean = LogSpace(log_f_mean)
    if log_f_var is not None:
        f_var = LogSpace(log_f_var)
        f2_mean = f_var + f_mean.square()
    else:
        f2_mean = LogSpace(log_f2_mean)
    mu = f_mean.square() / f2_mean.sqrt()
    sigma2 = f2_mean / f_mean.square()
    return mu.log_abs_val, sigma2.log_abs_val


def evidence_posterior_samples(key, num_live_points_per_sample, log_L_samples, S: int = 100):
    n_i = num_live_points_per_sample
    L = LogSpace(jnp.asarray([-jnp.inf], log_L_samples.dtype)).concatenate(
        LogSpace(log_L_samples))
    L_mid = (L[:-1] + L[1:]) * LogSpace(jnp.log(0.5))

    def evidence_chain(key):
        # T ~ Beta(n[i],1) <==> T ~ Kumaraswamy(n[i],1)
        log_T = jnp.log(random.uniform(key, n_i.shape, dtype=L_mid.dtype)) / n_i
        # log_T = jnp.log(random.beta(key, state.sample_collection.num_live_points, 1.))
        T = LogSpace(log_T)
        X = LogSpace(jnp.asarray([0.], log_L_samples.dtype)).concatenate(T).cumprod()
        dX = (X[:-1] - X[1:]).abs()
        dZ = dX * L_mid
        Z = dZ.sum()
        # ESS = Z.square() / dZ.square().sum()
        return Z.log_abs_val

    chain_key, key = random.split(key, 2)
    log_Z_chains = vmap(evidence_chain)(random.split(chain_key, S))
    Z_chains = LogSpace(log_Z_chains)
    return Z_chains.log_abs_val
