from jax import random, vmap, numpy as jnp, tree_map, local_device_count, devices as get_devices, pmap, jit, device_get, \
    tree_multimap
from jax.lax import scan, while_loop
from jax.scipy.special import logsumexp, gammaln
import numpy as np
import logging

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


def test_random_ortho_normal_matrix():
    for i in range(100):
        H = random_ortho_matrix(random.PRNGKey(0), 3)
        print(jnp.linalg.eigvals(H))
        assert jnp.all(jnp.isclose(H @ H.conj().T, jnp.eye(3), atol=1e-7))


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


def swap_dict_namedtuple(t):
    """
    Turn namedtuple({a:b},{a:c})
    to {a:namedtuple(b,c)}

    If dict is zero-level, i.e. scalar, then return the tuple.
    """
    for s in t:
        if not isinstance(s, dict):
            return t
        assert sorted(s.keys()) == sorted(t[0].keys())
    keys = t[0].keys()
    d = dict()
    for key in keys:
        d[key] = t.__class__(*[s[key] for s in t])
    return d


def swap_namedtuple_dict(d):
    """
    Turn {a:namedtuple(b,c)}
    to namedtuple({a:b},{a:c})
    """
    if not isinstance(d, dict):
        return d
    keys = list(d.keys())
    for k, v in d.items():
        assert v._fields == d[keys[0]]._fields
    fields = d[keys[0]]._fields
    t = []
    for i in range(len(fields)):
        t.append({key: d[key][i] for key in keys})
    return d[keys[0]].__class__(*t)


def test_swap():
    from collections import namedtuple

    Test = namedtuple('Test', ['a', 'b', 'c'])
    test_tuple = Test(dict(da=1., db=2.), dict(da=3., db=4.), dict(da=5., db=6.))
    test_dict = {'da': Test(a=1.0, b=3.0, c=5.0), 'db': Test(a=2.0, b=4.0, c=6.0)}
    _swap_dict = swap_dict_namedtuple(test_tuple)
    assert _swap_dict == test_dict
    _swap_tuple = swap_namedtuple_dict(test_dict)
    assert _swap_tuple == test_tuple

    test_tuple = Test(1., 2., 3.)
    test_dict = Test(1., 2., 3.)
    _swap_dict = swap_dict_namedtuple(test_tuple)
    assert _swap_dict == test_dict
    _swap_tuple = swap_namedtuple_dict(test_dict)
    assert _swap_tuple == test_tuple


def get_interval(s):
    ar_first = jnp.argmin(s)
    s = jnp.concatenate([jnp.where(jnp.arange(s.size) < ar_first, -1., s), jnp.array([1.])])
    ar_last = jnp.maximum(0, jnp.argmax(s) - 1)
    return ar_first, ar_last


def test_get_interval():
    s = jnp.array([1, -1, -1, -1, 1])
    assert get_interval(s) == (1, 3)
    s = jnp.array([1, -1, -1, -1])
    assert get_interval(s) == (1, 3)
    s = jnp.array([-1, -1, -1])
    assert get_interval(s) == (0, 2)
    s = jnp.array([-1, -1, -1, 1])
    assert get_interval(s) == (0, 2)
    s = jnp.array([-1])
    assert get_interval(s) == (0, 0)
    s = jnp.array([1])
    assert get_interval(s) == (0, 0)
    s = jnp.array([1, -1])
    assert get_interval(s) == (1, 1)


def masked_cluster_id(points, centers, K):
    """
    Computes the cluster cluster centre closest to points in `points`.
    Masks to only those clusters that are in use.

    Args:
        points: [N,D] Array of points to be sorted into clusters.
        centers: [K, D] centers of clusters
        K: int number of clusters in use, rest of masked.
    Returns: [N] int, cluster id's.
    """
    max_K = centers.shape[0]
    # max_K, N
    dist = jnp.linalg.norm(centers[:, None, :] - points[None, :, :], axis=-1)
    dist = jnp.where(jnp.arange(max_K)[:, None] > K - 1, jnp.full_like(dist, jnp.inf), dist)
    return jnp.argmin(dist, axis=0)


def cluster(key, points, max_K=6):
    """
    Cluster `init_U` automatically choosing K.
    Outdated.

    Args:
        key:
        points: [N,M]
        max_K:

    Returns: tuple of
        key: PRNG key
        cluster_centers: [max_K, M] (last max_K - K rows are zeros)
        K: int, the number of clusters found.

    """

    # init_U = init_U - jnp.mean(init_U, axis=0)
    # init_U = init_U / jnp.maximum(jnp.std(init_U, axis=0), 1e-8)

    def _init_points(key, points):
        def body(state, X):
            (key, i, center) = state
            key, new_point_key, t_key = random.split(key, 3)
            new_point = points[random.randint(new_point_key, (), 0, points.shape[0]), :]
            dx = points - center
            p = new_point - center
            p = p / jnp.linalg.norm(p)
            t_new = jnp.max(dx @ p, axis=0)
            new_point = center + random.uniform(t_key) * t_new * p
            center = (center * i + new_point) / (i + 1)
            return (key, i + 1, center), (new_point,)

        (key, _, _), (init_points,) = scan(body, (key, 0, points[0, :]), (jnp.arange(max_K),))
        return key, init_points

    def kmeans(key, points, K):
        # key, shuffle_key = random.split(key, 2)
        # centers = random.shuffle(shuffle_key, init_U, axis=0)
        # centers = centers[:K, :]
        key, centers = _init_points(key, points)
        # N
        cluster_id = masked_cluster_id(points, centers, K)

        def body(state):
            (done, i, centers, cluster_id) = state

            # [M, max_K]
            new_centers = vmap(lambda coords:
                               jnp.bincount(cluster_id, weights=coords, minlength=max_K, length=max_K))(points.T)
            # max_K, M
            new_centers = new_centers.T
            # max_K
            num_per_cluster = jnp.bincount(cluster_id, minlength=max_K, length=max_K)
            # max_K, M
            new_centers = jnp.where(num_per_cluster[:, None] == 0.,
                                    jnp.zeros_like(new_centers),
                                    new_centers / num_per_cluster[:, None])
            # N
            new_cluster_id = masked_cluster_id(points, new_centers, K)

            done = jnp.all(new_cluster_id == cluster_id)

            return (done, i + 1, new_centers, new_cluster_id)

        (_, _, centers, _) = while_loop(lambda state: ~state[0],
                                        body,
                                        (jnp.array(False), jnp.array(0), centers, cluster_id))
        return key, centers

    def metric(points, centers, K):
        # N
        cluster_id = masked_cluster_id(points, centers, K)
        # N,N
        dist = jnp.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        # N,N
        in_group = cluster_id[:, None] == cluster_id[None, :]
        in_group_dist, w = jnp.average(dist, weights=in_group, axis=-1, returned=True)
        in_group_dist *= w / (w - 1.)

        # max_K, N, N
        out_group = (~in_group) & (jnp.arange(max_K)[:, None, None] == cluster_id[None, None, :])
        # max_K, N
        out_group_dist = jnp.sum(dist * out_group, axis=-1) / jnp.sum(out_group, axis=-1)
        out_group_dist = jnp.where(jnp.isnan(out_group_dist), jnp.inf, out_group_dist)
        # N
        out_group_dist = jnp.min(out_group_dist, axis=0)
        out_group_dist = jnp.where(jnp.isinf(out_group_dist), jnp.max(in_group_dist), out_group_dist)
        sillohette = (out_group_dist - in_group_dist) / jnp.maximum(in_group_dist, out_group_dist)
        # condition for pos def cov
        sillohette = jnp.where(w < points.shape[1], -jnp.inf, sillohette)
        return jnp.mean(sillohette), cluster_id

    def cluster_probe(K, key):
        key, centers = kmeans(key, points, K)
        sillohette, _ = metric(points, centers, K)
        return centers, sillohette

    key, split_key = random.split(key, 2)
    # test_K = jnp.arange(1, max_K+1)
    # centers_list, sillohettes = vmap(cluster_probe)(test_K, random.split(split_key, max_K))
    # best = jnp.argmax(sillohettes)
    # K = test_K[best]
    # centers = centers_list[best, :, :]

    sillohettes = []
    centers_list = []
    for test_key, K in zip(random.split(split_key, max_K), range(1, max_K + 1)):
        clusters, sillohette = cluster_probe(K, test_key)
        centers_list.append(clusters)
        sillohettes.append(sillohette)
    sillohettes = jnp.stack(sillohettes)
    centers_list = jnp.stack(centers_list, axis=0)
    best = jnp.argmax(sillohettes)
    K = best + 1
    centers = centers_list[best, :, :]

    return key, centers, K, sillohettes


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


def test_broadcast_shapes():
    assert broadcast_shapes(1, 1) == (1,)
    assert broadcast_shapes(1, 2) == (2,)
    assert broadcast_shapes(1, (2, 2)) == (2, 2)
    assert broadcast_shapes((2, 2), 1) == (2, 2)
    assert broadcast_shapes((1, 1), (2, 2)) == (2, 2)
    assert broadcast_shapes((1, 2), (2, 2)) == (2, 2)
    assert broadcast_shapes((2, 1), (2, 2)) == (2, 2)
    assert broadcast_shapes((2, 1), (2, 1)) == (2, 1)
    assert broadcast_shapes((2, 1), (1, 1)) == (2, 1)
    assert broadcast_shapes((1, 1), (1, 1)) == (1, 1)
    assert broadcast_shapes((1,), (1, 1)) == (1, 1)
    assert broadcast_shapes((1, 1, 2), (1, 1)) == (1, 1, 2)
    assert broadcast_shapes((1, 2, 1), (1, 3)) == (1, 2, 3)
    assert broadcast_shapes((1, 2, 1), ()) == (1, 2, 1)


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


def test_iterative_topological_sort():
    dsk = {'a': [],
           'b': ['a'],
           'c': ['a', 'b']}
    assert iterative_topological_sort(dsk, ['a', 'b', 'c']) == ['c', 'b', 'a']
    assert iterative_topological_sort(dsk) == ['c', 'b', 'a']
    dsk = {'a': [],
           'b': ['a', 'd'],
           'c': ['a', 'b']}
    # print(iterative_topological_sort(dsk, ['a', 'b', 'c']))
    assert iterative_topological_sort(dsk, ['a', 'b', 'c']) == ['c', 'b', 'a', 'd']
    assert iterative_topological_sort(dsk) == ['c', 'b', 'a', 'd']


def stochastic_result_computation(key, results, S):
    """
    MC simulate the trajectory and compute statistics, and evidence.

    Args:
        key: PRNG seed
        results: NestedSamplerResult
        S: int

    Returns:
        ((mean logZ, std logZ), (mean ESS, std ESS), (mean H, std H))
    """

    def single_sample(n_per_sample, key, log_L_samples):
        # N
        t = jnp.where(n_per_sample == jnp.inf, 1., random.beta(key, n_per_sample, 1))
        log_t = jnp.log(t)
        log_X = jnp.cumsum(log_t)
        log_L_samples = jnp.concatenate([jnp.array([-jnp.inf]), log_L_samples])
        log_X = jnp.concatenate([jnp.array([0.]), log_X])
        # log_dX = log(1-t_i) + log(X[i-1])
        log_dX = jnp.log(1. - t) + log_X[:-1]  # jnp.log(-jnp.diff(jnp.exp(log_X))) #-inf where n_per_sample=inf
        log_avg_L = jnp.logaddexp(log_L_samples[:-1], log_L_samples[1:]) - jnp.log(2.)
        log_p = log_dX + log_avg_L
        # param calculation
        logZ = logsumexp(log_p)
        log_w = log_p - logZ
        # Kish's ESS = [sum weights]^2 / [sum weights^2]
        ESS = jnp.exp(2. * logsumexp(log_w) - logsumexp(2. * log_w))
        # H = sum w_i log(L)
        _H = jnp.exp(log_w) * log_avg_L
        H = jnp.sum(jnp.where(jnp.isnan(_H), 0., _H))
        return logZ, ESS, H

    logZ, ESS, H = vmap(lambda key: single_sample(results.n_per_sample, key, results.log_L_samples))(
        random.split(key, S))
    return (jnp.mean(logZ), jnp.std(logZ)), (jnp.mean(ESS), jnp.std(ESS)), (jnp.mean(H), jnp.std(H))


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
    if len(t) == 0:
        return 1
    res = t[0]
    for a in t[1:]:
        res *= a
    return res


def test_tuple_prod():
    assert tuple_prod(()) == 1
    assert tuple_prod((1, 2, 3)) == 6
    assert tuple_prod((4,)) == 4


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


def test_msqrt():
    for i in range(10):
        A = random.normal(random.PRNGKey(i), shape=(30, 30))
        A = A @ A.T
        L = msqrt(A)
        assert jnp.all(jnp.isclose(A, L @ L.T))


def logaddexp(x1, x2):
    """
    Equivalent to logaddexp but supporting complex arguments.

    see np.logaddexp
    """
    if is_complex(x1) or is_complex(x2):
        select1 = x1.real > x2.real
        amax = jnp.where(select1, x1, x2)
        delta = jnp.where(select1, x2 - x1, x1 - x2)
        return jnp.where(jnp.isnan(delta),
                         x1 + x2,  # NaNs or infinities of the same sign.
                         amax + jnp.log1p(jnp.exp(delta)))
    else:
        return jnp.logaddexp(x1, x2)


def test_logaddexp():
    a = jnp.log(1.)
    b = jnp.log(1.)
    assert logaddexp(a, b) == jnp.log(2.)
    a = jnp.log(1.)
    b = jnp.log(-2. + 0j)
    assert jnp.isclose(jnp.exp(logaddexp(a, b)).real, -1.)

    a = jnp.log(-1. + 0j)
    b = jnp.log(2. + 0j)
    assert jnp.isclose(jnp.exp(logaddexp(a, b)).real, 1.)

    for i in range(100):
        u = random.uniform(random.PRNGKey(i), shape=(2,)) * 20. - 10.
        a = jnp.log(u[0] + 0j)
        b = jnp.log(u[1] + 0j)
        assert jnp.isclose(jnp.exp(logaddexp(a, b)).real, u[0] + u[1])


def signed_logaddexp(log_abs_val1, sign1, log_abs_val2, sign2):
    """
    Equivalent of logaddexp but for signed quantities too.
    Broadcasting supported.

    Args:
        log_abs_val1: log(|val1|)
        sign1: sign(val1)
        log_abs_val2: log(|val2|)
        sign2: sign(val2)

    Returns:
        (log(|val1+val2|), sign(val1+val2))
    """
    amax = jnp.maximum(log_abs_val1, log_abs_val2)
    signmax = jnp.where(log_abs_val1 > log_abs_val2, sign1, sign2)
    delta = -jnp.abs(log_abs_val2 - log_abs_val1)  # nan iff inf - inf
    sign = sign1 * sign2
    return jnp.where(jnp.isnan(delta),
                     log_abs_val1 + log_abs_val2,  # NaNs or infinities of the same sign.
                     amax + jnp.log1p(sign * jnp.exp(delta))), signmax


def test_signed_logaddexp():
    for i in range(100):
        u = random.uniform(random.PRNGKey(i), shape=(2,)) * 20. - 10.
        a = jnp.log(jnp.abs(u[0]))
        b = jnp.log(jnp.abs(u[1]))
        sign1 = jnp.sign(u[0])
        sign2 = jnp.sign(u[1])
        ans = u[0] + u[1]
        ans_sign = jnp.sign(ans)
        log_abs_ans = jnp.log(jnp.abs(ans))
        log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
        assert sign_c == ans_sign
        assert jnp.isclose(log_abs_c, log_abs_ans)

    u = [1., 1.]
    a = jnp.log(jnp.abs(u[0]))
    b = jnp.log(jnp.abs(u[1]))
    sign1 = jnp.sign(u[0])
    sign2 = jnp.sign(u[1])
    ans = u[0] + u[1]
    ans_sign = jnp.sign(ans)
    log_abs_ans = jnp.log(jnp.abs(ans))
    log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
    assert sign_c == ans_sign
    assert jnp.isclose(log_abs_c, log_abs_ans)

    u = [1., -1.]
    a = jnp.log(jnp.abs(u[0]))
    b = jnp.log(jnp.abs(u[1]))
    sign1 = jnp.sign(u[0])
    sign2 = jnp.sign(u[1])
    ans = u[0] + u[1]
    ans_sign = jnp.sign(ans)
    log_abs_ans = jnp.log(jnp.abs(ans))
    log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
    # assert sign_c == ans_sign
    assert jnp.isclose(log_abs_c, log_abs_ans)

    u = [-1., 1.]
    a = jnp.log(jnp.abs(u[0]))
    b = jnp.log(jnp.abs(u[1]))
    sign1 = jnp.sign(u[0])
    sign2 = jnp.sign(u[1])
    ans = u[0] + u[1]
    ans_sign = jnp.sign(ans)
    log_abs_ans = jnp.log(jnp.abs(ans))
    log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
    # assert sign_c == ans_sign
    assert jnp.isclose(log_abs_c, log_abs_ans)

    u = [0., 0.]
    a = jnp.log(jnp.abs(u[0]))
    b = jnp.log(jnp.abs(u[1]))
    sign1 = jnp.sign(u[0])
    sign2 = jnp.sign(u[1])
    ans = u[0] + u[1]
    ans_sign = jnp.sign(ans)
    log_abs_ans = jnp.log(jnp.abs(ans))
    log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
    assert sign_c == ans_sign
    assert jnp.isclose(log_abs_c, log_abs_ans)

    u = [0., 1.]
    a = jnp.log(jnp.abs(u[0]))
    b = jnp.log(jnp.abs(u[1]))
    sign1 = jnp.sign(u[0])
    sign2 = jnp.sign(u[1])
    ans = u[0] + u[1]
    ans_sign = jnp.sign(ans)
    log_abs_ans = jnp.log(jnp.abs(ans))
    log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
    assert sign_c == ans_sign
    assert jnp.isclose(log_abs_c, log_abs_ans)

    u = [0., -1.]
    a = jnp.log(jnp.abs(u[0]))
    b = jnp.log(jnp.abs(u[1]))
    sign1 = jnp.sign(u[0])
    sign2 = jnp.sign(u[1])
    ans = u[0] + u[1]
    ans_sign = jnp.sign(ans)
    log_abs_ans = jnp.log(jnp.abs(ans))
    log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
    assert sign_c == ans_sign
    assert jnp.isclose(log_abs_c, log_abs_ans)


def is_complex(a):
    return a.dtype in [jnp.complex64, jnp.complex128]


def test_is_complex():
    assert is_complex(jnp.ones(1, dtype=jnp.complex_))


def cast_complex(a):
    return jnp.asarray(a, dtype=jnp.complex_)


def cumulative_logsumexp(u, reverse=False, unroll=2):
    def body(accumulant, u):
        new_accumulant = jnp.logaddexp(accumulant, u)
        return new_accumulant, new_accumulant

    _, v = scan(body,
                -jnp.inf * jnp.ones(u.shape[1:], dtype=u.dtype),
                u, reverse=reverse, unroll=unroll)
    return v


def test_cumulative_logsumexp():
    a = jnp.linspace(-1., 1., 100)
    v1 = jnp.log(jnp.cumsum(jnp.exp(a)))
    v2 = cumulative_logsumexp(a)
    print(v1)
    print(v2)
    assert jnp.isclose(v1, v2).all()


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
    p_cuml = jnp.exp(log_p_cuml)
    r = p_cuml[-1] * (1 - random.uniform(key, (S,)))
    idx = jnp.searchsorted(p_cuml, r)
    return dict_multimap(lambda s: s[idx, ...], samples)


def test_resample():
    x = random.normal(key=random.PRNGKey(0), shape=(50,))
    logits = -jnp.ones(50)
    samples = {'x': x}
    assert jnp.all(resample(random.PRNGKey(0), samples, logits)['x'] == resample(random.PRNGKey(0), x, logits))


def beta1_product_cdf(u, alpha, n):
    """
    u^alpha / (num_options-1)! Sum_m=0^(num_options-1) ( (-1)^m (num_options-1)!/m! alpha^m log(u)^m )
    u^alpha Sum_m=0^(num_options-1) ( (-alpha log(u))^m  / m! )
    Args:
        u:
        alpha:
        n:

    Returns:

    """

    def body(f, m):
        f = f + (-1.) ** m * jnp.exp(-gammaln(m + 1.)) * alpha ** m * jnp.log(u) ** m
        # f = jnp.logaddexp(f, m * (jnp.log(-jnp.log(u))  + jnp.log(alpha)) - gammaln(m+1.))
        return f, m

    f, _ = scan(body, 0., jnp.arange(0, n), unroll=2)
    # f, _ = scan(body, -jnp.inf, jnp.arange(0, num_options), unroll=2)
    return f * u ** alpha
    # return jnp.exp(jnp.logaddexp(alpha * jnp.log(u), f))


def log_beta1_product_cdf(u, alpha, n):
    """
    e^(alpha * u) / (num_options-1)! Sum_m=0^(num_options-1) ( (-1)^m (num_options-1)!/m! alpha^m u^m )
    e^(alpha * u) Sum_m=0^(num_options-1) ( (-alpha u)^m  / m! )
    Args:
        u:
        alpha:
        n:

    Returns:

    """

    def body(f, m):
        f = jnp.logaddexp(f, m * jnp.log(-alpha * u) - gammaln(m + 1.) + alpha * u)
        # f = jnp.logaddexp(f, m * (jnp.log(-jnp.log(u))  + jnp.log(alpha)) - gammaln(m+1.))
        return f, m

    f, _ = scan(body, jnp.log(0.), jnp.arange(0, n), unroll=2)
    # f, _ = scan(body, -jnp.inf, jnp.arange(0, num_options), unroll=2)
    return jnp.exp(f)
    # return jnp.exp(jnp.logaddexp(alpha * jnp.log(u), f))


def test_beta1_product_cdf():
    from jax import vmap, grad
    import pylab as plt
    alpha = 10
    n = 1000
    u = jnp.exp(jnp.linspace(-20, 0., 1000))
    cdf = vmap(lambda u: beta1_product_cdf(u, alpha, n))(u)
    plt.plot(u, cdf)
    plt.show()
    pdf = vmap(grad(lambda u: beta1_product_cdf(u, alpha, n)))(u)

    plt.plot(u, pdf)
    plt.show()


def test_log_beta1_product_cdf():
    from jax import vmap, grad
    import pylab as plt
    alpha = 1
    n = 10
    u = jnp.linspace(-20, 0., 1000)
    cdf = vmap(lambda u: log_beta1_product_cdf(u, alpha, n))(u)
    plt.plot(u, cdf)
    plt.show()
    pdf = vmap(grad(lambda u: log_beta1_product_cdf(u, alpha, n)))(u)

    plt.plot(u, pdf)
    plt.show()


def normal_to_lognormal(f, f2):
    """
    Convert normal parameters to log-normal parameters.
    Args:
        f:
        var:

    Returns:

    """
    ln_mu = 2. * jnp.log(f) - 0.5 * jnp.log(f2)
    ln_var = jnp.log(f2) - 2. * jnp.log(f)
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
    marginalised = jnp.mean(vmap(lambda d: fun(**d))(samples), axis=0)
    return marginalised


def marginalise(key, samples, log_weights, ESS, fun):
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
        _samples = tree_map(lambda v: v[0], resample(resample_key, samples, log_weights, S=1))
        marginalised += fun(**_samples)
        return (key, i + 1., marginalised)

    test_output = fun(**tree_map(lambda v: v[0], samples))
    (_, count, marginalised) = while_loop(lambda state: state[1] < ESS,
                                          body,
                                          (key, jnp.array(0.), jnp.zeros_like(test_output)))
    marginalised = marginalised / count
    return marginalised


def chunked_pmap(f, *args, chunksize=None, use_vmap=False, per_device_unroll=False):
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
        raise ValueError("chunksize should be <= {}".format(local_device_count()))
    N = args[0].shape[0]
    remainder = N % chunksize
    if (remainder != 0) and (N > chunksize):
        # only pad if not a zero remainder
        extra = chunksize - remainder
        args = tree_map(lambda arg: jnp.concatenate([arg, arg[:extra]], axis=0), args)
        N = args[0].shape[0]
    args = tree_map(lambda arg: jnp.reshape(arg, (chunksize, N // chunksize) + arg.shape[1:]), args)
    T = N // chunksize
    logger.info(f"Distributing {N} over {chunksize} devices in queues of length {T}.")

    # @jit
    def pmap_body(*args):
        def body(state, args):
            return state, f(*args)

        _, result = scan(body, (), args, unroll=1)
        return result

    if use_vmap:
        result = vmap(pmap_body)(*args)
    elif per_device_unroll:
        devices = get_devices()
        if len(devices) < chunksize:
            raise ValueError("Not enough devices {} for chunksize {}".format(len(devices), chunksize))
        result = []
        for i in range(chunksize):
            dev = devices[i]
            _func = jit(pmap_body, device=dev)
            _args = tree_map(lambda x: x[i], args)
            result.append(_func(*_args))
        result = [device_get(r) for r in result]
        result = tree_multimap(lambda *x: jnp.stack(x, axis=0), *result)
    else:
        result = pmap(pmap_body)(*args)

    result = tree_map(lambda arg: jnp.reshape(arg, (-1,) + arg.shape[2:]), result)
    if remainder != 0:
        # only slice if not a zero remainder
        result = tree_map(lambda x: x[:-extra], result)

    return result


def summary(results):
    print("ESS={}".format(results.ESS))

    max_like_idx = jnp.argmax(results.log_L_samples[:results.num_samples])
    max_like_points = tree_map(lambda x: x[max_like_idx], results.samples)
    tree_map(lambda x: print(x[max_like_idx-10:max_like_idx+10]), results.samples)
    print(results.log_L_samples[max_like_idx-10:max_like_idx+10])
    samples = resample(random.PRNGKey(23426), results.samples, results.log_p, S=int(results.ESS))

    for name in samples.keys():
        _samples = samples[name].reshape((samples[name].shape[0], -1))
        _max_like_points = max_like_points[name].reshape((-1,))
        ndims = _samples.shape[1]
        print("--------")
        print("{}: mean +- std.dev. | 10%ile / 50%ile / 90%ile | max(L) est.".format(
            name if ndims == 1 else "{}[#]".format(name), ))
        for dim in range(ndims):
            _uncert = jnp.std(_samples[:, dim])
            _max_like_point = _max_like_points[dim]
            sig_figs = -int("{:e}".format(_uncert).split('e')[1])

            def _round(ar):
                return round(float(ar), sig_figs)

            _uncert = _round(_uncert)
            print("{}: {} +- {} | {} / {} / {} | {}".format(
                name if ndims == 1 else "{}[{}]".format(name, dim),
                _round(jnp.mean(_samples[:, dim])), _uncert,
                *[_round(a) for a in jnp.percentile(_samples[:, dim], [10, 50, 90])],
                _round(_max_like_point)
            ))
    print("--------")
   