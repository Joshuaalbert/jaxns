from jax import random, vmap, numpy as jnp
from jax.lax import scan, while_loop
from jax.scipy.special import logsumexp
from scipy.stats.kde import gaussian_kde


def safe_gaussian_kde(samples, weights):
    try:
        return gaussian_kde(samples, weights=weights, bw_method='silverman')
    except:
        hist, bin_edges = jnp.histogram(samples,weights=weights, bins='auto')
        return lambda x: hist[jnp.searchsorted(bin_edges, x)]

def random_ortho_matrix(key, n):
    """
    Samples a random orthonormal num_parent,num_parent matrix from Stiefels manifold.

    From https://stackoverflow.com/a/38430739

    """
    H = random.normal(key, shape=(n, n))
    Q, R = jnp.linalg.qr(H)
    Q = Q @ jnp.diag(jnp.sign(jnp.diag(R)))
    return Q

def test_random_ortho_normal_matrix():
    H = random_ortho_matrix(random.PRNGKey(0), 3)
    assert jnp.all(jnp.isclose(H @ H.conj().T, jnp.eye(3), atol=1e-7))


def dict_multimap(f, d, *args):
    if not isinstance(d, dict):
        return f(d,*args)
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
        t.append({key:d[key][i] for key in keys})
    return d[keys[0]].__class__(*t)

def test_swap():
    from collections import namedtuple

    Test = namedtuple('Test',['a','b', 'c'])
    test_tuple = Test(dict(da=1., db=2.), dict(da=3., db=4.), dict(da=5., db=6.))
    test_dict = {'da': Test(a=1.0, b=3.0, c=5.0), 'db': Test(a=2.0, b=4.0, c=6.0)}
    _swap_dict = swap_dict_namedtuple(test_tuple)
    assert _swap_dict == test_dict
    _swap_tuple = swap_namedtuple_dict(test_dict)
    assert _swap_tuple == test_tuple

    test_tuple = Test(1.,2.,3.)
    test_dict = Test(1.,2.,3.)
    _swap_dict = swap_dict_namedtuple(test_tuple)
    assert _swap_dict == test_dict
    _swap_tuple = swap_namedtuple_dict(test_dict)
    assert _swap_tuple == test_tuple


def get_interval(s):
    ar_first = jnp.argmin(s)
    s = jnp.concatenate([jnp.where(jnp.arange(s.size) < ar_first, -1., s), jnp.array([1.])])
    ar_last = jnp.maximum(0,jnp.argmax(s) - 1)
    return ar_first, ar_last

def test_get_interval():
    s = jnp.array([1, -1,-1,-1, 1])
    assert get_interval(s) == (1,3)
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
    s = jnp.array([1,-1])
    assert get_interval(s) == (1,1)

def masked_cluster_id(points, centers, K):
    max_K = centers.shape[0]
    # max_K, N
    dist = jnp.linalg.norm(centers[:, None, :] - points[None, :, :], axis=-1)
    dist = jnp.where(jnp.arange(max_K)[:, None] > K - 1, jnp.full_like(dist, jnp.inf), dist)
    return jnp.argmin(dist, axis=0)

def cluster(key, points, max_K=6):
    """
    Cluster `points_U` automatically choosing K.

    Args:
        key:
        points: [N,M]
        niters:
        max_K:

    Returns: tuple of
        key: PRNG key
        cluster_centers: [max_K, M] (last max_K - K rows are zeros)
        K: int, the number of clusters found.

    """

    # points_U = points_U - jnp.mean(points_U, axis=0)
    # points_U = points_U / jnp.maximum(jnp.std(points_U, axis=0), 1e-8)

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
        # centers = random.shuffle(shuffle_key, points_U, axis=0)
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
    if isinstance(shape1, int):
        shape1 = (shape1,)
    if isinstance(shape2, int):
        shape2 = (shape2,)
    def left_pad_shape(shape,l):
        return tuple([1]*l + list(shape))
    l = max(len(shape1), len(shape2))
    shape1 = left_pad_shape(shape1,l- len(shape1))
    shape2 = left_pad_shape(shape2,l - len(shape2))
    out_shape = []
    for s1, s2 in zip(shape1, shape2):
        m = max(s1,s2)
        if ((s1 != m) and (s1 != 1)) or ((s2 != m) and (s2 != 1)):
            raise ValueError("Trying to broadcast {} with {}".format(shape1,shape2))
        out_shape.append(m)
    return tuple(out_shape)

def test_broadcast_shapes():

    assert broadcast_shapes(1,1) == (1,)
    assert broadcast_shapes(1,2) == (2,)
    assert broadcast_shapes(1,(2,2)) == (2,2)
    assert broadcast_shapes((2,2),1) == (2,2)
    assert broadcast_shapes((1,1),(2,2)) == (2,2)
    assert broadcast_shapes((1,2),(2,2)) == (2,2)
    assert broadcast_shapes((2,1),(2,2)) == (2,2)
    assert broadcast_shapes((2,1),(2,1)) == (2,1)
    assert broadcast_shapes((2,1),(1,1)) == (2,1)
    assert broadcast_shapes((1,1),(1,1)) == (1,1)
    assert broadcast_shapes((1,),(1,1)) == (1,1)
    assert broadcast_shapes((1,1,2),(1,1)) == (1,1,2)
    assert broadcast_shapes((1,2,1),(1,3)) == (1,2,3)
    assert broadcast_shapes((1,2,1),()) == (1,2,1)


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
    dsk = {'a':[],
           'b':['a'],
           'c':['a','b']}
    assert iterative_topological_sort(dsk,['a','b','c']) == ['c','b','a']
    assert iterative_topological_sort(dsk) == ['c','b','a']
    dsk = {'a': [],
           'b': ['a', 'd'],
           'c': ['a', 'b']}
    # print(iterative_topological_sort(dsk, ['a', 'b', 'c']))
    assert iterative_topological_sort(dsk, ['a', 'b', 'c']) == ['c', 'b', 'a', 'd']
    assert iterative_topological_sort(dsk) == ['c', 'b', 'a', 'd']


def stochastic_result_computation(n_per_sample, key, samples, log_L_samples):
    """

    Args:
        n_per_sample:
        key:
        samples:
        log_L_samples:

    Returns:

    """
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
    weights = jnp.exp(log_w)
    m = dict_multimap(lambda samples: jnp.sum(left_broadcast_mul(weights, samples), axis=0), samples)
    dsamples = dict_multimap(jnp.subtract, samples, m)
    cov = dict_multimap(lambda dsamples: jnp.sum(
        left_broadcast_mul(weights, (dsamples[..., :, None] * dsamples[..., None, :])), axis=0), dsamples)
    # Kish's ESS = [sum weights]^2 / [sum weights^2]
    ESS = jnp.exp(2. * logsumexp(log_w) - logsumexp(2. * log_w))
    # H = sum w_i log(L)
    _H = jnp.exp(log_w) * log_avg_L
    H = jnp.sum(jnp.where(jnp.isnan(_H), 0., _H))
    return logZ, m, cov, ESS, H


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
    res = t[0]
    for a in t[1:]:
        res *= a
    return res


def msqrt(A):
    U, s, Vh = jnp.linalg.svd(A)
    L = U * jnp.sqrt(s)
    return L


def test_msqrt():
    for i in range(10):
        A = random.normal(random.PRNGKey(i),shape=(30,30))
        A = A @ A.T
        L = msqrt(A)
        assert jnp.all(jnp.isclose(A, L @ L.T))


def logaddexp(x1, x2):
    if is_complex(x1) or is_complex(x2):
        select1 = x1.real > x2.real
        amax = jnp.where(select1, x1, x2)
        delta = jnp.where(select1, x2-x1, x1-x2)
        return jnp.where(jnp.isnan(delta),
                          x1+x2,  # NaNs or infinities of the same sign.
                          amax + jnp.log1p(jnp.exp(delta)))
    else:
        return jnp.logaddexp(x1, x2)

def test_logaddexp():
    a = jnp.log(1.)
    b = jnp.log(1.)
    assert logaddexp(a,b) == jnp.log(2.)
    a = jnp.log(1.)
    b = jnp.log(-2.+0j)
    assert jnp.isclose(jnp.exp(logaddexp(a, b)).real, -1.)

    a = jnp.log(-1.+0j)
    b = jnp.log(2. + 0j)
    assert jnp.isclose(jnp.exp(logaddexp(a, b)).real, 1.)

    for i in range(100):
        u = random.uniform(random.PRNGKey(i),shape=(2,))*20. - 10.
        a = jnp.log(u[0] + 0j)
        b = jnp.log(u[1] + 0j)
        assert jnp.isclose(jnp.exp(logaddexp(a,b)).real, u[0] + u[1])

def signed_logaddexp(log_abs_val1, sign1, log_abs_val2, sign2):
    amax = jnp.maximum(log_abs_val1, log_abs_val2)
    signmax = jnp.where(log_abs_val1 > log_abs_val2, sign1, sign2)
    delta = -jnp.abs(log_abs_val2 - log_abs_val1)
    sign = sign1*sign2
    return jnp.where(jnp.isnan(delta),
                      log_abs_val1 + log_abs_val2,  # NaNs or infinities of the same sign.
                      amax + jnp.log1p(sign * jnp.exp(delta))), signmax

def test_signed_logaddexp():
    for i in range(100):
        u = random.uniform(random.PRNGKey(i),shape=(2,))*20.-10.
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

    u = [0.,0.]
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

def is_complex(a):
    return a.dtype in [jnp.complex64, jnp.complex128]

def test_is_complex():
    assert is_complex(jnp.ones(1, dtype=jnp.complex_))


def cast_complex(a):
    return jnp.asarray(a, dtype = jnp.complex_)