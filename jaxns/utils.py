import jax.numpy as jnp
from jax import random, value_and_grad, vmap
from jax.lax import scan, while_loop
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

def quadratic_contour(key, points_U, log_likelihood, prior_transform):
    M = points_U.shape[1]
    N = points_U.shape[0]
    if N< M+1:
        raise ValueError("Need {} points_U, got {}".format(M+1, N))
    key, select_key = random.split(key, 2)
    spawn_point = points_U[random.randint(select_key, shape=(), minval=0, maxval=N), :]
    dist = jnp.linalg.norm(points_U - spawn_point, axis=1)
    closest = jnp.argsort(dist, axis=0)[1:1+M]
    simp = points_U[closest, :]
    v_grad = value_and_grad(lambda U: log_likelihood(**prior_transform(U)))
    v0, g0 = v_grad(spawn_point)#(), [M]
    v, g = vmap(v_grad)(simp)#(N), [N,M]
    dg = g - g0#[N,M]
    dx = simp - spawn_point#[N, M]
    C = 0.5*jnp.linalg.solve(dx.T, dg.T)
    print(C)


def test_quadratic_contour():
    from jax.lax_linalg import triangular_solve
    from jax import vmap
    from born_rime.nested_sampling.prior_transforms import MVNDiagPrior

    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = triangular_solve(L, dx, lower=True, transpose_a=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    def log_likelihood(x):
        return log_normal(x, jnp.zeros_like(x), jnp.eye(x.size))

    prior_transform = MVNDiagPrior(jnp.zeros(2), jnp.ones(2))

    U = random.uniform(random.PRNGKey(0), shape=(1000,2))
    log_L = vmap(lambda U: log_likelihood(**prior_transform(U)))(U)

    import pylab as plt
    plt.scatter(U[:,0], U[:,1], c= log_L)
    plt.show()

    quadratic_contour(random.PRNGKey(0), U, log_likelihood, prior_transform)


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


def recluster(state, max_K: int):
    """
    Cluster the live points_U into at most max_K clusters.

    Args:
        state:

    Returns:

    """
    # aug_points = jnp.concatenate([state.live_points_U, state.log_L_live[:, None]], axis=1)
    key, cluster_centers, K, _ = cluster(state.key,
                                         state.live_points,
                                         max_K=max_K)
    # cluster_centers = cluster_centers[:, :-1]

    # initialise clusters
    # K, N
    # dist = jnp.linalg.norm(state.live_points_U[None, :, :] - cluster_centers[:K, None, :], axis=-1)
    # # N
    # cluster_id = jnp.argmin(dist, axis=0)
    cluster_id = masked_cluster_id(state.live_points, cluster_centers, K)
    # max_K (only first K are non-empty)
    num_per_cluster = jnp.bincount(cluster_id, minlength=max_K, length=max_K)

    # initialise cluster evidence and volume
    cluster_evidence = ClusterEvidence(global_evidence=Evidence(state=state.evidence_state),
                                       num_parent=state.live_points.shape[0],
                                       num_per_cluster=num_per_cluster)
    state = state._replace(key=key,
                           num_clusters=K,
                           cluster_centers=cluster_centers,
                           cluster_id=cluster_id,
                           num_per_cluster=num_per_cluster,
                           cluster_evidence_state=cluster_evidence.state
                           )
    return state

def minimum_volume_enclosing_ellipsoid(points, tol, init_u=None, return_u=False):
    """
    Performs the algorithm of
    MINIMUM VOLUME ENCLOSING ELLIPSOIDS
    NIMA MOSHTAGH
    psuedo-code here:
    https://stackoverflow.com/questions/1768197/bounding-ellipse

    Args:
        points: [N, D]
    """
    N, D = points.shape
    Q = jnp.concatenate([points, jnp.ones([N, 1])], axis=1)#N,D+1
    def body(state):
        (count, err, u) = state
        V = Q.T @ jnp.diag(u) @ Q
        # g[i] = Q[i,j].V^-1_jk.Q[i,k]
        g = vmap(lambda q: q @ jnp.linalg.solve(V,q))(Q)
        #jnp.diag(Q @ jnp.linalg.solve(V, Q.T))
        j = jnp.argmax(g)
        g_max = g[j]
        step_size = \
            (g_max - D  - 1)/((D+1)*(g_max - 1))
        search_direction = jnp.where(jnp.arange(N)==j, 1.- u, -u)
        new_u = u + step_size * search_direction
        # new_u = (1. - step_size)*u
        # new_u = jnp.where(jnp.arange(N) == j, new_u + step_size, new_u)
        new_err = jnp.linalg.norm(u - new_u)
        return (count+1, new_err, new_u)
    if init_u is None:
        init_u = jnp.ones(N)/N
    (count, err, u) = while_loop(lambda state: state[1] > tol,
                                 body,
                                 (0, 1., init_u))
    U = jnp.diag(u)
    PU = (points.T @ u) # D, N
    A = jnp.reciprocal(D)* jnp.linalg.pinv(points.T @ U @ points - PU[:,None] @ PU[None,:] )
    c = points.T @ u
    W, Q, Vh = jnp.linalg.svd(A)
    radii = jnp.reciprocal(jnp.sqrt(Q))
    rotation = Vh.conj().T
    if return_u:
        return c, radii, rotation, u
    return c, radii, rotation

def sample_ellipsoid(key, center, radii, rotation):
    direction_key, radii_key = random.split(key, 2)
    direction = random.normal(direction_key, shape=radii.shape)
    log_norm = jnp.log(jnp.linalg.norm(direction))
    log_radius = jnp.log(random.uniform(radii_key))/radii.size
    # x = direction * (radius/norm)
    x = direction * jnp.exp(log_radius - log_norm)
    y = center + rotation @ (radii * x)
    return y

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
