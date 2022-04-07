from collections import namedtuple
from jax import numpy as jnp, random, vmap
from jax.lax import scan
from jax.lax import dynamic_update_slice
from jax.scipy.special import gammaln, logsumexp
from jax.lax import while_loop

from jaxns.internals.linalg import rank_one_update_matrix_inv
from jaxns.internals.types import int_type

MultiEllipsoidSamplerState = namedtuple('MultiEllipsoidSamplerState',
                                        ['cluster_id', 'mu', 'radii', 'rotation', 'num_k', 'num_fev_ma'])


def init_multi_ellipsoid_sampler_state(key, live_points_U, depth, log_X):
    cluster_id, (mu, radii, rotation) = ellipsoid_clustering(key, live_points_U, depth, log_X)
    num_k = jnp.bincount(cluster_id, minlength=0, length=mu.shape[0])
    return MultiEllipsoidSamplerState(cluster_id=cluster_id, mu=mu, radii=radii, rotation=rotation,
                                      num_k=num_k, num_fev_ma=jnp.asarray(live_points_U.shape[1] + 2.))



def bounding_ellipsoid(points, mask):
    """
    Use empirical mean and covariance as approximation to bounding ellipse, then scale so that all points are inside.

    for all i (points[i] - f) @ inv(scale * cov) @ (points[i] - f) <= 1
    for all i (points[i] - f) @ inv(cov) @ (points[i] - f) <= scale
    -> choose scale = max_i (points[i] - f) @ inv(cov) @ (points[i] - f)
    Args:
        points:
        mask:

    Returns: f, inv(scale * cov)

    """
    mu = jnp.average(points, weights=mask, axis=0)
    dx = points - mu
    cov = jnp.average(dx[:, :, None] * dx[:, None, :], weights=mask, axis=0)
    C = jnp.linalg.pinv(cov)
    maha = vmap(lambda dx: dx @ C @ dx)(dx)
    scale = jnp.max(jnp.where(mask, maha, 0.))
    C = C / scale
    return mu, C


def ellipsoid_params(C):
    """
    If C satisfies the sectional inequality,

    (x - f)^T C (x - f) <= 1

    then this returns the radius and rotation matrix of the ellipsoid.

    Args:
        C: [D,D]

    Returns: radii [D] rotation [D,D]

    """
    W, Q, Vh = jnp.linalg.svd(C)
    radii = jnp.reciprocal(jnp.sqrt(Q))
    radii = jnp.where(jnp.isnan(radii), 0., radii)
    rotation = Vh.conj().T
    return radii, rotation


def kmeans(key, points, mask, K=2):
    """
    Perform kmeans clustering with Euclidean metric.

    Args:
        key:
        points: [N, D]
        mask: [N] bool
        K: int

    Returns: cluster_id [N], centers [K, D]

    """
    N, D = points.shape

    def body(state):
        (i, done, old_cluster_id, centers) = state
        new_centers = vmap(lambda k: jnp.average(points, weights=(old_cluster_id == k) & mask, axis=0))(jnp.arange(K))
        dx = points - new_centers[:, None, :]  # K, N, D
        squared_norm = jnp.sum(jnp.square(dx), axis=-1)  # K, N
        new_cluster_id = jnp.argmin(squared_norm, axis=0)  # N
        done = jnp.all(new_cluster_id == old_cluster_id)
        # print("kmeans reassigns", jnp.sum(old_cluster_id!=new_cluster_k))
        return i + 1, done, new_cluster_id, new_centers

    do_kmeans = jnp.sum(mask) > K
    i, _, cluster_id, centers = while_loop(lambda state: ~state[1],
                                           body,
                                           (jnp.array(0), ~do_kmeans,
                                            random.randint(key, shape=(N,), minval=jnp.asarray(0),
                                                           maxval=jnp.asarray(2)),
                                            jnp.zeros((K, D))))
    return cluster_id, centers


def generic_kmeans(key, points, mask, K=2, meta=None, method='euclidean'):
    """
    Perform kmeans clustering with Euclidean metric.

    Args:
        key:
        points: [N, D]
        mask: [N] bool
        K: int

    Returns: cluster_id [N], centers [K, D]

    """
    N, D = points.shape
    if meta is None:
        meta = dict()
    if method == 'ellipsoid':
        log_VS = meta.get('log_VS')
    MetricState = namedtuple("MetricState", ['cluster_centers', 'num_k', 'C', 'radii'])
    weights = mask.astype(int_type)
    num_S = jnp.sum(mask)

    def cluster_dist_metric(point, metric_state: MetricState):
        if method == 'euclidean':
            dist = metric_state.cluster_centers - point
            return jnp.sum(jnp.square(dist), axis=-1)
        if method == 'mahalanobis':
            dist = metric_state.cluster_centers - point
            maha = vmap(lambda dist, C: dist @ C @ dist)(dist, metric_state.C)
            return maha
        if method == 'ellipsoid':
            dist = metric_state.cluster_centers - point
            weighted_maha = vmap(lambda dist, C, radii, num_k: (dist @ C @ dist)
                                                               * jnp.exp(log_ellipsoid_volume(radii) - jnp.log(num_k)
                                                                         + jnp.log(num_S) - log_VS))(dist,
                                                                                                     metric_state.C,
                                                                                                     metric_state.radii,
                                                                                                     metric_state.num_k)

            return weighted_maha

    def update_metric_state(cluster_id):
        num_k = jnp.bincount(cluster_id, weights, minlength=0, length=K)
        if method == 'euclidean':
            def get_mu(k):
                weights = (cluster_id == k) & mask
                mu = jnp.average(points, weights=weights, axis=0)
                mu = jnp.where(num_k[k] == 0, 0., mu)
                return mu

            mu = vmap(get_mu)(jnp.arange(K))
            return MetricState(cluster_centers=mu, num_k=num_k, C=None, radii=None)
        if method == 'mahalanobis':
            def get_mu_and_C(k):
                weights = (cluster_id == k) & mask
                mu = jnp.average(points, weights=weights, axis=0)
                dist = points - mu
                Cov = jnp.average(dist[:, :, None] * dist[:, None, :], weights=weights, axis=0)
                C = jnp.linalg.pinv(Cov)
                mu = jnp.where(num_k[k] == 0, 0., mu)
                C = jnp.where(num_k[k] < D + 1, 0., C)
                return mu, C

            mu, C = vmap(get_mu_and_C)(jnp.arange(K))
            return MetricState(cluster_centers=mu, num_k=num_k, C=C, radii=None)
        if method == 'ellipsoid':
            def get_mu_and_C_radii(k):
                weights = (cluster_id == k) & mask
                mu = jnp.average(points, weights=weights, axis=0)
                dist = points - mu
                Cov = jnp.average(dist[:, :, None] * dist[:, None, :], weights=weights, axis=0)
                C = jnp.linalg.pinv(Cov)
                mu = jnp.where(num_k[k] == 0, 0., mu)
                C = jnp.where(num_k[k] < D + 1, 0., C)
                radii, rotation = ellipsoid_params(C)
                return mu, C, radii

            mu, C, radii = vmap(get_mu_and_C_radii)(jnp.arange(K))
            return MetricState(cluster_centers=mu, num_k=num_k, C=C, radii=radii)

    State = namedtuple("State", ['i', 'done', 'cluster_id', 'metric_state'])

    def body(state: State):
        # N, K
        cluster_dist = vmap(lambda point: cluster_dist_metric(point, state.metric_state))(points)
        # N
        current_cluster_dist = vmap(lambda n, k: cluster_dist[n, k])(jnp.arange(N, dtype=int_type), state.cluster_id)
        # N, K
        rel_dist = cluster_dist - current_cluster_dist[:, None]
        # N
        min_dist = jnp.min(rel_dist, axis=-1)
        proposed_cluster_id = jnp.argmin(rel_dist, axis=-1)
        can_take_from = state.metric_state.num_k[state.cluster_id] > D + 1
        min_dist = jnp.where(mask & can_take_from, min_dist, jnp.inf)
        amin = jnp.argmin(min_dist)
        k_to = proposed_cluster_id[amin]
        cluster_id = dynamic_update_slice(state.cluster_id, k_to[None], amin[None])
        # # update cluster_id
        # cluster_id = jnp.where(state.metric_state.num_k[state.cluster_id] < D+1, state.cluster_id, jnp.argmin(rel_dist, axis=-1))
        # proposed_num_k = jnp.bincount(proposed_cluster_id, weights, minlength=0, length=K)
        # cluster_id = jnp.where(proposed_num_k[proposed_cluster_id] < D + 1, state.cluster_id, proposed_cluster_id)
        metric_state = update_metric_state(cluster_id)
        # print()
        # print(state.i, jnp.sum(state.cluster_id!=cluster_id), amin, state.cluster_id[amin], k_to, jnp.min(rel_dist))
        done = jnp.all(cluster_id == state.cluster_id)
        state = state._replace(i=state.i + 1,
                               done=done,
                               cluster_id=cluster_id,
                               metric_state=metric_state)
        return state

    do_kmeans = jnp.sum(mask) > K * (D + 1)
    cluster_id = random.randint(key, shape=(N,), minval=0, maxval=K)
    init_state = State(i=jnp.asarray(0),
                       done=~do_kmeans,
                       cluster_id=cluster_id,
                       metric_state=update_metric_state(cluster_id))
    state = while_loop(lambda state: ~state.done,
                       body,
                       init_state)
    return state


def decide_split(points, mask, cluster_id, log_VS, method='volume'):
    """
    V(S1+S2) = V(S1)+V(S2)-V(S1.S2)
    If V(S1.S2) is very small, then we should split.
    If V(S1.S2)=V(S1)+V(S2)-V(S1+S2) < alpha * V(S1+S2) then split
    If V(S1)+V(S2) < (1+alpha) * V(S1+S2) then split

    Args:
        points:
        mask:
        cluster_id:

    Returns:

    """
    n = jnp.sum(mask)
    if method == 'volume':
        def get_log_VE(mask):
            mu, C = bounding_ellipsoid(points, mask)
            radii, rotation = ellipsoid_params(C)
            log_VE = log_ellipsoid_volume(radii)
            return log_VE

        log_VE = get_log_VE(mask)
        log_VE1 = get_log_VE(mask & (cluster_id == 0))
        log_VE2 = get_log_VE(mask & (cluster_id == 1))
        # print(jnp.exp(log_VE) , jnp.exp(log_VS + jnp.log(2.)))
        return (jnp.logaddexp(log_VE1, log_VE2) < log_VE) | (log_VE > log_VS + jnp.log(2.))

    elif method == 'AIC':
        def get_log_L(weights):
            mu = jnp.average(points, weights=weights, axis=0)
            dx = points - mu
            Cov = jnp.average(dx[:, :, None] * dx[:, None, :], weights=weights, axis=0)
            logdetCov = jnp.log(jnp.linalg.det(Cov))
            C = jnp.linalg.pinv(Cov)
            # logdetCov = 0.
            # C = jnp.eye(f.size)
            maha = vmap(lambda dx: dx @ C @ dx)(dx)
            n_i = jnp.sum(weights)
            log_L_1 = -0.5 * jnp.sum(jnp.where(weights, maha, 0.)) \
                      - 0.5 * n_i * mu.size * jnp.log(2. * jnp.pi) - 0.5 * n_i * logdetCov \
                      + jnp.log(n_i) - jnp.log(n)
            # log_L_1 = -0.5 * jnp.sum(jnp.where(weights, maha, 0.)) \
            #           - 0.5 * n_i * f.size * jnp.log(2. * jnp.pi) - 0.5 * n_i * logdetCov
            return jnp.where(jnp.isnan(log_L_1), -jnp.inf, log_L_1)

        def AICc(k, log_L):
            # return (2. * k ** 2 + 2. * k) / (num_options - k - 1.) + 2. * k - 2. * log_L_samples
            return 2. * k - 2. * log_L

        # k = {mu_all, Cov_all}
        k_all = points.shape[1]  # + points.shape[1] * (points.shape[1]-1)
        log_L_all = get_log_L(mask)
        AICc_all = AICc(k_all, log_L_all)
        # k = {mu_0, mu_1, Cov0, Cov1}
        k_split = 2. * k_all

        log_L_k = jnp.logaddexp(get_log_L((cluster_id == 0) & mask),
                                get_log_L((cluster_id == 1) & mask))
        AICc_split = AICc(k_split, log_L_k)
        print(AICc_split, AICc_all)
        diff = AICc_split - AICc_all
        return (diff < 0.) & (~jnp.isnan(diff))


def hierarchical_clustering(key, points, depth, log_VS):
    """
    Partition live_points into 2^depth clusters in depth-first.
    0:    0 | # 2^0
    1:    1 2 | # 2^1
    2:    3 4 5 6
    3:    7 8 9 10 11 12 13 14

    index(r,c) = index(r,0) + c
    index(r,0) = sum_p=0^(r-1) 2^p = 2^r - 1
    index(r,c) = 2^r - 1 + c

    index(r,c) = index(r-1, 2^(r-1) - 1) + c
    child0(r,c) = index(r+1, 2*c) = index(r+1, 0) + 2*c = index(r,0) + 2^r + 2*c = index(r,c) + 2^r + c
    child1(r,c) = index(r+1, 2*c+1) = index(r+1, 0) + 2*c + 1 = = index(r,c) + 2^r + c + 1
    parent(r,c) = index(r-1, c//2)

    To walk this and modify memory in place we follow the rule:

    Find lowest index, replace with left node, append right node.
    0
    1 2
    3 2 4
    3 5 4 6
    7 5 4 6 8
    7 5 9 6 8 10
    7 11 9 6 8 10 12
    7 11 9 13 8 10 12 14 <- final splitting
    0 4 2 6 1 3 5 7 subtract offset
    rearange with argsort


    In general we will do sum_p=0^(depth-1) 2^p = 2^depth - 1 splittings

    for depth = 4 -> 2^4 - 1 = 15
    for depth = 3 -> 2^2 - 1 = 7


    Args:
        key:
        points: [N, D]
        depth:
        log_VS: expected true volume of points

    Returns: cluster_id, ellipsoid_parameters

    """
    N, D = points.shape

    num_clusters = 2 ** (depth - 1)
    cluster_id = jnp.zeros(N, dtype=int_type)
    num_splittings = 2 ** (depth - 1) - 1
    keys = random.split(key, num_splittings)
    order = jnp.zeros(num_clusters, dtype=int_type)
    log_VS_subclusters = jnp.array([log_VS] + [0] * num_splittings)

    def body(state, X):
        (cluster_id, order, log_VS_subclusters) = state
        (key, splitting) = X
        splitting_select = jnp.arange(num_clusters) <= splitting
        child0 = jnp.max(jnp.where(splitting_select, order, -jnp.inf)) + 1
        child0 = child0.astype(int_type)
        child1 = child0 + 1
        i_lowest = jnp.argmin(jnp.where(splitting_select, order, jnp.inf))

        def _replace_result(operand, update1, update2):
            operand = dynamic_update_slice(operand, update1, jnp.asarray([i_lowest] + [0] * (len(operand.shape) - 1)))
            operand = dynamic_update_slice(operand, update2,
                                           jnp.asarray([splitting + 1] + [0] * (len(operand.shape) - 1)))
            return operand

        mask = cluster_id == order[i_lowest]
        log_VS_subcluster = log_VS_subclusters[i_lowest]
        # log_VE_parent = log_ellipsoid_volume(radii_result[i_lowest, :])
        state = generic_kmeans(key, points, mask, K=2, meta=dict(log_VS=log_VS_subcluster), method='euclidean')
        unsorted_cluster_id = state.cluster_id
        log_VS1 = jnp.log(state.metric_state.num_k[0]) - jnp.log(jnp.sum(state.metric_state.num_k)) + log_VS_subcluster
        log_VS2 = jnp.log(state.metric_state.num_k[1]) - jnp.log(jnp.sum(state.metric_state.num_k)) + log_VS_subcluster
        # decide if to do it.
        do_split = jnp.all(state.metric_state.num_k >= D + 1) & decide_split(points, mask, unsorted_cluster_id,
                                                                             log_VS_subcluster)
        unsorted_cluster_id = jnp.where(unsorted_cluster_id == 0, child0, child1)
        cluster_id = jnp.where(mask, unsorted_cluster_id, cluster_id)
        order = _replace_result(order, child0[None], child1[None])
        # if do_split then keep else
        # we replace child0 with parent and child1 gets zero-size ellipsoid that has no members.
        log_VS1 = jnp.where(do_split, log_VS1, log_VS_subcluster)
        log_VS2 = jnp.where(do_split, log_VS2, jnp.array(-jnp.inf))
        cluster_id = jnp.where((~do_split) & (cluster_id == child1), child0, cluster_id)
        # update tracked objects
        log_VS_subclusters = _replace_result(log_VS_subclusters, log_VS1[None], log_VS2[None])
        return (cluster_id, order, log_VS_subclusters), ()

    (cluster_id, order, log_VS_subclusters), _ = \
        scan(body,
             (cluster_id, order, log_VS_subclusters),
             (keys, jnp.arange(num_splittings)),
             unroll=1)
    cluster_id = cluster_id - (2 ** (depth - 1) - 1)
    order = order - (2 ** (depth - 1) - 1)
    # order results so that cluster_id corresponds to the correct row
    log_VS_subclusters = log_VS_subclusters[order]
    return cluster_id, log_VS_subclusters


def log_coverage_scale(log_VE, log_VS, D):
    """
    Computes the required scaling relation such that
    V(E) = max(V(E), V(S))
    where the scaling is applied to each radius.

    Args:
        log_VE:
        log_VS:
        D:

    Returns:

    """
    return jnp.maximum(0., (log_VS - log_VE) / D)


def cluster_split_matching_pursuit(key, points, mask, log_VS, log_VE, kmeans_init=True, K=2):
    """
    Splits a set of points into two ellipsoids such that the enclosed volume is as close to V(S) without being less.
    V(S) should be an estimate of the true volume contained by the points.

    Args:
        key:
        points: [N, D]
        mask: [N] only split on these points
        log_VS: logV(S) of the set of points
        log_VE: logV(E) of the parent ellipsoid
        kmeans_init: whether to use kmeans to initialise the clustering

    Returns:
        cluster_id: ids of the points, places where ~mask are random assignments
        mu1, radii1, rotation1: ellipsoid params of first subcluster
        mu2, radii2, rotation2: ellipsoid paras of second subcluster

    """
    N, D = points.shape
    num_S = jnp.sum(mask)
    print(mask)
    a_k = jnp.arange(K)

    def log_ellipsoid_volume(logdetC_k, num_k, log_f_k):
        """
        Computes area of u_k @ Lamba @ u_k <= 1
        using
            Lambda = f_k n_k C_k)
            and |Lambda| = f_k^d n_k^d |C_k|
        """
        logdetLambda = D * (log_f_k + jnp.log(num_k)) + logdetC_k
        return (jnp.log(2.) + 0.5 * D * jnp.log(jnp.pi)
                - jnp.log(D) - gammaln(0.5 * D)
                - 0.5 * logdetLambda)

    def log_factor_k(cluster_id, log_maha_k, num_k, logdetC_k):
        """
        Computes f_k such that,
            u_k @ f0_k n_k C_k @ u_k <= 1
        and
            f_k^d V(n_k C_k) = max(V(S_k), V(f0_k n_k C_k))
            log_f_k = (log max(V(S)*n_k/n_S, V(f0_k n_k C_k)) - log V(n_k C_k))/D
            log_f_k = (max(log(V(S)*n_k/n_S), logV(n_k C_k)) - log V(n_k C_k))/D
        """
        # K
        log_f_expand_k = -jnp.max(jnp.where(cluster_id == a_k[:, None], log_maha_k, -jnp.inf),
                                  axis=-1)
        log_VE_expand_k = log_ellipsoid_volume(logdetC_k, num_k, log_f_expand_k)
        log_VE_k = log_ellipsoid_volume(logdetC_k, num_k, 0.)

        log_scale_k = (jnp.maximum(log_VS + jnp.log(num_k) - jnp.log(num_S), log_VE_expand_k) - log_VE_k) / D
        # K
        return log_scale_k

    # # calculate bounding ellipsoid
    # f, C =// bounding_ellipsoid(points, mask)
    # radii, _ = ellipsoid_params(C)
    # log_VE = log_ellipsoid_volume(radii)
    # # enlarge so that V(E) = max(V(E), V(S))
    # # (const * r**D) >= V(S) -> scale = 1 else log_scale = (log_V(S) - log(const * r**D))/D
    # log_scale = jnp.maximum(0., (log_VS - log_VE) / D)
    # C = C / jnp.exp(log_scale)
    ###
    # input is essentially log_VS
    if kmeans_init:
        # do Euclidean kmean clustering
        cluster_id, centers = kmeans(key, points, mask, K=K)
    else:
        # assign to random clusters: child0 or child1
        cluster_id = random.randint(key, shape=(N,), minval=0, maxval=K)
    # K, N
    # log_maha_k is with f_k=1

    State = namedtuple('State', ['i', 'done', 'cluster_id', 'C_k', 'logdetC_k',
                                 'mu_k', 'log_maha_k', 'num_k',
                                 'log_VE_k', 'log_VS_k',
                                 'min_loss', 'delay'])

    def init_state(cluster_id):
        num_k = jnp.sum(mask & (cluster_id == a_k[:, None]), axis=-1)
        mu_k = vmap(lambda k: jnp.average(points, axis=0, weights=k == cluster_id))(a_k)
        C_k = vmap(lambda k, mu_k: jnp.linalg.pinv(
            jnp.average((points - mu_k)[:, :, None] * (points - mu_k)[:, None, :], axis=0,
                        weights=k == cluster_id)))(a_k, mu_k)
        logdetC_k = vmap(lambda C_k: jnp.sum(jnp.log(jnp.linalg.eigvals(C_k).real)))(C_k)
        precision_k = C_k * num_k[:, None, None]
        # K, N
        log_maha_k = vmap(lambda mu_k, precision_k:
                          jnp.log(vmap(lambda point:
                                       (point - mu_k) @ precision_k @ (point - mu_k))(points)))(mu_k, precision_k)
        log_f_k = log_factor_k(cluster_id, log_maha_k, num_k, logdetC_k)
        log_VE_k = vmap(log_ellipsoid_volume)(logdetC_k, num_k, log_f_k)

        log_VS_k = jnp.log(num_k) - jnp.log(num_S)
        return State(i=jnp.asarray(0),
                     done=num_S < K * (D + 1),
                     cluster_id=cluster_id,
                     C_k=C_k,
                     logdetC_k=logdetC_k,
                     mu_k=mu_k,
                     log_maha_k=log_maha_k,
                     num_k=num_k,
                     log_VE_k=log_VE_k,
                     log_VS_k=log_VS_k,
                     min_loss=jnp.asarray(jnp.inf),
                     delay=jnp.asarray(0)
                     )

    def body(state: State):
        new_state_date = dict()
        # upon the start of each iteration the state is consistent.
        # we use the consistent state to calculate the reassignment metrics.
        # we then reassign and update the state so that it is consistent again.
        # K, N
        # K
        log_f_k = log_factor_k(state.cluster_id, state.log_maha_k, state.num_k, state.logdetC_k)

        def single_log_h(log_f_k, log_maha_k, num_k, logdetC_k):
            log_d = log_maha_k + log_f_k
            log_VS_k = log_VS + jnp.log(num_k) - jnp.log(num_S)
            return log_ellipsoid_volume(logdetC_k, num_k, log_f_k) + log_d - log_VS_k

        # K, N
        log_h_k = vmap(single_log_h)(log_f_k, state.log_maha_k, state.num_k, state.logdetC_k)
        h_k = jnp.exp(log_h_k)
        # # K, K, N
        delta_F = h_k[:, None, :] - h_k
        # Can reassign if mask says we are working on that node and there would be at least D+1 points in that cluster_from_nn_dist
        # after taking from it. And, if delta_F < 0.
        able_to_reassign = mask & (state.num_k[state.cluster_id] > D + 1)
        delta_F_masked = jnp.where(able_to_reassign, delta_F, jnp.inf)

        # (k_to, k_from, n_reassign) = jnp.where(delta_F == min_delta_F)
        (k_to, k_from, n_reassign) = jnp.unravel_index(jnp.argmin(delta_F_masked.flatten()), delta_F.shape)
        # dynamic update index arrays of sufficient length for all
        dyn_k_to_idx = jnp.concatenate([k_to[None], jnp.asarray([0, 0])])
        dyn_k_from_idx = jnp.concatenate([k_from[None], jnp.asarray([0, 0])])

        ###
        # update the state

        ###
        # cluster_from_nn_dist id
        cluster_id = dynamic_update_slice(state.cluster_id, dyn_k_to_idx[0:1], n_reassign[None])

        ###
        # num_k
        num_from = state.num_k[k_from] - 1
        num_to = state.num_k[k_from] + 1
        num_k = dynamic_update_slice(state.num_k, num_from[None], dyn_k_from_idx[0:1])
        num_k = dynamic_update_slice(num_k, num_to[None], dyn_k_to_idx[0:1])

        ###
        # ellipsoid parameters
        x_n = points[n_reassign, :]
        mu_from = state.mu_k[k_from, :] + (state.mu_k[k_from, :] - x_n) / (state.num_k[k_from] - 1)
        C_from, logdetC_from = rank_one_update_matrix_inv(state.C_k[k_from, :, :], state.logdetC_k[k_from],
                                                          x_n - mu_from, x_n - state.mu_k[k_from, :], add=False)
        # print(C_from, logdetC_from)
        mu_to = state.mu_k[k_to, :] + (x_n - state.mu_k[k_to, :]) / (state.num_k[k_to] + 1)
        C_to, logdetC_to = rank_one_update_matrix_inv(state.C_k[k_to, :, :], state.logdetC_k[k_to],
                                                      x_n - mu_to, x_n - state.mu_k[k_to, :], add=True)
        print('from', state.logdetC_k[k_from])
        # print(C_to, logdetC_to)
        mu_k = dynamic_update_slice(state.mu_k, mu_from[None, :], dyn_k_from_idx[0:2])
        mu_k = dynamic_update_slice(mu_k, mu_to[None, :], dyn_k_to_idx[0:2])
        C_k = dynamic_update_slice(state.C_k, C_from[None, :, :], dyn_k_from_idx)
        C_k = dynamic_update_slice(C_k, C_to[None, :, :], dyn_k_to_idx)
        logdetC_k = dynamic_update_slice(state.logdetC_k, logdetC_from[None], dyn_k_from_idx[0:1])
        logdetC_k = dynamic_update_slice(logdetC_k, logdetC_to[None], dyn_k_to_idx[0:1])

        ###
        # maha

        precision_from = C_from * num_from
        precision_to = C_to * num_to
        log_maha_from = jnp.log(vmap(lambda point: (point - mu_from) @ precision_from @ (point - mu_from))(points))
        log_maha_to = jnp.log(vmap(lambda point: (point - mu_to) @ precision_to @ (point - mu_to))(points))
        log_maha_k = dynamic_update_slice(state.log_maha_k, log_maha_from[None, :], dyn_k_from_idx[0:2])
        log_maha_k = dynamic_update_slice(log_maha_k, log_maha_to[None, :], dyn_k_to_idx[0:2])

        # estimate volumes of current clustering
        log_f_k = log_factor_k(cluster_id, log_maha_k, num_k, logdetC_k)
        log_VE_k = vmap(log_ellipsoid_volume)(logdetC_k, num_k, log_f_k)
        log_VS_k = jnp.log(num_k) - jnp.log(num_S)
        log_V_sum = logsumexp(log_VE_k)
        new_loss = log_V_sum - log_VS
        loss_decreased = new_loss < state.min_loss
        delay = jnp.where(loss_decreased, 0, state.delay + 1)
        min_loss = jnp.where(loss_decreased, new_loss, state.min_loss)
        print(jnp.min(delta_F_masked), log_V_sum, logdetC_k)
        done = jnp.all(cluster_id == state.cluster_id) \
               | (delay >= 10) \
               | jnp.any(num_k < D + 1) \
               | jnp.isnan(log_V_sum) \
               | (jnp.min(delta_F_masked) >= 0.)
        # ['i', 'done', 'cluster_id', 'C_k', 'logdetC_k',
        # 'mu_k', 'log_maha_k', 'num_k',
        # 'log_VE_k', 'log_VS_k',
        # 'min_loss', 'delay']
        state = state._replace(i=state.i + 1,
                               done=done,
                               cluster_id=cluster_id,
                               C_k=C_k,
                               logdetC_k=logdetC_k,
                               mu_k=mu_k,
                               log_maha_k=log_maha_k,
                               num_k=num_k,
                               log_VE_k=log_VE_k,
                               log_VS_k=log_VS_k,
                               min_loss=min_loss,
                               delay=delay)
        return state

    init_state = init_state(cluster_id)
    state = while_loop(lambda state: ~state.done,
                       body,
                       init_state)

    log_f_k = log_factor_k(state.cluster_id, state.log_maha_k, state.num_k, state.logdetC_k)
    log_VE_k = vmap(log_ellipsoid_volume)(state.logdetC_k, state.num_k, log_f_k)
    log_V_sum = logsumexp(log_VE_k)

    do_split = ((log_V_sum < log_VE) | (log_VE > log_VS + jnp.log(K))) \
               & (~jnp.any(jnp.isnan(state.logdetC_k))) \
               & jnp.all(state.num_k >= D + 1)

    precision_k = state.C_k * jnp.exp(jnp.log(state.num_k) + log_f_k)[:, None, None]
    radii_k, rotation_k = vmap(lambda C_k: ellipsoid_params(C_k))(precision_k)

    return state.cluster_id, state.log_VS_k, state.mu_k, radii_k, rotation_k, do_split


def cluster_split(key, points, mask, log_VS, log_VE, kmeans_init=True):
    """
    Splits a set of points into two ellipsoids such that the enclosed volume is as close to V(S) without being less.
    V(S) should be an estimate of the true volume contained by the points.

    Args:
        key:
        points: [N, D]
        mask: [N] only split on these points
        log_VS: logV(S) of the set of points
        log_VE: logV(E) of the parent ellipsoid
        kmeans_init: whether to use kmeans to initialise the clustering

    Returns:
        cluster_id: ids of the points, places where ~mask are random assignments
        mu1, radii1, rotation1: ellipsoid params of first subcluster
        mu2, radii2, rotation2: ellipsoid paras of second subcluster

    """

    init_key, volume_key = random.split(key, 2)
    N, D = points.shape
    n_S = jnp.sum(mask)
    # calculate bounding ellipsoid
    ###
    # input is essentially log_VS
    if kmeans_init:
        # do Euclidean kmean clustering
        cluster_id, centers = kmeans(init_key, points, mask, K=2)
    else:
        # assign to random clusters: child0 or child1
        cluster_id = random.randint(init_key, shape=(N,), minval=0, maxval=2)

    def body(state):
        (i, done, old_cluster_id, _, _, _, _, _, _, _, _, min_loss, delay) = state
        mask1 = mask & (old_cluster_id == 0)
        mask2 = mask & (old_cluster_id == 1)
        # estimate volumes of current clustering
        n1 = jnp.sum(mask1)
        n2 = jnp.sum(mask2)
        log_VS1 = log_VS + jnp.log(n1) - jnp.log(n_S)
        log_VS2 = log_VS + jnp.log(n2) - jnp.log(n_S)
        # construct E_1, E_2 and compute volumes
        mu1, C1 = bounding_ellipsoid(points, mask1)
        radii1, rotation1 = ellipsoid_params(C1)
        log_VE1 = log_ellipsoid_volume(radii1)
        mu2, C2 = bounding_ellipsoid(points, mask2)
        radii2, rotation2 = ellipsoid_params(C2)
        log_VE2 = log_ellipsoid_volume(radii2)
        # enlarge to at least cover V(S1) and V(S2)
        log_scale1 = log_coverage_scale(log_VE1, log_VS1, D)
        log_scale2 = log_coverage_scale(log_VE2, log_VS2, D)
        C1 = C1 / jnp.exp(log_scale1)
        radii1 = jnp.exp(jnp.log(radii1) + log_scale1)
        C2 = C2 / jnp.exp(log_scale2)
        radii2 = jnp.exp(jnp.log(radii2) + log_scale2)
        log_VE1 = log_VE1 + log_scale1 * D
        log_VE2 = log_VE2 + log_scale2 * D
        # compute reassignment metrics
        maha1 = vmap(lambda point: (point - mu1) @ C1 @ (point - mu1))(points)
        maha2 = vmap(lambda point: (point - mu2) @ C2 @ (point - mu2))(points)
        log_h1 = log_VE1 - log_VS1 + jnp.log(maha1)
        log_h2 = log_VE2 - log_VS2 + jnp.log(maha2)
        # reassign
        delta_F = jnp.exp(log_h1) - jnp.exp(log_h2)
        reassign_idx = jnp.argmax(jnp.abs(delta_F))
        new_cluster_id = dynamic_update_slice(cluster_id, (delta_F[reassign_idx, None] > 0).astype(int_type),
                                              reassign_idx[None])
        # new_cluster_k = jnp.where(log_h1 < log_h2, 0, 1)
        log_V_sum = jnp.logaddexp(log_VE1, log_VE2)
        new_loss = jnp.exp(log_V_sum - log_VS)
        loss_decreased = new_loss < min_loss
        delay = jnp.where(loss_decreased, 0, delay + 1)
        min_loss = jnp.where(loss_decreased, new_loss, min_loss)
        ###
        # i / delay / loss_decreased / new_loss / min_loss
        # 0 / 0 / True / a / a
        # 1 / 1 / False / b / a
        # 2 / 2 / False / a / a
        # 3 / 3 / False / b / a
        # 4 / 4 / False / a / a
        done = jnp.all(new_cluster_id == old_cluster_id) \
               | (delay >= 10) \
               | (n1 < D + 1) \
               | (n2 < D + 1) \
               | jnp.isnan(log_V_sum)
        # print(i, "reassignments", jnp.sum(new_cluster_k != old_cluster_id), 'F', log_V_sum)
        # print(i, done, jnp.abs(delta_F).max())
        return (i + 1, done, new_cluster_id, log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2,
                min_loss, delay)

    done = (jnp.sum(mask) < 2 * (D + 1))
    (i, _, cluster_id, log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2, min_loss, delay) = \
        while_loop(lambda state: ~state[1],
                   body,
                   (jnp.array(0), done, cluster_id,
                    jnp.array(-jnp.inf), jnp.zeros(D), jnp.zeros(D), jnp.eye(D),
                    jnp.array(-jnp.inf), jnp.zeros(D), jnp.zeros(D), jnp.eye(D),
                    jnp.asarray(jnp.inf), 0))
    mask1 = mask & (cluster_id == 0)
    mask2 = mask & (cluster_id == 1)
    log_V_sum = jnp.logaddexp(log_ellipsoid_volume(radii1), log_ellipsoid_volume(radii2))

    mu_joint = jnp.stack([mu1, mu2], axis=0)
    radii_joint = jnp.stack([radii1, radii2], axis=0)
    rotation_joint = jnp.stack([rotation1, rotation2], axis=0)
    log_V_union = union_volume_approximation_ellipsoids(volume_key, mu_joint, radii_joint, rotation_joint,
                                                        S=50, constraint_unit_cube=True)

    # V(A v B) = V(A) + V(B) - V(A ^ B) <= V(A) + V(B)
    # V(A v B) / (V(A) + V(B)) = 1 - V(A ^ B)/(V(A) + V(B))
    # (1 - V(A v B) / (V(A) + V(B))) = V(A ^ B)/(V(A) + V(B))
    # V(A ^ B) / V(A v B) = (1 - V(A v B) / (V(A) + V(B))) (V(A) + V(B)) / V(A v B)
    # intersection_ratio = jnp.log(1. - jnp.exp(log_V_union - log_V_sum)) + log_V_sum - log_V_union
    # no_intersection = jnp.exp(intersection_ratio) < 0.05
    no_intersection = (log_V_sum < log_VE)


    do_split = (no_intersection | (log_VE > log_VS + jnp.log(2.))) \
               & (~jnp.any(jnp.isnan(radii1))) \
               & (~jnp.any(jnp.isnan(radii2))) \
               & (jnp.sum(mask1) >= (D + 1)) \
               & (jnp.sum(mask2) >= (D + 1))

    # do_split = (log_VE > log_VS + jnp.log(2.)) \
    #            | ((~jnp.any(jnp.isnan(radii1)))
    #               & (~jnp.any(jnp.isnan(radii2)))
    #               & (jnp.sum(mask1) >= (D + 1))
    #               & (jnp.sum(mask2) >= (D + 1)))
    # & (cond1 < 50.) \
    # & (cond2 < 50.)

    return cluster_id, log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2, do_split


def ellipsoid_clustering(key, points, depth, log_VS):
    """
    Partition live_points into 2^depth clusters in depth-first.
    0:    0 | # 2^0
    1:    1 2 | # 2^1
    2:    3 4 5 6
    3:    7 8 9 10 11 12 13 14

    index(r,c) = index(r,0) + c
    index(r,0) = sum_p=0^(r-1) 2^p = 2^r - 1
    index(r,c) = 2^r - 1 + c

    index(r,c) = index(r-1, 2^(r-1) - 1) + c
    child0(r,c) = index(r+1, 2*c) = index(r+1, 0) + 2*c = index(r,0) + 2^r + 2*c = index(r,c) + 2^r + c
    child1(r,c) = index(r+1, 2*c+1) = index(r+1, 0) + 2*c + 1 = = index(r,c) + 2^r + c + 1
    parent(r,c) = index(r-1, c//2)

    To walk this and modify memory in place we follow the rule:

    Find lowest index, replace with left node, append right node.
    0
    1 2
    3 2 4
    3 5 4 6
    7 5 4 6 8
    7 5 9 6 8 10
    7 11 9 6 8 10 12
    7 11 9 13 8 10 12 14 <- final splitting
    0 4 2 6 1 3 5 7 subtract offset
    rearange with argsort


    In general we will do sum_p=0^(depth-1) 2^p = 2^depth - 1 splittings

    for depth = 4 -> 2^4 - 1 = 15
    for depth = 3 -> 2^2 - 1 = 7


    Args:
        key:
        points: [N, D]
        depth:
        log_VS: expected true volume of points

    Returns: cluster_id, ellipsoid_parameters

    """
    N, D = points.shape

    num_clusters = 2 ** (depth - 1)
    cluster_id = jnp.zeros(N, dtype=int_type)
    mu, C = bounding_ellipsoid(points, cluster_id == 0)
    radii, rotation = ellipsoid_params(C)

    num_splittings = 2 ** (depth - 1) - 1
    keys = random.split(key, num_splittings)
    mu_result = jnp.zeros((num_clusters, D))
    mu_result = dynamic_update_slice(mu_result, mu[None, :], [0, 0])
    radii_result = jnp.zeros((num_clusters, D))
    radii_result = dynamic_update_slice(radii_result, radii[None, :], [0, 0])
    rotation_result = jnp.tile(jnp.eye(D)[None, :, :], (num_clusters, 1, 1))
    rotation_result = dynamic_update_slice(rotation_result, rotation[None, :], [0, 0, 0])
    order = jnp.zeros(num_clusters, dtype=int_type)
    log_VS_subclusters = jnp.array([log_VS] + [0] * num_splittings)

    def body(state, X):
        (cluster_id, mu_result, radii_result, rotation_result, order, log_VS_subclusters) = state
        (key, splitting) = X
        split_key, volume_key = random.split(key, 2)
        splitting_select = jnp.arange(num_clusters) <= splitting
        child0 = jnp.max(jnp.where(splitting_select, order, -jnp.inf)) + 1
        child0 = child0.astype(int_type)
        child1 = child0 + 1
        i_lowest = jnp.argmin(jnp.where(splitting_select, order, jnp.inf))

        def _replace_result(operand, update1, update2):
            operand = dynamic_update_slice(operand, update1, jnp.asarray([i_lowest] + [0] * (len(operand.shape) - 1)))
            operand = dynamic_update_slice(operand, update2,
                                           jnp.asarray([splitting + 1] + [0] * (len(operand.shape) - 1)))
            return operand

        mask = cluster_id == order[i_lowest]
        log_VS_subcluster = log_VS_subclusters[i_lowest]
        log_VE_parent = log_ellipsoid_volume(radii_result[i_lowest, :])
        unsorted_cluster_id, log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2, do_split = cluster_split(
            split_key, points, mask, log_VS_subcluster, log_VE_parent, kmeans_init=True)

        # assuming we do the split.
        _log_VS_subclusters = _replace_result(log_VS_subclusters, log_VS1[None], log_VS2[None])
        _mu_result = _replace_result(mu_result, mu1[None, :], mu2[None, :])
        _radii_result = _replace_result(radii_result, radii1[None, :], radii2[None, :])
        _rotation_result = _replace_result(rotation_result, rotation1[None, :, :], rotation2[None, :, :])

        unsorted_cluster_id = jnp.where(unsorted_cluster_id == 0, child0, child1)
        cluster_id = jnp.where(mask, unsorted_cluster_id, cluster_id)


        # if no split we replace child0 with parent and child1 gets zero-size ellipsoid that has no members.

        __log_VS_subclusters = _replace_result(log_VS_subclusters, log_VS_subcluster[None], jnp.array(-jnp.inf)[None])
        __mu_result = _replace_result(mu_result, mu_result[i_lowest][None, :], jnp.zeros(D)[None, :])
        __radii_result = _replace_result(radii_result, radii_result[i_lowest][None, :], jnp.zeros(D)[None, :])
        __rotation_result = _replace_result(rotation_result, rotation_result[i_lowest][None, :, :],
                                            jnp.eye(D)[None, :, :])

        log_VS_subclusters = jnp.where(do_split, _log_VS_subclusters, __log_VS_subclusters)
        mu_result = jnp.where(do_split, _mu_result, __mu_result)
        radii_result = jnp.where(do_split, _radii_result, __radii_result)
        rotation_result = jnp.where(do_split, _rotation_result, __rotation_result)

        cluster_id = jnp.where((~do_split) & (cluster_id == child1), child0, cluster_id)

        # order[i_lowest] = child0
        # order.append(child1)
        order = _replace_result(order, child0[None], child1[None])

        return (cluster_id, mu_result, radii_result, rotation_result, order, log_VS_subclusters), ()

    (cluster_id, mu_result, radii_result, rotation_result, order, log_VS_subclusters), _ = \
        scan(body,
             (cluster_id, mu_result, radii_result, rotation_result, order, log_VS_subclusters),
             (keys, jnp.arange(num_splittings)),
             length=num_splittings)
    cluster_id = cluster_id - (2 ** (depth - 1) - 1)
    order = order - (2 ** (depth - 1) - 1)
    # order results so that cluster_id corresponds to the correct row
    mu_result = mu_result[order, :]
    radii_result = radii_result[order, :]
    rotation_result = rotation_result[order, :, :]
    return cluster_id, (mu_result, radii_result, rotation_result)


def maha_ellipsoid(u, mu, radii, rotation):
    u_circ = ellipsoid_to_circle(u, mu, radii, rotation)
    return u_circ @ u_circ


def point_in_ellipsoid(u, mu, radii, rotation):
    return maha_ellipsoid(u, mu, radii, rotation) <= 1.


def sample_multi_ellipsoid(key, mu, radii, rotation, unit_cube_constraint=True):
    """
    Sample from a set of overlapping ellipsoids.
    When unit_cube_constraint=True then during the sampling when a random radius is chosen, the radius is constrained.

    u(t) = R @ (x + t * num_options) + c
    u(t) == 1
    1-c = R@x + t * R@num_options
    t = ((1 - c) - R@x)/R@num_options

    Args:
        key:
        mu: [K, D]
        radii: [K, D]
        rotation: [K,D,D]

    Returns: point uniformly sampled from intersection of ellipsoids [D]

    """
    K, D = radii.shape
    log_VE = vmap(log_ellipsoid_volume)(radii)
    log_p = log_VE  # - logsumexp(log_VE)

    def body(state):
        (i, _, key, done, _) = state
        key, accept_key, sample_key, select_key = random.split(key, 4)
        k = random.categorical(select_key, log_p)
        mu_k = mu[k, :]
        radii_k = radii[k, :]
        rotation_k = rotation[k, :, :]
        u_test = sample_ellipsoid(sample_key, mu_k, radii_k, rotation_k, unit_cube_constraint=False)
        depth = compute_depth_ellipsoids(u_test, mu, radii, rotation, constraint_unit_cube=unit_cube_constraint)
        done = random.uniform(accept_key) < jnp.reciprocal(depth)
        return (i + 1, k, key, done, u_test)

    _, k, _, _, u_accept = while_loop(lambda state: ~state[3],
                                      body,
                                      (jnp.array(0), jnp.array(0), key, jnp.array(False), jnp.zeros(D)))
    return k, u_accept


def log_ellipsoid_volume(radii):
    D = radii.shape[0]
    return jnp.log(2.) - jnp.log(D) + 0.5 * D * jnp.log(jnp.pi) - gammaln(0.5 * D) + jnp.sum(jnp.log(radii))


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
    Q = jnp.concatenate([points, jnp.ones([N, 1])], axis=1)  # N,D+1

    def body(state):
        (count, err, u) = state
        V = Q.T @ jnp.diag(u) @ Q  # D+1, D+1
        # g[i] = Q[i,j].V^-1_jk.Q[i,k]
        g = vmap(lambda q: q @ jnp.linalg.solve(V, q))(Q)  # difference
        # jnp.diag(Q @ jnp.linalg.solve(V, Q.T))
        j = jnp.argmax(g)
        g_max = g[j]

        step_size = \
            (g_max - D - 1) / ((D + 1) * (g_max - 1))
        search_direction = jnp.where(jnp.arange(N) == j, 1. - u, -u)
        new_u = u + step_size * search_direction
        # new_u = (1. - step_size)*u
        new_u = jnp.where(jnp.arange(N) == j, u + step_size * (1. - u), u * (1. - step_size))
        new_err = jnp.linalg.norm(u - new_u)
        return (count + 1, new_err, new_u)

    if init_u is None:
        init_u = jnp.ones(N) / N
    (count, err, u) = while_loop(lambda state: state[1] > tol * jnp.linalg.norm(init_u),
                                 body,
                                 (0, jnp.inf, init_u))
    U = jnp.diag(u)
    PU = (points.T @ u)  # D, N
    A = jnp.reciprocal(D) * jnp.linalg.pinv(points.T @ U @ points - PU[:, None] @ PU[None, :])
    c = PU
    W, Q, Vh = jnp.linalg.svd(A)
    radii = jnp.reciprocal(jnp.sqrt(Q))
    rotation = Vh.conj().T
    if return_u:
        return c, radii, rotation, u
    return c, radii, rotation


def sample_ellipsoid(key, mu, radii, rotation, unit_cube_constraint=False):
    """
    Sample uniformly inside an ellipsoid.
    When unit_cube_constraint=True then reject points outside box.

    Args:
        key:
        mu: [D]
        radii: [D]
        rotation: [D,D]

    Returns: [D]

    """

    def _single_sample(key):
        direction_key, radii_key = random.split(key, 2)
        direction = random.normal(direction_key, shape=radii.shape)
        direction = direction / jnp.linalg.norm(direction)
        t = random.uniform(radii_key) ** (1. / radii.size)
        u_circ = direction * t
        R = rotation * radii
        u = R @ u_circ + mu
        return u

    def body(state):
        (key, _, _) = state
        key, sample_key = random.split(key, 2)
        u = _single_sample(sample_key)
        done = jnp.all((u < 1) & (u > 0))
        return (key, done, u)

    if unit_cube_constraint:
        (_, _, u) = while_loop(lambda s: ~s[1],
                               body,
                               (key, jnp.asarray(False), mu))
    else:
        u = _single_sample(key)
    return u


def ellipsoid_to_circle(points, center, radii, rotation):
    """
    Scale and rotate an ellipse to a circle.
    Args:
        points: [N, D] or [D]
        center: [D]
        radii: [D]
        rotation: [D,D]

    Returns: [N, D] or [D]

    """
    if len(points.shape) == 1:
        return (rotation.T / radii[:, None]) @ (points - center)
    else:
        return vmap(lambda point: ellipsoid_to_circle(point, center, radii, rotation))(points)


def circle_to_ellipsoid(points, center, radii, rotation):
    """
    Scale and rotate and translate an ellipse to a circle.
    Args:
        points: [N, D] or [D]
        center: [D]
        radii: [D]
        rotation: [D,D]

    Returns: [N, D] or [D]

    """
    if len(points.shape) == 1:
        return (rotation * radii[None, :]) @ points + center
    else:
        return vmap(lambda point: circle_to_ellipsoid(point, center, radii, rotation))(points)


def compute_depth_ellipsoids(point, mu, radii, rotation, constraint_unit_cube=False):
    """
    Compute overlap of ellipsoids at point. Points outside the domain are given infinite depth.

    Args:
        point: point to compute depth at.
        mu: [K, D] means of ellispoids
        radii: [K, D] radii of ellipsoids
        rotation: [K, D, D] rotation matrices of ellipsoids
        constraint_unit_cube: bool, whether domain is clipped to unit-cube.

    Returns:
        scalar representing overlap of ellipsoids.
    """
    # in any of the ellipsoids
    depth = jnp.sum(vmap(
        lambda mu, radii, rotation: point_in_ellipsoid(point, mu, radii, rotation)
    )(mu, radii, rotation))
    if constraint_unit_cube:
        # outside cube
        outside_unit_cube = jnp.any(point < 0.) | jnp.any(point > 1.)
        depth = jnp.where(outside_unit_cube, jnp.inf, depth)
    return depth


def union_volume_approximation_ellipsoids(key, mu, radii, rotation, S, constraint_unit_cube=False):
    def _single_cell(key, _mu, _radii, _rotation):
        def _single_point_estimate(key):
            point = sample_ellipsoid(key, _mu, _radii, _rotation)
            depth = compute_depth_ellipsoids(point, mu, radii, rotation, constraint_unit_cube=constraint_unit_cube)
            return jnp.log(depth)

        log_depth = vmap(_single_point_estimate)(random.split(key, S))
        log_h = logsumexp(-log_depth)

        # def body(state):
        #     (m, log_h, key) = state
        #     key, sample_key = random.split(key, 2)
        #     point = sample_piston(sample_key, _y, _z, log_radius)
        #     depth = compute_depth_pistons(point, y, z, log_radius)
        #     log_h = jnp.logaddexp(log_h, -jnp.log(depth))
        #     return (m + 1, log_h, key)
        #
        # (m, log_h, key) = while_loop(lambda s: s[0] < S,
        #                              body,
        #                              (jnp.asarray(0), jnp.asarray(-jnp.inf), key))

        log_V = log_ellipsoid_volume(_radii) + log_h - jnp.log(S)
        # all are depth of zero times volume of zero => 0/0 and should be zero
        log_V = jnp.where(jnp.isinf(log_h), -jnp.inf, log_V)
        return log_V

    keys = random.split(key, mu.shape[0])
    log_V = vmap(_single_cell)(keys, mu, radii, rotation)
    return logsumexp(log_V)