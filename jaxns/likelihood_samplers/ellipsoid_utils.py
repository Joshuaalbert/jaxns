from jax import numpy as jnp, vmap, random
from jax.lax import while_loop, dynamic_update_slice, scan
from jax.scipy.special import gammaln, logsumexp


def test_binary_tree_bf_walk():
    """
    Test
    0:    0 | # 2^0
    1:    1 2 | # 2^1
    2:    3 4 5 6
    3:    7 8 9 10 11 12 13 14

    index(r,c) = index(r,0) + c
    index(r,0) = sum_p=0^(r-1) 2^p = 2^r - 1
    index(r,c) = 2^r - 1 + c

    index(r,c) = index(r-1, 2^(r-1) - 1) + c
    child0(r,c) = index(r+1, 2*c)
    child1(r,c) = index(r+1, 2*c+1)
    parent(r,c) = index(r-1, c//2)
    Returns:

    """

    def index(r, c):
        return 2 ** r - 1 + c

    def child0(r, c):
        return index(r + 1, 2 * c)

    def child1(r, c):
        return index(r + 1, 2 * c + 1)

    def parent(r, c):
        return index(r - 1, c // 2)

    def row(i):
        return jnp.ceil(jnp.log2(i + 1))

    depth = 3
    for r in range(depth):
        for c in range(2 ** r):
            print("({},{}), row={}, index={}, child0={}, child1={}, parent={}".format(r, c,
                                                                                      row(index(r, c)),
                                                                                      index(r, c),
                                                                                      child0(r, c),
                                                                                      child1(r, c),
                                                                                      parent(r, c)))


def bounding_ellipsoid(points, mask):
    """
    Use empirical mean and covariance as approximation to bounding ellipse, then scale so that all points are inside.

    for all i (points[i] - mu) @ inv(scale * cov) @ (points[i] - mu) <= 1
    for all i (points[i] - mu) @ inv(cov) @ (points[i] - mu) <= scale
    -> choose scale = max_i (points[i] - mu) @ inv(cov) @ (points[i] - mu)
    Args:
        points:
        mask:

    Returns: mu, inv(scale * cov)

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

    (x - mu)^T C (x - mu) <= 1

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


def test_bounding_ellipsoid():
    points = random.normal(random.PRNGKey(0), shape=(10, 2))
    mu, C = bounding_ellipsoid(points, jnp.ones(points.shape[0]))
    radii, rotation = ellipsoid_params(C)
    print(mu, C, radii, rotation)
    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = mu[:, None] + rotation @ jnp.diag(radii) @ jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    import pylab as plt
    plt.scatter(points[:, 0], points[:, 1])
    plt.plot(x[0, :], x[1, :])
    plt.show()


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
        # print("kmeans reassigns", jnp.sum(old_cluster_id!=new_cluster_id))
        return i + 1, done, new_cluster_id, new_centers

    do_kmeans = jnp.sum(mask) > K
    i, _, cluster_id, centers = while_loop(lambda state: ~state[1],
                                           body,
                                           (jnp.array(0), ~do_kmeans,
                                            random.randint(key, shape=(N,), minval=0, maxval=2),
                                            jnp.zeros((K, D))))
    return cluster_id, centers


def test_kmeans():
    points = jnp.concatenate([random.normal(random.PRNGKey(0), shape=(30, 2)),
                              3. + random.normal(random.PRNGKey(0), shape=(10, 2))],
                             axis=0)

    cluster_id, centers = kmeans(random.PRNGKey(0), points, jnp.ones(points.shape[0], dtype=jnp.bool_), K=2)

    mu, C = bounding_ellipsoid(points, jnp.ones(points.shape[0]))
    radii, rotation = ellipsoid_params(C)
    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = mu[:, None] + rotation @ jnp.diag(radii) @ jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    import pylab as plt
    mask = cluster_id == 0
    plt.scatter(points[mask, 0], points[mask, 1])
    mask = cluster_id == 1
    plt.scatter(points[mask, 0], points[mask, 1])
    plt.plot(x[0, :], x[1, :])
    plt.show()


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
    N, D = points.shape
    n_S = jnp.sum(mask)
    # # calculate bounding ellipsoid
    # mu, C =// bounding_ellipsoid(points, mask)
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
        cluster_id, centers = kmeans(key, points, mask, K=2)
    else:
        # assign to random clusters: child0 or child1
        cluster_id = random.randint(key, shape=(N,), minval=0, maxval=2)

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
        new_cluster_id = jnp.where(log_h1 < log_h2, 0, 1)
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
        # print(i, "reassignments", jnp.sum(new_cluster_id != old_cluster_id), 'F', log_V_sum)
        return (i + 1, done, new_cluster_id, log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2,
                min_loss, delay)

    done = jnp.sum(mask) < 2 * (D + 1)
    (i, _, cluster_id, log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2, min_loss, delay) = \
        while_loop(lambda state: ~state[1],
                   body,
                   (jnp.array(0), done, cluster_id,
                    jnp.array(-jnp.inf), jnp.zeros(D), jnp.zeros(D), jnp.eye(D),
                    jnp.array(-jnp.inf), jnp.zeros(D), jnp.zeros(D), jnp.eye(D),
                    jnp.asarray(jnp.inf), 0))
    mask1 = mask & (cluster_id == 0)
    mask2 = mask & (cluster_id == 1)
    cond1 = jnp.max(radii1) / jnp.min(radii1)
    cond2 = jnp.max(radii2) / jnp.min(radii2)
    log_V_sum = jnp.logaddexp(log_ellipsoid_volume(radii1), log_ellipsoid_volume(radii2))

    do_split = ((log_V_sum < log_VE) | (log_VE > log_VS + jnp.log(2.))) \
               & (~jnp.any(jnp.isnan(radii1))) \
               & (~jnp.any(jnp.isnan(radii2))) \
               & (jnp.sum(mask1) >= (D + 1)) \
               & (jnp.sum(mask2) >= (D + 1)) \
               & (cond1 < 50.) \
               & (cond2 < 50.)

    return cluster_id, log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2, do_split


def test_cluster_split():
    import pylab as plt
    from jax import disable_jit
    points = jnp.concatenate([random.uniform(random.PRNGKey(0), shape=(30, 2)),
                              1.25 + random.uniform(random.PRNGKey(0), shape=(10, 2))],
                             axis=0)
    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    mask = jnp.zeros(points.shape[0], jnp.bool_)
    mu, C = bounding_ellipsoid(points, jnp.ones(points.shape[0], jnp.bool_))
    radii, rotation = ellipsoid_params(C)
    y = mu[:, None] + rotation @ jnp.diag(radii) @ x
    plt.plot(y[0, :], y[1, :])
    log_VS = log_ellipsoid_volume(radii) - jnp.log(5)
    with disable_jit():
        cluster_id, log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2, do_split = \
            cluster_split(random.PRNGKey(0), points, mask, log_VS, log_ellipsoid_volume(radii), kmeans_init=True)
        print(jnp.logaddexp(log_ellipsoid_volume(radii1), log_ellipsoid_volume(radii2)), log_ellipsoid_volume(radii))
        print(log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2, do_split)
        print(cluster_id)

    y = mu1[:, None] + rotation1 @ jnp.diag(radii1) @ x
    plt.plot(y[0, :], y[1, :])

    y = mu2[:, None] + rotation2 @ jnp.diag(radii2) @ x
    plt.plot(y[0, :], y[1, :])

    mask = cluster_id == 0
    plt.scatter(points[mask, 0], points[mask, 1])
    mask = cluster_id == 1
    plt.scatter(points[mask, 0], points[mask, 1])

    plt.show()


def test_accept_order():
    def constraint(u):
        return u ** 2 > 0.5

    def accept_prob(u):
        if u > 0.9:
            return 1
        if u > 0.8:
            return 0.5
        if u > 0.7:
            return 0.25
        return 0.

    def f1(key):
        while True:
            key, u_key = random.split(key, 2)
            u = random.uniform(u_key)
            if constraint(u):
                key, a_key = random.split(key, 2)
                if random.uniform(a_key) < accept_prob(u):
                    return u

    def f2(key):
        while True:
            key, u_key, a_key = random.split(key, 3)
            u = random.uniform(u_key)
            if random.uniform(a_key) < accept_prob(u):
                if constraint(u):
                    return u

    from jax import vmap, disable_jit
    import pylab as plt
    keys = random.split(random.PRNGKey(0), 1000)
    with disable_jit():
        u1 = jnp.array([f1(key) for key in keys])
        u2 = jnp.array([f2(key) for key in keys])

    print(u1)

    plt.hist(u1, bins='auto', alpha=0.5)
    plt.hist(u2, bins='auto', alpha=0.5)
    plt.show()


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
    cluster_id = jnp.zeros(N, dtype=jnp.int_)
    mu, C = bounding_ellipsoid(points, cluster_id == 0)
    radii, rotation = ellipsoid_params(C)

    num_splittings = 2 ** (depth - 1) - 1
    keys = random.split(key, num_splittings)
    mu_result = jnp.zeros((num_clusters, D))
    mu_result = dynamic_update_slice(mu_result, mu[None, :], [0, 0])
    radii_result = jnp.zeros((num_clusters, D))
    radii_result = dynamic_update_slice(radii_result, radii[None, :], [0, 0])
    rotation_result = jnp.zeros((num_clusters, D, D))
    rotation_result = dynamic_update_slice(rotation_result, rotation[None, :], [0, 0, 0])
    order = jnp.zeros(num_clusters, dtype=jnp.int_)
    log_VS_subclusters = jnp.array([log_VS] + [0] * num_splittings)

    def body(state, X):
        (cluster_id, mu_result, radii_result, rotation_result, order, log_VS_subclusters) = state
        (key, splitting) = X
        splitting_select = jnp.arange(num_clusters) <= splitting
        child0 = jnp.max(jnp.where(splitting_select, order, -jnp.inf)) + 1
        child0 = child0.astype(jnp.int_)
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
        # print(log_VE_parent, order, i_lowest, radii_result)
        unsorted_cluster_id, log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2, do_split = cluster_split(
            key, points, mask, log_VS_subcluster, log_VE_parent, kmeans_init=True)
        # print(do_split, radii1, radii2)
        unsorted_cluster_id = jnp.where(unsorted_cluster_id == 0, child0, child1)
        cluster_id = jnp.where(mask, unsorted_cluster_id, cluster_id)
        # order[i_lowest] = child0
        # order.append(child1)
        order = _replace_result(order, child0[None], child1[None])
        # print(order)
        # if do_split then keep else
        # we replace child0 with parent and child1 gets zero-size ellipsoid that has no members.
        log_VS1 = jnp.where(do_split, log_VS1, log_VS_subcluster)
        mu1 = jnp.where(do_split, mu1, mu_result[i_lowest, :])
        radii1 = jnp.where(do_split, radii1, radii_result[i_lowest, :])
        rotation1 = jnp.where(do_split, rotation1, rotation_result[i_lowest, :])
        log_VS2 = jnp.where(do_split, log_VS2, jnp.array(-jnp.inf))
        mu2 = jnp.where(do_split, mu2, jnp.zeros(D))
        radii2 = jnp.where(do_split, radii2, jnp.zeros(D))
        rotation2 = jnp.where(do_split, rotation2, jnp.eye(D))
        cluster_id = jnp.where((~do_split) & (cluster_id == child1), child0, cluster_id)
        # update tracked objects
        # log_VS_subclusters[i_lowest] = log_VS1
        # log_VS_subclusters.append(log_VS2)
        log_VS_subclusters = _replace_result(log_VS_subclusters, log_VS1[None], log_VS2[None])
        mu_result = _replace_result(mu_result, mu1[None, :], mu2[None, :])
        radii_result = _replace_result(radii_result, radii1[None, :], radii2[None, :])
        # print(i_lowest, splitting+1, radii_result)
        rotation_result = _replace_result(rotation_result, rotation1[None, :, :], rotation2[None, :, :])
        # print(cluster_id, mu_result, radii_result, rotation_result, order, log_VS_subclusters)
        return (cluster_id, mu_result, radii_result, rotation_result, order, log_VS_subclusters), ()

    (cluster_id, mu_result, radii_result, rotation_result, order, log_VS_subclusters), _ = \
        scan(body,
             (cluster_id, mu_result, radii_result, rotation_result, order, log_VS_subclusters),
             (keys, jnp.arange(num_splittings)),
             unroll=2)
    cluster_id = cluster_id - (2 ** (depth - 1) - 1)
    order = order - (2 ** (depth - 1) - 1)
    # order results so that cluster_id corresponds to the correct row
    mu_result = mu_result[order, :]
    radii_result = radii_result[order, :]
    rotation_result = rotation_result[order, :, :]
    return cluster_id, (mu_result, radii_result, rotation_result)


def maha_ellipsoid(u, mu, radii, rotation):
    dx = u - mu
    dx = jnp.diag(jnp.reciprocal(radii)) @ rotation.T @ dx
    return dx @ dx


def point_in_ellipsoid(u, mu, radii, rotation):
    return maha_ellipsoid(u, mu, radii, rotation) <= 1.


def sample_multi_ellipsoid(key, mu, radii, rotation, unit_cube_constraint=True):
    """
    Sample from a set of overlapping ellipsoids.
    When unit_cube_constraint=True then during the sampling when a random radius is chosen, the radius is constrained.

    u(t) = R @ (x + t * n) + c
    u(t) == 1
    1-c = R@x + t * R@n
    t = ((1 - c) - R@x)/R@n

    Args:
        key:
        mu: [K, D]
        radii: [K, D]
        rotation: [K,D,D]

    Returns: point uniformly sampled from intersection of ellipsoids [D]

    """
    K, D = radii.shape
    log_VE = vmap(log_ellipsoid_volume)(radii)
    log_p = log_VE #- logsumexp(log_VE)
    if unit_cube_constraint:
        center_in_unit_cube = vmap(lambda mu: jnp.all(mu < 1.) & jnp.all(mu > 0.))(mu)
        log_p = jnp.where(center_in_unit_cube, log_p, -jnp.inf)
        # print(log_p)

    def body(state):
        (i, _, key, done, _) = state
        key, accept_key, sample_key, select_key = random.split(key, 4)
        k = random.categorical(select_key, log_p)
        mu_k = mu[k, :]
        radii_k = radii[k, :]
        rotation_k = rotation[k, :, :]
        u_test = sample_ellipsoid(sample_key, mu_k, radii_k, rotation_k, unit_cube_constraint=unit_cube_constraint)
        inside = vmap(lambda mu, radii, rotation: point_in_ellipsoid(u_test, mu, radii, rotation))(mu, radii, rotation)
        n_intersect = jnp.sum(inside)
        done = (random.uniform(accept_key) < jnp.reciprocal(n_intersect))
        return (i + 1, k, key, done, u_test)

    _, k, _, _, u_accept = while_loop(lambda state: ~state[3],
                                      body,
                                      (jnp.array(0), jnp.array(0), key, jnp.array(False), jnp.zeros(D)))
    return k, u_accept


def test_sample_multi_ellipsoid():
    import pylab as plt
    from jax import disable_jit, jit
    points = jnp.concatenate([random.uniform(random.PRNGKey(0), shape=(30, 2)),
                              1.25 + random.uniform(random.PRNGKey(0), shape=(10, 2))],
                             axis=0)
    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    mask = jnp.ones(points.shape[0], jnp.bool_)
    mu, C = bounding_ellipsoid(points, mask)
    radii, rotation = ellipsoid_params(C)
    y = mu[:, None] + rotation @ jnp.diag(radii) @ x
    # plt.plot(y[0, :], y[1, :])
    log_VS = log_ellipsoid_volume(radii) - jnp.log(5)

    with disable_jit():
        cluster_id, ellipsoid_parameters = \
            jit(lambda key, points, log_VS: ellipsoid_clustering(random.PRNGKey(0), points, 4, log_VS)
                )(random.PRNGKey(0), points, log_VS)

        mu, radii, rotation = ellipsoid_parameters
        # print(mu, radii, rotation)
        u = jnp.stack(
            [sample_multi_ellipsoid(random.PRNGKey(i), mu, radii, rotation, unit_cube_constraint=True)[1] for i in
             range(100)], axis=0)
    plt.scatter(u[:, 0], u[:, 1])
    for i, (mu, radii, rotation) in enumerate(zip(mu, radii, rotation)):
        y = mu[:, None] + rotation @ jnp.diag(radii) @ x
        plt.plot(y[0, :], y[1, :])
        mask = cluster_id == i
        # plt.scatter(points[mask, 0], points[mask, 1], c=plt.cm.jet(i / len(ellipsoid_parameters)))
    plt.show()


def test_ellipsoid_clustering():
    import pylab as plt
    from jax import disable_jit, jit
    points = jnp.concatenate([random.uniform(random.PRNGKey(0), shape=(30, 2)),
                              1.25 + random.uniform(random.PRNGKey(0), shape=(10, 2))],
                             axis=0)
    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    mask = jnp.ones(points.shape[0], jnp.bool_)
    mu, C = bounding_ellipsoid(points, mask)
    radii, rotation = ellipsoid_params(C)
    # plt.plot(y[0, :], y[1, :])
    log_VS = log_ellipsoid_volume(radii) - jnp.log(5)

    with disable_jit():
        cluster_id, ellipsoid_parameters = \
            jit(lambda key, points, log_VS: ellipsoid_clustering(random.PRNGKey(0), points, 4, log_VS)
                )(random.PRNGKey(0), points, log_VS)
        mu, radii, rotation = ellipsoid_parameters
        print(mu, radii, rotation, jnp.bincount(cluster_id, minlength=0, length=4))

    for i, (mu, radii, rotation) in enumerate(zip(mu, radii, rotation)):
        y = mu[:, None] + rotation @ jnp.diag(radii) @ x
        plt.plot(y[0, :], y[1, :])
        mask = cluster_id == i
        plt.scatter(points[mask, 0], points[mask, 1], c=plt.cm.jet(i / len(ellipsoid_parameters)))

    plt.show()


def log_ellipsoid_volume(radii):
    D = radii.shape[0]
    return jnp.log(2.) - jnp.log(D) + 0.5 * D * jnp.log(jnp.pi) - gammaln(0.5 * D) + jnp.sum(jnp.log(radii))


def test_log_ellipsoid_volume():
    radii = jnp.ones(2)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(jnp.pi))
    radii = jnp.ones(3)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(4. * jnp.pi / 3.))


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


def sample_ellipsoid(key, center, radii, rotation, unit_cube_constraint=False):
    """
    Sample uniformly inside an ellipsoid.
    When unit_cube_constraint=True then during the sampling when a random radius is chosen, the radius is constrained.

    u(t) = R @ (t * n) + c
    u(t) == 1
    1-c = t * R@n
    t = (1 - c)/R@n
    Args:
        key:
        center: [D]
        radii: [D]
        rotation: [D,D]

    Returns: [D]

    """
    direction_key, radii_key = random.split(key, 2)
    direction = random.normal(direction_key, shape=radii.shape)
    if unit_cube_constraint:
        direction = direction / jnp.linalg.norm(direction)
        R = rotation * radii
        D = R @ direction
        t0 = -center / D
        t1 = jnp.reciprocal(D) + t0
        t0 = jnp.where(t0 < 0., jnp.inf, t0)
        t1 = jnp.where(t1 < 0., jnp.inf, t1)
        t = jnp.minimum(jnp.min(t0), jnp.min(t1))
        t = jnp.minimum(t, 1.)
        return random.uniform(radii_key, minval=0., maxval=t) * D + center
    log_norm = jnp.log(jnp.linalg.norm(direction))
    log_radius = jnp.log(random.uniform(radii_key)) / radii.size
    # x = direction * (radius/norm)
    x = direction * jnp.exp(log_radius - log_norm)
    return circle_to_ellipsoid(x, center, radii, rotation)


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
    Scale and rotate an ellipse to a circle.
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


def test_mvee():
    import pylab as plt
    from jax import disable_jit
    ndims = 2
    N = 10 * ndims
    points = random.normal(random.PRNGKey(0), shape=(N, ndims))
    center, radii, rotation, next_u = minimum_volume_enclosing_ellipsoid(points, 0.01, return_u=True)
    points = random.normal(random.PRNGKey(1), shape=(N, ndims))
    with disable_jit():
        center, radii, rotation = minimum_volume_enclosing_ellipsoid(points, 0.01, init_u=next_u)

    circ_points = ellipsoid_to_circle(points[:-1, :], center, radii, rotation)
    # plt.scatter(circ_points[:,0], circ_points[:,1])
    plt.scatter(points[:, 0], points[:, 1])
    # plt.show()
    print(jnp.max(jnp.linalg.norm(circ_points, axis=-1) ** 2 - 1.))
    # assert jnp.all(jnp.linalg.norm(circ_points, axis=-1) <= 1.03)

    theta = jnp.linspace(0., jnp.pi * 2, 100)
    circle = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    ellipse = circle_to_ellipsoid(circle, center, radii, rotation)
    plt.plot(ellipse[:, 0], ellipse[:, 1])
    plt.show()
