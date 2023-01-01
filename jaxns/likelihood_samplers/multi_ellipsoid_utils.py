from typing import NamedTuple, Tuple

from etils.array_types import IntArray, FloatArray, PRNGKey, BoolArray
from jax import numpy as jnp, vmap, random, tree_map
from jax._src.lax.control_flow import while_loop
from jax._src.scipy.special import gammaln

from jaxns.internals.log_semiring import LogSpace
from jaxns.types import UType, int_type, float_type

__all__ = ['ellipsoid_clustering',
           'sample_multi_ellipsoid',
           'MultEllipsoidState']


class MultiEllipsoidParams(NamedTuple):
    mu: FloatArray  # [K, D] Ellipsoids centres
    radii: FloatArray  # [K, D] Ellsipoids radii
    rotation: FloatArray  # [K, D, D] Ellipsoids rotation matrices


class MultEllipsoidState(NamedTuple):
    params: MultiEllipsoidParams
    cluster_id: IntArray  # [N] the cluster index of each point


def log_ellipsoid_volume(radii):
    D = radii.shape[0]
    return jnp.log(2.) - jnp.log(D) + 0.5 * D * jnp.log(jnp.pi) - gammaln(0.5 * D) + jnp.sum(jnp.log(radii))


def bounding_ellipsoid(points: UType, mask: FloatArray) -> Tuple[FloatArray, FloatArray]:
    """
    Use empirical mean and covariance as approximation to bounding ellipse.

    Args:
        points: [N, D] points to fit ellipsoids to
        mask: [N] mask of which points to consider

    Returns:
        mu, cov
    """

    mu = jnp.average(points, weights=mask, axis=0)
    dx = points - mu
    cov = jnp.average(dx[:, :, None] * dx[:, None, :], weights=mask, axis=0)
    return mu, cov


def ellipsoid_params(points: UType, mask: FloatArray) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    If the ellipsoid is defined by

    (x - mu)^T cov (x - mu) = 1

    where cov = L @ L.T and L = diag(1/radii) @ rotation.T

    then this returns the mu, radius and rotation matrices of the ellipsoid.

    Args:
        points: [N, D] points to fit ellipsoids to
        mask: [N] mask of which points to consider

    Returns:
        mu [D], radii [D] rotation [D,D]
    """
    # get ellipsoid mean and covariance
    mu, cov = bounding_ellipsoid(points=points, mask=mask)

    # compute factorisation
    u, s, vh = jnp.linalg.svd(cov)
    W, Q, Vh = vh.T, jnp.where(s > 1e-15, jnp.reciprocal(s), jnp.zeros_like(s)), u.T
    C = (W * Q) @ Vh
    radii = jnp.reciprocal(jnp.sqrt(Q))
    radii = jnp.where(jnp.isnan(radii), 0., radii)
    rotation = u  # Vh.conj().T

    # Compute scale factor for radii to enclose all points.
    # for all i (points[i] - f) @ inv(scale * cov) @ (points[i] - f) <= 1
    # for all i (points[i] - f) @ inv(cov) @ (points[i] - f) <= scale
    # -> choose scale = max_i (points[i] - f) @ inv(cov) @ (points[i] - f)
    dx = points - mu
    maha = vmap(lambda dx: dx @ C @ dx)(dx)
    scale = jnp.max(jnp.where(mask, maha, 0.))
    radii /= scale

    return mu, radii, rotation


###
# clustering

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


###
# Sampling


def ellipsoid_to_circle(point: FloatArray, mu: FloatArray, radii: FloatArray, rotation: FloatArray) -> FloatArray:
    """
    Apply a linear map that would turn an ellipsoid into a sphere.
    Args:
        point: [D] point to transform
        mu: [D] center of ellipse
        radii: [D] radii of ellipse
        rotation: [D,D] rotation matrix of ellipse

    Returns:
        a transformed point of shape [D]
    """
    return (rotation.T / radii[:, None]) @ (point - mu)


def circle_to_ellipsoid(point: FloatArray, mu: FloatArray, radii: FloatArray, rotation: FloatArray) -> FloatArray:
    """
    Apple a linear map that would turn a sphere into an ellipsoid

    Args:
        point: [D] point to transform
        mu: [D] center of ellipse
        radii: [D] radii of ellipse
        rotation: [D,D] rotation matrix of ellipse

    Returns:
        a transformed point of shape [D]
    """
    return (rotation * radii[None, :]) @ point + mu


def maha_ellipsoid(x: FloatArray, mu: FloatArray, radii: FloatArray, rotation: FloatArray) -> FloatArray:
    """
    Compute the Mahalanobis distance.

    Args:
        x: point [D]
        mu: center of ellipse [D]
        radii: radii of ellipse [D]
        rotation: rotation matrix [D, D]

    Returns:
        The Mahalanobis distance of `x` to `mu`.
    """
    u_circ = ellipsoid_to_circle(x, mu, radii, rotation)
    return u_circ @ u_circ


def point_in_ellipsoid(x: FloatArray, mu: FloatArray, radii: FloatArray, rotation: FloatArray) -> BoolArray:
    """
    Determine if a given point is inside a closed ellipse.

    Args:
        x: point [D]
        mu: center of ellipse [D]
        radii: radii of ellipse [D]
        rotation: rotation matrix [D, D]

    Returns:
        True iff x is inside the closed ellipse
    """
    return jnp.less_equal(maha_ellipsoid(x, mu, radii, rotation), jnp.asarray(1., x.dtype))


def sample_ellipsoid(key: PRNGKey, mu: FloatArray, radii: FloatArray, rotation: FloatArray,
                     unit_cube_constraint: bool = False) -> FloatArray:
    """
    Sample uniformly inside an ellipsoid. When unit_cube_constraint=True then reject points outside unit-cube.

    Args:
        key:
        mu: [D]
        radii: [D]
        rotation: [D,D]
        unit_cube_constraint: whether to restrict to the closed unit-cube.

    Returns:
        i.i.d. sample from ellipsoid of shape [D]
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
        done = jnp.all((u <= 1) & (u >= 0))
        return (key, done, u)

    if unit_cube_constraint:
        (_, _, u) = while_loop(lambda s: ~s[1],
                               body,
                               (key, jnp.asarray(False), mu))
    else:
        u = _single_sample(key)
    return u


def compute_depth_ellipsoids(point: FloatArray, mu: FloatArray, radii: FloatArray, rotation: FloatArray,
                             constraint_unit_cube: bool = False) -> IntArray:
    """
    Compute overlap of ellipsoids at point. Points outside the domain are given infinite depth.

    Args:
        point: [D] point to compute depth at.
        mu: [K, D] means of ellispoids
        radii: [K, D] radii of ellipsoids
        rotation: [K, D, D] rotation matrices of ellipsoids
        constraint_unit_cube: bool, whether domain is clipped to closed unit-cube.

    Returns:
        scalar representing overlap of ellipsoids.
    """
    # in any of the ellipsoids
    contained_in = vmap(lambda mu, radii, rotation: point_in_ellipsoid(point, mu, radii, rotation))(mu, radii, rotation)
    depth = jnp.sum(contained_in)
    if constraint_unit_cube:
        # outside cube
        outside_unit_cube = jnp.any(point < 0.) | jnp.any(point > 1.)
        depth = jnp.where(outside_unit_cube, jnp.iinfo(depth.dtype).max, depth)
    return depth


def sample_multi_ellipsoid(key: PRNGKey, mu: FloatArray, radii: FloatArray, rotation: FloatArray,
                           unit_cube_constraint: bool = True) -> FloatArray:
    """
    Sample from a set of intersecting ellipsoids.
    When unit_cube_constraint=True then reject points outside the closed unit-cube.

    Args:
        key: PRNGKey
        mu: [K, D] centres of ellipses
        radii: [K, D] radii of ellipses
        rotation: [K,D,D] rotation matrices of ellipses

    Returns:
        a sample point i.i.d. sampled from union of ellipsoids, of shape [D]
    """
    # u(t) = R @ (x + t * num_options) + c
    #     u(t) == 1
    #     1-c = R@x + t * R@num_options
    #     t = ((1 - c) - R@x)/R@num_options
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


def cluster_split(key: PRNGKey, points: FloatArray, mask: BoolArray, log_VS: FloatArray, log_VE: FloatArray,
                  kmeans_init: bool = True) \
        -> Tuple[IntArray, FloatArray, MultiEllipsoidParams, FloatArray, MultiEllipsoidParams, BoolArray]:
    """
    Splits a set of points into two ellipsoids such that the enclosed volume is as close to V(S) without being less.
    V(S) should be an estimate of the true volume contained by the points.

    Args:
        key: PRNGKey
        points: [N, D] points to split
        mask: [N] mask only those points which should be split
        log_VS: estimate of logV(S) of the set of points
        log_VE: logV(E) of the minimum volume enclosing ellipsoid of masked points
        kmeans_init: whether to use kmeans to initialise the clustering

    Returns:
        cluster indices of points (0 | 1),
        logV(S1),
        params of ellipsoid 1
        logV(S2),
        params of ellipsoid 2,
        whether split occured -- If False then above return values are undefined
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
        mu1, radii1, rotation1 = ellipsoid_params(points=points, mask=mask1)
        log_VE1 = log_ellipsoid_volume(radii1)
        mu2, radii2, rotation2 = ellipsoid_params(points=points, mask=mask2)
        log_VE2 = log_ellipsoid_volume(radii2)
        # enlarge to at least cover V(S1) and V(S2)
        log_scale1 = log_coverage_scale(log_VE1, log_VS1, D)
        log_scale2 = log_coverage_scale(log_VE2, log_VS2, D)
        radii1 = jnp.exp(jnp.log(radii1) + log_scale1)
        radii2 = jnp.exp(jnp.log(radii2) + log_scale2)
        log_VE1 = log_VE1 + log_scale1 * D
        log_VE2 = log_VE2 + log_scale2 * D
        # compute reassignment metrics
        maha1 = vmap(lambda point: maha_ellipsoid(point, mu=mu1, radii=radii1, rotation=rotation1))(points)
        maha2 = vmap(lambda point: maha_ellipsoid(point, mu=mu2, radii=radii2, rotation=rotation2))(points)
        log_h1 = log_VE1 - log_VS1 + jnp.log(maha1)
        log_h2 = log_VE2 - log_VS2 + jnp.log(maha2)
        # reassign
        delta_F = LogSpace(log_h1) - LogSpace(log_h2)
        reassign_idx = jnp.argmax(delta_F.abs().log_abs_val)
        new_id = jnp.asarray(delta_F[reassign_idx] > 0, int_type)
        new_cluster_id = cluster_id.at[reassign_idx].set(new_id)

        # new_cluster_k = jnp.where(log_h1 < log_h2, 0, 1)
        log_V_sum = jnp.logaddexp(log_VE1, log_VE2)
        new_loss = log_V_sum - log_VS
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
        return (
            i + jnp.asarray(1, int_type), done, new_cluster_id, log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2,
            rotation2,
            min_loss, delay)

    done = (jnp.sum(mask) < 2 * (D + 1))
    (i, _, cluster_id, log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2, min_loss, delay) = \
        while_loop(lambda state: ~state[1],
                   body,
                   (jnp.array(0), done, cluster_id,
                    jnp.array(-jnp.inf), jnp.zeros(D), jnp.zeros(D), jnp.eye(D),
                    jnp.array(-jnp.inf), jnp.zeros(D), jnp.zeros(D), jnp.eye(D),
                    jnp.asarray(jnp.inf), jnp.asarray(0, int_type)))
    mask1 = mask & (cluster_id == 0)
    mask2 = mask & (cluster_id == 1)
    log_V_sum = jnp.logaddexp(log_ellipsoid_volume(radii1), log_ellipsoid_volume(radii2))

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

    params1 = MultiEllipsoidParams(mu=mu1,
                                   radii=radii1,
                                   rotation=rotation1)

    params2 = MultiEllipsoidParams(mu=mu2, radii=radii2, rotation=rotation2)

    return cluster_id, log_VS1, params1, log_VS2, params2, do_split


def _log_coverage_scale(log_VE, log_VS, D):
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


def ellipsoid_clustering(key: PRNGKey, points: FloatArray, log_VS: FloatArray,
                         max_num_ellipsoids: int) -> MultEllipsoidState:
    """
    Partition live_points into 2^depth ellipsoids in depth-first order.

    Args:
        key:PRNGKey
        points: [N, D] points to partition
        log_VS: expected true volume of points
        max_num_ellipsoids: the maximum number of ellipsoids

    Returns:
        params of multi-ellipsoids and cluster id of points
    """

    N, D = points.shape
    if max_num_ellipsoids < 1:
        raise ValueError(f"max_num_ellipsoids should be >= 1, got {max_num_ellipsoids}.")
    K = max_num_ellipsoids

    cluster_id = jnp.zeros(N, dtype=int_type)

    params = MultiEllipsoidParams(
        mu=jnp.zeros((K, D), float_type),
        radii=jnp.zeros((K, D), float_type),
        rotation=jnp.zeros((K, D, D), float_type)
    )

    mu, radii, rotation = ellipsoid_params(points=points, mask=jnp.ones(N, jnp.bool_))

    params = params._replace(
        mu=params.mu.at[0].set(mu),
        radii=params.radii.at[0].set(radii),
        rotation=params.rotation.at[0].set(rotation)
    )
    state = MultEllipsoidState(cluster_id=cluster_id,
                               params=params)

    log_VS_subclusters = jnp.array([log_VS] + [-jnp.inf] * (K - 1))
    done_splitting = jnp.asarray([False] + [True] * (K - 1), jnp.bool_)
    split_depth = jnp.zeros([K], int_type)

    CarryType = Tuple[PRNGKey, IntArray, MultEllipsoidState, BoolArray, IntArray, FloatArray]

    def body(body_state: CarryType) -> CarryType:
        (key, next_k, state, done_splitting, split_depth, log_VS_subclusters) = body_state
        key, split_key = random.split(key, 2)
        # bread first selection
        select_split = jnp.argmin(jnp.where(done_splitting, jnp.iinfo(split_depth.dtype).max, split_depth))

        mask = state.cluster_id == select_split
        log_VS_subcluster = log_VS_subclusters[select_split]
        log_VE_parent = log_ellipsoid_volume(state.params.radii[select_split])

        unsorted_cluster_id, log_VS1, params1, log_VS2, params2, do_split = cluster_split(
            split_key, points, mask, log_VS_subcluster, log_VE_parent, kmeans_init=True)

        params = tree_map(lambda x, y: jnp.where(do_split, x.at[select_split].set(y), x),
                          state.params,
                          params1)
        params = tree_map(lambda x, y: jnp.where(do_split, x.at[next_k].set(y), x),
                          params,
                          params2)

        # update those points where mask is true, unsorted id is 1, and do_split is true
        cluster_id = jnp.where(do_split & (unsorted_cluster_id == 1) & mask, next_k, state.cluster_id)

        state = state._replace(params=params, cluster_id=cluster_id)

        done_splitting = jnp.where(do_split,
                                   done_splitting.at[next_k].set(False),
                                   done_splitting.at[select_split].set(True))

        new_depth = split_depth[select_split] + jnp.asarray(1, int_type)
        split_depth = jnp.where(do_split, split_depth.at[select_split].set(new_depth), split_depth)
        split_depth = jnp.where(do_split, split_depth.at[next_k].set(new_depth), split_depth)

        log_VS_subclusters = jnp.where(do_split, log_VS_subclusters.at[select_split].set(log_VS1), log_VS_subclusters)
        log_VS_subclusters = jnp.where(do_split, log_VS_subclusters.at[next_k].set(log_VS2), log_VS_subclusters)

        # if no split we replace child0 with parent and child1 gets zero-size ellipsoid that has no members.

        next_k = next_k + jnp.asarray(1, int_type)

        return (key, next_k, state, done_splitting, split_depth, log_VS_subclusters)

    def cond(body_state: CarryType) -> BoolArray:
        (key, next_k, state, done_splitting, split_depth, log_VS_subclusters) = body_state
        done = jnp.all(done_splitting) | (next_k == K)
        return jnp.bitwise_not(done)

    init_body_state = (key, jnp.asarray(1, int_type), state, done_splitting, split_depth, log_VS_subclusters)
    (key, next_k, state, done_splitting, split_depth, log_VS_subclusters) = \
        while_loop(cond,
                   body,
                   init_body_state)

    return state
