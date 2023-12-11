from typing import NamedTuple, Tuple, Literal

import numpy as np
import pylab as plt
from jax import numpy as jnp, vmap, random, tree_map, lax
from jax._src.scipy.special import gammaln

from jaxns.internals.log_semiring import LogSpace
from jaxns.samplers.multi_ellipsoid.em_gmm import em_gmm
from jaxns.internals.types import IntArray, FloatArray, PRNGKey, BoolArray
from jaxns.internals.types import UType, int_type, float_type

__all__ = [
    'ellipsoid_clustering',
    'sample_multi_ellipsoid',
    'MultEllipsoidState',
]


class EllipsoidParams(NamedTuple):
    mu: FloatArray  # [K, D] Ellipsoids centres
    radii: FloatArray  # [K, D] Ellsipoids radii
    rotation: FloatArray  # [K, D, D] Ellipsoids rotation matrices


class MultEllipsoidState(NamedTuple):
    params: EllipsoidParams
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


def covariance_to_rotational(cov: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    (x - mu)^T inv(cov) (x - mu) = (x - mu)^T J @ J.T (x - mu)

    where J.T is composed of un-rotation and un-scaling:

    J.T = diag(1/radii) @ rotation.T <==> J = rotation @ diag(1/radii)

    Now since, cov = U @ diag(s) @ V.H we have

    J @ J.T = inv(U @ diag(s) @ V.H) = V @ diag(1/s) @ U.H

    ==> J.T = diag(1/sqrt(s)) @ U.H
    ==> radii = sqrt(s), rotation = U

    Args:
        cov:

    Returns:
        radii, rotation
    """
    u, s, vh = jnp.linalg.svd(cov)
    radii_min = jnp.finfo(s.dtype).eps
    radii = jnp.maximum(jnp.sqrt(s), radii_min)
    rotation = u
    return radii, rotation


def ellipsoid_params(points: UType, mask: FloatArray) -> EllipsoidParams:
    """
    If the ellipsoid is defined by

    (x - mu)^T C (x - mu) = 1

    where C = L @ L.T and L = diag(1/radii) @ rotation.T

    then this returns the mu, radius and rotation matrices of the ellipsoid.

    Args:
        points: [N, D] points to fit ellipsoids to
        mask: [N] mask of which points to consider

    Returns:
        mu [D], radii [D] rotation [D,D]
    """
    # get ellipsoid mean and covariance
    mu, Sigma = bounding_ellipsoid(points=points, mask=mask)
    radii, rotation = covariance_to_rotational(Sigma)

    # Compute scale factor for radii to enclose all points.
    # for all i (points[i] - mu) @ inv(Sigma) / scale**2 @ (points[i] - mu) <= 1
    # for all i (points[i] - mu) @ (L @ L.T) @ (points[i] - mu) <= scale**2
    rho = vmap(lambda x: maha_ellipsoid(x=x, mu=mu, radii=radii, rotation=rotation))(points)

    rho_max = jnp.max(jnp.where(mask, rho, 0.))
    radii *= jnp.sqrt(rho_max)

    return EllipsoidParams(mu=mu, radii=radii, rotation=rotation)


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
    return jnp.diag(jnp.reciprocal(radii)) @ rotation.T @ (point - mu)


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
    return mu + (rotation @ jnp.diag(radii) @ point)


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
        (_, _, u) = lax.while_loop(lambda s: ~s[1],
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
                           unit_cube_constraint: bool = True) -> Tuple[IntArray, FloatArray]:
    """
    Sample from a set of intersecting ellipsoids.
    When unit_cube_constraint=True then reject points outside the closed unit-cube.

    Args:
        key: PRNGKey
        mu: [K, D] centres of ellipses
        radii: [K, D] radii of ellipses
        rotation: [K,D,D] rotation matrices of ellipses

    Returns:
        ellipsoid selected, and a sample point i.i.d. sampled from union of ellipsoids, of shape [D]
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

    _, k, _, _, u_accept = lax.while_loop(lambda state: ~state[3],
                                          body,
                                          (jnp.array(0), jnp.array(0), key, jnp.array(False), jnp.zeros(D)))
    return k, u_accept


def log_coverage_scale(log_VE, log_VS, D):
    """
    Computes the required scaling relation such that
    V(E) = max(V(E), V(S))
    where the scaling is to be applied to each radius.
    Args:
        log_VE:
        log_VS:
        D:
    Returns:
    """
    return jnp.maximum(0., (log_VS - log_VE) / D)


class ClusterSplitResult(NamedTuple):
    unsorted_cluster_id: IntArray  # unsorted cluster id, using 0/1 to indicate the child.
    log_VS0: FloatArray
    params0: EllipsoidParams
    log_VS1: FloatArray
    params1: EllipsoidParams
    successful_split: BoolArray


def _multinest_split(key: PRNGKey, params: EllipsoidParams, points: FloatArray, mask: BoolArray, log_VS: FloatArray,
                     em_init: bool = False,
                     patience: int = 1):
    """
    Use's Multinest's method to partition points

    Args:
        key: PRNGKey
        params: ellipsoid params of points that are being split (same as used for log VE)
        points: [N, D] points to split
        mask: [N] mask only those points which should be split
        log_VS: estimate of logV(S) of the set of points
        em_init: whether to use kmeans to initialise the clustering
        patience: how long to wait before seeing improvement

    Returns:
        cluster_id, log_VS0, params0, log_VS1, params1
    """
    init_key, volume_key = random.split(key, 2)

    N, D = points.shape
    n_S = jnp.sum(mask)
    # calculate bounding ellipsoid

    ###
    # input is essentially log_VS
    if em_init:
        # do Euclidean kmean clustering
        cluster_id, (_, _, _), _ = em_gmm(
            key=init_key,
            data=points,
            mask=mask,
            n_components=2,
            n_iters=100
        )
    else:
        # Split the ellipsoid in half
        j_max = jnp.argmax(params.radii)
        n = jnp.where(jnp.arange(params.radii.size) == j_max,
                      jnp.asarray(1., float_type),
                      jnp.asarray(0., float_type)
                      )
        p = params.rotation @ (jnp.diag(params.radii) @ n)
        q = points - params.mu
        proj = q @ p
        cluster_id = jnp.where(proj >= jnp.asarray(0., float_type),
                               jnp.asarray(0, int_type),
                               jnp.asarray(1, int_type))
        # # assign to random clusters: child0 or child1
        # cluster_id = random.randint(init_key, shape=(N,), minval=0, maxval=2)

    class CarryState(NamedTuple):
        iter: IntArray
        done: BoolArray
        cluster_id: IntArray
        log_VS0: FloatArray
        params0: EllipsoidParams
        log_VS1: FloatArray
        params1: EllipsoidParams
        min_loss: FloatArray
        iters_no_improvement: IntArray

    def body(body_state: CarryState):
        mask0 = mask & (body_state.cluster_id == 0)
        mask1 = mask & (body_state.cluster_id == 1)
        # estimate volumes of current clustering
        n0 = jnp.sum(mask0)
        n1 = jnp.sum(mask1)
        log_VS0 = log_VS + jnp.log(n0) - jnp.log(n_S)
        log_VS1 = log_VS + jnp.log(n1) - jnp.log(n_S)
        # construct E_1, E_2 and compute volumes
        params0 = ellipsoid_params(points=points, mask=mask0)
        log_VE0 = log_ellipsoid_volume(params0.radii)
        params1 = ellipsoid_params(points=points, mask=mask1)
        log_VE1 = log_ellipsoid_volume(params1.radii)
        # enlarge to at least cover V(S1) and V(S2)
        log_scale0 = log_coverage_scale(log_VE0, log_VS0, D)
        log_scale1 = log_coverage_scale(log_VE1, log_VS1, D)
        radii0 = jnp.exp(jnp.log(params0.radii) + log_scale0)
        radii1 = jnp.exp(jnp.log(params1.radii) + log_scale1)
        log_VE0 = log_VE0 + log_scale0 * D
        log_VE1 = log_VE1 + log_scale1 * D
        params0 = params0._replace(radii=radii0)
        params1 = params1._replace(radii=radii1)

        # compute reassignment metrics
        maha0 = vmap(lambda point: maha_ellipsoid(point,
                                                  mu=params0.mu,
                                                  radii=params0.radii,
                                                  rotation=params0.radii))(points)
        maha1 = vmap(lambda point: maha_ellipsoid(point, mu=params1.mu,
                                                  radii=params1.radii,
                                                  rotation=params1.radii))(points)
        h0 = LogSpace(log_VE0) * LogSpace(jnp.log(maha0)) / LogSpace(log_VS0)
        h1 = LogSpace(log_VE1) * LogSpace(jnp.log(maha1)) / LogSpace(log_VS1)
        # reassign biggest violator
        abs_delta_F = (h0 - h1).abs()  # N
        masked_log_abs_delta_F = jnp.where(mask, abs_delta_F.log_abs_val, -jnp.inf)
        reassign_idx = jnp.argmax(masked_log_abs_delta_F)
        new_id = jnp.where(masked_log_abs_delta_F[reassign_idx] > -jnp.inf,
                           jnp.asarray(1, int_type) - cluster_id[reassign_idx],
                           cluster_id[reassign_idx])
        new_cluster_id = cluster_id.at[reassign_idx].set(new_id)

        # new_cluster_id = jnp.where(mask & (h0.log_abs_val < h1.log_abs_val),
        #                            jnp.asarray(0., int_type),
        #                            jnp.asarray(1., int_type))

        # new_cluster_k = jnp.where(log_h1 < log_h2, 0, 1)
        log_V_sum = jnp.logaddexp(log_VE0, log_VE1)
        new_loss = log_V_sum - log_VS  # If scaling happened, then this will be zero
        loss_decreased = new_loss < body_state.min_loss
        iters_no_improvement = jnp.where(loss_decreased, 0, body_state.iters_no_improvement + 1)
        min_loss = jnp.where(loss_decreased, new_loss, body_state.min_loss)
        ###
        # i / delay / loss_decreased / new_loss / min_loss
        # 0 / 0 / True / a / a
        # 1 / 1 / False / b / a
        # 2 / 2 / False / a / a
        # 3 / 3 / False / b / a
        # 4 / 4 / False / a / a
        cluster_mapping_unchanged = jnp.all((new_cluster_id == body_state.cluster_id) | jnp.bitwise_not(mask))
        done = cluster_mapping_unchanged \
               | (iters_no_improvement >= patience) \
               | (n0 < D + 1) \
               | (n1 < D + 1) \
               | jnp.isnan(log_V_sum)

        return CarryState(
            iter=body_state.iter + jnp.asarray(1, int_type),
            done=done,
            cluster_id=new_cluster_id,
            log_VS0=log_VS0,
            params0=params0,
            log_VS1=log_VS1,
            params1=params1,
            min_loss=min_loss,
            iters_no_improvement=iters_no_improvement
        )

    # Done to start with if not at least D+1 points per ellipsoid possible
    done = (n_S < 2 * (D + 1))

    init_state = CarryState(
        iter=jnp.array(0),
        done=done,
        cluster_id=cluster_id,
        log_VS0=jnp.array(-jnp.inf),
        params0=EllipsoidParams(mu=jnp.zeros(D), radii=jnp.zeros(D), rotation=jnp.eye(D)),
        log_VS1=jnp.array(-jnp.inf),
        params1=EllipsoidParams(mu=jnp.zeros(D), radii=jnp.zeros(D), rotation=jnp.eye(D)),
        min_loss=jnp.asarray(jnp.inf),
        iters_no_improvement=jnp.asarray(0, int_type)
    )

    output_state: CarryState = lax.while_loop(lambda state: ~state.done,
                                              body,
                                              init_state)
    return output_state.cluster_id, output_state.log_VS0, output_state.params0, output_state.log_VS1, output_state.params1


def _em_gmm_split(key: PRNGKey, points: FloatArray, mask: BoolArray, log_VS: FloatArray):
    """
    Use's EM Gaussian mixture model to partition points.

    Args:
        key: PRNGKey
        points: [N, D] points to split
        mask: [N] mask only those points which should be split
        log_VS: estimate of logV(S) of the set of points

    Returns:
        cluster_id, log_VS0, params0, log_VS1, params1
    """
    N, D = points.shape
    n_S = jnp.sum(mask)

    # do Euclidean kmean clustering
    cluster_id, (_, _, _), _ = em_gmm(
        key=key,
        data=points,
        mask=mask,
        n_components=2,
        n_iters=100
    )

    mask0 = mask & (cluster_id == 0)
    mask1 = mask & (cluster_id == 1)
    # estimate volumes of current clustering
    n0 = jnp.sum(mask0)
    n1 = jnp.sum(mask1)
    log_VS0 = log_VS + jnp.log(n0) - jnp.log(n_S)
    log_VS1 = log_VS + jnp.log(n1) - jnp.log(n_S)
    # construct E_1, E_2 and compute volumes
    params0 = ellipsoid_params(points=points, mask=mask0)
    log_VE0 = log_ellipsoid_volume(params0.radii)
    params1 = ellipsoid_params(points=points, mask=mask1)
    log_VE1 = log_ellipsoid_volume(params1.radii)
    # enlarge to at least cover V(S1) and V(S2)
    log_scale0 = log_coverage_scale(log_VE0, log_VS0, D)
    log_scale1 = log_coverage_scale(log_VE1, log_VS1, D)
    radii0 = jnp.exp(jnp.log(params0.radii) + log_scale0)
    radii1 = jnp.exp(jnp.log(params1.radii) + log_scale1)
    params0 = params0._replace(radii=radii0)
    params1 = params1._replace(radii=radii1)

    return cluster_id, log_VS0, params0, log_VS1, params1


def cluster_split(key: PRNGKey, params: EllipsoidParams, points: FloatArray, mask: BoolArray, log_VS: FloatArray,
                  method: Literal['multinest', 'em_gmm']) -> ClusterSplitResult:
    """
    Splits a set of points into two ellipsoids such that the enclosed volume is as close to V(S) without being less.
    V(S) should be an estimate of the true volume contained by the points.

    Args:
        key: PRNGKey
        params: ellipsoid params of points that are being split (same as used for log VE)
        points: [N, D] points to split
        mask: [N] mask only those points which should be split
        log_VS: estimate of logV(S) of the set of points
        method: what method to use for splitting. Available are: 'multinest','em_gmm'

    Returns:
        cluster split results
    """

    N, D = points.shape
    # calculate bounding ellipsoid
    # volume of ellipsoid, already have E scaled so that V(E) >= V(S)
    log_VE = log_ellipsoid_volume(params.radii)

    # We always have
    if method == 'em_gmm':
        cluster_id, log_VS0, params0, log_VS1, params1 = _em_gmm_split(key=key, points=points, mask=mask,
                                                                       log_VS=log_VS)
    elif method == 'multinest':
        cluster_id, log_VS0, params0, log_VS1, params1 = _multinest_split(key=key, params=params, points=points,
                                                                          mask=mask,
                                                                          log_VS=log_VS,
                                                                          em_init=False,
                                                                          patience=1)
    else:
        raise ValueError(f"Invalid method {method}")

    # Imperfect sampling condition
    # (0)       V(A) <= V(S1), and V(B) <= V(S2)

    # V(A) <= V(S1) = V(S) V(A) / (V(A) + V(B))
    # V(A) (V(A) + V(B)) <= V(S) V(A)
    # V(A)^2 + V(A)V(B) - V(S) V(A) <= 0
    # V(B)^2 + V(A)V(B) - V(S) V(B) <= 0
    # V(A)^2 - V(B)^2 - V(S) (V(A) - V(B)) <= 0
    # (V(A) + V(B)) (V(A) - V(B)) - V(S) (V(A) - V(B)) <= 0
    # (V(A) + V(B)) - V(S) <= 0, (V(A) - V(B)) != 0
    #
    # Bounding ellipsoid condition
    # (1)       V(S1) <= V(E_A), and V(S2) <= V(E_B)
    # (1.1)     from (0) => V(A) + V(B) <= V(S1) + V(S2) <= V(E_A) + V(E_B)

    # Disjoint partitioning condition
    # (2)       V(S1) + V(S2) = V(S)
    # (2.1)     from (1.1) => V(A) + V(B) <= V(S1) + V(S2) = V(S) <= V(E_A) + V(E_B)
    # (2.2)     from (2) => V(S1 ^ S2) = 0

    # Disjoint partitioning condition
    # (3)       V(A) + V(B) = V(A v B)
    # (3.1)     from (2) => V(A ^ B) = 0

    # Bounding ellipsoid condition
    # (4)       V(S) <= V(E)
    # (4.1)     with (2.1) => V(A) + V(B) <= V(S1) + V(S2) = V(S) <= V(E)

    # (5)       Good split <=> V(S1) ~ V(E_A), and V(S2) ~ V(E_B)
    # (5.1)     from (4.1) Good split <=> V(A) + V(B) <= V(S1) + V(S2) = V(S) <~ V(E_B) + V(E_B)  <= V(E)

    # (6)       Good sampling <=> V(A) ~ V(S1), and V(B) ~ V(S2)
    # (6.1)     from (5.1) Good sampling <=> V(A) + V(B) <~ V(S1) + V(S2) = V(S) <= V(E)

    # (7)       Good split and Good sampling <=> V(A) + V(B) <~ V(S1) + V(S2) = V(S) <~ V(E_B) + V(E_B) <= V(E)

    # We take the condition for success:
    #  V(E_B) + V(E_B) is closer to V(S) than V(E) (from 5.1)

    mask0 = mask & (cluster_id == 0)
    mask1 = mask & (cluster_id == 1)

    log_VE_A = log_ellipsoid_volume(params0.radii)
    log_VE_B = log_ellipsoid_volume(params1.radii)

    V_sum = LogSpace(log_VE_A) + LogSpace(log_VE_B)

    good_split = (V_sum.log_abs_val < log_VE)

    successful_split = good_split \
                       & jnp.bitwise_not(jnp.isnan(log_VE_A)) \
                       & jnp.bitwise_not(jnp.isnan(log_VE_A)) \
                       & (jnp.sum(mask0) >= (D + 1)) \
                       & (jnp.sum(mask1) >= (D + 1))

    return ClusterSplitResult(unsorted_cluster_id=cluster_id, log_VS0=log_VS0,
                              params0=params0, log_VS1=log_VS1,
                              params1=params1, successful_split=successful_split)


def plot_ellipses(params: EllipsoidParams, show: bool = True):
    """
    Plots ellipses.

    Args:
        params: ellipsoid parameters to plot
        show: whether to show figure
    """
    theta = jnp.linspace(0., 2 * jnp.pi, 100)
    circle = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    for mu, radii, rotation in zip(params.mu, params.radii, params.rotation):
        ellipse = vmap(lambda point: circle_to_ellipsoid(point, mu, radii, rotation))(circle)
        plt.plot(ellipse[:, 0], ellipse[:, 1], c=np.random.uniform(size=3))
    if show:
        plt.show()


def ellipsoid_clustering(key: PRNGKey, points: FloatArray, log_VS: FloatArray,
                         max_num_ellipsoids: int,
                         method: Literal['multinest', 'em_gmm'] = 'em_gmm') -> MultEllipsoidState:
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

    # Construct the initial state
    init_ellipsoid = ellipsoid_params(points=points, mask=jnp.ones(N, jnp.bool_))

    log_VE = log_ellipsoid_volume(init_ellipsoid.radii)
    log_scale = log_coverage_scale(log_VE, log_VS, D)
    radii = jnp.exp(jnp.log(init_ellipsoid.radii) + log_scale)
    # log_VE = log_VE + log_scale * D
    init_ellipsoid = init_ellipsoid._replace(radii=radii)

    # state is zeros except first ellipsoid
    cluster_id = jnp.zeros(N, dtype=int_type)
    params = EllipsoidParams(
        mu=jnp.zeros((K, D), float_type),
        radii=jnp.zeros((K, D), float_type),
        rotation=jnp.zeros((K, D, D), float_type)
    )
    params: EllipsoidParams = tree_map(lambda x, y: x.at[0].set(y), params, init_ellipsoid)
    state = MultEllipsoidState(
        cluster_id=cluster_id,
        params=params
    )

    # Initial tracking parameters
    log_VS_subclusters = jnp.asarray([log_VS] + [-jnp.inf] * (K - 1))
    done_splitting = jnp.isneginf(log_VS_subclusters)
    split_depth = jnp.zeros([K], int_type)

    # TODO: compare performance with scan

    class CarryType(NamedTuple):
        key: PRNGKey
        next_k: IntArray
        state: MultEllipsoidState
        done_splitting: BoolArray
        split_depth: IntArray
        log_VS_subclusters: FloatArray

    def body(body_state: CarryType) -> CarryType:
        key, split_key = random.split(body_state.key, 2)
        # Select the depth we work on now: bread first selection ==> min depth first (excluding done splits)
        select_split = jnp.argmin(
            jnp.where(body_state.done_splitting, jnp.iinfo(body_state.split_depth.dtype).max, body_state.split_depth)
        )
        mask = body_state.state.cluster_id == select_split
        # estimated volume in sub-cluster
        log_VS = body_state.log_VS_subclusters[select_split]

        # params of ellipsoid
        params = tree_map(lambda x: x[select_split], body_state.state.params)

        # Perform a split on points in the given mask
        # Strategy: if no split we replace child0 with parent and child1 gets zero-size ellipsoid that has no members.
        cluster_split_result: ClusterSplitResult = cluster_split(
            key=split_key,
            params=params,
            points=points,
            mask=mask,
            log_VS=log_VS,
            method=method
        )

        # Update the parameters in given component that is being split with child 0
        params = tree_map(lambda x, y: jnp.where(cluster_split_result.successful_split, x.at[select_split].set(y), x),
                          body_state.state.params,
                          cluster_split_result.params0)
        # Update the parameters in `next_k` with child 1
        params = tree_map(
            lambda x, y: jnp.where(cluster_split_result.successful_split, x.at[body_state.next_k].set(y), x),
            params,
            cluster_split_result.params1)
        # select_split stays the same cluster_id taking on child 0, but next_k gets child 1
        cluster_id = jnp.where(
            cluster_split_result.successful_split & (cluster_split_result.unsorted_cluster_id == 1) & mask,
            body_state.next_k,
            body_state.state.cluster_id)
        state = body_state.state._replace(params=params, cluster_id=cluster_id)

        # If success => next_k is not done, (and select_k is not done, as previously set)
        # Else select_k is done (next_k stays done, as previously set)
        done_splitting = jnp.where(cluster_split_result.successful_split,
                                   body_state.done_splitting.at[body_state.next_k].set(False),
                                   body_state.done_splitting.at[select_split].set(True))

        # If success => update split depth
        new_depth = body_state.split_depth[select_split] + jnp.asarray(1, int_type)
        split_depth = jnp.where(cluster_split_result.successful_split,
                                body_state.split_depth.at[select_split].set(new_depth),
                                body_state.split_depth)
        split_depth = jnp.where(cluster_split_result.successful_split,
                                split_depth.at[body_state.next_k].set(new_depth),
                                split_depth)

        # If success => update estimated subcluster volumes
        log_VS_subclusters = jnp.where(cluster_split_result.successful_split,
                                       body_state.log_VS_subclusters.at[select_split].set(cluster_split_result.log_VS0),
                                       body_state.log_VS_subclusters)
        log_VS_subclusters = jnp.where(cluster_split_result.successful_split,
                                       log_VS_subclusters.at[body_state.next_k].set(cluster_split_result.log_VS1),
                                       log_VS_subclusters)

        # TODO: (verify) I think next_k should only increment if successful split, as otherwise it uses up space.
        next_k = jnp.where(cluster_split_result.successful_split,
                           body_state.next_k + jnp.asarray(1, int_type),
                           body_state.next_k)

        return CarryType(
            key=key,
            next_k=next_k,
            state=state,
            done_splitting=done_splitting,
            split_depth=split_depth,
            log_VS_subclusters=log_VS_subclusters
        )

    def cond(body_state: CarryType) -> BoolArray:
        done = jnp.all(body_state.done_splitting) | (body_state.next_k == K)
        return jnp.bitwise_not(done)

    init_body_state = CarryType(
        key=key,
        next_k=jnp.asarray(1, int_type),
        state=state,
        done_splitting=done_splitting,
        split_depth=split_depth,
        log_VS_subclusters=log_VS_subclusters
    )

    output_state = lax.while_loop(cond, body, init_body_state)

    return output_state.state
