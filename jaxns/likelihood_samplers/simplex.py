from collections import namedtuple

from jaxns.likelihood_samplers.cubes_utils import (log_cubes_intersect_volume, points_in_box, determine_log_cube_width,
                                                   squared_norm)
from jaxns.utils import random_ortho_matrix
from jax import numpy as jnp, vmap, random, jit
from jax.lax import while_loop, dynamic_update_slice
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln, logsumexp
from functools import partial

SimplexSamplerState = namedtuple('SimplexSamplerState',
                                   ['knn_indices'])


def init_simplex_sampler_state(live_points_U):
    N,D = live_points_U.shape
    inter_point_distance = squared_norm(live_points_U, live_points_U)
    inter_point_distance = jnp.where(inter_point_distance == 0., jnp.inf, inter_point_distance)
    knn_indices = jnp.argsort(inter_point_distance, axis=-1)[:, :D+1]
    return SimplexSamplerState(knn_indices=knn_indices)


def simplex(key, log_L_constraint, live_points_U,
                       loglikelihood_from_constrained,
                       prior_transform, sampler_state, replace_id):
    """
    Samples from the prior restricted to the likelihood constraint.
    This undoes the shrinkage at each step to approximate a bound on the contours.
    First it does a scaling on each dimension.

    Args:
        key:
        log_L_constraint:
        live_points_U:
        loglikelihood_from_constrained:

    Returns:

    """

    N,D = live_points_U.shape
    key, width_key = random.split(key, 2)

    def body(state):
        (key, i, u_test, x_test, log_L_test) = state

        key, sample_key, select_key, R_key = random.split(key, 4)
        i = random.randint(select_key, shape=(), minval=0, maxval=N + 1)

        # M,M
        R = random_ortho_matrix(R_key, D)
        # initial L, R for each direction
        # t_R[i] = max_(k) (points[k,j] - spawn_point_U[j]) @ R[j,i]
        # t_L[i] = max_(k) (points[k,j] - spawn_point_U[j]) @ -R[j,i]
        # N, M
        dx = live_points_U[sampler_state.knn_indices[i, :], :] - live_points_U[i, :]
        # [N, M]
        t = dx @ R
        # [M]
        t_R = jnp.maximum(jnp.max(t, axis=0), 0.)
        t_L = jnp.minimum(jnp.min(t, axis=0), 0.)

        u_test = live_points_U[i,:] + R @ random.uniform(sample_key, shape=[D], minval=t_L, maxval=t_R)
        u_test = jnp.clip(u_test, 0., 1.)

        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(**x_test)
        return (key, i + 1, u_test, x_test, log_L_test)

    (key, num_likelihood_evaluations, u_new, x_new, log_L_new) = while_loop(lambda state: state[-1] <= log_L_constraint,
                                                                            body,
                                                                            (key, 0, live_points_U[0, :],
                                                                             prior_transform(live_points_U[0, :]),
                                                                             log_L_constraint))

    new_dist = jnp.linalg.norm(u_new - dynamic_update_slice(live_points_U, u_new[None, :], [replace_id,0]), axis=1)
    new_dist = jnp.where(new_dist == 0., jnp.inf, new_dist)
    new_indices = jnp.argsort(new_dist)[:D+1]
    knn_indices = dynamic_update_slice(sampler_state.knn_indices,
                                       new_indices[None, :],
                                       [replace_id, 0])
    sampler_state = sampler_state._replace(knn_indices=knn_indices)

    CubesResults = namedtuple('CubesResults',
                                          ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new',
                                           'sampler_state'])
    return CubesResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new, sampler_state)
