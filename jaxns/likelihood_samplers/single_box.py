from collections import namedtuple

from jaxns.utils import random_ortho_matrix
from jax import numpy as jnp, vmap, random
from jax.lax import while_loop
from jax.scipy.linalg import solve_triangular


def expanded_box(key, log_L_constraint, live_points_U,
                 spawn_point_U, loglikelihood_from_constrained,
                 prior_transform, whiten=False):
    """
    Samples from the prior restricted to the likelihood constraint.
    This undoes the shrinkage at each step to approximate a bound on the contours.
    First it does a scaling on each dimension.

    Args:
        key:
        log_L_constraint:
        live_points_U:
        spawn_point_U:
        loglikelihood_from_constrained:

    Returns:

    """
    if whiten:
        u_mean = jnp.mean(live_points_U, axis=0)
        L = jnp.linalg.cholesky(jnp.cov(live_points_U, rowvar=False, bias=True))
        live_points_U = vmap(lambda u: solve_triangular(L, u, lower=True))(live_points_U - u_mean)
        spawn_point_U = solve_triangular(L, spawn_point_U-u_mean, lower=True)

    key, R_key = random.split(key, 2)
    # M,M
    R = random_ortho_matrix(R_key, spawn_point_U.size)

    # initial L, R for each direction
    # t_R[i] = max_(k) (live_points_U[k,j] - spawn_point_U[j]) @ R[j,i]
    # t_L[i] = max_(k) (live_points_U[k,j] - spawn_point_U[j]) @ -R[j,i]
    # t = ((live_points_U - center)/scale - (spawn_point_U - center)/scale) . R
    # t = (live_points_U - spawn_point_U) . R/scale
    # y_test = (spawn_point_U - center)/scale + U[t_L, t_R].R
    # x_test = scale y_test + center
    # N, M
    dx = live_points_U - spawn_point_U
    # [N, M]
    t = dx @ R
    # [M]
    t_R = jnp.maximum(jnp.max(t, axis=0), 0.)
    t_L = jnp.minimum(jnp.min(t, axis=0), 0.)

    # import pylab as plt
    # _live_points = live_points_U
    # _spawn_point = spawn_point_U
    #
    # plt.scatter(_live_points[:, 0], _live_points[:, 1])
    # #x_i = x0_i + R_ij u_j
    # plt.plot([_spawn_point[0] + t_L[0] * R[0,0], _spawn_point[0] + t_R[0] * R[0,0]], [_spawn_point[1] + t_L[0] * R[1,0], _spawn_point[1] + t_R[0] * R[1,0]])
    # plt.plot([_spawn_point[0] + t_L[1] * R[0,1], _spawn_point[0] + t_R[1] * R[0,1]], [_spawn_point[1] + t_L[1] * R[1,1], _spawn_point[1] + t_R[1] * R[1,1]])
    # plt.show()

    def body(state):
        (key, i, u_test, x_test, log_L_test) = state
        key, uniform_key, beta_key = random.split(key, 3)
        # [M]
        U_scale = random.uniform(uniform_key, shape=spawn_point_U.shape,
                                 minval=t_L, maxval=t_R)
        t_shrink = random.beta(beta_key, live_points_U.shape[0], 1) ** jnp.reciprocal(spawn_point_U.size)
        u_test_white = U_scale / t_shrink
        # y_j =
        #    = dx + sum_i p_i * u_i
        #    = dx + R @ u
        # x_i = x0_i + R_ij u_j
        if whiten:
            u_test = L @ (spawn_point_U + R @ u_test_white) + u_mean
        else:
            u_test = u_test_white
        u_test = jnp.clip(u_test, 0., 1.)
        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(**x_test)
        return (key, i + 1, u_test, x_test, log_L_test)

    (key, num_likelihood_evaluations, u_new, x_new, log_L_new) = while_loop(lambda state: state[-1] <= log_L_constraint,
                                                                            body,
                                                                            (key, 0, spawn_point_U,
                                                                             prior_transform(spawn_point_U),
                                                                             log_L_constraint))

    ExpandedBoundResults = namedtuple('ExpandedBoundResults',
                                      ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new'])
    return ExpandedBoundResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new)
