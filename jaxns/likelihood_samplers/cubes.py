from collections import namedtuple

from jaxns.likelihood_samplers.cubes_utils import (log_cubes_intersect_volume, points_in_box, determine_log_cube_width,
                                                   squared_norm)
from jax import numpy as jnp, vmap, random, jit
from jax.lax import while_loop
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln, logsumexp
from functools import partial

CubesSamplerState = namedtuple('CubesSamplerState',
                                   ['log_cube_width'])


def init_cubes_sampler_state(num_live_points, dim):
    log_X_init = 0.
    log_volume_per_cube = log_X_init - jnp.log(num_live_points)
    log_cube_width = log_volume_per_cube/dim
    return CubesSamplerState(log_cube_width=log_cube_width)


# @partial(jit, static_argnums=[3,4])
def cubes(key, log_L_constraint, live_points_U,
                       loglikelihood_from_constrained,
                       prior_transform, sampler_state, log_mean_X):
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

    log_ideal_cube_volume = log_mean_X - jnp.log(N)
    log_ideal_cube_width = log_ideal_cube_volume / D
    key, width_key = random.split(key, 2)

    inter_point_distance = jnp.sqrt(squared_norm(live_points_U, live_points_U))
    inter_point_distance = jnp.where(inter_point_distance == 0., jnp.inf, inter_point_distance)
    nearest_dist = jnp.min(inter_point_distance, axis=0)
    log_cube_width = jnp.log(jnp.mean(nearest_dist))
    # log_cube_width = determine_log_cube_width(width_key, init_U, log_mean_X, log_init_cube_width=sampler_state.log_cube_width,
    #                          tol=0.1, shrink_amount=0.5, grow_amount=1.5, log_vol_samples=1)

    # log_cube_width = jnp.where(jnp.isfinite(log_cube_width), log_cube_width, log_ideal_cube_width)

    # resolve center, radii if f_e
    next_sampler_state = CubesSamplerState(log_cube_width=log_cube_width)

    cube_width = jnp.exp(log_cube_width)
    points_lower = jnp.maximum(live_points_U - 0.5*cube_width, 0.)
    points_upper = jnp.minimum(live_points_U + 0.5*cube_width, 1.)

    log_Vp_i = jnp.sum(jnp.log(points_upper - points_lower), axis=1)

    def body(state):
        (key, i, u_test, x_test, log_L_test) = state
        def inner_body(inner_state):
            (key,accept, _) = inner_state
            key, sample_key, accept_key, select_key = random.split(key, 4)
            i = random.categorical(select_key,logits=log_Vp_i)
            mean_trials = jnp.exp(jnp.log(N) + D * jnp.log(cube_width))
            u_test = random.uniform(sample_key,
                                          shape=(10, D,),
                                          minval=points_lower[i,None,:],
                                          maxval=points_upper[i,None,:])
            n_intersect = vmap(lambda u_test: jnp.sum(vmap(lambda y_lower, y_upper: points_in_box(u_test, y_lower, y_upper))(points_lower, points_upper)))(u_test)
            accept = n_intersect * random.uniform(accept_key, shape=(10,)) < 1.
            u_test = u_test[jnp.argmax(accept),:]
            accept = jnp.any(accept)
            return (key, accept, u_test)
        (_key, accept, u_test) = while_loop(lambda inner_state: ~inner_state[1],
                                            inner_body,
                                            (key, jnp.array(False), u_test))
        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(**x_test)
        return (key, i + 1, u_test, x_test, log_L_test)

    (key, num_likelihood_evaluations, u_new, x_new, log_L_new) = while_loop(lambda state: state[-1] <= log_L_constraint,
                                                                            body,
                                                                            (key, 0, live_points_U[0, :],
                                                                             prior_transform(live_points_U[0, :]),
                                                                             log_L_constraint))

    CubesResults = namedtuple('CubesResults',
                                          ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new',
                                           'sampler_state'])
    return CubesResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new, next_sampler_state)
