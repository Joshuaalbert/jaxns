from collections import namedtuple

from jaxns.likelihood_samplers.ellipsoid_utils import minimum_volume_enclosing_ellipsoid, sample_ellipsoid
from jax import numpy as jnp, vmap, random
from jax.lax import while_loop, cond
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln

EllipsoidSamplerState = namedtuple('EllipsoidSamplerState',
                                   ['mvee_u','i', 'radii', 'center', 'rotation'])


def init_ellipsoid_sampler_state(live_points_U):
    center, radii, rotation, mvee_u = minimum_volume_enclosing_ellipsoid(live_points_U, 0.01, return_u=True)
    return EllipsoidSamplerState(mvee_u=mvee_u, i=0, radii=radii, center=center, rotation=rotation)


def ellipsoid_sampler(key, log_L_constraint, live_points_U,
                      loglikelihood_from_constrained,
                      prior_transform, sampler_state, whiten=True):
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
    center, radii, rotation, next_mvee_u = minimum_volume_enclosing_ellipsoid(live_points_U, 0.01,
                                                                              init_u=sampler_state.mvee_u,
                                                                              return_u=True)
    # D = sampler_state.radii.size
    # log_mean_X = -sampler_state.i/points.shape_dict[0]
    # log_ellipsoid_volume = jnp.log(2.) - jnp.log(D) + 0.5*D * jnp.log(jnp.pi) - gammaln(0.5*D) + jnp.sum(jnp.log(sampler_state.radii))
    # #V(E)/V(X) = max(f_e, 1) because they already enclose the points so shouldn't be smaller.
    # log_expansion_factor = jnp.maximum(log_ellipsoid_volume - log_mean_X, 0.)
    # center, radii, rotation, next_mvee_u = cond(log_expansion_factor < jnp.log(1.1),
    #                                             lambda x:(sampler_state.center, sampler_state.radii,sampler_state.rotation, sampler_state.mvee_u),
    #                                             lambda x:minimum_volume_enclosing_ellipsoid(points, 0.01,
    #                                                                           init_u=sampler_state.mvee_u,
    #                                                                           return_u=True),
    #                                             None
    #                                             )
    # log_ellipsoid_volume = jnp.log(2.) - jnp.log(D) + 0.5 * D * jnp.log(jnp.pi) - gammaln(0.5 * D) + jnp.sum(
    #     jnp.log(radii))
    # # V(E)/V(X) = max(f_e, 1) because they already enclose the points so shouldn't be smaller.
    # log_expansion_factor = jnp.maximum(log_ellipsoid_volume - log_mean_X, 0.)
    #
    # # resolve center, radii if f_e
    next_sampler_state = EllipsoidSamplerState(mvee_u=next_mvee_u,i=sampler_state.i+1,center=center,
                                               radii=radii, rotation=rotation)
    # radii = radii * jnp.exp(log_expansion_factor / D)

    def body(state):
        (key, i, u_test, x_test, log_L_test) = state
        key, sample_key = random.split(key, 2)
        # V_i+1 = V_i * t
        # r_i+1^d = r_i^d * t
        # r_i+1 = r_i * t^1/d
        # t_shrink = random.beta(beta_key, points.shape_dict[0], 1) ** jnp.reciprocal(radii.size)
        u_test_white = sample_ellipsoid(sample_key, center, radii, rotation)
        u_test = u_test_white
        u_test = jnp.clip(u_test, 0., 1.)
        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(**x_test)
        return (key, i + 1, u_test, x_test, log_L_test)

    (key, num_likelihood_evaluations, u_new, x_new, log_L_new) = while_loop(lambda state: state[-1] <= log_L_constraint,
                                                                            body,
                                                                            (key, 0, live_points_U[0, :],
                                                                             prior_transform(live_points_U[0, :]),
                                                                             log_L_constraint))

    ExpandedEllipsoidResults = namedtuple('ExpandedEllipsoidResults',
                                          ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new',
                                           'sampler_state'])
    return ExpandedEllipsoidResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new, next_sampler_state)
