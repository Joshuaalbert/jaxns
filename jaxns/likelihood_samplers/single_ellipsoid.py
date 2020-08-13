from collections import namedtuple

from jaxns.utils import minimum_volume_enclosing_ellipsoid, sample_ellipsoid
from jax import numpy as jnp, vmap, random
from jax.lax import while_loop
from jax.scipy.linalg import solve_triangular

EllipsoidSamplerState = namedtuple('EllipsoidSamplerState',
                                   ['mvee_u'])

def init_ellipsoid_sampler_state(num_live_points, whiten=True):
    mvee_u = jnp.ones(num_live_points)/num_live_points
    return EllipsoidSamplerState(mvee_u=mvee_u)

def expanded_ellipsoid(key, log_L_constraint, live_points_U,
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
    if whiten:
        u_mean = jnp.mean(live_points_U, axis=0)
        L = jnp.linalg.cholesky(jnp.cov(live_points_U, rowvar=False, bias=True))
        live_points_U = vmap(lambda u: solve_triangular(L, u, lower=True))(live_points_U - u_mean)
    center, radii, rotation, next_mvee_u = minimum_volume_enclosing_ellipsoid(live_points_U, 0.01,
                                                                              init_u=sampler_state.mvee_u,
                                                                              return_u=True)
    # t_expand_mean = live_points_U.shape[0]/(live_points_U.shape[0] + 1)
    # print(jnp.prod(radii))
    # radii = 1.2*radii / t_expand_mean**(1./radii.size)
    # import pylab as plt
    # theta = jnp.linspace(0., jnp.pi * 2, 100)
    # ellipsis = center[:, None] + rotation @ jnp.stack([radii[0] * jnp.cos(theta), radii[1] * jnp.sin(theta), radii[2]*jnp.zeros_like(theta)], axis=0)
    # plt.plot(ellipsis[0, :], ellipsis[1, :])
    # plt.scatter(live_points_U[:, 0], live_points_U[:, 1])
    # plt.show()

    def body(state):
        (key, i, u_test, x_test, log_L_test) = state
        key, sample_key, beta_key = random.split(key, 3)
        # V_i+1 = V_i * t
        # r_i+1^d = r_i^d * t
        # r_i+1 = r_i * t^1/d
        t_shrink = random.beta(beta_key, live_points_U.shape[0], 1) ** jnp.reciprocal(radii.size)
        u_test_white = sample_ellipsoid(sample_key, center, radii / t_shrink, rotation)
        if whiten:
            u_test = L @ u_test_white + u_mean
        else:
            u_test = u_test_white
        u_test = jnp.clip(u_test, 0., 1.)
        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(**x_test)
        return (key, i + 1, u_test, x_test, log_L_test)

    (key, num_likelihood_evaluations, u_new, x_new, log_L_new) = while_loop(lambda state: state[-1] <= log_L_constraint,
                                                                            body,
                                                                            (key, 0, live_points_U[0,:],
                                                                             prior_transform(live_points_U[0,:]),
                                                                             log_L_constraint))

    ExpandedEllipsoidResults = namedtuple('ExpandedEllipsoidResults',
                                      ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new', 'sampler_state'])
    return ExpandedEllipsoidResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new, EllipsoidSamplerState(mvee_u=next_mvee_u))

