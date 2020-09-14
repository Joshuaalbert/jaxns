from collections import namedtuple

from jaxns.likelihood_samplers.ellipsoid_utils import minimum_volume_enclosing_ellipsoid, sample_ellipsoid, \
    log_ellipsoid_volume, ellipsoid_clustering, log_coverage_scale, sample_multi_ellipsoid
from jax import numpy as jnp, vmap, random
from jax.lax import while_loop, cond, dynamic_update_slice
from jax.scipy.special import logsumexp
import pylab as plt

MultiEllipsoidSamplerState = namedtuple('MultiEllipsoidSamplerState',
                                        ['cluster_id', 'mu', 'radii', 'rotation'])


def init_multi_ellipsoid_sampler_state(key, live_points_U, depth, log_X):
    cluster_id, (mu, radii, rotation) = ellipsoid_clustering(key, live_points_U, depth, log_X)
    return MultiEllipsoidSamplerState(cluster_id=cluster_id, mu=mu, radii=radii, rotation=rotation)


def recalculate_sampler_state(key, live_points_U, sampler_state, log_X):
    print('recalculating ellipsoids')
    num_clusters = sampler_state.mu.shape[0]  # 2^(depth-1)
    depth = jnp.log2(num_clusters) + 1
    depth = depth.astype(jnp.int_)
    cluster_id, (mu, radii, rotation) = ellipsoid_clustering(key, live_points_U, depth, log_X)
    return MultiEllipsoidSamplerState(cluster_id=cluster_id, mu=mu, radii=radii, rotation=rotation)


def evolve_sampler_state(sampler_state, num_live_points, log_X, k_from):
    print('evolving ellipsoids')
    # E_from radii -> radii * t_k**(1/D)
    D = sampler_state.radii.shape[1]
    num_k = jnp.bincount(sampler_state.cluster_id, minlength=0, length=sampler_state.mu.shape[0])
    radii_shrunk = jnp.exp(jnp.log(sampler_state.radii[k_from,:])
                           + (jnp.log(num_k[k_from])  - jnp.log(num_k[k_from])/D + 1.))

    log_scale = log_coverage_scale(log_ellipsoid_volume(radii_shrunk),
                       log_X + jnp.log(num_k[k_from]) - jnp.log(num_live_points + 1),
                       D)
    radii_shrunk = jnp.exp(jnp.log(radii_shrunk) + log_scale)
    new_radii = dynamic_update_slice(sampler_state.radii, radii_shrunk[None, :], [k_from, 0])
    sampler_state = sampler_state._replace(radii = new_radii)
    return sampler_state


def multi_ellipsoid_sampler(key, log_L_constraint, live_points_U,
                            loglikelihood_from_constrained,
                            prior_transform, sampler_state, log_X, iteration, i_min):
    """
    Does multi-nest sampling.

    Order of updating:
    Start program: initialise ellipsoids -> E_0, cluster_id0
    Loop
        find i_min
        sample new point
            replace ellipsoids or evolve ellipsoids
            draw new point



    https://arxiv.org/pdf/0809.3437.pdf

    Args:
        key:
        log_L_constraint:
        live_points_U:
        loglikelihood_from_constrained:

    Returns:

    """
    N, D = live_points_U.shape
    ###
    # evolve ellipsoids or potentially recalculate ellipsoids
    h = 1.1
    log_volumes = vmap(lambda radii: log_ellipsoid_volume(radii))(sampler_state.radii)
    do_recalculate = (logsumexp(log_volumes) - log_X > jnp.log(h)) | (jnp.mod(iteration, live_points_U.shape[0]) == 0)
    print(jnp.exp(logsumexp(log_volumes) - log_X))
    key, recalc_key = random.split(key, 2)
    # sampler_state = cond(do_recalculate,
    #                      (recalc_key, live_points_U, sampler_state, log_X), lambda arg: recalculate_sampler_state(*arg),
    #                      (sampler_state, N, log_X, sampler_state.cluster_id[i_min]), lambda arg: evolve_sampler_state(*arg))
    sampler_state =  recalculate_sampler_state(recalc_key, live_points_U, sampler_state, log_X)

    def body(state):
        (key, i, _, u_test, x_test, log_L_test) = state
        key, sample_key = random.split(key, 2)
        print(i, 'Sampling multi ellipsoid', sampler_state.mu, sampler_state.radii)
        # plt.scatter(live_points_U[:, 0], live_points_U[:, 1])
        # plt.show()


        # plt.scatter(live_points_U[:, 0], live_points_U[:, 1])
        #
        # theta = jnp.linspace(0., jnp.pi * 2, 100)
        # x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
        # for i, (mu, radii, rotation) in enumerate(zip(sampler_state.mu, sampler_state.radii, sampler_state.rotation)):
        #     y = mu[:, None] + rotation @ jnp.diag(radii) @ x
        #     plt.plot(y[0, :], y[1, :])
        #     mask = sampler_state.cluster_id == i
        #     plt.scatter(live_points_U[mask, 0], live_points_U[mask, 1], c=plt.cm.jet(i / sampler_state.mu.shape[0]))

        k, u_test = sample_multi_ellipsoid(sample_key, sampler_state.mu, sampler_state.radii, sampler_state.rotation,
                                           unit_cube_constraint=True)
        # plt.scatter(u_test[0], u_test[1], c='red')
        # plt.show()
        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(**x_test)
        return (key, i + 1, k, u_test, x_test, log_L_test)

    (key, num_likelihood_evaluations, ellipsoid_select, u_new, x_new, log_L_new) = while_loop(
        lambda state: state[-1] <= log_L_constraint,
        body,
        (key, 0, 0, live_points_U[0, :],
         prior_transform(live_points_U[0, :]),
         log_L_constraint))

    cluster_id = dynamic_update_slice(sampler_state.cluster_id, jnp.asarray([ellipsoid_select]), jnp.asarray([i_min]))
    sampler_state = sampler_state._replace(cluster_id=cluster_id)

    MultiEllipsoidResults = namedtuple('MultiEllipsoidResults',
                                       ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new',
                                        'sampler_state'])
    return MultiEllipsoidResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new, sampler_state)
