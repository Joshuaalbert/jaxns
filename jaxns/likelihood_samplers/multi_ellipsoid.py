from collections import namedtuple

from jaxns.likelihood_samplers.ellipsoid_utils import minimum_volume_enclosing_ellipsoid, sample_ellipsoid, \
    log_ellipsoid_volume, ellipsoid_clustering, log_coverage_scale, sample_multi_ellipsoid, maha_ellipsoid
from jax import numpy as jnp, vmap, random
from jax.lax import while_loop, cond, dynamic_update_slice
from jax.scipy.special import logsumexp
import numpy as np

MultiEllipsoidSamplerState = namedtuple('MultiEllipsoidSamplerState',
                                        ['cluster_id', 'mu', 'radii', 'rotation', 'num_k', 'log_F'])


def init_multi_ellipsoid_sampler_state(key, live_points_U, depth, log_X):
    cluster_id, (mu, radii, rotation) = ellipsoid_clustering(key, live_points_U, depth, log_X)
    num_k = jnp.bincount(cluster_id, minlength=0, length=mu.shape[0])
    log_volumes = vmap(lambda radii: log_ellipsoid_volume(radii))(radii)
    log_F = logsumexp(log_volumes) - log_X
    return MultiEllipsoidSamplerState(cluster_id=cluster_id, mu=mu, radii=radii, rotation=rotation,
                                      num_k=num_k, log_F=log_F)


def recalculate_sampler_state(key, live_points_U, sampler_state, log_X):
    # print('recalculating ellipsoids')
    num_clusters = sampler_state.mu.shape[0]  # 2^(depth-1)
    depth = np.log2(num_clusters) + 1
    depth = depth.astype(np.int_)
    cluster_id, (mu, radii, rotation) = ellipsoid_clustering(key, live_points_U, depth, log_X)
    num_k = jnp.bincount(cluster_id, minlength=0, length=mu.shape[0])
    log_volumes = vmap(lambda radii: log_ellipsoid_volume(radii))(radii)
    log_F = logsumexp(log_volumes) - log_X
    # print(num_k)
    # if jnp.any(jnp.isnan(radii)):
    #     print('recalc error',MultiEllipsoidSamplerState(cluster_id=cluster_id, mu=mu, radii=radii, rotation=rotation,
    #                                   num_k=num_k))
    #     raise ValueError()
    return MultiEllipsoidSamplerState(cluster_id=cluster_id, mu=mu, radii=radii, rotation=rotation,
                                      num_k=num_k, log_F=log_F)


def evolve_sampler_state(sampler_state, live_points_U, log_X, k_from):
    """
    Evolve ellipsoid k_from by scaling the radii such that V(E_k) = max(V(E_k), V(S_k))
    where E_k is chosen such that E_k bounds the points in cluster k_from.
    Args:
        sampler_state:
        num_live_points:
        log_X:
        k_from:

    Returns:

    """
    N, D = live_points_U.shape
    # scale ellipsoid to bound points
    maha = vmap(lambda u: maha_ellipsoid(u, sampler_state.mu[k_from, :], sampler_state.radii[k_from, :],
                                         sampler_state.rotation[k_from, :, :]))(live_points_U)
    mask = sampler_state.cluster_id == k_from
    maha = jnp.where(mask, maha, 0.)
    log_bound_scale = jnp.log(jnp.max(maha)) / D
    radii_bound = jnp.exp(jnp.log(sampler_state.radii[k_from,:]) + log_bound_scale)
    # make sure it's at least as big as the volume of S_k
    log_S_scale = log_coverage_scale(log_ellipsoid_volume(radii_bound),
                                   log_X + jnp.log(sampler_state.num_k[k_from]) - jnp.log(N + 1),
                                   D)
    radii = jnp.exp(jnp.log(radii_bound) + log_S_scale)
    new_radii = dynamic_update_slice(sampler_state.radii, radii[None, :], [k_from, 0])
    sampler_state = sampler_state._replace(radii=new_radii)
    # if jnp.any(jnp.isnan(sampler_state.radii)):
    #     print('Evolve error',
    #           radii_bound,
    #           log_bound_scale,
    #           radii, log_S_scale, log_X, sampler_state.num_k[k_from],
    #           log_ellipsoid_volume(radii_bound),
    #           sampler_state, maha_)
    #     raise ValueError()
    return sampler_state


def multi_ellipsoid_sampler(key, log_L_constraint, live_points_U,
                            loglikelihood_from_constrained,
                            prior_transform, sampler_state, log_X, iteration, i_min):
    """
    Does iterative multi-nest sampling with a few extra features to improve over the original algorithm.

    References:

    [1] MULTINEST: an efficient and robust Bayesian inference tool for cosmology and particle physics,
        F. Feroz et al. 2008. https://arxiv.org/pdf/0809.3437.pdf

    Args:
        key:
        log_L_constraint:
        live_points_U:
        loglikelihood_from_constrained:

    Returns:

    """
    N, D = live_points_U.shape

    # subtract the i_min
    k_from = sampler_state.cluster_id[i_min]
    num_k = dynamic_update_slice(sampler_state.num_k, sampler_state.num_k[k_from, None] - 1, k_from[None])
    sampler_state = sampler_state._replace(num_k=num_k)
    ###
    # evolve ellipsoids or potentially recalculate ellipsoids
    h = 1.2
    log_volumes = vmap(lambda radii: log_ellipsoid_volume(radii))(sampler_state.radii)
    log_F = logsumexp(log_volumes) - log_X
    # print(jnp.exp(log_F), jnp.exp(sampler_state.log_F), jnp.exp(log_F - sampler_state.log_F))
    do_recalculate = (log_F - sampler_state.log_F > jnp.log(h)) \
                     | (sampler_state.num_k[k_from] < D + 1) \
                     | (jnp.mod(iteration+1, live_points_U.shape[0]) == 0) \

    key, recalc_key = random.split(key, 2)
    sampler_state = cond(do_recalculate,
                         (recalc_key, live_points_U, sampler_state, log_X),
                         lambda arg: recalculate_sampler_state(*arg),
                         (sampler_state, live_points_U, log_X, k_from),
                         lambda arg: evolve_sampler_state(*arg))

    def body(state):
        (key, i, _, u_test, x_test, log_L_test) = state
        key, sample_key = random.split(key, 2)
        # print(i, 'Sampling multi ellipsoid', sampler_state.mu, sampler_state.radii)


        k, u_test = sample_multi_ellipsoid(sample_key, sampler_state.mu, sampler_state.radii, sampler_state.rotation,
                                           unit_cube_constraint=True)

        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(**x_test)
        return (key, i + 1, k, u_test, x_test, log_L_test)

    # plt.scatter(live_points_U[:, 0], live_points_U[:, 1])
    # theta = jnp.linspace(0., jnp.pi * 2, 100)
    # x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    # for i, (mu, radii, rotation) in enumerate(zip(sampler_state.mu, sampler_state.radii, sampler_state.rotation)):
    #     y = mu[:, None] + rotation @ jnp.diag(radii) @ x
    #     plt.plot(y[0, :], y[1, :])
    #     mask = sampler_state.cluster_id == i
    #     plt.scatter(live_points_U[mask, 0], live_points_U[mask, 1])
    # plt.savefig('/home/albert/git/jaxns/debug_figs/fig_{:04d}.png'.format(
    #     len(glob.glob('/home/albert/git/jaxns/debug_figs/fig_*.png'))))
    # plt.close('all')


    (key, num_likelihood_evaluations, ellipsoid_select, u_new, x_new, log_L_new) = while_loop(
        lambda state: state[-1] <= log_L_constraint,
        body,
        (key, 0, 0, live_points_U[0, :],
         prior_transform(live_points_U[0, :]),
         log_L_constraint))

    # plt.scatter(live_points_U[:, 0], live_points_U[:, 1])
    # theta = jnp.linspace(0., jnp.pi * 2, 100)
    # x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    # for i, (mu, radii, rotation) in enumerate(zip(sampler_state.mu, sampler_state.radii, sampler_state.rotation)):
    #     y = mu[:, None] + rotation @ jnp.diag(radii) @ x
    #     plt.plot(y[0, :], y[1, :])
    #     mask = sampler_state.cluster_id == i
    #     plt.scatter(live_points_U[mask, 0], live_points_U[mask, 1])
    # plt.scatter(u_new[0], u_new[1], c='red')
    # plt.savefig('/home/albert/git/jaxns/debug_figs/fig_{:04d}.png'.format(
    #     len(glob.glob('/home/albert/git/jaxns/debug_figs/fig_*.png'))))
    # plt.close('all')

    cluster_id = dynamic_update_slice(sampler_state.cluster_id, ellipsoid_select[None], i_min[None])
    num_k = dynamic_update_slice(sampler_state.num_k, sampler_state.num_k[ellipsoid_select, None] + 1,
                                 ellipsoid_select[None])
    sampler_state = sampler_state._replace(cluster_id=cluster_id, num_k=num_k)

    MultiEllipsoidResults = namedtuple('MultiEllipsoidResults',
                                       ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new',
                                        'sampler_state'])
    return MultiEllipsoidResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new, sampler_state)
