from collections import namedtuple
from jaxns.likelihood_samplers.ellipsoid_utils import log_ellipsoid_volume, ellipsoid_clustering, sample_multi_ellipsoid
from jax import numpy as jnp, vmap, random
from jax.lax import while_loop, cond, dynamic_update_slice
from jax.scipy.special import logsumexp
import numpy as np

MultiEllipsoidSamplerState = namedtuple('MultiEllipsoidSamplerState',
                                        ['cluster_id', 'mu', 'radii', 'rotation', 'num_k', 'num_fev_ma'])


def init_multi_ellipsoid_sampler_state(key, live_points_U, depth, log_X):
    cluster_id, (mu, radii, rotation) = ellipsoid_clustering(key, live_points_U, depth, log_X)
    num_k = jnp.bincount(cluster_id, minlength=0, length=mu.shape[0])
    return MultiEllipsoidSamplerState(cluster_id=cluster_id, mu=mu, radii=radii, rotation=rotation,
                                      num_k=num_k, num_fev_ma=jnp.asarray(1.))


def recalculate_sampler_state(key, live_points_U, sampler_state, log_X):
    num_clusters = sampler_state.mu.shape[0]  # 2^(depth-1)
    depth = np.log2(num_clusters) + 1
    depth = depth.astype(np.int_)
    return init_multi_ellipsoid_sampler_state(key, live_points_U, depth, log_X)._replace(num_fev_ma=sampler_state.num_fev_ma)


def evolve_sampler_state(sampler_state, live_points_U):
    """
    Evolve all ellipsoids by shrinking each ellipsoid by n/(n+1)
    V(S) = sum_k V(S_k)
         = (sum_k n_k/n) V(S)
         = (sum_k n_k/n) sum_k V(E_k)
    V(S) -> V(S) n/(n+1)
    => sum_k V(E_k) -> (sum_k V(E_k)) n/(n+1)
    Args:
        sampler_state:
        num_live_points:
        log_X:
        k_from:

    Returns:

    """
    N, D = live_points_U.shape
    new_radii = jnp.exp(jnp.log(sampler_state.radii)
                        + (jnp.log(N) - jnp.log(N + 1.)) / D)
    sampler_state = sampler_state._replace(radii=new_radii)
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
    sampler_state = evolve_sampler_state(sampler_state, live_points_U)
    scale = 1.1**(1./D)

    def body(state):
        (key, i, _, u_test, x_test, log_L_test) = state
        key, sample_key = random.split(key, 2)
        k, u_test = sample_multi_ellipsoid(sample_key,
                                           sampler_state.mu,
                                           sampler_state.radii * scale,
                                           sampler_state.rotation,
                                           unit_cube_constraint=True)

        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(**x_test)
        return (key, i + 1, k, u_test, x_test, log_L_test)

    (key, num_likelihood_evaluations, ellipsoid_select, u_new, x_new, log_L_new) = while_loop(
        lambda state: state[-1] <= log_L_constraint,
        body,
        (key, 0, 0, live_points_U[0, :],
         prior_transform(live_points_U[0, :]),
         log_L_constraint))

    cluster_id = dynamic_update_slice(sampler_state.cluster_id,
                                      ellipsoid_select[None],
                                      i_min[None])

    num_k = dynamic_update_slice(sampler_state.num_k,
                                 sampler_state.num_k[ellipsoid_select, None] + 1,
                                 ellipsoid_select[None])
    sampler_state = sampler_state._replace(cluster_id=cluster_id, num_k=num_k)

    log_volumes = vmap(lambda radii: log_ellipsoid_volume(radii))(sampler_state.radii)
    log_F = logsumexp(log_volumes) - log_X

    # V(E_k) > 2 V(S_k)
    # |S_k| < D+1
    # jnp.any(log_volumes > jnp.log(sampler_state.num_k) - jnp.log(N) + log_X + jnp.log(2.))
    do_recalculate =  jnp.any(sampler_state.num_k == D) \
                    | (num_likelihood_evaluations > 3. * sampler_state.num_fev_ma) \
                    | (log_F < 0.)
    tau = 1./N
    sampler_state = sampler_state._replace(num_fev_ma=sampler_state.num_fev_ma*(1. - tau) + tau * num_likelihood_evaluations)

    # print('do_recalculate', do_recalculate, 'num fev', num_likelihood_evaluations, '/',sampler_state.num_fev_ma,'V(E)/V(S)', jnp.exp(log_F), 'V(E_k)', jnp.exp(log_volumes), '2V(S_k)',jnp.exp(jnp.log(sampler_state.num_k) - jnp.log(N) + log_X + jnp.log(2.)))


    key, recalc_key = random.split(key, 2)

    sampler_state = cond(do_recalculate,
                         lambda args: recalculate_sampler_state(*args),
                         lambda _: sampler_state,
                         (recalc_key, live_points_U, sampler_state, log_X))

    MultiEllipsoidResults = namedtuple('MultiEllipsoidResults',
                                       ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new',
                                        'sampler_state'])
    return MultiEllipsoidResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new, sampler_state)
