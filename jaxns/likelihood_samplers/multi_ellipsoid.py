from collections import namedtuple
import numpy as np
from jax import vmap, numpy as jnp, random
from jax.scipy.special import logsumexp
from jax.lax import while_loop, scan, cond, dynamic_update_slice
from jaxns.likelihood_samplers.ellipsoid_utils import ellipsoid_clustering, maha_ellipsoid, circle_to_ellipsoid, \
    ellipsoid_to_circle, log_ellipsoid_volume, sample_multi_ellipsoid

MultiEllipsoidSamplerState = namedtuple('MultiEllipsoidSamplerState',
                                        ['cluster_id', 'mu', 'radii', 'rotation', 'num_k', 'num_fev_ma'])


def init_multi_ellipsoid_sampler_state(key, live_points_U, depth, log_X):
    cluster_id, (mu, radii, rotation) = ellipsoid_clustering(key, live_points_U, depth, log_X)
    num_k = jnp.bincount(cluster_id, minlength=0, length=mu.shape[0])
    return MultiEllipsoidSamplerState(cluster_id=cluster_id, mu=mu, radii=radii, rotation=rotation,
                                      num_k=num_k, num_fev_ma=jnp.asarray(live_points_U.shape[1] + 2.))


def recalculate_sampler_state(key, live_points_U, sampler_state, log_X):
    num_clusters = sampler_state.mu.shape[0]  # 2^(depth-1)
    depth = np.log2(num_clusters) + 1
    depth = depth.astype(np.int_)
    return init_multi_ellipsoid_sampler_state(key, live_points_U, depth, log_X)._replace(
        num_fev_ma=sampler_state.num_fev_ma)


def evolve_sampler_state(sampler_state, N, D):
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
    new_radii = jnp.exp(jnp.log(sampler_state.radii)
                        + (jnp.log(N) - jnp.log(N + 1.)) / D)
    sampler_state = sampler_state._replace(radii=new_radii)
    return sampler_state


def multi_ellipsoid_sampler(key,
                            log_L_constraint,
                            live_points_U,
                            loglikelihood_from_constrained,
                            prior_transform,
                            log_X,
                            sampler_state: MultiEllipsoidSamplerState):
    N, D = live_points_U.shape

    def while_body(state):
        (key, num_f_eval0, _, _, _) = state
        key, sample_key = random.split(key)
        k_select, u_test = sample_multi_ellipsoid(sample_key,
                                                  sampler_state.mu, sampler_state.radii, sampler_state.rotation,
                                                  unit_cube_constraint=True)
        num_f_eval = 1
        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(**x_test)

        return (key, num_f_eval0 + num_f_eval, u_test, x_test, log_L_test)

    u_init = live_points_U[0, :]

    (key, num_likelihood_evaluations, u_new, x_new, log_L_new) = while_loop(lambda s: s[-1] <= log_L_constraint,
                                                                            while_body,
                                                                            (key, jnp.asarray(0), u_init,
                                                                             prior_transform(u_init), log_L_constraint))

    log_volumes = vmap(lambda radii: log_ellipsoid_volume(radii))(sampler_state.radii)
    log_F = logsumexp(log_volumes) - log_X

    # V(E_k) > 2 V(S_k)
    # |S_k| < D+1
    too_many_func_evals = D * 2  # step out and a shrink per dimension
    do_recalculate = (num_likelihood_evaluations > sampler_state.num_fev_ma + too_many_func_evals) \
                     | (log_F < 0.)

    tau = 1. / N
    sampler_state = sampler_state._replace(
        num_fev_ma=sampler_state.num_fev_ma * (1. - tau) + tau * num_likelihood_evaluations)

    key, recalc_key = random.split(key, 2)

    sampler_state = cond(do_recalculate,
                         lambda args: recalculate_sampler_state(*args),
                         lambda _: sampler_state,# evolve_sampler_state(sampler_state, N, D),
                         (recalc_key, live_points_U, sampler_state, log_X))

    # if do_recalculate:
    #     plot(-1,1)

    MultiEllipsoidSamplingResults = namedtuple('MultiEllipsoidSamplingResults',
                                               ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new',
                                                'sampler_state'])
    return MultiEllipsoidSamplingResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new, sampler_state)
