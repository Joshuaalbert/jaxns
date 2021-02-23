from collections import namedtuple
from jax import numpy as jnp, random
from jax.lax import while_loop
from jaxns.likelihood_samplers.ellipsoid_utils import ellipsoid_clustering, sample_multi_ellipsoid

MultiEllipsoidSamplerState = namedtuple('MultiEllipsoidSamplerState',
                                        ['cluster_id', 'mu', 'radii', 'rotation', 'num_k', 'num_fev_ma'])
MultiEllipsoidSamplingResults = namedtuple('MultiEllipsoidSamplingResults',
                                           ['key', 'num_likelihood_evaluations', 'u_new', 'log_L_new'])


def init_multi_ellipsoid_sampler_state(key, live_points_U, depth, log_X):
    cluster_id, (mu, radii, rotation) = ellipsoid_clustering(key, live_points_U, depth, log_X)
    num_k = jnp.bincount(cluster_id, minlength=0, length=mu.shape[0])
    return MultiEllipsoidSamplerState(cluster_id=cluster_id, mu=mu, radii=radii, rotation=rotation,
                                      num_k=num_k, num_fev_ma=jnp.asarray(live_points_U.shape[1] + 2.))


def multi_ellipsoid_sampler(key,
                            log_L_constraint,
                            log_likelihood_from_U,
                            sampler_state: MultiEllipsoidSamplerState):
    def while_body(state):
        (key, num_f_eval0, _, _, _) = state
        key, sample_key = random.split(key)
        k_select, u_test = sample_multi_ellipsoid(sample_key,
                                                  sampler_state.mu, sampler_state.radii, sampler_state.rotation,
                                                  unit_cube_constraint=True)
        num_f_eval = 1
        log_L_test = log_likelihood_from_U(u_test)

        return (key, num_f_eval0 + num_f_eval, u_test, log_L_test)

    u_init = sampler_state.mu[0]

    (key, num_likelihood_evaluations, u_new, log_L_new) = while_loop(lambda s: s[-1] <= log_L_constraint,
                                                                     while_body,
                                                                     (key, jnp.asarray(0), u_init, log_L_constraint))

    return MultiEllipsoidSamplingResults(key, num_likelihood_evaluations, u_new, log_L_new)
