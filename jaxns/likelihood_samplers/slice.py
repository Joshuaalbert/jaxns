from collections import namedtuple
from jax import numpy as jnp, random
from jax.lax import scan
from jaxns.likelihood_samplers.ellipsoid_utils import ellipsoid_clustering
from jaxns.likelihood_samplers.slice_utils import slice_sample_1d, compute_init_interval_size

SliceSamplerState = namedtuple('SliceSamplerState',
                               ['cluster_id', 'mu', 'radii', 'rotation', 'num_k', 'num_fev_ma'])
SliceSamplingResults = namedtuple('SliceSamplingResults',
                                  ['key', 'num_likelihood_evaluations', 'u_new', 'log_L_new'])


def init_slice_sampler_state(key, live_points_U, depth, log_X, num_slices):
    cluster_id, (mu, radii, rotation) = ellipsoid_clustering(key, live_points_U, depth, log_X)
    num_k = jnp.bincount(cluster_id, minlength=0, length=mu.shape[0])
    return SliceSamplerState(cluster_id=cluster_id, mu=mu, radii=radii, rotation=rotation,
                             num_k=num_k, num_fev_ma=jnp.asarray(num_slices * live_points_U.shape[1] + 2.))


def slice_sampling(key,
                   log_L_constraint,
                   init_U,
                   num_slices,
                   log_likelihood_from_U,
                   sampler_state: SliceSamplerState):

    def slice_body(state, X):
        (key, num_f_eval0, u_current, _) = state

        key, n_key = random.split(key, 2)

        n = random.normal(n_key, shape=u_current.shape)
        n /= jnp.linalg.norm(n)
        origin = jnp.zeros(n.shape)

        # w = compute_init_interval_size(n, origin, u_current, sampler_state.mu, sampler_state.radii, sampler_state.rotation)
        w = jnp.inf

        (key, u_prop, log_L_prop, num_f_eval) = slice_sample_1d(key, u_current, n, w, log_L_constraint,
                                                                log_likelihood_from_U,
                                                                do_stepout=False, midpoint_shrink=False,
                                                                midpoint_shrink_thresh=100)

        return (key, num_f_eval0 + num_f_eval, u_prop, log_L_prop), ()

    (key, num_likelihood_evaluations, u_new, log_L_new), _ = scan(slice_body,
                                                                  (key, jnp.asarray(0), init_U, log_L_constraint),
                                                                  (jnp.arange(num_slices),),
                                                                  unroll=1)

    return SliceSamplingResults(key, num_likelihood_evaluations, u_new, log_L_new)
