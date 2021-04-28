from collections import namedtuple
from jax import numpy as jnp, random
from jax.lax import scan
from jaxns.likelihood_samplers.common import SamplingResults
from jaxns.likelihood_samplers.slice_utils import slice_sample_1d

SliceSamplerState = namedtuple('SliceSamplerState',
                               [])



def init_slice_sampler_state(key, live_points_U, depth, log_X, num_slices):
    return SliceSamplerState()


def slice_sampling(key,
                   log_L_constraint,
                   init_U,
                   init_log_L,
                   num_slices,
                   log_likelihood_from_U,
                   sampler_state: SliceSamplerState):

    def slice_body(state, X):
        (key, num_f_eval0, u_current, logL_current) = state

        key, n_key = random.split(key, 2)

        n = random.normal(n_key, shape=u_current.shape)
        n /= jnp.linalg.norm(n)

        # w = compute_init_interval_size(n, origin, u_current, sampler_state.mu, sampler_state.radii, sampler_state.rotation)
        w = jnp.inf

        (key, u_prop, log_L_prop, num_f_eval) = slice_sample_1d(key, u_current, logL_current, n, w, log_L_constraint,
                                                                log_likelihood_from_U,
                                                                do_stepout=False, midpoint_shrink=True)

        return (key, num_f_eval0 + num_f_eval, u_prop, log_L_prop), ()

    (key, num_likelihood_evaluations, u_new, log_L_new), _ = scan(slice_body,
                                                                  (key, jnp.asarray(0), init_U, init_log_L),
                                                                  (jnp.arange(num_slices),),
                                                                  unroll=1)

    return SamplingResults(key, num_likelihood_evaluations, u_new, log_L_new)
