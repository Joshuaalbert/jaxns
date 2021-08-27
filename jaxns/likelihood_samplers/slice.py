from collections import namedtuple
from jax import numpy as jnp, random, vmap
from jax.lax import scan
from jaxns.likelihood_samplers.common import SamplingResults
from jaxns.likelihood_samplers.slice_utils import slice_sample_1d, prepare_bayesian_opt
from jaxns.utils import random_ortho_matrix

SliceSamplerState = namedtuple('SliceSamplerState',
                               ['live_points'])



def init_slice_sampler_state(key, live_points_U, depth, log_X, num_slices, points_all_U, log_L_all):
    return SliceSamplerState(live_points=live_points_U)


def slice_sampling(key,
                   log_L_constraint,
                   init_U,
                   init_log_L,
                   num_slices,
                   log_likelihood_from_U,
                   sampler_state: SliceSamplerState):
    """
    Slice sampling from initial set of U (continuous) in the unit cube, with parallel likelihood evaluations using vmap.

    Note, speedups occur if the likelihood is composed of primitives with efficient vectorisation properties, like
    linear algebra ops.

    Args:
        key: PRNG key
        log_L_constraint: scalar or [S], log-likelihood level to slice above.
        init_U: [D] or [S,D], initial points to slice from.
        init_log_L: scalar or [S], log-likelihood at init_U.
        num_slices: int, number of transitions to take to decorrelate the samples
        log_likelihood_from_U: callable(u) that takes compact representation of U, and evaluates log-likelihood
        sampler_state: SliceSamplerState

    Returns: SamplingResults
    """

    D = init_U.shape[-1]
    # num_slices = D*L if num_slices % D == 0
    extra = num_slices % D
    L = num_slices // D
    if extra > 0:
        L += 1
    key, n_key = random.split(key, 2)

    R = vmap(lambda key: random_ortho_matrix(key, D).T)(random.split(n_key, L)).reshape((L*D,D))



    def slice_body(state, i):
        (key, num_f_eval0, u_current, logL_current) = state

        # key, n_key = random.split(key, 2)
        # n = random.normal(n_key, shape=u_current.shape)
        # n /= jnp.linalg.norm(n)
        n = R[i, :]

        # w = compute_init_interval_size(n, origin, u_current, sampler_state.mu, sampler_state.radii, sampler_state.rotation)
        w = jnp.inf

        (key, u_prop, log_L_prop, num_f_eval) = slice_sample_1d(key,
                                                                u_current,
                                                                logL_current,
                                                                n, w,
                                                                log_L_constraint,
                                                                log_likelihood_from_U,
                                                                sampler_state.live_points,
                                                                do_init_try_bracket=False,
                                                                do_stepout=False,
                                                                midpoint_shrink=False)




        return (key, num_f_eval0 + num_f_eval, u_prop, log_L_prop), ()

    (key, num_likelihood_evaluations, u_new, log_L_new), _ = scan(slice_body,
                                                                  (key, jnp.asarray(0), init_U, init_log_L),
                                                                  jnp.arange(num_slices))

    return SamplingResults(key, num_likelihood_evaluations, u_new, log_L_new)
