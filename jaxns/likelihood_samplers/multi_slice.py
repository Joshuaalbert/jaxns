from collections import namedtuple
from jax import vmap, numpy as jnp, random
from jax.lax import while_loop, scan, cond
from jaxns.likelihood_samplers.ellipsoid_utils import ellipsoid_clustering, maha_ellipsoid, ellipsoid_to_circle, sample_box

MultiSliceSamplerState = namedtuple('MultiSliceSamplerState',
                                    ['cluster_id', 'mu', 'radii', 'rotation', 'num_k', 'num_fev_ma'])
MultiSliceSamplingResults = namedtuple('MultiSliceSamplingResults',
                                       ['key', 'num_likelihood_evaluations', 'u_new', 'log_L_new'])


def init_multi_slice_sampler_state(key, live_points_U, depth, log_X, num_slices):
    cluster_id, (mu, radii, rotation) = ellipsoid_clustering(key, live_points_U, depth, log_X)
    num_k = jnp.bincount(cluster_id, minlength=0, length=mu.shape[0])
    return MultiSliceSamplerState(cluster_id=cluster_id, mu=mu, radii=radii, rotation=rotation,
                                  num_k=num_k, num_fev_ma=jnp.asarray(num_slices * live_points_U.shape[1] + 2.))


def which_cluster(x, sampler_state):
    dist = vmap(lambda mu, radii, rotation: maha_ellipsoid(x, mu, radii, rotation))(sampler_state.mu,
                                                                                    sampler_state.radii,
                                                                                    sampler_state.rotation)
    return jnp.argmin(jnp.where(jnp.isnan(dist), jnp.inf, dist))
def multi_slice_sampling(key,
                         log_L_constraint,
                         init_U,
                         num_slices,
                         log_likelihood_from_U,
                         sampler_state: MultiSliceSamplerState):


    def slice_sample_multi(key, x):
        """
        Performs a single multi-dimensional slice sample from point x.

        Args:
            key: PRNG
            x: point to sample from
        """
        k = which_cluster(x, sampler_state)
        radii = sampler_state.radii[k]
        rotation = sampler_state.rotation[k]

        key, placement_key = random.split(key, 2)

        left = random.uniform(placement_key, shape=(radii.size,), minval=-2., maxval=0.)
        right = left + 2.

        def shrink_body(state):
            (_, num_f_eval, key, left, right, _, _) = state

            key, sample_key = random.split(key, 2)

            # growth = random.uniform(growth_key, shape=(radii.size,), minval=-0.5, maxval=0.5)
            base_point = random.uniform(sample_key, shape=(radii.size,), minval=left, maxval=right)
            u_test = (rotation @ jnp.diag(radii)) @ (base_point) + x

            logL_test = log_likelihood_from_U(u_test)
            done = logL_test > log_L_constraint

            left = jnp.where(base_point < 0, jnp.maximum(base_point, left), left)
            # left = jnp.where(jnp.arange(left.size) == jnp.argmax(jnp.abs(_left - left)), _left, left)
            right = jnp.where(base_point > 0, jnp.minimum(base_point, right), right)
            # right = jnp.where(jnp.arange(right.size) == jnp.argmax(jnp.abs(_right - right)), _right, right)

            return (done, num_f_eval + 1, key, left, right, u_test, logL_test)

        (done, num_f_eval, key, left, right, u_test, logL_test) = while_loop(lambda state: ~state[0],
                                                                    shrink_body,
                                                                    (jnp.asarray(False),
                                                                     jnp.zeros((), dtype=jnp.int_), key, left,
                                                                        right, x, log_L_constraint))

        return key, u_test, logL_test, num_f_eval

    def slice_body(state, X):
        (key, num_f_eval0, u_current, _) = state
        (key, u_prop, log_L_prop, num_f_eval) = slice_sample_multi(key, u_current)

        return (key, num_f_eval0 + num_f_eval, u_prop, log_L_prop), ()

    (key, num_likelihood_evaluations, u_new, log_L_new), _ = scan(slice_body,
                                                                  (key, jnp.asarray(0, dtype=jnp.int_), init_U, log_L_constraint),
                                                                  (jnp.arange(num_slices),),
                                                                  unroll=1)

    return MultiSliceSamplingResults(key, num_likelihood_evaluations, u_new, log_L_new)
