from collections import namedtuple

from jaxns.likelihood_samplers.cubes_utils import cubes_intersect_volume, point_in_cube
from jax import numpy as jnp, vmap, random, jit
from jax.lax import while_loop
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln, logsumexp
from functools import partial

CubesSamplerState = namedtuple('CubesSamplerState',
                                   [])


def init_cubes_sampler_state():
    return CubesSamplerState()

def test_cubes():
    from jax import disable_jit
    from jaxns.prior_transforms import UniformPrior, PriorChain
    log_L_constraint = jnp.array(-jnp.inf)
    prior = PriorChain().push(UniformPrior('x', jnp.array([0.]), jnp.array([1.])))
    N = 10
    live_points_U = random.uniform(random.PRNGKey(1),shape=(N, prior.U_ndims))
    def log_likelihood(x,**kwargs):
        return jnp.sum(x)
    key = random.PRNGKey(0)
    sampler_state = init_cubes_sampler_state()


    def body(state):
        (i,log_L_constraint, live_points_U,sampler_state, log_X) = state
        sampler_state = cubes(key, log_L_constraint, live_points_U, log_likelihood, prior, sampler_state, log_X)

        return (i+1,log_L_constraint*jnp.array(1.), live_points_U*jnp.array(1.),sampler_state, log_X-0.01)

    state = while_loop(lambda state: state[0]<3,
                       body,
                       (0, log_L_constraint, live_points_U,sampler_state, jnp.array(0.)))
    print(state)


@partial(jit, static_argnums=[3,4])
def cubes(key, log_L_constraint, live_points_U,
                       loglikelihood_from_constrained,
                       prior_transform, sampler_state, log_mean_X):
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

    N,D = live_points_U.shape
    # log_mean_X = -sampler_state.i/live_points_U.shape[0]

    # cube_volume = X/N
    # cube_size = (X/N)**(1./D)
    log_cube_volume = log_mean_X - jnp.log(N)
    log_cube_size = log_cube_volume/D
    # def body(state):
    #     (i, log_cube_size,) = state
    #     log_cube_half_width = log_cube_size - jnp.log(2.)
    #     intersection_volume = vmap(
    #         lambda x: vmap(
    #             lambda y: cubes_intersect_volume(x,y,jnp.exp(log_cube_half_width)))(live_points_U))(live_points_U)
    #     log_intersection_volume = logsumexp(intersection_volume * jnp.tri(intersection_volume.shape[0], k=-1))
    #     # X = N * (f * l_n)**D - intersection_volume_n
    #     # log(f * l_n)= (log(X + I_n) - log(N)) / D
    #     log_cube_size_next = (jnp.logaddexp(log_mean_X, log_intersection_volume) - jnp.log(N))/D
    #     return (i+1, log_cube_size_next)
    #
    # _, log_cube_size = while_loop(lambda state: state[0] < 3,
    #                               body,
    #                               (0, log_cube_size))
    log_cube_half_width = log_cube_size - jnp.log(2.)
    # resolve center, radii if f_e
    next_sampler_state = CubesSamplerState()

    def body(state):
        (key, i, u_test, x_test, log_L_test) = state
        key, sample_key, select_key, accept_key = random.split(key, 4)
        i = random.categorical(select_key,logits=jnp.zeros(N))
        u_test = random.uniform(sample_key,
                                      shape=live_points_U.shape[1:],
                                      minval=live_points_U[i,:] - jnp.exp(log_cube_half_width),
                                      maxval=live_points_U[i,:] + jnp.exp(log_cube_half_width))
        n_intersect = jnp.sum(vmap(lambda cube: point_in_cube(u_test, cube, jnp.exp(log_cube_half_width) ))(live_points_U))
        accept = n_intersect * random.uniform(accept_key) < 1.
        u_test = jnp.clip(u_test, 0., 1.)
        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(**x_test)
        log_L_test = jnp.where(accept, log_L_test, -jnp.inf)
        return (key, i + 1, u_test, x_test, log_L_test)

    (key, num_likelihood_evaluations, u_new, x_new, log_L_new) = while_loop(lambda state: state[-1] <= log_L_constraint,
                                                                            body,
                                                                            (key, 0, live_points_U[0, :],
                                                                             prior_transform(live_points_U[0, :]),
                                                                             log_L_constraint))

    CubesResults = namedtuple('CubesResults',
                                          ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new',
                                           'sampler_state'])
    return CubesResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new, next_sampler_state)
