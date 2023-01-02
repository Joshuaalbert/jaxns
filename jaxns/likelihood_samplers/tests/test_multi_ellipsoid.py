import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp, random, tree_map, disable_jit

from jaxns import PriorModelGen, Prior, Model
from jaxns.initial_state import get_uniform_init_live_points
from jaxns.likelihood_samplers.multi_ellipsoid_utils import log_ellipsoid_volume, ellipsoid_clustering
from jaxns.types import float_type

tfpd = tfp.distributions


def test_ellipsoid_clustering():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2))
        y = yield Prior(tfpd.Normal(loc=0, scale=2))

        return x, y

    def log_likelihood(x,y):
        return jnp.log(jnp.exp(-0.5*((x-0.5)/0.1)**2-0.5*((y-0.5)/0.1)**2) + jnp.exp(-0.5*((x-1.5)/0.1)**2-0.5*((y-1.5)/0.1)**2))
    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    n = 1000
    live_points = get_uniform_init_live_points(random.PRNGKey(43),
                                               num_live_points=n,
                                               model=model)
    keep = live_points.reservoir.log_L > log_likelihood(1.1, 1.1)
    reservoir = tree_map(lambda x: x[keep], live_points.reservoir)
    # import pylab as plt
    # plt.scatter(reservoir.point_U[:,0], reservoir.point_U[:,1])
    # plt.show()
    with disable_jit():
        params = ellipsoid_clustering(random.PRNGKey(42), points=reservoir.point_U,
                                  log_VS=jnp.asarray(0., float_type),
                                  max_num_ellipsoids=10)


def test_log_ellipsoid_volume():
    radii = jnp.ones(2)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(jnp.pi))
    radii = jnp.ones(3)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(4. * jnp.pi / 3.))
