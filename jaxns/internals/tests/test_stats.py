import numpy as np
from jax import numpy as jnp, random

from jaxns.internals.stats import linear_to_log_stats, density_estimation


def test_linear_to_log_stats():
    Z = jnp.exp(random.normal(key=random.PRNGKey(42), shape=(1000000,)))
    Z_mu = jnp.mean(Z)
    Z_var = jnp.var(Z)
    log_mu1, log_var1 = linear_to_log_stats(jnp.log(Z_mu), log_f_var=jnp.log(Z_var))
    log_mu2, log_var2 = linear_to_log_stats(jnp.log(Z_mu), log_f2_mean=jnp.log(Z_var + Z_mu ** 2))
    assert jnp.isclose(log_mu1, 0., atol=1e-2)
    assert jnp.isclose(log_var1, 1., atol=1e-2)
    assert jnp.isclose(log_mu1,  log_mu2, atol=1e-4)
    assert jnp.isclose(log_var1, log_var2, atol=1e-4)


def test_density_estimation():
    np.random.seed(42)
    x = jnp.asarray(np.random.standard_gamma(1., 100))[:, None]
    xstar = jnp.linspace(0., 20., 1000)[:, None]
    assert density_estimation(xstar, x).size == 1000
    #
    # import pylab as plt
    #
    # plt.plot(xstar, density_estimation(xstar, x))
    # plt.hist(np.random.standard_gamma(1., 10000), bins=np.linspace(0, 20, 100), density=True, alpha=0.5)
    # plt.hist(x[:, 0], bins=np.linspace(0., 20, 100), density=True, alpha=0.5)
    # plt.show()
