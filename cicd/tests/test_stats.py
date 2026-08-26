from jax import numpy as jnp, random

from jaxns.stats_utils import linear_to_log_stats


def test_linear_to_log_stats():
    Z = jnp.exp(random.normal(key=random.PRNGKey(42), shape=(1000000,)))
    Z_mu = jnp.mean(Z)
    Z_var = jnp.var(Z)
    log_mu1, log_var1 = linear_to_log_stats(jnp.log(Z_mu), log_f_var=jnp.log(Z_var))
    log_mu2, log_var2 = linear_to_log_stats(jnp.log(Z_mu), log_f2_mean=jnp.log(Z_var + Z_mu ** 2))
    assert jnp.isclose(log_mu1, 0., atol=1e-2)
    assert jnp.isclose(log_var1, 1., atol=1e-2)
    assert jnp.isclose(log_mu1, log_mu2, atol=1e-4)
    assert jnp.isclose(log_var1, log_var2, atol=1e-4)
