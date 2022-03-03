from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as jnp, vmap

from jaxns.modules.gaussian_process.utils import product_log
from jaxns.internals.linalg import squared_norm


def test_squared_norm():
    x = jnp.linspace(0., 1., 100)[:, None]
    y = jnp.linspace(1., 2., 50)[:, None]
    assert jnp.all(jnp.isclose(squared_norm(x, x), jnp.sum(jnp.square(x[:, None, :] - x[None, :, :]), axis=-1)))
    assert jnp.all(jnp.isclose(squared_norm(x, y), jnp.sum(jnp.square(x[:, None, :] - y[None, :, :]), axis=-1)))


def test_product_log():
    from scipy.special import lambertw

    w = jnp.linspace(-1./jnp.exp(1)+0.001, 0., 100)
    assert jnp.all(jnp.isclose(vmap(product_log)(w), lambertw(w, 0), atol=1e-2))

