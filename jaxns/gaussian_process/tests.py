from jax import numpy as jnp, vmap

from jaxns.gaussian_process.utils import squared_norm, product_log


def test_squared_norm():
    x = jnp.linspace(0., 1., 100)[:, None]
    y = jnp.linspace(1., 2., 50)[:, None]
    assert jnp.all(jnp.isclose(squared_norm(x, x), jnp.sum(jnp.square(x[:, None, :] - x[None, :, :]), axis=-1)))
    assert jnp.all(jnp.isclose(squared_norm(x, y), jnp.sum(jnp.square(x[:, None, :] - y[None, :, :]), axis=-1)))


def test_integrate_sqrt_quadratic():
    a = 1.
    b = 2.
    c = 3.
    ans = jnp.sqrt(a + b + c) - ((b + c) * jnp.log(jnp.sqrt(a) * jnp.sqrt(b + c))) / jnp.sqrt(a) + (
                (b + c) * jnp.log(a + jnp.sqrt(a) * jnp.sqrt(a + b + c))) / jnp.sqrt(a)

    u = jnp.linspace(0., 1., 1000)
    ref = jnp.sum(jnp.sqrt(a * u**2 + b * u + c))/(u.size-1)
    print(ref,ans)


def test_product_log():
    from scipy.special import lambertw

    # import pylab as plt
    # w = jnp.linspace(-1./jnp.exp(1)+0.001, 0., 100)
    # plt.plot(w, lambertw(w, 0))
    # plt.plot(w, vmap(product_log)(w))
    # plt.show()

    assert jnp.all(jnp.isclose(vmap(product_log)(w), lambertw(w, 0), atol=1e-2))