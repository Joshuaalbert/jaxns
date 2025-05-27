import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_common.common.ad_utils import grad_and_hvp, build_hvp


def test_grad_and_hvp():
    def f(params):
        W, b = params
        x = jnp.ones_like(b)
        return jnp.sum(jax.nn.sigmoid(W @ x + b))

    W = jnp.array([[1., 2.], [3., 4.]])
    b = jnp.array([1., 2.])

    params = (W, b)

    v = params

    grad, hvp = grad_and_hvp(f, params, v)

    print(grad)
    print(hvp)


def test_build_hvp():
    def f_crazy(x):
        return jnp.sum(jnp.cos(x) ** 2) + jnp.sum(jnp.sin(x) ** 2)

    x = jnp.ones((10,))
    v = jnp.ones((10,))

    matvec = build_hvp(f_crazy, x, linearise=False)
    matvec_lin = build_hvp(f_crazy, x, linearise=True)

    np.testing.assert_allclose(matvec(v), matvec_lin(v), atol=1e-8)
