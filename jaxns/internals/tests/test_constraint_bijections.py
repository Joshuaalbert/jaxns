import jax
import numpy as np
from jax import numpy as jnp
from jax._src.scipy.special import logit

from jaxns.internals.constraint_bijections import quick_unit, quick_unit_inverse


def test_quick_unit():
    x = jnp.linspace(-10, 10, 1000000)
    y = quick_unit(x)
    assert np.all(y <= 1)
    assert np.all(y >= 0)
    x_reconstructed = quick_unit_inverse(y)
    np.testing.assert_allclose(x, x_reconstructed, atol=2e-5)

    g = jax.grad(quick_unit)
    assert np.all(np.isfinite(jax.vmap(g)(x)))
    assert np.isfinite(g(0.))

    h = jax.grad(quick_unit_inverse)
    assert np.all(np.isfinite(jax.vmap(h)(y)))
    assert np.isfinite(h(0.5))

    # Test performance against sigmoid and logit
    import time
    for f in [quick_unit, jax.nn.sigmoid]:
        g = jax.jit(f).lower(x).compile()
        t0 = time.time()
        for _ in range(1000):
            g(x).block_until_ready()
        print(f"{f.__name__} {time.time() - t0}s")

    for f in [quick_unit_inverse, logit]:
        g = jax.jit(f).lower(y).compile()
        t0 = time.time()
        for _ in range(1000):
            g(y).block_until_ready()
        print(f"{f.__name__} {time.time() - t0}s")
