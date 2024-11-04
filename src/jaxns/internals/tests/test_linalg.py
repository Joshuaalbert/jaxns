from jax import numpy as jnp, random

import jaxns.internals.maps
from jaxns.internals.linalg import msqrt


def test_msqrt():
    for i in range(10):
        A = random.normal(random.PRNGKey(i), shape=(30, 30))
        B = A @ A.T
        L = msqrt(B)
        assert jnp.allclose(B, L @ L.T, atol=2e-4)
