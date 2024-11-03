from jax import numpy as jnp

import jaxns.internals.maps


def msqrt(A):
    """
    Computes the matrix square-root using SVD, which is robust to poorly conditioned covariance matrices.
    Computes, M such that M @ M.T = A

    Args:
        A: [N,N] Square matrix to take square root of.

    Returns: [N,N] matrix.
    """
    U, s, Vh = jnp.linalg.svd(A)
    L = U * jnp.sqrt(s)
    return L


def squared_norm(x1, x2):
    # r2_ij = sum_k (x_ik - x_jk)^2
    #       = sum_k x_ik^2 - 2 x_jk x_ik + x_jk^2
    #       = sum_k x_ik^2 + x_jk^2 - 2 U U^T
    # r2_ij = sum_k (x_ik - y_jk)^2
    #       = sum_k x_ik^2 - 2 y_jk x_ik + y_jk^2
    #       = sum_k x_ik^2 + y_jk^2 - 2 U Y^T
    x1 = x1
    x2 = x2
    r2 = jnp.sum(jnp.square(x1), axis=1)[:, None] + jnp.sum(jnp.square(x2), axis=1)[None, :]
    r2 = r2 - 2. * (x1 @ x2.T)
    return r2

