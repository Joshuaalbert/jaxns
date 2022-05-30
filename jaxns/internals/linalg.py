from jax import numpy as jnp
from jax.lax import scan


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


def cholesky_update(L, x, alpha=1., beta=1.):
    """
    Perform rank-1 Cholesky update equivalent to,
        Cholesky(alpha * (L @ L.T) + beta * x @ x.T)

    Algorithm 3.1 from https://christian-igel.github.io/paper/AMERCMAUfES.pdf

    Args:
        L: Cholesky of pos-def matrix prior to rank-1 update
        x: update vector
        alpha: positive scalar
        beta: scalar, some negative values might result in non-pos-def update

    Returns: Cholesky of the updated matrix
    """
    n = x.size
    b = 1.

    def body(state, X):
        (x, b) = state
        (j,) = X
        old_Ljj = L[j, j]
        gamma = b * alpha * old_Ljj ** 2 + beta * x[j] ** 2
        new_Ljj = jnp.sqrt(gamma / b)
        xj = x[j]
        x = x - xj * (jnp.sqrt(alpha) / old_Ljj) * L[:, j]
        next_column = (jnp.sqrt(alpha) / old_Ljj) * new_Ljj * L[:, j] + (new_Ljj * beta * xj / gamma) * x
        b = b + beta * xj ** 2 / (alpha * old_Ljj ** 2)
        return (x, b), (next_column,)

    (x, b), (Lt,) = scan(body, (x, b), (jnp.arange(n),))
    return Lt.T


def rank_one_update_matrix_inv(Ainv, logdetAinv, u, v, add=True):
    """
    Uses the Woodbury matrix lemma to update a matrix inverse under either addition and subtraction.

    Equivalent to,
        (A + u v^T)^{-1} if add==True else (A - u v^T)^{-1}

    Args:
        Ainv: [N, N]
        u: [N]
        v: [N]
        add: bool

    Returns:

    """
    U = u[:, None]
    V = v[None, :]
    AinvU = Ainv @ U
    if add:
        S = 1. + V @ AinvU
        Ainv_update = (jnp.eye(Ainv.shape[-1]) - (AinvU @ V) / S) @ Ainv
        logdetAinv_update = logdetAinv - jnp.log(S[0, 0])
        return Ainv_update, logdetAinv_update
    else:
        S = V @ AinvU - 1.
        Ainv_update = (jnp.eye(Ainv.shape[-1]) - (AinvU @ V) / S) @ Ainv
        logdetAinv_update = logdetAinv - jnp.log(-S[0, 0])
        return Ainv_update, logdetAinv_update
