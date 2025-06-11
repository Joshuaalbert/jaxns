from typing import NamedTuple, Callable, TypeVar, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxns.experimental.solvers.ad_utils import tree_vdot_real_part, tree_scalar_mul, tree_add, tree_sub
from jaxns.internals.types import IntArray, FloatArray


def _identity(x):
    return x


class CGDiagnostics(NamedTuple):
    iterations: IntArray
    final_res_norm: FloatArray


DomainType = TypeVar('DomainType')


def cg_solve(A: Callable[[DomainType], DomainType], b: DomainType, x0: DomainType,
             M: Callable[[DomainType], DomainType] = _identity, maxiter: int | None = 100, tol: float = 1e-5,
             atol: float = 0.0) -> Tuple[DomainType, CGDiagnostics]:
    """
    Solve a linear system Ax = b using the conjugate gradient method.

    Args:
        A: a square PSD linear operator
        b: the right-hand side
        x0: an initial guess for the solution
        M: a preconditioner for A
        maxiter: the maximum number of iterations, if None then size of b
        tol: the relative tolerance for the residual norm
        atol: the absolute tolerance for the residual norm

    Returns:
        the solution x and diagnostics
    """

    if maxiter is None:
        maxiter = sum(jax.tree.leaves(jax.tree.map(np.size, b)))

    class CGState(NamedTuple):
        x: DomainType  # current solution estimate
        r: DomainType  # residual: r = b - A(x)
        z: DomainType  # preconditioned residual: z = M(r)
        p: DomainType  # search direction
        gamma: FloatArray  # inner product gamma = <r, z>
        iter: IntArray  # current iteration count
        atol2: FloatArray  # squared absolute tolerance for residual norm

    def _create_initial_state(x0: DomainType) -> CGState:
        b_norm_sq = tree_vdot_real_part(b, b)
        atol2 = jnp.maximum(tol ** 2 * b_norm_sq, atol ** 2)
        r = tree_sub(b, A(x0))
        z = M(r)
        p = z
        dtype = jnp.result_type(*jax.tree.leaves(p))
        gamma = tree_vdot_real_part(r, z).astype(dtype)
        return CGState(x=x0, r=r, z=z, p=p, gamma=gamma, iter=0, atol2=atol2)

    def cond_fn(state: CGState):
        res_norm_sq = state.gamma.real if M == _identity else tree_vdot_real_part(state.r, state.r)
        return (res_norm_sq > state.atol2) & (state.iter < maxiter)

    def body_fn(state: CGState):
        # Cache A(p)
        q = A(state.p)
        dtype = jnp.result_type(*jax.tree.leaves(state.p))
        alpha = state.gamma / tree_vdot_real_part(state.p, q).astype(dtype)
        x_new = tree_add(state.x, tree_scalar_mul(alpha, state.p))
        r_new = tree_sub(state.r, tree_scalar_mul(alpha, q))
        z_new = M(r_new)
        gamma_new = tree_vdot_real_part(r_new, z_new).astype(dtype)
        beta = gamma_new / state.gamma
        p_new = tree_add(z_new, tree_scalar_mul(beta, state.p))
        return CGState(x=x_new, r=r_new, z=z_new, p=p_new,
                       gamma=gamma_new, iter=state.iter + 1, atol2=state.atol2)

    init_state = _create_initial_state(x0=x0)
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    final_res_sq = final_state.gamma.real if M == _identity else tree_vdot_real_part(final_state.r, final_state.r)

    diag = CGDiagnostics(iterations=final_state.iter,
                         final_res_norm=jnp.sqrt(final_res_sq))
    return final_state.x, diag
