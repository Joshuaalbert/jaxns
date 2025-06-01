from functools import partial

import jax
import jax.numpy as jnp

_dot = partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)
_vdot = partial(jnp.vdot, precision=jax.lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


def tree_dot(x, y):
    dots = jax.tree.leaves(jax.tree.map(jnp.vdot, x, y))
    return sum(dots[1:], start=dots[0])


def tree_norm(x):
    return jnp.sqrt(tree_dot(x, x).real)


def tree_mul(x, y):
    return jax.tree.map(jax.lax.mul, x, y)


def tree_div(x, y):
    return jax.tree.map(jax.lax.div, x, y)


# aliases for working with pytrees
def _vdot_real_part(x, y):
    """Vector dot-product guaranteed to have a real valued result despite
       possibly complex input. Thus neglects the real-imaginary cross-terms.
       The result is a real float.
    """
    # all our uses of vdot() in CG are for computing an operator of the form
    #  z^H M z
    #  where M is positive definite and Hermitian, so the result is
    # real valued:
    # https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Definitions_for_complex_matrices
    result = _vdot(x.real, y.real)
    if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
        result += _vdot(x.imag, y.imag)
    return result


def tree_vdot(x, y):
    z = jax.tree.leaves(jax.tree.map(_vdot, x, y))
    return sum(z[1:], z[0])


def tree_vdot_real_part(x, y):
    z = jax.tree.leaves(jax.tree.map(_vdot_real_part, x, y))
    return sum(z[1:], z[0])


def tree_add(x, y):
    return jax.tree.map(jax.lax.add, x, y)


def tree_scalar_mul(alpha, x):
    return jax.tree.map(lambda x: alpha * x, x)


def tree_neg(x):
    return jax.tree.map(jax.lax.neg, x)


def tree_sub(x, y):
    return jax.tree.map(jax.lax.sub, x, y)


def hvp_linearized(f, params):
    # Compute the gradient function and linearize it at params
    grad_f = jax.grad(f)
    _, jvp_lin = jax.linearize(grad_f, params)
    # lin_fun is a function that computes the JVP of grad_f at params
    return jvp_lin  # This function computes HVPs for different v


def hvp_forward_over_reverse(f, params):
    def hvp(v):
        return jax.jvp(jax.grad(f), (params,), (v,))[1]

    return hvp


def hvp_reverse_over_reverse(f, params):
    def hvp(v):
        return jax.grad(lambda y: jnp.vdot(jax.grad(f)(y), v))(params)

    return hvp


def hvp_reverse_over_forward(f, params):
    def hvp(v):
        jvp_fun = lambda params: jax.jvp(f, (params,), (v,))[1]
        return jax.grad(jvp_fun)(params)

    return hvp


def grad_and_hvp(f, params, v):
    """
    Compute the gradient and Hessian-vector product of a function.

    Args:
        f: the function to differentiate, should be scalar output
        params: the parameters to differentiate with respect to
        v: the vector to multiply the Hessian with

    Returns:
        the gradient and Hessian-vector product
    """
    return jax.jvp(jax.grad(f), (params,), (v,))


def build_hvp(f, params, linearise: bool = True):
    """
    Build a function that computes the Hessian-vector product of a function.

    Args:
        f: scalar function to differentiate
        params: the parameters to differentiate with respect to
        linearise: whether to linearize the gradient function at params, can be better for reapplying the HVP multiple
            times.

    Returns:
        a function that computes the Hessian-vector product
    """
    if linearise:
        # Compute the gradient function and linearize it at params
        grad_f = jax.grad(f)
        # lin_fun is a function that computes the JVP of grad_f at params
        _, grad_jvp_lin = jax.linearize(grad_f, params)

        def matvec(v):
            return grad_jvp_lin(v)
    else:
        def matvec(v):
            return grad_and_hvp(f, params, v)[1]
    return matvec
