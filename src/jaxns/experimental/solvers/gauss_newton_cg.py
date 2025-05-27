from typing import NamedTuple, TypeVar, Tuple, Callable, Any

import jax
import jax.numpy as jnp

from jaxns.experimental.solvers.ad_utils import tree_neg, tree_vdot_real_part, tree_scalar_mul, tree_add, tree_sub, \
    build_hvp
from jaxns.experimental.solvers.cg import cg_solve
from jaxns.internals.types import IntArray, FloatArray, BoolArray

# ----------------------------------------------------------------
# Type helpers
# ----------------------------------------------------------------
DomainType = TypeVar("DomainType")  # parameter pytree
ObjectiveRet = TypeVar("ObjectiveRet")  # scalar objective (FloatArray or 0-D ndarray)

CT = TypeVar("CT")
_CT = TypeVar("_CT")


# ----------------------------------------------------------------
# Utility: split complex pytrees into real pairs so Wirtinger calculus
#          works out of the box with JAX’s real autodiff.
# ----------------------------------------------------------------
def convert_to_real(x: CT) -> Tuple[_CT, Callable[[_CT], CT]]:
    """Return a real-valued twin of `x`   and   a merge-back function."""

    def _maybe_split(a: jax.Array | Any):
        if isinstance(a, jax.Array) and jnp.iscomplexobj(a):
            return (a.real, a.imag)
        return a

    leaves, treedef = jax.tree.flatten(x)
    split_leaves = jax.tree.map(_maybe_split, leaves)

    def merge(split_x: _CT) -> CT:
        def _maybe_merge(a):
            if isinstance(a, tuple):
                return jax.lax.complex(a[0], a[1])
            return a

        merged = list(map(_maybe_merge, split_x))
        return jax.tree.unflatten(treedef, merged)

    return split_leaves, merge


# ----------------------------------------------------------------
# Diagnostics – patterned after LMDiagnostic
# ----------------------------------------------------------------
class NewtonDiagnostic(NamedTuple):
    iteration: IntArray
    g_norm: FloatArray  # |∇f|
    mu: FloatArray  # damping parameter
    damping: FloatArray  # g_norm / mu
    cg_iters: IntArray
    f: FloatArray  # f(x_k)
    f_prop: FloatArray  # f(x_k + δx_k)
    f_quad: FloatArray  # quadratic model at proposal
    delta_f_pred: FloatArray  # predicted decrease
    delta_f_actual: FloatArray  # actual decrease
    gain_ratio: FloatArray  # delta_f_actual / delta_f_pred
    accepted: BoolArray
    in_trust_region: BoolArray
    delta_x_norm: FloatArray  # ‖δx‖
    ddelta_x_norm: FloatArray  # ‖δx – δx⁰‖


# ----------------------------------------------------------------
# Main solver
# ----------------------------------------------------------------
def newton_cg_solver(
        obj_fn: Callable[..., ObjectiveRet],
        x0: DomainType,
        args: tuple = (),
        maxiter: int = 100,
        maxiter_cg: int = 100,
        gtol: float = 3e-5,
        p_accept: float = 0.01,
        p_lower: float = 0.25,
        p_upper: float = 1.10,
        mu_init: float = 1.0,
        mu_min: float = 1e-6,
        mu_in_factor: float = 10,
        mu_out_factor: float = 0.1,
        approx_hvp: bool = False,  # reuse H·v between rejections
        verbose: bool = False,
) -> Tuple[DomainType, NewtonDiagnostic]:
    """
    Trust-region Newton-CG minimiser.

    Identical call signature and adaptive-μ logic as `lm_solver`, but uses a
    scalar objective instead of residuals, and solves
        (H  +  damping·I) δx = -∇f
    by CG with Hessian–vector products.

    Returns
    -------
    x_final : pytree matching `x0`     (merged back to complex if needed)
    diagnostics : NewtonDiagnostic[...] array with length = `maxiter`
    """
    # ---- 1.  Handle complex inputs -----------------------------------------
    x0_real, merge_back = convert_to_real(x0)

    # Wrap obj_fn so that it consumes / produces purely real pytrees
    def _obj_fn(x):
        val = obj_fn(merge_back(x), *args)
        if not isinstance(val, jax.Array):
            raise RuntimeError("Objective function must return a JAX scalar array.")
        if jnp.ndim(val) != 0:
            raise RuntimeError("Objective function must return a scalar.")
        return val

    # ---- 2.  State container -----------------------------------------------
    class NState(NamedTuple):
        x: DomainType  # current parameters (real)
        f: FloatArray  # f(x)
        g: DomainType  # -∇f  (descent direction)
        g_norm: FloatArray
        mu: FloatArray
        delta_x_prev: DomainType  # δx⁻¹
        delta_x_prev2: DomainType  # δx⁻²
        iter: IntArray

    # ---- 3.  Helpers --------------------------------------------------------
    def _gradient(x):
        return jax.grad(_obj_fn)(x)

    def _initial_state(x):
        f0 = _obj_fn(x)
        grad_f = _gradient(x)
        g0 = tree_neg(grad_f)  # -∇f
        g_norm0 = jnp.sqrt(tree_vdot_real_part(g0, g0))
        g_unit = tree_scalar_mul(1.0 / (g_norm0 + 1e-12), g0)

        # Simple backtracking line search along -∇f to pick starting μ
        def ls_cond(mu):
            step = tree_scalar_mul(mu, g_unit)
            f_new = _obj_fn(tree_add(x, step))
            return (f_new >= f0) & (mu > mu_min)

        def ls_body(mu):
            return mu * mu_out_factor

        mu0 = jax.lax.while_loop(ls_cond, ls_body, mu_init)

        z = jax.tree.map(jnp.zeros_like, x)
        return NState(
            x=x, f=f0, g=g0, g_norm=g_norm0, mu=mu0,
            delta_x_prev=z, delta_x_prev2=z, iter=0
        )

    # ---- 4.  Iteration ------------------------------------------------------
    def cond_fn(carry):
        state, _ = carry
        return (state.g_norm > gtol) & (state.iter < maxiter)

    def step_fn(state: NState):
        # 4.1  Warm-start: δx⁰ = 2δx⁻¹ − δx⁻²
        delta_x0 = tree_sub(tree_add(state.delta_x_prev, state.delta_x_prev),
                            state.delta_x_prev2)

        # 4.2  Build Hessian-vector product operator (maybe reused)
        hvp = build_hvp(_obj_fn, state.x, linearise=True)
        damping = state.g_norm / state.mu

        def A_op(v):
            return tree_add(hvp(v), tree_scalar_mul(damping, v))

        # 4.3  Solve Newton system with CG
        delta_x, cg_diag = cg_solve(
            A=A_op, b=state.g, x0=delta_x0,
            maxiter=maxiter_cg, tol=1e-5, atol=0.0
        )

        # 4.4  Book-keeping norms
        delta_x_norm = jnp.sqrt(tree_vdot_real_part(delta_x, delta_x))
        ddelta_x = tree_sub(delta_x, delta_x0)
        ddelta_x_norm = jnp.sqrt(tree_vdot_real_part(ddelta_x, ddelta_x))

        # 4.5  Evaluate objective at proposal
        x_prop = tree_add(state.x, delta_x)
        f_prop = _obj_fn(x_prop)

        # Quadratic model prediction
        hvp_dx = hvp(delta_x)
        quad_term = 0.5 * tree_vdot_real_part(delta_x, hvp_dx)
        lin_term = tree_vdot_real_part(state.g, delta_x)  # g·δx
        f_quad = state.f - (lin_term - quad_term)  # f(x)+...
        delta_f_pred = state.f - f_quad  # should be >0
        delta_f_actual = state.f - f_prop
        gain_ratio = delta_f_actual / delta_f_pred
        gain_ratio = jnp.where(jnp.isnan(gain_ratio), 0.0, gain_ratio)  # handle NaN

        # 4.6  Trust-region logic
        in_trust = (delta_f_pred > 0) & (delta_f_actual > p_lower * delta_f_pred) & (
                delta_f_actual < p_upper * delta_f_pred)
        new_mu = jax.lax.select(in_trust, mu_in_factor * state.mu, state.mu * mu_out_factor)
        new_mu = jnp.maximum(new_mu, mu_min)

        accepted = (delta_f_pred > 0) & (delta_f_actual > p_accept * delta_f_pred)

        # 4.7  Diagnostics -----------------------------------------------------
        diag = NewtonDiagnostic(
            iteration=state.iter,
            g_norm=state.g_norm,
            mu=state.mu,
            damping=damping,
            cg_iters=cg_diag.iterations,
            f=state.f,
            f_prop=f_prop,
            f_quad=f_quad,
            delta_f_pred=delta_f_pred,
            delta_f_actual=delta_f_actual,
            gain_ratio=gain_ratio,
            accepted=accepted,
            in_trust_region=in_trust,
            delta_x_norm=delta_x_norm,
            ddelta_x_norm=ddelta_x_norm
        )
        if verbose:
            jax.debug.print(
                "iter {iteration:3d}  f={f:9.3e}  |g|={g_norm:9.3e}  "
                "μ={mu:9.3e}  r={gain_ratio:6.2f}  "
                "Δf_pred={delta_f_pred:9.3e}  Δf_act={delta_f_actual:9.3e}  "
                "CG={cg_iters}",
                iteration=state.iter,
                f=state.f, g_norm=state.g_norm, mu=state.mu,
                gain_ratio=diag.gain_ratio,
                delta_f_pred=diag.delta_f_pred,
                delta_f_actual=diag.delta_f_actual,
                cg_iters=diag.cg_iters
            )

        # 4.8  Accept / reject step
        (
            x_new, f_new, delta_x_prev_new, delta_x_prev2_new
        ) = jax.tree.map(
            lambda a, b: jax.lax.select(accepted, a, b),
            (x_prop, f_prop, delta_x, state.delta_x_prev),
            (state.x, state.f, state.delta_x_prev, state.delta_x_prev2)
        )

        # 4.9  Gradient at (possibly) new point
        if approx_hvp:
            # keep previous hvp & gradient
            g_new = state.g
            g_norm_new = state.g_norm
        else:
            grad_f_new = _gradient(x_new)
            g_new = tree_neg(grad_f_new)
            g_norm_new = jnp.sqrt(tree_vdot_real_part(g_new, g_new))

        # 4.10  Next state
        return NState(
            x=x_new, f=f_new, g=g_new, g_norm=g_norm_new, mu=new_mu,
            delta_x_prev=delta_x_prev_new, delta_x_prev2=delta_x_prev2_new,
            iter=state.iter + 1
        ), diag

    # --------------------------------------------------------------------
    # 5.  Main while-loop with diagnostic accumulation
    # --------------------------------------------------------------------
    init_state = _initial_state(x0_real)

    # prototype diag for shape inference
    diag_aval = jax.eval_shape(lambda s: step_fn(s)[1], init_state)
    empty_diag = jax.tree.map(
        lambda arr: jnp.zeros((maxiter,) + arr.shape, arr.dtype),
        diag_aval
    )

    def body_fn(carry):
        st, diag_arr = carry
        new_st, new_d = step_fn(st)
        diag_arr = jax.tree.map(
            lambda arr, d: arr.at[st.iter].set(d),
            diag_arr, new_d
        )
        return new_st, diag_arr

    final_state, final_diag = jax.lax.while_loop(
        cond_fn, body_fn, (init_state, empty_diag)
    )

    # Merge complex components back to the user space
    return merge_back(final_state.x), final_diag
