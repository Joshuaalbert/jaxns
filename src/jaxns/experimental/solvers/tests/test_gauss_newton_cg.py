import numpy as np
import pytest
from jax import numpy as jnp

from jaxns.experimental.solvers.gauss_newton_cg import newton_cg_solver


def test_newton_cg_solver():
    """A simple test for the newton_cg_solver function."""

    def obj_fn(x):
        return jnp.cos(jnp.sum(x ** 2))  # simple quadratic function

    x0 = jnp.array([1.0, 2.0, 3.0])  # initial guess
    x_final, diag = newton_cg_solver(obj_fn, x0, maxiter=10, verbose=True)

    print("Final solution:", x_final)
    print("Diagnostics:", diag)


@pytest.mark.parametrize("n", [1000])
def test_approx_cg_newton(n):
    def rosenbrock_nd(x):
        a = 1.
        b = 100.
        return jnp.sum((a - x[:-1]) ** 2 + b * (x[1:] - x[:-1] ** 2) ** 2)

    x0 = 10 * jnp.ones(n)

    solution, diagnostics = newton_cg_solver(rosenbrock_nd, x0, verbose=True)
    np.testing.assert_allclose(solution, jnp.ones(n), atol=1e-4)
    import pylab as plt
    plt.plot(np.log(diagnostics.g_norm), label="error")
    plt.plot(np.log(diagnostics.ddelta_x_norm), label="delta_norm")
    plt.plot(np.log(diagnostics.g_norm / diagnostics.ddelta_x_norm), label="error/delta_norm")
    plt.plot(np.log(diagnostics.damping), label="damping")
    plt.plot(np.log(diagnostics.mu), label="mu")
    plt.legend()
    plt.show()
