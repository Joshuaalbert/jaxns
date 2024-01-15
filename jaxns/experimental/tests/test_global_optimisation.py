import jax.numpy as jnp
import jax.random
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

from jaxns import Prior, Model
from jaxns.experimental.global_optimisation import GlobalOptimisationTerminationCondition
from jaxns.experimental.public import DefaultGlobalOptimisation

tfpd = tfp.distributions


@pytest.fixture(scope='package')
def drop_wave_problem_2d():
    ndims = 2

    def prior_model():
        z = yield Prior(tfpd.Uniform(low=-5.12 * jnp.ones(ndims), high=5.12 * jnp.ones(ndims)), name='z')
        return z

    def log_likelihood(z):
        return (1 + jnp.cos(12. * jnp.sqrt(jnp.sum(z ** 2)))) / (0.5 * jnp.sum(z ** 2) + 2) - 1.

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    optimum = 0. * jnp.ones(ndims)
    # import pylab as plt
    # X, Y = np.meshgrid(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), indexing='ij')
    # U_test = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    # Z = jax.vmap(lambda U: model.forward(U, allow_nan=True))(U_test)
    # Z = jnp.reshape(Z, X.shape)
    # plt.imshow(Z, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    # plt.colorbar()
    # plt.show()
    a_tol = 0.01
    log_L_tol = -0.01
    return model, optimum, a_tol, log_L_tol


@pytest.fixture(scope='package')
def drop_wave_problem_5d():
    ndims = 5

    def prior_model():
        z = yield Prior(tfpd.Uniform(low=-5.12 * jnp.ones(ndims), high=5.12 * jnp.ones(ndims)), name='z')
        return z

    def log_likelihood(z):
        return (1 + jnp.cos(12. * jnp.sqrt(jnp.sum(z ** 2)))) / (0.5 * jnp.sum(z ** 2) + 2) - 1.

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    optimum = 0. * jnp.ones(ndims)
    a_tol = 0.01
    log_L_tol = -0.01
    return model, optimum, a_tol, log_L_tol


@pytest.fixture(scope='package')
def xin_she_yang_1_problem():
    ndims = 2

    def prior_model():
        z = yield Prior(tfpd.Uniform(low=-2. * jnp.pi * jnp.ones(ndims), high=2. * jnp.pi * jnp.ones(ndims)), name='z')
        return z

    def log_likelihood(z):
        return - jnp.sum(jnp.abs(z)) * jnp.exp(-jnp.sum(jnp.sin(z ** 2)))

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    optimum = 0. * jnp.ones(ndims)
    a_tol = 0.01
    log_L_tol = -0.01
    return model, optimum, a_tol, log_L_tol


@pytest.fixture(scope='package')
def all_global_optimisation_problems(
        xin_she_yang_1_problem,
        drop_wave_problem_2d,
        drop_wave_problem_5d
):
    # Return tuples with names
    return [
        ('xin_she_yang_1_problem', xin_she_yang_1_problem),
        ('drop_wave_problem_2d', drop_wave_problem_2d),
        ('drop_wave_problem_5d', drop_wave_problem_5d),
    ]


def test_all_global_optimisation(all_global_optimisation_problems):
    for name, (model, optimum, a_tol, log_L_tol) in all_global_optimisation_problems:
        print(f"Checking {name}")
        go = DefaultGlobalOptimisation(model)
        results = jax.jit(go)(
            key=jax.random.PRNGKey(0),
            term_cond=GlobalOptimisationTerminationCondition(log_likelihood_contour=log_L_tol)
        )
        assert len(results.X_solution) > 0
        # print(results)
        go.summary(results)

        np.testing.assert_allclose(results.solution[0], optimum, atol=a_tol)
