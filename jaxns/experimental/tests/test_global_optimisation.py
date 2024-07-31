import jax.numpy as jnp
import jax.random
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from jaxns import Prior, Model
from jaxns.experimental import GlobalOptimisationResults
from jaxns.experimental.global_optimisation import GlobalOptimisationTerminationCondition, gradient_based_optimisation, \
    summary
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


def test_gradient_based_optimisation(all_global_optimisation_problems):
    for name, (model, optimum, a_tol, log_L_tol) in all_global_optimisation_problems:
        print(f"Checking {name}")
        U_init = model.sample_U(jax.random.PRNGKey(42))
        log_L_init = model.log_prob_likelihood(U_init, allow_nan=False)
        U_opt, log_L_solution, num_likelihood_evals = gradient_based_optimisation(model=model, init_U_point=U_init)
        assert log_L_solution >= log_L_init


def test_all_global_optimisation(all_global_optimisation_problems):
    for name, (model, optimum, a_tol, log_L_tol) in all_global_optimisation_problems:
        print(f"Checking {name}")
        go = DefaultGlobalOptimisation(model, gradient_slice=True)
        results = jax.jit(go, static_argnames=['finetune'])(
            key=jax.random.PRNGKey(0),
            term_cond=GlobalOptimisationTerminationCondition(log_likelihood_contour=log_L_tol),
            finetune=True
        )
        assert len(results.X_solution) > 0
        # print(results)
        go.summary(results)

        np.testing.assert_allclose(results.solution[0], optimum, atol=a_tol)


def test_summary():
    mock_results = GlobalOptimisationResults(
        U_solution=jnp.array([1., 2.]),
        X_solution={
            'x': jnp.array([1., 2.]),
            'y': jnp.array(1.),
            'z': jnp.array([]),
            'w': jnp.ones((2, 0, 2)),
            't': jnp.ones((2, 1, 3))
        },
        solution=jnp.array([1., 2.]),
        log_L_solution=jnp.array(1.),
        num_likelihood_evaluations=jnp.array(1),
        num_samples=jnp.array(1),
        termination_reason=jnp.array(1),
        relative_spread=jnp.array(1.),
        absolute_spread=jnp.array(1.)
    )
    summary(mock_results)
