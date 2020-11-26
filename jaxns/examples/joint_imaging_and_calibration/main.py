from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_cornerplot, plot_diagnostics, plot_samples_development
from jaxns.examples.joint_imaging_and_calibration.generate_data import fake_vis
from jaxns.examples.joint_imaging_and_calibration.build_prior import build_prior
from timeit import default_timer
from jax import random, jit
from jax import numpy as jnp
from jax.scipy.optimize import minimize
from jax.nn import sigmoid
import pylab as plt


def main():

    nant, ndir = 5, 20
    uncert = 0.1

    theta_true, gamma_true, y, y_obs = fake_vis(nant,ndir,uncert)

    def log_likelihood(delta, **kwargs):
        dy = delta - y_obs
        r2 = jnp.sum(jnp.real(dy)**2)/uncert**2
        r2 = r2 + jnp.sum(jnp.imag(dy)**2)/uncert**2
        logL = -0.5*r2 -jnp.log(2.*jnp.pi*uncert**2)*dy.size
        return logL


    prior_transform = build_prior(nant, ndir)

    ### MAP with  BFGS
    def constrain(U):
        return 0.05 + sigmoid(U)*0.9

    def loss(U):
        U = constrain(U)
        return -log_likelihood(**prior_transform(U))

    print(loss(jnp.zeros(prior_transform.U_ndims)))
    @jit
    def do_minimisation():
        results = minimize(loss, jnp.zeros(prior_transform.U_ndims),method='BFGS',
                 options=dict(gtol=1e-10, line_search_maxiter=200))
        print(results.message)
        return prior_transform(constrain(results.x)), constrain(results.x), results.status

    results = do_minimisation()
    print('Status',results[2])
    print(results)
    plt.scatter(jnp.arange(nant * ndir), results[0]['theta'], label='inferred')
    plt.scatter(jnp.arange(nant * ndir), theta_true, label='true')
    plt.legend()
    plt.show()

    plt.scatter(jnp.arange(ndir), results[0]['gamma'], label='inferred')
    plt.scatter(jnp.arange(ndir), gamma_true, label='true')
    plt.legend()
    plt.show()
    return


    ns = NestedSampler(log_likelihood, prior_transform, sampler_name='multi_ellipsoid')

    def run_with_n(n):
        @jit
        def run(key):
            return ns(key=key,
                      num_live_points=n,
                      max_samples=1e5,
                      collect_samples=False,
                      termination_frac=0.01,
                      stochastic_uncertainty=False,
                      sampler_kwargs=dict(depth=3))

        t0 = default_timer()
        results = run(random.PRNGKey(0))
        print(results.efficiency)
        print("Time to run including compile:", default_timer() - t0)
        print("Time efficiency normalised:", results.efficiency*(default_timer() - t0))
        return results

    for n in [100]:
        results = run_with_n(n)
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)

    plt.show()

    # plot_diagnostics(results)
    plt.errorbar(jnp.arange(nant*ndir),results.param_mean['theta'], yerr=jnp.sqrt(jnp.diag(results.param_covariance['theta'])), label='inferred')
    plt.scatter(jnp.arange(nant*ndir),theta_true, label='true')
    plt.legend()
    plt.show()

    plt.errorbar(jnp.arange(ndir), results.param_mean['gamma'],
                 yerr=jnp.sqrt(jnp.diag(results.param_covariance['gamma'])), label='inferred')
    plt.scatter(jnp.arange(ndir), gamma_true, label='true')
    plt.legend()
    plt.show()
    # plot_cornerplot(results)


if __name__ == '__main__':
    main()
