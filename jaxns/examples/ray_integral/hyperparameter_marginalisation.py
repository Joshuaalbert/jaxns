from jaxns.gaussian_process.kernels import RBF, M12
from jaxns.gaussian_process.tomographic_kernel import dtec_tomographic_kernel
from jaxns.examples.ray_integral.generate_data import rbf_dtec
from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, UniformPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jax import random, jit, make_jaxpr
from jax import numpy as jnp
import pylab as plt


def main(kernel):
    def log_normal(x, mean, cov):
        # L = jnp.linalg.cholesky(cov)
        dx = x - mean
        # dx = jnp.solve_triangular(L, dx, lower=True)
        maha = dx @ jnp.linalg.solve(cov, dx)
        logdet = 0.5*jnp.log(jnp.linalg.det(cov))
        return -0.5 * x.size * jnp.log(2. * jnp.pi) -logdet -0.5*maha


    N = 100
    X = jnp.linspace(-2., 2., N)[:, None]
    true_height, true_width, true_sigma, true_l, true_uncert = 200., 100., 0.2, 10., 2.5

    X, Y, Y_obs = rbf_dtec(1, 1, true_height, true_width, true_sigma, true_l, true_uncert)
    a = X[:, 0:3]
    k = X[:, 3:6]

    def log_likelihood(sigma, l, height, width, uncert, **kwargs):
        """
        P(Y|sigma, half_width) = N[Y, mu, K]
        Args:
            sigma:
            l:

        Returns:

        """
        K = dtec_tomographic_kernel(random.PRNGKey(1), a[0,:], a, a, k, k, jnp.zeros(3), kernel, height, width, l, S=15, sigma=sigma)
        data_cov = jnp.square(uncert)*jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return log_normal(Y_obs, mu , K + data_cov)

    def predict_f(sigma, l,height, width, uncert, **kwargs):
        K = dtec_tomographic_kernel(random.PRNGKey(1), a[0, :], a, a, k, k, jnp.zeros(3), kernel, height, width, l,
                                    S=15, sigma=sigma)

        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return mu + K @ jnp.linalg.solve(K + data_cov, Y_obs)

    def predict_fvar(sigma, l,height, width, uncert, **kwargs):
        K = dtec_tomographic_kernel(random.PRNGKey(1), a[0, :], a, a, k, k, jnp.zeros(3), kernel, height, width, l,
                                    S=15, sigma=sigma)

        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return jnp.diag(K - K @ jnp.linalg.solve(K + data_cov, K))

    prior_chain = PriorChain()\
        .push(UniformPrior('sigma', 0., 0.3))\
        .push(UniformPrior('l', 0., 20.))\
        .push(UniformPrior('height', 100., 300.))\
        .push(UniformPrior('width', 50., 150.))\
        .push(UniformPrior('uncert',0.,5.))

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='ellipsoid', predict_f=predict_f, predict_fvar=predict_fvar)

    def run_with_n(n):
        key = random.PRNGKey(0)
        @jit
        def run(key):
            return ns(key=key,
                      num_live_points=n,
                      max_samples=1e5,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=True)
        print(make_jaxpr(run)(key))
        results = run(key)
        return results

    for n in [100]:
        results = run_with_n(n)
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)
    plt.title("Kernel: {}".format(kernel.__name__))
    plt.ylabel('log Z')
    plt.show()

    plt.scatter(X[:, 0], Y_obs, label='data')
    plt.plot(X[:, 0], Y, label='underlying')
    plt.plot(X[:,0], results.marginalised['predict_f'], label='marginalised')
    plt.plot(X[:,0], results.marginalised['predict_f'] + jnp.sqrt(results.marginalised['predict_fvar']), ls='dotted', c='black')
    plt.plot(X[:,0], results.marginalised['predict_f'] - jnp.sqrt(results.marginalised['predict_fvar']), ls='dotted', c='black')
    plt.title("Kernel: {}".format(kernel.__name__))
    plt.legend()
    plt.show()

    plot_diagnostics(results)
    plot_cornerplot(results)
    return results.logZ, results.logZerr



if __name__ == '__main__':
    logZ_rbf, logZerr_rbf = main(RBF)
    logZ_m12, logZerr_m12 = main(M12)
    plt.errorbar(['rbf', 'm12'],
                 [logZ_rbf, logZ_m12],
                 [logZerr_rbf, logZerr_m12])
    plt.ylabel("log Z")
    plt.show()
