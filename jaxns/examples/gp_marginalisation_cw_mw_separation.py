from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, NormalPrior, UniformPrior, GaussianProcessKernelPrior, HalfLaplacePrior,MVNPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.gaussian_process.kernels import RBF
from jax.scipy.linalg import solve_triangular
from jax import random, jit
from jax import numpy as jnp
import pylab as plt

def main():
    def log_normal(x, mean, uncert):
        dx = x - mean
        dx = dx / uncert
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - x.size*jnp.log(uncert) - 0.5 * dx @ dx

    N = 100
    X = jnp.linspace(-2., 2., N)[:, None]
    true_alpha, true_sigma, true_l, true_uncert = 1., 1., 0.2, 0.25
    data_mu = jnp.zeros((N, ))
    prior_cov = RBF()(X, X, true_l, true_sigma)
    Y = jnp.linalg.cholesky(prior_cov) @ random.normal(random.PRNGKey(0), shape=(N, )) + data_mu
    Y_obs = Y + true_uncert * random.normal(random.PRNGKey(1), shape=(N, ))


    def predict_f(sigma, K,  uncert, **kwargs):
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return mu + K @ jnp.linalg.solve(K + data_cov, Y_obs)

    def predict_fvar(sigma, K, uncert, **kwargs):
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return jnp.diag(K - K @ jnp.linalg.solve(K + data_cov, K))

    ###
    # define the prior chain
    # Here we assume each image is represented by pixels.
    # Alternatively, you could choose regions arranged non-uniformly over the image.

    image_shape = (128, 128)
    npix = image_shape[0] * image_shape[1]
    I150 = jnp.ones(image_shape)

    alpha_cw_gp_sigma = HalfLaplacePrior('alpha_cw_gp_sigma', 1.)
    alpha_mw_gp_sigma = HalfLaplacePrior('alpha_mw_gp_sigma', 1.)
    l_cw = UniformPrior('l_cw', 0., 0.5)#degrees
    l_mw = UniformPrior('l_mw', 0.5, 2.)#degrees
    K_cw = GaussianProcessKernelPrior('K_cw',RBF(), X, l_cw, alpha_cw_gp_sigma)
    K_mw = GaussianProcessKernelPrior('K_mw',RBF(), X, l_mw, alpha_mw_gp_sigma)
    alpha_cw = MVNPrior('alpha_cw', -1.5, K_cw)
    alpha_mw = MVNPrior('alpha_mw', -2.5, K_mw)
    S_cw_150 = UniformPrior('S150_cw', 0., I150)
    S_mw_150 = UniformPrior('S150_mw', 0., I150)
    uncert = HalfLaplacePrior('uncert', 1.)

    def log_likelihood(uncert, alpha_cw, alpha_mw, S_cw_150, S_mw_150):
        log_prob = 0
        for img, freq in zip(images, freqs): # <- need to define these
            I_total = S_mw_150 * (freq/150e6) ** (alpha_mw) + S_cw_150 * (freq/150e6) ** (alpha_cw)
            log_prob += log_normal(img, I_total, uncert)
        return log_prob

    prior_chain = PriorChain()\
        .push(alpha_cw).push(S_cw_150)\
        .push(alpha_mw).push(S_mw_150)\
        .push(uncert)
    print(prior_chain)

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='ellipsoid', predict_f=predict_f, predict_fvar=predict_fvar)

    def run_with_n(n):
        @jit
        def run():
            return ns(key=random.PRNGKey(0),
                      num_live_points=n,
                      max_samples=1e3,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=True)

        results = run()
        return results

    # for n in [200]:
    #     results = run_with_n(n)
    #     plt.scatter(n, results.logZ)
    #     plt.errorbar(n, results.logZ, yerr=results.logZerr)
    # plt.title("Kernel: {}".format(rational_quadratic.__name__))
    # plt.ylabel('log Z')
    # plt.show()
    #
    # plt.scatter(X[:, 0], Y_obs, label='data')
    # plt.plot(X[:, 0], Y, label='underlying')
    # plt.plot(X[:,0], results.marginalised['predict_f'], label='marginalised')
    # plt.plot(X[:,0], results.marginalised['predict_f'] + jnp.sqrt(results.marginalised['predict_fvar']), ls='dotted', c='black')
    # plt.plot(X[:,0], results.marginalised['predict_f'] - jnp.sqrt(results.marginalised['predict_fvar']), ls='dotted', c='black')
    # plt.title("Kernel: {}".format(rational_quadratic.__name__))
    # plt.legend()
    # plt.show()
    #
    # plot_diagnostics(results)
    # plot_cornerplot(results)
    # return results.logZ, results.logZerr



if __name__ == '__main__':
    main()
