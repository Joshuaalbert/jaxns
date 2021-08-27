from jax import numpy as jnp, random, jit, vmap
from jax.scipy.special import gammaln
from jaxns.nested_sampling import NestedSampler, save_results, load_results
from jaxns.prior_transforms import UniformPrior, PriorChain, NormalPrior, StudentT
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.utils import summary, resample
import numpy as np
import pylab as plt

def main():

    true_nu, true_mu, true_sigma = 2., 0., 1.

    def log_likelihood(x, **kwargs):
        """
        Unit likelihood
        """
        return 0.

    x = StudentT('x', true_nu, true_mu, true_sigma)
    prior_chain = x.prior_chain()
    ns = NestedSampler(loglikelihood=log_likelihood,
                       prior_chain=prior_chain,
                       num_live_points=1000)

    results = ns(random.PRNGKey(32564))
    summary(results)
    plot_diagnostics(results)
    plot_cornerplot(results)

    samples = resample(random.PRNGKey(43083245),results.samples, results.log_p, S=int(results.ESS))

    plt.hist(samples['x'], bins='auto', ec='blue', alpha=0.5, density=True, fc='none')

    _x = np.random.standard_t(true_nu, size=100000)

    plt.hist(_x, bins='auto', ec='orange', alpha=0.5, density=True, fc='none')
    plt.xlim(-10., 10.)
    plt.show()



if __name__ == '__main__':
    main()