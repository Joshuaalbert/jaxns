from jax import numpy as jnp, random, jit
from jax.scipy.special import gammaln
from jaxns.nested_sampler.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, GammaPrior, UniformPrior
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.utils import summary, resample
import numpy as np
import pylab as plt

def main():


    def log_likelihood(x, **kwargs):
        """
        Poisson likelihood.
        """
        return jnp.asarray(0.)

    true_logZ = 0.
    print(f"True Evidence = {true_logZ}")

    with PriorChain() as prior_chain:
        UniformPrior('x', 0., 1.)

    ns = NestedSampler(loglikelihood=log_likelihood,
                       prior_chain=prior_chain)

    f = jit(ns)

    results = f(random.PRNGKey(3452345), termination_evidence_frac=0.001, num_live_points=100.)

    summary(results)

    plot_diagnostics(results)
    plot_cornerplot(results)


if __name__ == '__main__':
    main()