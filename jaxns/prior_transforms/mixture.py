from jax.scipy.special import ndtri
from jax import numpy as jnp
from jaxns.prior_transforms.common import ContinuousPrior
from jaxns.prior_transforms.discrete import GumbelCategoricalPrior
from jaxns.prior_transforms import prior_docstring, get_shape
from jaxns.internals.shapes import broadcast_shapes


class GMMDiagPrior(ContinuousPrior):
    """
    More efficient version of a mixture of diagonal Gaussians because it avoids computing and stacking
    all components before selecting.
    """
    @prior_docstring
    def __init__(self, name, logits, mu, sigma, tracked=True):
        """
        Mixture of diagonal Gaussians.

        Args:
            logits: log-weights of the mixture components
            mu: mean of components, should have first dim like pi
            sigma: std-dev of components, should have first dim like pi
        """
        logits = self._prepare_parameter(name, 'logits', logits)
        select_component = GumbelCategoricalPrior('_{}_select'.format(name),logits,False)
        mu = self._prepare_parameter(name, 'mu', mu)
        sigma = self._prepare_parameter(name, 'sigma', sigma)

        shape = broadcast_shapes(get_shape(mu), get_shape(sigma))[len(get_shape(logits)):]
        super(GMMDiagPrior, self).__init__(name, shape, [select_component, mu, sigma], tracked)


    def transform_U(self, U, select, mu, sigma, **kwargs):
        mu, sigma = jnp.broadcast_arrays(mu, sigma)
        sigma = sigma[select, ...]
        mu = mu[select, ...]
        return sigma * ndtri(U) + mu
