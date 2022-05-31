from jax import numpy as jnp
from jax.lax import scan

from jaxns.prior_transforms.common import ContinuousPrior
from jaxns.prior_transforms import prior_docstring, get_shape
from jaxns.internals.shapes import broadcast_shapes


class ForcedIdentifiabilityPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, n, low, high, tracked=True):
        """
        Prior for a sequence of `n` random variables uniformly distributed on U[low, high] such that U[i,...] <= U[i+1,...].
        For broadcasting the resulting random variable is sorted on the first dimension elementwise.

        Args:
            n: number of samples within [low,high]
            low: minimum of distribution
            high: maximum of distribution
        """
        low = self._prepare_parameter(name, 'low', low)
        high = self._prepare_parameter(name, 'high', high)
        self._n = n
        shape = (self._n,) + broadcast_shapes(get_shape(low), get_shape(high))
        super(ForcedIdentifiabilityPrior, self).__init__(name, shape, [low, high], tracked)

    def transform_U(self, U, low, high, **kwargs):
        log_x = jnp.log(U)

        # theta[i] = theta[i-1] * (1 - x[i]) + theta_max * x[i]
        def body(state, X):
            (log_theta,) = state
            (log_x, i) = X
            log_theta = log_x / i + log_theta
            return (log_theta,), (log_theta,)

        log_init_theta = jnp.zeros(broadcast_shapes(low.shape, high.shape))
        _, (log_theta,) = scan(body, (log_init_theta,), (log_x, jnp.arange(1, self._n + 1)), reverse=True)
        theta = low + (high - low) * jnp.exp(log_theta)
        return theta
