from jax import numpy as jnp
from jax.lax import scan

from jaxns.prior_transforms.common import ContinuousPrior
from jaxns.prior_transforms import prior_docstring, get_shape
from jaxns.internals.shapes import broadcast_shapes


class ForcedIdentifiabilityPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, n, low, high, tracked=True):
        """
        Prior for a sequence of `n` random variables uniformly distributed on U[low, high] such that X[i,...] <= X[i+1,...].
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


class PiecewiseLinearPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, n, low, high, tracked=True):
        """
        Sample from a piece-wise linear approximation to a prior when the quantile is given by
        a piecewise linear approximation.

        Args:
            name: str, name of prior
            n: int, number of nodes in the piece wise linear approximation, excluding endpoints.
                Thus, there are n + 1 linear segments defining the quantile.
            low: Prior, the lower end of support
            high: Prior, the upper end of support
            tracked:
        """
        low = self._prepare_parameter(name, 'low', low)
        high = self._prepare_parameter(name, 'high', high)
        x = ForcedIdentifiabilityPrior(f"_{name}_x", n, low, high, tracked=tracked)
        y = ForcedIdentifiabilityPrior(f"_{name}_y", n, 0., 1., tracked=tracked)
        shape = broadcast_shapes(get_shape(x), get_shape(y))[:-1]

        super(PiecewiseLinearPrior, self).__init__(name, shape, [x, y, low, high], tracked=tracked)

    def transform_U(self, U, x, y, low, high, **kwargs):
        return jnp.interp(U, y, x, left=low, right=high)

        low = jnp.broadcast_to(low, x.shape[:-1])
        high = jnp.broadcast_to(high, x.shape[:-1])
        x = jnp.concatenate([low[...,None], x, high[...,None]], axis=-1)
        y_low = jnp.zeros(y.shape[:-1], y.dtype)
        y_high = jnp.ones(y.shape[:-1], y.dtype)
        y = jnp.concatenate([y_low[...,None], y, y_high[...,None]], axis=-1)
        return jnp.interp(U, y, x)
