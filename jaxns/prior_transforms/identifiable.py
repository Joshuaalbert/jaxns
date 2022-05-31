from jax import numpy as jnp, random
from jax.lax import scan

from jaxns.prior_transforms.common import ContinuousPrior
from jaxns.prior_transforms import prior_docstring, get_shape
from jaxns.internals.shapes import broadcast_shapes
from jaxns.internals.random import resample_indicies
from jaxns.internals.log_semiring import LogSpace

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

class FromSamplesPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, samples:jnp.ndarray, log_weights=None, tracked=True):
        """
        Construct a piecewise linear approximation to a distribution given by a set of samples.

        Args:
            name: str, name of prior
            samples: [M] M equally-weighted samples of a scalar RV
            n: int, number of nodes in the piece wise linear approximation, excluding endpoints.
                Thus, there are n + 1 linear segments defining the quantile.
            log_weights: optinal if given then used to compute ESS.
            tracked:
        """
        assert len(samples.shape) == 1, "Only 1D samples allowed."
        if log_weights is not None:
            idx = resample_indicies(random.PRNGKey(4212498765), log_weights, replace=True)
            samples = samples[idx]
        bins = max(10, int(jnp.sqrt(samples.size)))
        freq, bins = jnp.histogram(samples, bins=bins)
        bin_centers = 0.5*(bins[1:] + bins[:-1])
        cum_freq = jnp.cumsum(freq)
        cum_freq /= cum_freq[-1]
        def icdf(u):
            return jnp.interp(u,cum_freq, bin_centers)
        self.icdf = icdf
        shape = ()

        super(FromSamplesPrior, self).__init__(name, shape, [], tracked=tracked)

    def transform_U(self, U, **kwargs):
        return self.icdf(U)
