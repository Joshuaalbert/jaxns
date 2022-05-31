from jax import numpy as jnp, random

from jaxns import ContinuousPrior, prior_docstring, ForcedIdentifiabilityPrior, broadcast_shapes, get_shape
from jaxns.internals.random import resample_indicies


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
        # TODO: add joint information through homogeneous measure with at least 1 count per bin
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