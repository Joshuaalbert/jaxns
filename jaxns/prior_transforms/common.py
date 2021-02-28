from jax import numpy as jnp
from jax.scipy.special import ndtri, gammaln

from jaxns.prior_transforms.prior_chain import Prior, UniformBase, PriorBase, DiscreteBase
from jaxns.prior_transforms.prior_utils import get_shape, prior_docstring, convert_to_array, broadcast_dtypes
from jaxns.utils import broadcast_shapes, msqrt


class DeltaPrior(Prior):

    @prior_docstring
    def __init__(self, name, value, tracked=False):
        """
        Delta distribution. Always returns the same value.

        Args:
            value: singular value to always return.
        """
        value = convert_to_array(value)
        super(DeltaPrior, self).__init__(name, get_shape(value), [], tracked, PriorBase((), value.dtype))
        if isinstance(value, Prior):
            raise ValueError('value should be array-like, got {}'.format(type(value)))
        self.value = jnp.asarray(value)

    def __repr__(self):
        return "DeltaPrior({})".format(self.value if self.value.size == 1 else "array<{}>".format(self.value.shape))

    @property
    def shape(self):
        return self.value.shape

    def transform_U(self, U, **kwargs):
        """
        Just returns the singular value.

        Args:
            U:
            **kwargs:

        Returns: jnp.ndarray
        """
        del U
        return self.value


class HierarchicalPrior(Prior):
    """
    For Priors that depend on other RVs.
    """

    @staticmethod
    def _prepare_parameter(parent_name, name, param):
        """
        If param is not a Prior, then this creates a DeltaPrior.

        Args:
            name: name of this param, which will be made unique.
            param: array-like or Prior

        Returns: Prior
        """
        if not isinstance(param, Prior):
            return DeltaPrior('_{}_{}'.format(parent_name, name), convert_to_array(param), False)
        return param


class ContinuousPrior(HierarchicalPrior):
    @prior_docstring
    def __init__(self, name, shape, parents, tracked, prior_base=None):
        """
        This is a general prior for continuous RVs that have an implemented bijective transform from the PriorBase.
        In this case, the dimension of the prior_base is equal to the dimension codomain, and even the shapes are
        assumed to be the same.
        """
        if prior_base is None:
            prior_base = UniformBase(shape, broadcast_dtypes(jnp.float_, *[p.dtype for p in parents]))
        super(ContinuousPrior, self).__init__(name, shape, parents, tracked, prior_base)


class DiscretePrior(HierarchicalPrior):
    @prior_docstring
    def __init__(self, name, shape, parents, tracked, num_outcomes=None, prior_base:DiscreteBase=None):
        """
        Represents a general prior for discrete RVs where the log-homogeneous measure is used to modulate the uniform
        samples from the PriorBase. An exception are discrete RVs that use some non-bijective trick, e.g. the
        CategoricalPrior which uses the Gumbel trick.

        Args:
            num_outcomes: int, number of discrete choices.
        """
        if prior_base is None:
            if num_outcomes is None:
                raise ValueError("Either prior_base or num_outcomes must be given.")
            prior_base = DiscreteBase(num_outcomes, shape, jnp.int_)
        super(DiscretePrior, self).__init__(name, shape, parents, tracked, prior_base)


class LogNormalPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, mu, sigma, tracked=True):
        """
        Log-normal prior.

        X ~ N[mu, sigma^2]
        Y ~ exp(X)

        Args:
            mu: mean of underlying
            sigma: std-dev of underlying
        """
        mu = self._prepare_parameter(name, name, 'mu', mu)
        sigma = self._prepare_parameter(name, name, 'sigma', sigma)
        shape = broadcast_shapes(get_shape(mu), get_shape(sigma))
        super(LogNormalPrior, self).__init__(name, shape, [mu, sigma], tracked)

    def transform_U(self, U, mu, sigma, **kwargs):
        return jnp.exp(ndtri(U) * sigma + mu)


class NormalPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, mu, sigma, tracked=True):
        """
        Normal prior.

        Y ~ N[mu, sigma^2]

        Args:
            mu: mean
            sigma: std-dev
        """
        mu = self._prepare_parameter(name, 'mu', mu)
        sigma = self._prepare_parameter(name, 'sigma', sigma)
        shape = broadcast_shapes(get_shape(mu), get_shape(sigma))
        super(NormalPrior, self).__init__(name, shape, [mu, sigma], tracked)

    def transform_U(self, U, mu, sigma, **kwargs):
        return ndtri(U) * sigma + mu


class MVNPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, mu, Sigma, ill_cond=False, tracked=True):
        """
        Multivate Normal distribution.

        Y ~ N[mu, Sigma]

        Args:
            mu: mean
            Sigma: covariance matrix
            ill_cond: bool, whether to use SVD for matrix square-root.
        """
        self._ill_cond = ill_cond
        mu = self._prepare_parameter(name, 'mu', mu)
        Sigma = self._prepare_parameter(name, 'Sigma', Sigma)
        shape = broadcast_shapes(get_shape(mu), get_shape(Sigma)[-1:])
        super(MVNPrior, self).__init__(name, shape, [mu, Sigma], tracked)

    def transform_U(self, U, mu, Sigma, **kwargs):
        if self._ill_cond:
            L = msqrt(Sigma)
        else:
            L = jnp.linalg.cholesky(Sigma)
        return L @ ndtri(U) + mu


class LaplacePrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, mu, b, tracked=True):
        """
        Laplace distribution.

        Y~L[mu,b]

        Args:
            mu: mean
            b: scale
        """
        mu = self._prepare_parameter(name, 'mu', mu)
        b = self._prepare_parameter(name, 'b', b)
        shape = broadcast_shapes(get_shape(mu), get_shape(b))
        super(LaplacePrior, self).__init__(name, shape, [mu, b], tracked)

    def transform_U(self, U, mu, b, **kwargs):
        return mu - b * jnp.sign(U - 0.5) * jnp.log(1. - 2. * jnp.abs(U - 0.5))


class HalfLaplacePrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, b, tracked=True):
        """
        Half-Laplace distribution.

        X ~ L[0,b]
        Y = abs(X)

        Args:
            b: scale
        """
        b = self._prepare_parameter(name, 'b', b)
        shape = get_shape(b)
        super(HalfLaplacePrior, self).__init__(name, shape, [b], tracked)

    def transform_U(self, U, b, **kwargs):
        return - b * jnp.sign(0.5 * U) * jnp.log(1. - 2. * jnp.abs(0.5 * U))


class UniformPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, low, high, tracked=True):
        """
        Uniform distribution.

        X ~ U[low, high]

        Args:
            low: minimum value
            high: maximum value
        """
        low = self._prepare_parameter(name, 'low', low)
        high = self._prepare_parameter(name, 'high', high)

        shape = broadcast_shapes(get_shape(low), get_shape(high))

        super(UniformPrior, self).__init__(name, shape, [low, high], tracked)

    def transform_U(self, U, low, high, **kwargs):
        return low + U * (high - low)


class Gumbel(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, shape, tracked=True):
        """
        Standard Gumbel distribution.
        Args:
            shape: shape of output
        """
        super(Gumbel, self).__init__(name, shape, [], tracked)

    def transform_U(self, U, **kwargs):
        return -jnp.log(-jnp.log(jnp.maximum(U, jnp.finfo(U.dtype).eps)))


class CauchyPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, mu, gamma, tracked=True):
        """
        Cauchy distribution
        Args:
            mu: mean
            gamma: scale
        """
        mu = self._prepare_parameter(name, 'mu', mu)
        gamma = self._prepare_parameter(name, 'gamma', gamma)
        shape = broadcast_shapes(get_shape(mu), get_shape(gamma))
        super(CauchyPrior, self).__init__(name, shape, [mu, gamma], tracked)

    def transform_U(self, U, mu, gamma, **kwargs):
        return jnp.tan(jnp.pi * (U - 0.5)) * gamma + mu


class GammaPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, k, theta, tracked=True):
        """
        Gamma distribution
        Args:
            k: shape
            theta: scale
        """
        k = self._prepare_parameter(name, 'k', k)
        theta = self._prepare_parameter(name, 'theta', theta)
        shape = broadcast_shapes(get_shape(k), get_shape(theta))
        super(GammaPrior, self).__init__(name, shape, [k, theta], tracked)

    def log_homogeneous_measure(self, X, k, theta):
        """
        Gamma probability density function.
        """
        return -gammaln(k) - k * jnp.log(theta) + (k - 1.) * X - X / theta

    def transform_U(self, U, k, theta, **kwargs):
        """
        Transforms U to a range that covers most of the prior volume.
        We use the homogeneous measure to do the rest.
        """
        mean = k * theta
        stddev = jnp.sqrt(k * theta**2)
        return U * (mean + 5.*stddev)
