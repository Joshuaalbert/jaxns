import numpy as np
from jax import numpy as jnp
from jax.scipy.special import gammaln

from jaxns.internals.shapes import broadcast_shapes
from jaxns.prior_transforms import ContinuousPrior, prior_docstring, get_shape, Prior


class StrictlyPositivePrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, shape, params, tracked=True):
        """
        Manages a strictly positive RV, without a quantile, but with a density.
        Args:
            shape: shape of RV
            *params: parameters, which must already be Prior type.
        """
        for param in params:
            if not isinstance(param, Prior):
                raise TypeError(f"Expected Prior, got {type(param)}.")
        super(StrictlyPositivePrior, self).__init__(name, shape, params, tracked)

    def _support_width(self, *params):
        """
                Estimate of prior width, ideally the standard-deviation for distributions with well-defined
                standard-deviation.
        """
        raise NotImplementedError()

    def _support_center(self, *params):
        """
        Estimate of prior center, ideally the mean for distributions with well-defined means.
        """
        raise NotImplementedError()

    def _log_prob(self, X, *params):
        """
        Returns the log-probability of U given the *params.

            i.e. log(p(U | params))

        Args:
            X: Value of U
            *params: parameters of distribution

        Returns:
            log(p(U | params)) of event-size dimension. This will be reduced with a sum.
        """
        raise NotImplementedError()

    def log_homogeneous_measure(self, X, *params):
        """
        Provides the log-homogenous measure, which is the prior density divided by surrogate density,
        which is the exponential distribution.
        """
        try:
            mean = self._support_center(*params)
        except NotImplementedError:
            mean = jnp.asarray(1.)
        try:
            variance = self._support_width(*params)
        except NotImplementedError:
            variance = jnp.asarray(1.)

        stddev = jnp.sqrt(jnp.maximum(jnp.zeros_like(variance), variance))

        b = mean + 2. * stddev
        log_prob_prior = self._log_prob(X, *params)
        log_prob_exponential = -jnp.log(b) - X / b
        return jnp.sum(log_prob_prior - log_prob_exponential)

    def transform_U(self, U, *params, **kwargs):
        """
        Transforms U to a range that covers the prior support.
        We use the homogeneous measure to do the rest.
        """
        try:
            center = self._support_center(*params)
        except NotImplementedError:
            center = jnp.asarray(1.)
        try:
            width = self._support_width(*params)
        except NotImplementedError:
            width = jnp.asarray(1.)

        b = center + 2. * width
        exponential = - b * jnp.sign(0.5 * U) * jnp.log(1. - 2. * jnp.abs(0.5 * U))
        return exponential

class GammaPrior(StrictlyPositivePrior):
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

    def _support_center(self, k, theta):
        mean = k * theta
        return mean

    def _support_width(self, k, theta):
        variance = k * theta ** 2
        return jnp.sqrt(variance)

    def _log_prob(self, X, k, theta):
        return -gammaln(k) - k * jnp.log(theta) + (k - 1.) * jnp.log(X) - X / theta

class RealPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, shape, params, tracked=True):
        """
        Manages a strictly positive RV, without a quantile, but with a density.
        Args:
            shape: shape of RV
            *params: parameters, which must already be Prior type.
        """
        for param in params:
            if not isinstance(param, Prior):
                raise TypeError(f"Expected Prior, got {type(param)}.")
        super(RealPrior, self).__init__(name, shape, params, tracked)

    def _support_width(self, *params):
        """
        Estimate of prior width, ideally the FWHM of the probability density function of prior.
        """
        raise NotImplementedError()

    def _support_center(self, *params):
        """
        Estimate of prior center, ideally the mean for distributions with well-defined means.
        """
        raise NotImplementedError()

    def _log_prob(self, X, *params):
        """
        Returns the log-probability of U given the *params.

            i.e. log(p(U | params))

        Args:
            X: Value of U
            *params: parameters of distribution

        Returns:
            log(p(U | params)) of event-size dimension. This will be reduced with a sum.
        """
        raise NotImplementedError()

    def log_homogeneous_measure(self, X, *params):
        """
        Provides the log-homogenous measure, which is the prior density divided by surrogate density,
        which is the exponential distribution.
        """
        try:
            center = self._support_center(*params)
        except NotImplementedError:
            center = jnp.asarray(0.)
        try:
            width = self._support_width(*params)
        except NotImplementedError:
            width = jnp.asarray(1.)
        mu = center
        sigma = width / (2. * np.sqrt(2.**(2./3.) - 1.))

        log_prob_prior = self._log_prob(X, *params)
        log_prob_cauchy = gammaln(1.) - gammaln(0.5) - 0.5 * jnp.log(jnp.pi) - jnp.log(sigma) \
                          - jnp.log(1. + (X - mu) ** 2 / sigma ** 2)
        return jnp.sum(log_prob_prior - log_prob_cauchy)

    def transform_U(self, U, *params, **kwargs):
        """
        Transforms U to a range that covers the prior support.
        We use the homogeneous measure to do the rest.
        """
        try:
            center = self._support_center(*params)
        except NotImplementedError:
            center = jnp.asarray(0.)
        try:
            width = self._support_width(*params)
        except NotImplementedError:
            width = jnp.asarray(1.)
        mu = center
        sigma = width / (2. * np.sqrt(2.**(2./3.) - 1.))

        cauchy = jnp.tan(jnp.pi * (U - 0.5)) * sigma + mu
        return cauchy

class StudentT(RealPrior):
    @prior_docstring
    def __init__(self, name, nu, mu, sigma, tracked=True):
        """
        Student-T distribution
        Args:
            nu: degree
            mu: mean
            sigma: scale
        """
        nu = self._prepare_parameter(name, 'nu', nu)
        mu = self._prepare_parameter(name, 'mu', mu)
        sigma = self._prepare_parameter(name, 'sigma', sigma)
        shape = broadcast_shapes(broadcast_shapes(get_shape(nu), get_shape(mu)),
                                 get_shape(sigma))
        super(StudentT, self).__init__(name, shape, [nu, mu, sigma], tracked)

    def _support_center(self, nu, mu, sigma):
        """
        We return mode, and also mean when nu > 1.
        """
        return mu

    def _support_width(self, nu, mu, sigma):
        """
        We use the FWHM of the student-T.
        """
        fwhm = 2. * sigma * jnp.sqrt(nu * jnp.maximum(jnp.zeros_like(mu), jnp.power(2., 2./(nu+1)) - 1.))
        return fwhm

    def _log_prob(self, X, nu, mu, sigma):
        """
        Student-T log-prob.
        """
        log_prob_student_t = gammaln(0.5 * (nu + 1.)) - gammaln(0.5 * nu) - 0.5 * jnp.log(jnp.pi * nu) - jnp.log(sigma) \
                             + (-0.5 * (nu + 1.)) * jnp.log(1. + (X - mu) ** 2 / sigma ** 2 / nu)
        return log_prob_student_t
