from typing import Tuple, Union, Optional, Literal

import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp, vmap, lax
from jax._src.scipy.special import gammaln
from tensorflow_probability.substrates.jax.math import lbeta, betaincinv

from jaxns.internals.log_semiring import cumulative_logsumexp
from jaxns.framework.bases import BaseAbstractPrior
from jaxns.internals.types import FloatArray, IntArray, BoolArray, float_type

tfpd = tfp.distributions

__all__ = [
    "Bernoulli",
    "Beta",
    "Categorical",
    "ForcedIdentifiability",
    "Poisson",
    "UnnormalisedDirichlet"
]


class Bernoulli(BaseAbstractPrior):
    def __init__(self, *, logits=None, probs=None, name: Optional[str] = None):
        super(Bernoulli, self).__init__(name=name)
        self.dist = tfpd.Bernoulli(logits=logits, probs=probs)

    def _dtype(self):
        return self.dist.dtype

    def _base_shape(self) -> Tuple[int, ...]:
        return self._shape()

    def _shape(self) -> Tuple[int, ...]:
        return tuple(self.dist.batch_shape_tensor()) + tuple(self.dist.event_shape_tensor())

    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        return self._quantile(U)

    def _inverse(self, X) -> FloatArray:
        return self.dist.cdf(X)

    def _log_prob(self, X) -> FloatArray:
        return self.dist.log_prob(X)

    def _quantile(self, U):
        probs = self.dist._probs_parameter_no_checks()
        sample = jnp.less(U, probs)
        return sample.astype(self.dtype)


class Beta(BaseAbstractPrior):
    def __init__(self, *, concentration0=None, concentration1=None, name: Optional[str] = None):
        super(Beta, self).__init__(name=name)
        self.dist = tfpd.Beta(concentration0=concentration0, concentration1=concentration1)

    def _dtype(self):
        return self.dist.dtype

    def _base_shape(self) -> Tuple[int, ...]:
        return self._shape()

    def _shape(self) -> Tuple[int, ...]:
        return tuple(self.dist.batch_shape_tensor()) + tuple(self.dist.event_shape_tensor())

    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        return self._quantile(U)

    def _inverse(self, X) -> FloatArray:
        return self.dist.cdf(X)

    def _log_prob(self, X) -> FloatArray:
        return self.dist.log_prob(X)

    def _quantile(self, U):
        alpha = self.dist.concentration0
        beta = self.dist.concentration1
        # cdf(x, a, b) = I(x,a,b) = B(x,a,b)/B(a,b)
        # G = B(a,b) cdf(x, a, b) = B(a,b) I(x,a,b) = B(x,a,b)
        # quantile(u, a, b) = B^{-1}(B(a,b) U, a, b)
        Y = jnp.exp(lbeta(alpha, beta) + jnp.log(U))
        X = betaincinv(alpha, beta, Y)
        return X.astype(self.dtype)


class Categorical(BaseAbstractPrior):
    def __init__(self, parametrisation: Literal['gumbel_max', 'cdf'], *, logits=None, probs=None,
                 name: Optional[str] = None):
        """
        Initialised Categorical special prior.

        Args:
            parametrisation: 'cdf' is good for discrete params with correlation between neighbouring categories,
                otherwise gumbel is better.
            logits: log-prob of each category
            probs: prob of each category
            name: optional name
        """
        super(Categorical, self).__init__(name=name)
        self.dist = tfpd.Categorical(logits=logits, probs=probs)
        self._parametrisation = parametrisation

    def _dtype(self):
        return self.dist.dtype

    def _base_shape(self) -> Tuple[int, ...]:
        if self._parametrisation == 'gumbel_max':
            return self._shape() + (self.dist._num_categories(),)
        elif self._parametrisation == 'cdf':
            return self._shape()

    def _shape(self) -> Tuple[int, ...]:
        return tuple(self.dist.batch_shape_tensor()) + tuple(self.dist.event_shape_tensor())

    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        if self._parametrisation == 'gumbel_max':
            return self._quantile_gumbelmax(U)
        elif self._parametrisation == 'cdf':
            return self._quantile_cdf(U)

    def _inverse(self, X) -> FloatArray:
        return self.dist.cdf(X)

    def _log_prob(self, X) -> FloatArray:
        return self.dist.log_prob(X)

    def _quantile_gumbelmax(self, U):
        logits = self.dist._logits_parameter_no_checks()
        sample_dtype = self.dtype
        z = -jnp.log(-jnp.log(U))  # gumbel
        draws = jnp.argmax(logits + z, axis=-1).astype(sample_dtype)
        return draws

    def _quantile_cdf(self, U):
        """
        The quantile for CDF parametrisation.

        Args:
            U: [...]

        Returns:
            [...]
        """
        logits = self.dist._logits_parameter_no_checks()  # [..., N]
        cum_logits = cumulative_logsumexp(logits, axis=-1)  # [..., N]
        cum_logits -= cum_logits[..., -1:]
        N = cum_logits.shape[-1]
        cum_logits_flat = jnp.reshape(cum_logits, (-1, N))
        log_U_flat = jnp.reshape(jnp.log(U), (-1,))

        category_flat = vmap(lambda a, v: jnp.searchsorted(a, v, side='left'))(cum_logits_flat, log_U_flat)
        category = jnp.reshape(category_flat, U.shape)
        return category


class ForcedIdentifiability(BaseAbstractPrior):
    """
    Prior for a sequence of `n` random variables uniformly distributed on U[low, high] such that U[i,...] <= U[i+1,...].
    For broadcasting the resulting random variable is sorted on the first dimension elementwise.

    Args:
        n: number of samples within [low,high]
        low: minimum of distribution
        high: maximum of distribution
    """

    def __init__(self, *, n: int, low=None, high=None, name: Optional[str] = None):
        super(ForcedIdentifiability, self).__init__(name=name)
        self.n = n
        low, high = jnp.broadcast_arrays(low, high)
        self.low = low
        self.high = high

    def _dtype(self):
        return float_type

    def _base_shape(self) -> Tuple[int, ...]:
        return (self.n,) + jnp.shape(self.low)

    def _shape(self) -> Tuple[int, ...]:
        return (self.n,) + jnp.shape(self.low)

    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        return self._quantile(U)

    def _inverse(self, X) -> FloatArray:
        return self._cdf(X)

    def _log_prob(self, X) -> FloatArray:
        log_n_fac = gammaln(self.n + 1)
        diff = self.high - self.low
        log_prob = - log_n_fac - self.n * jnp.log(diff)
        # no check that X is inside high and low
        return log_prob

    def _cdf(self, X):
        log_theta = jnp.log((X - self.low) / (self.high - self.low))  # [n, ...]

        def inv_body(state, X):
            (log_theta_prev,) = state
            (log_theta, i) = X
            log_x = i * (log_theta - log_theta_prev)
            return (log_theta,), (log_x,)

        # Initial log_x value
        log_init_x = jnp.zeros(self.shape[1:], self.dtype)  # [...]
        _, (log_x,) = lax.scan(inv_body, (log_init_x,), (log_theta, jnp.arange(1, self.n + 1)), reverse=True)
        U = jnp.exp(log_x)
        return U.astype(self.dtype)

    def _quantile(self, U):
        log_x = jnp.log(U)  # [n, ...]

        # theta[i] = theta[i-1] * (1 - x[i]) + theta_max * x[i]
        def body(state, X):
            (log_theta,) = state
            (log_x, i) = X
            log_theta = log_x / i + log_theta  # shrinkage
            return (log_theta,), (log_theta,)

        log_init_theta = jnp.zeros(self.shape[1:], self.dtype)  # [...] -- log(1)
        _, (log_theta,) = lax.scan(body, (log_init_theta,), (log_x, jnp.arange(1, self.n + 1)), reverse=True)
        theta = self.low + (self.high - self.low) * jnp.exp(log_theta)
        return theta.astype(self.dtype)


class Poisson(BaseAbstractPrior):
    def __init__(self, *, rate=None, log_rate=None, name: Optional[str] = None):
        super(Poisson, self).__init__(name=name)
        self.dist = tfpd.Poisson(rate=rate, log_rate=log_rate)

    def _dtype(self):
        return self.dist.dtype

    def _base_shape(self) -> Tuple[int, ...]:
        return self._shape()

    def _shape(self) -> Tuple[int, ...]:
        return tuple(self.dist.batch_shape_tensor()) + tuple(self.dist.event_shape_tensor())

    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        return self._quantile(U)

    def _inverse(self, X) -> FloatArray:
        return self.dist.cdf(X)

    def _log_prob(self, X) -> FloatArray:
        return self.dist.log_prob(X)

    def _quantile(self, U):
        """
        Algorithmic Poisson generator based upon the inversion by sequential search

        .. code-block::

                init:
                    Let log_x ← -inf, log_p ← −λ, log_s ← log_p.
                    Generate uniform random number u in [0,1].
                while log_u > log_s do:
                    log_x ← logaddexp(log_x, 0).
                    log_p ← log_p + log_λ - log_x.
                    log_s ← logaddexp(log_s, log_p).
                return exp(log_x).
        """
        log_rate = self.dist.log_rate_parameter()
        rate = self.dist.rate_parameter()
        tiny = 1e-7  # For anything smaller it never terminates
        U = jnp.minimum(U, jnp.asarray(1., U.dtype) - tiny)
        log_u = jnp.log(U)
        log_x = jnp.full(log_u.shape, -jnp.inf)
        log_p = -rate
        log_s = log_p

        def body(state):
            (_log_x, _log_p, _log_s) = state
            impute = log_u <= _log_s
            log_x = jnp.logaddexp(_log_x, 0.)
            log_p = _log_p + log_rate - log_x
            log_s = jnp.logaddexp(_log_s, log_p)

            log_x = jnp.where(impute, _log_x, log_x)
            log_p = jnp.where(impute, _log_p, log_p)
            log_x = jnp.where(impute, _log_x, log_x)

            return (log_x, log_p, log_s)

        (log_x, log_p, log_s) = lax.while_loop(lambda s: jnp.any(log_u > s[2]),
                                               body,
                                               (log_x, log_p, log_s))
        return jnp.exp(log_x).astype(self.dtype)


class UnnormalisedDirichlet(BaseAbstractPrior):
    """
    Represents an unnormalised dirichlet distribution of K classes.
    That is, the output is related to the K-simplex via normalisation.

    X ~ UnnormalisedDirichlet(alpha)
    Y = X / sum(X) ==> Y ~ Dirichlet(alpha)
    """

    def __init__(self, *, concentration, name: Optional[str] = None):
        super(UnnormalisedDirichlet, self).__init__(name=name)
        self._dirichlet_dist = tfpd.Dirichlet(concentration=concentration)
        self._gamma_dist = tfpd.Gamma(concentration=concentration, log_rate=0.)

    def _dtype(self):
        return self._dirichlet_dist.dtype

    def _base_shape(self) -> Tuple[int, ...]:
        return self._shape()

    def _shape(self) -> Tuple[int, ...]:
        return tuple(self._dirichlet_dist.batch_shape_tensor()) + tuple(self._dirichlet_dist.event_shape_tensor())

    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        return self._quantile(U)

    def _inverse(self, X) -> FloatArray:
        return self._gamma_dist.cdf(X)

    def _log_prob(self, X) -> FloatArray:
        return jnp.sum(self._gamma_dist.log_prob(X), axis=-1)

    def _quantile(self, U):
        gamma = self._gamma_dist.quantile(U).astype(self.dtype)
        return gamma
