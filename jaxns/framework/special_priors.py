from functools import partial
from typing import Tuple, Union, Optional, Literal

import jax
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp, vmap, lax
from jax._src.scipy.special import gammaln
from tensorflow_probability.substrates.jax.math import lbeta, betaincinv

from jaxns.framework.bases import BaseAbstractPrior
from jaxns.internals.log_semiring import cumulative_logsumexp
from jaxns.internals.types import FloatArray, IntArray, BoolArray, float_type, int_type

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
        return (self.n,) + np.shape(self.low)

    def _shape(self) -> Tuple[int, ...]:
        return (self.n,) + np.shape(self.low)

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
        log_output = jnp.log((X - self.low) / (self.high - self.low))

        # Step 2: Undo cumulative sum operation
        inner = jnp.diff(log_output[::-1], prepend=0, axis=0)[::-1]

        # Step 3: Undo reshaping and division
        k = jnp.arange(self.n) + 1
        log_x = inner * jnp.reshape(k, (self.n,) + (1,) * (len(self.shape) - 1))

        # Step 4: Find U by exponentiating log_x
        U = jnp.exp(log_x)

        return U.astype(self.dtype)

    def _quantile(self, U):
        log_x = jnp.log(U)  # [n, ...]
        k = jnp.arange(self.n) + 1
        inner = log_x / jnp.reshape(k, (self.n,) + (1,) * (len(self.shape) - 1))
        log_output = jnp.cumsum(inner[::-1], axis=0)[::-1]
        output = self.low + (self.high - self.low) * jnp.exp(log_output)
        return output.astype(self.dtype)


def _poisson_quantile_bisection(U, rate, max_iter=15, unroll: bool = True):
    # max_iter is set so that error < 1 up to rate of 1e4
    rate = jnp.maximum(jnp.asarray(rate), 1e-5)
    if np.size(rate) > 1:
        raise ValueError("Rate must be a scalar")

    if np.size(U) > 1:
        U_flat = U.ravel()
        x_final, x_results = vmap(lambda u: _poisson_quantile_bisection(u, rate, max_iter, unroll))(U_flat)
        return x_final.reshape(U.shape), x_results.reshape(U.shape + (max_iter,))

    def smooth_cdf(x, rate):
        return lax.igammac(x + 1., rate)

    def fixed_point_update(x, args):
        (a, b, f_a, f_b) = x

        c = 0.5 * (a + b)
        f_c = smooth_cdf(c, rate)

        left = f_c > U
        a1 = jnp.where(left, a, c)
        f_a1 = jnp.where(left, f_a, f_c)
        b1 = jnp.where(left, c, b)
        f_b1 = jnp.where(left, f_c, f_b)

        a2 = a
        f_a2 = f_a
        b2 = b * 2.
        f_b2 = smooth_cdf(b2, rate)

        bounded = f_b >= U  # a already bounds.

        a = jnp.where(bounded, a1, a2)
        b = jnp.where(bounded, b1, b2)
        f_a = jnp.where(bounded, f_a1, f_a2)
        f_b = jnp.where(bounded, f_b1, f_b2)

        new_x = (a, b, f_a, f_b)

        return new_x, 0.5 * (a + b)

    a = jnp.asarray(0.)
    b = jnp.asarray(rate)
    f_a = jnp.asarray(0.)
    f_b = smooth_cdf(b, rate)
    init = (a, b, f_a, f_b)

    # Dummy array to facilitate using scan for a fixed number of iterations
    (a, b, f_a, f_b), x_results = lax.scan(
        fixed_point_update,
        init,
        jnp.arange(max_iter),
        unroll=max_iter if unroll else 1
    )

    c = 0.5 * (a + b)

    return c, x_results


@partial(jax.jit, static_argnames=("unroll",))
def _poisson_quantile(U, rate, unroll: bool = False):
    x, _ = _poisson_quantile_bisection(U, rate, unroll=unroll)
    return x.astype(int_type)


@pytest.mark.parametrize("rate, error", (
        [2.0, 1.],
        [10., 1.],
        [100., 1.],
        [1000., 1.],
        [10000., 1.]
)
                         )
def test_poisson_quantile_bisection(rate, error):
    U = jnp.linspace(0., 1. - np.spacing(1.), 1000)
    x, x_results = _poisson_quantile_bisection(U, rate, unroll=False)
    diff_last_two = jnp.abs(x_results[..., -1] - x_results[..., -2])

    # Make sure less than 1 apart
    assert jnp.all(diff_last_two <= error)


@pytest.mark.parametrize("rate", [2.0, 10., 100., 1000., 10000.])
def test_poisson_quantile(rate):
    U = jnp.linspace(0., 1. - np.spacing(1.), 10000)
    x = _poisson_quantile(U, rate)
    assert jnp.all(jnp.isfinite(x))


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
        rate = self.dist.rate_parameter()
        if np.size(rate) > 1:
            return jax.vmap(lambda u, r: _poisson_quantile(u, r))(U.ravel(), rate.ravel()).reshape(self.shape)
        return _poisson_quantile(U, rate)


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
