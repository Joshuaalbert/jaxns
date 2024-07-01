from functools import partial
from typing import Tuple, Union, Optional, Literal

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp, vmap, lax
from jax._src.scipy.special import gammaln

from jaxns.framework.bases import BaseAbstractPrior
from jaxns.framework.prior import SingularPrior, prior_to_parametrised_singular
from jaxns.internals.interp_utils import InterpolatedArray
from jaxns.internals.log_semiring import cumulative_logsumexp
from jaxns.internals.types import FloatArray, IntArray, BoolArray, float_type, int_type, UType, RandomVariableType, \
    MeasureType

tfpd = tfp.distributions

__all__ = [
    "Bernoulli",
    "Beta",
    "Categorical",
    "ForcedIdentifiability",
    "Poisson",
    "UnnormalisedDirichlet",
    "Empirical",
    "TruncationWrapper"
]


class SpecialPrior(BaseAbstractPrior):
    def parametrised(self, random_init: bool = False) -> SingularPrior:
        """
        Convert this prior into a non-Bayesian parameter, that takes a single value in the model, but still has an associated
        log_prob. The parameter is registered as a `hk.Parameter` with added `_param` name suffix.

        Args:
            random_init: whether to initialise the parameter randomly or at the median of the distribution.

        Returns:
            A singular prior.
        """
        return prior_to_parametrised_singular(self, random_init=random_init)


class Bernoulli(SpecialPrior):
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


class Beta(SpecialPrior):
    def __init__(self, *, concentration0=None, concentration1=None, name: Optional[str] = None):
        super(Beta, self).__init__(name=name)
        # Special cases for Beta that are faster use the Kumaraswamy distribution
        if isinstance(concentration0, (float, int)) and concentration0 == 1:
            self.dist = tfpd.Kumaraswamy(concentration0=concentration0, concentration1=concentration1)
        elif isinstance(concentration1, (float, int)) and concentration1 == 1:
            self.dist = tfpd.Kumaraswamy(concentration0=concentration0, concentration1=concentration1)
        elif isinstance(concentration0, np.ndarray) and np.all(concentration0 == 1):
            self.dist = tfpd.Kumaraswamy(concentration0=concentration0, concentration1=concentration1)
        elif isinstance(concentration1, np.ndarray) and np.all(concentration1 == 1):
            self.dist = tfpd.Kumaraswamy(concentration0=concentration0, concentration1=concentration1)
        else:
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
        X = self.dist.quantile(U)
        return X.astype(self.dtype)


class Categorical(SpecialPrior):
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
        if self._parametrisation == 'cdf':
            return self.dist.cdf(X)
        raise NotImplementedError()

    def _log_prob(self, X) -> FloatArray:
        return self.dist.log_prob(X)

    def _quantile_gumbelmax(self, U):
        logits = self.dist._logits_parameter_no_checks()
        sample_dtype = self.dtype
        z = -jnp.log(-jnp.log(U))  # gumbel
        draws = jnp.argmax(logits + z, axis=-1).astype(sample_dtype)
        return draws

    def _cdf_gumbelmax(self, X):
        logits = self.dist._logits_parameter_no_checks()
        z = jnp.max(logits, axis=-1, keepdims=True)
        logits -= z
        return jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)

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


class ForcedIdentifiability(SpecialPrior):
    """
    Prior for a sequence of `n` random variables uniformly distributed on U[low, high] such that U[i,...] <= U[i+1,...].
    For broadcasting the resulting random variable is sorted on the first dimension elementwise.

    Args:
        n: number of samples within [low,high]
        low: minimum of distribution
        high: maximum of distribution
        fix_left: if True, the leftmost value is fixed to `low`
        fix_right: if True, the rightmost value is fixed to `high`
    """

    def __init__(self, *, n: int, low=None, high=None, fix_left: bool = False, fix_right: bool = False,
                 name: Optional[str] = None):
        super(ForcedIdentifiability, self).__init__(name=name)
        n_min = (1 if fix_left else 0) + (1 if fix_right else 0)
        if n < n_min:
            raise ValueError(f'`n` too small for fix_left={fix_left} and fix_right={fix_right}')
        self.n = n
        low, high = jnp.broadcast_arrays(low, high)
        self.low = low
        self.high = high
        self.fix_left = fix_left
        self.fix_right = fix_right

    def _dtype(self):
        return float_type

    def _base_shape(self) -> Tuple[int, ...]:
        num_base = self.n
        if self.fix_left:
            num_base -= 1
        if self.fix_right:
            num_base -= 1
        return (num_base,) + np.shape(self.low)

    def _shape(self) -> Tuple[int, ...]:
        return (self.n,) + np.shape(self.low)

    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        return self._quantile(U)

    def _inverse(self, X) -> FloatArray:
        return self._cdf(X)

    def _log_prob(self, X) -> FloatArray:
        n = self.n
        if self.fix_left:
            n -= 1
        if self.fix_right:
            n -= 1
        log_n_fac = gammaln(n + 1)
        diff = self.high - self.low
        log_prob = - log_n_fac - n * jnp.log(diff)
        # no check that X is inside high and low
        return log_prob

    def _cdf(self, X):
        if self.fix_left:
            X = X[1:]
        if self.fix_right:
            X = X[:-1]

        n = self.n
        if self.fix_left:
            n -= 1
        if self.fix_right:
            n -= 1

        log_output = jnp.log((X - self.low) / (self.high - self.low))

        # Step 2: Undo cumulative sum operation
        inner = jnp.diff(log_output[::-1], prepend=0, axis=0)[::-1]

        # Step 3: Undo reshaping and division
        k = jnp.arange(n) + 1
        log_x = inner * jnp.reshape(k, (n,) + (1,) * (len(self.shape) - 1))

        # Step 4: Find U by exponentiating log_x
        U = jnp.exp(log_x)

        return U.astype(self.dtype)

    def _quantile(self, U):
        n = self.n
        if self.fix_left:
            n -= 1
        if self.fix_right:
            n -= 1
        log_x = jnp.log(U)  # [n, ...]
        k = jnp.arange(n) + 1
        inner = log_x / lax.reshape(k, (n,) + (1,) * (len(self.shape) - 1))
        log_output = jnp.cumsum(inner[::-1], axis=0)[::-1]
        output = self.low + (self.high - self.low) * jnp.exp(log_output)
        if self.fix_left:
            output = jnp.concatenate([self.low[None], output], axis=0)
        if self.fix_right:
            output = jnp.concatenate([output, self.high[None]], axis=0)
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


class Poisson(SpecialPrior):
    def __init__(self, *, rate=None, log_rate=None, name: Optional[str] = None):
        super(Poisson, self).__init__(name=name)
        self.dist = tfpd.Poisson(rate=rate, log_rate=log_rate)

    def _dtype(self):
        return int_type

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


class UnnormalisedDirichlet(SpecialPrior):
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


class Empirical(SpecialPrior):
    """
    Represents the empirical distribution of a set of 1D samples, with arbitrary batch dimension.
    """

    def __init__(self, *, samples: jax.Array, resolution: int = 100, name: Optional[str] = None):
        super(Empirical, self).__init__(name=name)
        if len(np.shape(samples)) < 1:
            raise ValueError("Samples must have at least one dimension")
        if np.size(samples) == 0:
            raise ValueError("Samples must have at least one element")
        if resolution < 1:
            raise ValueError("Resolution must be at least 1")
        samples = jnp.asarray(samples)
        self._q = jnp.linspace(0., 100., resolution + 1)
        self._percentiles = jnp.reshape(jnp.percentile(samples, self._q, axis=-1), (resolution + 1, -1))

        self._batch_shape = np.shape(samples)[:-1]
        self._samples_dtype = samples.dtype

    def _dtype(self):
        return self._samples_dtype

    def _base_shape(self) -> Tuple[int, ...]:
        return self._batch_shape

    def _shape(self) -> Tuple[int, ...]:
        return self._batch_shape

    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        return self._quantile(U)

    def _inverse(self, X) -> FloatArray:
        return self._cdf(X)

    def _cdf(self, X) -> FloatArray:
        X_flat = jnp.ravel(X)
        u = jax.vmap(lambda x, per: jnp.interp(x, per, self._q * 1e-2), in_axes=(0, 1))(X_flat, self._percentiles)
        u = lax.reshape(u, X.shape)
        return u

    def _log_prob(self, X) -> FloatArray:
        X_flat = jnp.ravel(X)

        def _cdf(x):
            return self._cdf(x)

        def log_pdf(x):
            return jnp.log(jax.grad(_cdf)(x))

        log_prob = lax.reshape(jax.vmap(log_pdf)(X_flat), X.shape)
        return log_prob

    def _quantile(self, U):
        U_flat = jnp.ravel(U)
        x = jax.vmap(lambda u, per: jnp.interp(u * 100., self._q, per), in_axes=(0, 1))(U_flat, self._percentiles)
        x = lax.reshape(x, U.shape)
        return x


class TruncationWrapper(SpecialPrior):
    """
    Wraps another prior to make it truncated.

    For truncated distribution the quantile transforms to:

        Q_truncated(p) = Q_untruncated( p * (F_truncated(high) - F_truncated(low)) + F_truncated(low))

    And the CDF transforms to:

        F_truncated(x) = (F_untruncated(x) - F_untruncated(low)) / (F_untruncated(high) - F_untruncated(low))
    """

    def __init__(self, prior: BaseAbstractPrior, low: Union[jax.Array, float], high: Union[jax.Array, float],
                 name: Optional[str] = None):
        super(TruncationWrapper, self).__init__(name=name)
        self.prior = prior
        self.low = jnp.minimum(low, high)
        self.high = jnp.maximum(low, high)
        self.cdf_low = self.prior._inverse(self.low)
        self.cdf_diff = self.prior._inverse(self.high) - self.prior._inverse(self.low)

    def _inverse(self, X: RandomVariableType) -> UType:
        return jnp.clip((self.prior._inverse(X) - self.cdf_low) / jnp.maximum(self.cdf_diff, 1e-6),
                        0., 1.)

    def _forward(self, U: UType) -> RandomVariableType:
        return jnp.clip(self.prior._forward(jnp.clip(U * self.cdf_diff + self.cdf_low,
                                                     0., 1.)),
                        self.low, self.high)

    def _log_prob(self, X: RandomVariableType) -> MeasureType:
        outside_mask = jnp.bitwise_or(X < self.low, X > self.high)
        return jnp.where(outside_mask, -jnp.inf, self.prior._log_prob(X) - jnp.log(self.cdf_diff))

    def _base_shape(self) -> Tuple[int, ...]:
        return self.prior._base_shape()

    def _shape(self) -> Tuple[int, ...]:
        return self.prior._shape()

    def _dtype(self) -> jnp.dtype:
        return self.prior._dtype()


class ExplicitDensityPrior(SpecialPrior):
    def __init__(self, *, axes: Tuple[jax.Array, ...], density: jax.Array, regular_grid: bool = False,
                 name: Optional[str] = None):
        super(ExplicitDensityPrior, self).__init__(name=name)
        self._num_dims = num_dims = len(np.shape(density))
        for i in range(num_dims):
            if len(np.shape(axes[i])) != 1:
                raise ValueError(f"Each axis must be 1D, got {np.shape(axes[i])}")
            if np.shape(density)[i] < 2:
                raise ValueError(f"Each dimension of density must have at least 2 elements, got {np.shape(density)}")
            if np.shape(density)[i] != np.size(axes[i]):
                raise ValueError(
                    f"Each dimension of density must have the same number of elements as the corresponding axis, "
                    f"got {np.shape(density)} and {np.shape(axes[i])}"
                )

        norm = density
        for i in range(num_dims)[::-1]:
            axis = axes[i]
            d = axis[1:] - axis[:-1]
            norm = 0.5 * jnp.sum((norm[..., :-1] + norm[..., 1:]) * d, axis=-1)

        self._density = density / norm
        self._axes = axes
        self._regular_grid = regular_grid

    def _dtype(self):
        return self._density.dtype

    def _base_shape(self) -> Tuple[int, ...]:
        return (self._num_dims,)

    def _shape(self) -> Tuple[int, ...]:
        return (self._num_dims,)

    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        return self._quantile(U)

    def _inverse(self, X) -> FloatArray:
        return self._cdf(X)

    def _cdf(self, X) -> FloatArray:
        U = []
        for i in range(self._num_dims):
            # Marginalise the last dims
            density = self._density
            for j in range(i + 1, self._num_dims)[::-1]:
                axis = self._axes[j]
                d = axis[1:] - axis[:-1]
                density = 0.5 * jnp.sum((density[..., :-1] + density[..., 1:]) * d, axis=-1)

            # Now we have P(X_0, ..., X_i), interpolate to get norm * P(X_i | X_0, ...)

            for j in range(i):
                P = InterpolatedArray(
                    x=self._axes[j],
                    values=density,
                    axis=0,
                    regular_grid=self._regular_grid
                )
                density = P(X[j])  # P(X_1, ..., X_i)   [N1, ..., Ni]
            # Now we have P(X_i | X_0, ...), and we compute percentiles
            d = self._axes[i][1:] - self._axes[i][:-1]
            norm = 0.5 * jnp.sum((density[1:] + density[:-1]) * d)
            density /= norm
            percentiles = jnp.concatenate([jnp.asarray([0.]), jnp.cumsum(0.5 * (density[1:] + density[:-1]) * d)])
            u_i = jnp.interp(X[i], self._axes[i], percentiles)
            U.append(u_i)
        return jnp.stack(U, axis=-1)

    def _log_prob(self, X) -> FloatArray:
        density = self._density
        for i in range(self._num_dims):
            P = InterpolatedArray(
                x=self._axes[i],
                values=density,
                axis=0,
                regular_grid=self._regular_grid
            )
            density = P(X[i])
        return jnp.log(density)

    def _quantile(self, U):
        # Sequentially sample from: P(X_0), P(X_1 | X_0), P(X_2 | X_0, X_1), ...
        # P(A| B) = P(A, B) / sum_A P(B, A)
        X = []
        for i in range(self._num_dims):
            # Marginalise the last dims
            density = self._density
            for j in range(i + 1, self._num_dims)[::-1]:
                axis = self._axes[j]
                d = axis[1:] - axis[:-1]
                density = 0.5 * jnp.sum((density[..., :-1] + density[..., 1:]) * d, axis=-1)

            # Now we have P(X_0, ..., X_i), interpolate to get norm * P(X_i | X_0, ...)

            for j in range(i):
                P = InterpolatedArray(
                    x=self._axes[j],
                    values=density,
                    axis=0,
                    regular_grid=self._regular_grid
                )
                density = P(X[j])  # P(X_1, ..., X_i)   [N1, ..., Ni]
            # Now we have P(X_i | X_0, ...), and we compute percentiles
            d = self._axes[i][1:] - self._axes[i][:-1]
            norm = 0.5 * jnp.sum((density[1:] + density[:-1]) * d)
            density /= norm
            percentiles = jnp.concatenate([jnp.asarray([0.]), jnp.cumsum(0.5 * (density[1:] + density[:-1]) * d)])
            x_i = jnp.interp(U[i], percentiles, self._axes[i])
            X.append(x_i)
        return jnp.stack(X, axis=-1)
