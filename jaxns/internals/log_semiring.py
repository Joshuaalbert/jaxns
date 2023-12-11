from typing import Union

from jax import numpy as jnp, lax
from jax.scipy.special import logsumexp

from jaxns.internals.types import SignedLog, float_type


def logaddexp(x1, x2):
    """
    Equivalent to logaddexp but supporting complex arguments.

    see np.logaddexp
    """
    if is_complex(x1) or is_complex(x2):
        select1 = x1.real > x2.real
        amax = jnp.where(select1, x1, x2)
        delta = jnp.where(select1, x2 - x1, x1 - x2)
        return jnp.where(jnp.isnan(delta),
                         x1 + x2,  # NaNs or infinities of the same sign.
                         amax + jnp.log1p(jnp.exp(delta)))
    else:
        return jnp.logaddexp(x1, x2)


def signed_logaddexp(log_abs_val1, sign1, log_abs_val2, sign2):
    r"""
    Equivalent of logaddexp but for signed quantities too.
    Broadcasting supported.

    Args:
        log_abs_val1: Logarithm of absolute value of val1, :math:`\log(|x_1|)`
        sign1: Sign of val1, :math:`\mathrm{sign}(x_1)`
        log_abs_val2: Logarithm of absolute value of val2, :math:`\log(|x_2|)`
        sign2: Sign of val2, :math:`\mathrm{sign}(x_2)`

    Returns:
        (:math:`\log(|x_1+x_2|)`, :math:`\mathrm{sign}(x_1+x_2)`)
    """
    amax = jnp.maximum(log_abs_val1, log_abs_val2)
    signmax = jnp.where(log_abs_val1 > log_abs_val2, sign1, sign2)
    delta = -jnp.abs(log_abs_val2 - log_abs_val1)  # nan iff inf - inf
    sign = sign1 * sign2
    return jnp.where(jnp.isnan(delta),
                     log_abs_val1 + log_abs_val2,  # NaNs or infinities of the same sign.
                     amax + jnp.log1p(sign * jnp.exp(delta))), signmax


def cumulative_logsumexp(u, sign=None, reverse=False, axis=0):
    if sign is not None:
        u, sign = jnp.broadcast_arrays(u, sign)

    def body(state, X):
        if sign is not None:
            (u, u_sign) = X
            (accumulant, accumulant_sign) = state
            new_accumulant, new_accumulant_sign = signed_logaddexp(accumulant, accumulant_sign, u, u_sign)
            return (new_accumulant, accumulant_sign), (new_accumulant, accumulant_sign)
        else:
            u = X
            accumulant = state
            new_accumulant = jnp.logaddexp(accumulant, u)
            return new_accumulant, new_accumulant

    if sign is not None:
        if axis != 0:
            sign = jnp.swapaxes(sign, axis, 0)
            u = jnp.swapaxes(u, axis, 0)
        state = (-jnp.inf * jnp.ones(u.shape[1:], dtype=u.dtype), jnp.ones(u.shape[1:], dtype=u.dtype))
        X = (u, sign)
    else:
        if axis != 0:
            u = jnp.swapaxes(u, axis, 0)
        state = -jnp.inf * jnp.ones(u.shape[1:], dtype=u.dtype)
        X = u
    _, result = lax.scan(body,
                         state,
                         X,
                         reverse=reverse)
    if sign is not None:
        v, v_sign = result
        if axis != 0:
            v = jnp.swapaxes(v, axis, 0)
            v_sign = jnp.swapaxes(v_sign, axis, 0)
        return v, v_sign
    else:
        v = result
        if axis != 0:
            v = jnp.swapaxes(v, axis, 0)
        return v


class LogSpace(object):
    def __init__(self, log_abs_val: Union[jnp.ndarray, float], sign: Union[jnp.ndarray, float] = None):
        self._log_abs_val = jnp.asarray(log_abs_val, float_type)
        if sign is None:
            self._sign = jnp.asarray(1., float_type)
            self._naked = True
        else:
            self._sign = jnp.asarray(sign, float_type)
            self._naked = False

    @property
    def dtype(self):
        return self.log_abs_val.dtype

    @property
    def log_abs_val(self):
        return self._log_abs_val

    @property
    def sign(self):
        return self._sign

    @property
    def value(self):
        if self._naked:
            return jnp.exp(self.log_abs_val)
        return self.sign * jnp.exp(self.log_abs_val)

    def __neg__(self):
        if self._naked:
            return LogSpace(self.log_abs_val, -jnp.ones_like(self.log_abs_val))
        return LogSpace(self.log_abs_val, -self.sign)

    def __add__(self, other):
        """
        Implements addition in log space

            log(exp(log_A) + exp(log_B))

        Args:
            other: ndarray or LogSpace, if ndarray assumed to be log(B)

        Returns:
             LogSpace
        """
        if not isinstance(other, LogSpace):
            raise TypeError(f"Expected type {type(self)} got {type(other)}")
        if self._naked and other._naked:  # no coefficients
            return LogSpace(jnp.logaddexp(self._log_abs_val, other._log_abs_val))
        return LogSpace(*signed_logaddexp(self._log_abs_val, self._sign, other._log_abs_val, other._sign))

    def __sub__(self, other):
        """
        Implements addition in log space

            log(exp(log_A) - exp(log_B))

        Args:
            other: ndarray or LogSpace, if ndarray assumed to be log(B)

        Returns:
             LogSpace
        """
        if not isinstance(other, LogSpace):
            raise TypeError(f"Expected type {type(self)} got {type(other)}")
        return LogSpace(*signed_logaddexp(self._log_abs_val, self._sign, other._log_abs_val, -other._sign))

    def __mul__(self, other):
        """
        Implements addition in log space

            log(exp(log_A) * exp(log_B))

        Args:
            other: ndarray or LogSpace, if ndarray assumed to be log(B)

        Returns:
             LogSpace
        """
        if not isinstance(other, LogSpace):
            raise TypeError(f"Expected type {type(self)} got {type(other)}")
        if self._naked and other._naked:  # no coefficients
            return LogSpace(self._log_abs_val + other._log_abs_val)
        return LogSpace(self._log_abs_val + other._log_abs_val, self._sign * other._sign)

    def __repr__(self):
        if self._naked:
            return f"LogSpace({self.log_abs_val})"
        return f"LogSpace({self.log_abs_val}, {self.sign})"


    def sum(self, axis=-1, keepdims=False):
        if not self._naked:  # no coefficients
            return LogSpace(*logsumexp(self.log_abs_val, b=self.sign, axis=axis, keepdims=keepdims, return_sign=True))
        return LogSpace(logsumexp(self._log_abs_val, axis=axis, keepdims=keepdims))

    def nansum(self, axis=-1, keepdims=False):
        log_abs_val = jnp.where(jnp.isnan(self.log_abs_val), -jnp.inf, self.log_abs_val)
        if not self._naked:  # no coefficients
            return LogSpace(*logsumexp(log_abs_val, b=self.sign, axis=axis, keepdims=keepdims, return_sign=True))
        return LogSpace(logsumexp(log_abs_val, axis=axis, keepdims=keepdims))

    def cumsum(self, axis=0, reverse=False):
        if not self._naked:  # no coefficients
            return LogSpace(*cumulative_logsumexp(self.log_abs_val, sign=self.sign, axis=axis, reverse=reverse))
        return LogSpace(cumulative_logsumexp(self._log_abs_val, axis=axis, reverse=reverse))

    def cumprod(self, axis=0):
        if not self._naked:  # no coefficients
            log_abs_val, sign = jnp.broadcast_arrays(self.log_abs_val, self.sign)
            return LogSpace(jnp.cumsum(log_abs_val, axis=axis), jnp.cumprod(sign, axis=axis))
        return LogSpace(jnp.cumsum(self._log_abs_val, axis=axis))

    def mean(self, axis=-1, keepdims=False):
        N = self._log_abs_val.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / LogSpace(jnp.log(N))

    def var(self, axis=-1, keepdims=False):
        return (self - self.mean(axis=axis, keepdims=True)).mean(axis=axis, keepdims=keepdims)

    def log(self):
        assert self._naked
        return LogSpace(jnp.log(jnp.abs(self.log_abs_val)), jnp.sign(self.log_abs_val))

    def exp(self):
        return LogSpace(self.value)

    def sqrt(self):
        return self ** 0.5

    def abs(self):
        return LogSpace(self.log_abs_val)

    def diff(self):
        if self._naked:
            log_abs_val, sign = jnp.broadcast_arrays(self.log_abs_val, self.sign)
            return LogSpace(log_abs_val[1:], sign[1:]) - LogSpace(log_abs_val[:-1], sign[:-1])
        else:
            return LogSpace(self.log_abs_val[1:]) - LogSpace(self.log_abs_val[:-1])

    def square(self):
        return self * self

    def argmax(self):
        return jnp.argmax(self.log_abs_val)

    def maximum(self, other: "LogSpace"):
        assert self._naked and other._naked
        return LogSpace(jnp.maximum(self.log_abs_val, other.log_abs_val))

    def minimum(self, other: "LogSpace"):
        assert self._naked and other._naked
        return LogSpace(jnp.minimum(self.log_abs_val, other.log_abs_val))

    def max(self):
        assert self._naked
        return LogSpace(jnp.max(self.log_abs_val))

    def min(self):
        assert self._naked
        return LogSpace(jnp.min(self.log_abs_val))

    def concatenate(self, other: "LogSpace", axis=0):
        if self._naked and other._naked:
            return LogSpace(jnp.concatenate([self.log_abs_val, other.log_abs_val], axis=axis))
        log_abs_val, sign = jnp.broadcast_arrays(self.log_abs_val, self.sign)
        _log_abs_val, _sign = jnp.broadcast_arrays(other.log_abs_val, other.sign)
        return LogSpace(jnp.concatenate([log_abs_val, _log_abs_val], axis=axis),
                        jnp.concatenate([sign, _sign], axis=axis))

    def __getitem__(self, item):
        if self._naked:
            return LogSpace(self.log_abs_val[item])
        log_abs_val, sign = jnp.broadcast_arrays(self.log_abs_val, self.sign)
        return LogSpace(log_abs_val[item], sign[item])

    @property
    def signed_log(self):
        return SignedLog(self.log_abs_val, self.sign)

    def __gt__(self, other):
        if not isinstance(other, LogSpace):
            raise TypeError(f"Expected type {type(self)} got {type(other)}")
        if self._naked and other._naked:
            return self.log_abs_val > other.log_abs_val
        return (self / other).value > 1.

    def __lt__(self, other):
        if not isinstance(other, LogSpace):
            raise TypeError(f"Expected type {type(self)} got {type(other)}")
        if self._naked and other._naked:
            return self.log_abs_val < other.log_abs_val
        return (self / other).value < 1.

    def __ge__(self, other):
        if not isinstance(other, LogSpace):
            raise TypeError(f"Expected type {type(self)} got {type(other)}")
        if self._naked and other._naked:
            return self.log_abs_val >= other.log_abs_val
        return (self / other).value >= 1.

    def __le__(self, other):
        if not isinstance(other, LogSpace):
            raise TypeError(f"Expected type {type(self)} got {type(other)}")
        if self._naked and other._naked:
            return self.log_abs_val <= other.log_abs_val
        return (self / other).value <= 1.

    @property
    def size(self):
        if self._naked:
            return self.log_abs_val.size
        log_abs_val, sign = jnp.broadcast_arrays(self.log_abs_val, self.sign)
        return log_abs_val.size

    def __pow__(self, n):
        """
        Implements power in log space

            log(exp(log_A)**n)

        Args:
            n: int or float

        Returns:
             LogSpace
        """
        if not isinstance(n, (int, float, jnp.ndarray)):
            raise NotImplementedError("Not implemented for non-int powers.")
        n = jnp.asarray(n, float_type)
        if self._naked:
            return LogSpace(n * self.log_abs_val)
        # complex values can occur if n is not even
        return LogSpace(n * self.log_abs_val, sign=self.sign ** n)
        # for _ in range(n-1):
        #     output = output * self
        # return output

    def __truediv__(self, other):
        """
        Implements addition in log space

            log(exp(log_A) / exp(log_B))

        Args:
            other: ndarray or LogSpace, if ndarray assumed to be log(B)

        Returns:
             LogSpace
        """
        if not isinstance(other, LogSpace):
            raise TypeError(f"Expected type {type(self)} got {type(other)}")
        if self._naked and other._naked:  # no coefficients
            return LogSpace(self._log_abs_val - other._log_abs_val)
        return LogSpace(self._log_abs_val - other._log_abs_val, self._sign * other._sign)


def is_complex(a):
    return a.dtype in [jnp.complex64, jnp.complex128]


def normalise_log_space(x: LogSpace) -> LogSpace:
    """
    Safely normalise a LogSpace, accounting for zero-sum.
    """
    norm = x.sum()
    x /= norm
    x = LogSpace(jnp.where(jnp.isneginf(norm.log_abs_val), -jnp.inf, x.log_abs_val))
    return x
