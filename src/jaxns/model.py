import dataclasses
from functools import partial
from typing import Any, Callable

import jax
import numpy as np
from jax import numpy as jnp
from jaxctx import transform

from jaxns.logging import jaxns_logger
from jaxns.pytree import PureDataclassPytree
from jaxns.types import PRNGKey, FloatArray, UType, XType

__all__ = [
    'Model'
]


@dataclasses.dataclass(slots=True, frozen=True)
class Model(PureDataclassPytree):
    """
    Represents a Bayesian model in terms of a generative prior, and likelihood function.
    """
    prior_model: Callable

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(this, ['prior_model'])

    @classmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        return cls.build_unflatten(aux_data, children)

    def U_ndims(self, args=(), params=None) -> int:
        """
        Get the number of dimensions in flattened U-space.
        Will cause a problem if traced.

        Args:
            args: additional arguments for the model
            params: parameters of the model

        Returns:
            the number of dimensions in flattened U-space
        """
        return int(_U_ndims(self, args=args, params=params))

    def sample_U(self, key: PRNGKey, args=(), params=None) -> UType:
        """
        Sample uniformly from the prior in U-space.

        Args:
            key: PRNGKey
            args: additional arguments for the sampling function
            params: parameters of the model

        Returns:
            U-space sample
        """
        return _sample_U(self, key, args=args, params=params)

    def transform_to_X(self, U: UType, args=(), params=None) -> XType:
        """
        Compute the X-space sample corresponding to the U-space sample.

        Args:
            U: The U-space sample
            args: args for the model
            params: params of th emodel

        Returns:
            X-space sample
        """
        return _transform_to_X(self, U, args=args, params=params)

    def log_likelihood(self, U: UType, args=(), params=None, *, allow_nan: bool = True) -> FloatArray:
        """
        Compute the log-likelihood.

        Args:
            U: U-space sample
            args: additional arguments for the likelihood function
            params: parameters of the model
            allow_nan: whether to allow nans in likelihood

        Returns:
            log likelihood at the sample
        """
        return _log_likelihood(self, U, args=args, params=params, allow_nan=allow_nan)

    def log_prior(self, U: UType, args=(), params=None) -> FloatArray:
        """
        Computes the log-probability of the prior.

        Args:
            U: The U-space sample
            args: additional arguments for the prior function
            params: parameters of the model

        Returns:
            the log probability of prior
        """
        return _log_prior(self, U, args=args, params=params)

    def log_joint(self, U: UType, args=(), params=None, *, allow_nan: bool = True) -> FloatArray:
        """
        Computes the log-joint probability of the model.

        Args:
            U: The U-space sample
            args: additional arguments for the joint function
            params: parameters of the model
            allow_nan: whether to allow nans in likelihood

        Returns:
            the log joint probability of the model
        """
        return _log_joint(self, U, args=args, params=params, allow_nan=allow_nan)

    def sanity_check(self, key: PRNGKey, args=(), params=None, num_samples: int = 100):
        """
        Performs a sanity check on the model.

        Args:
            key: PRNGKey
            args: additional arguments for the sanity check
            params: parameters of the model
            num_samples: number of samples to check

        Raises:
            AssertionError: if any of the sampled prior variables are nan, or log_likelihood is +inf.
        """
        jaxns_logger.info("Sanity check...")
        for key in jax.random.split(key, num_samples):
            u_sample = self.sample_U(key, args=args, params=params)

            log_likelihood = self.log_likelihood(u_sample, args=args, params=params, allow_nan=True)
            if not jnp.isfinite(log_likelihood):
                jaxns_logger.info(f"Found bad point:"
                                  f"\n{u_sample} -> {self.transform_to_X(u_sample, args=args, params=params)}"
                                  f"\nlog_likelihood: {log_likelihood}")
        jaxns_logger.info("Sanity check passed")


Model.register_pytree()


def _U_ndims(self: Model, args=(), params=None) -> int:
    u_example = jax.eval_shape(self.sample_U, jax.random.PRNGKey(0), args=args, params=params)
    U_ndims = sum(map(lambda x: np.prod(x.shape), jax.tree.leaves(u_example)))
    return int(U_ndims)


@partial(jax.jit, inline=True)
def _sample_U(self: Model, key: PRNGKey, args=(), params=None) -> UType:
    u_key, params_key = jax.random.split(key, 2)
    init_return = transform(self.prior_model).init(
        rngs={'params': params_key, 'U': u_key},
        collections={'params': params},
        *args
    )
    return init_return.collections['U']


@partial(jax.jit, inline=True)
def _transform_to_X(self: Model, U: UType, args=(), params=None) -> XType:
    apply_return = transform(self.prior_model).apply(
        rngs=None,
        collections={'params': params, 'U': U},
        *args
    )
    return apply_return.collections['X']


@partial(jax.jit, inline=True, static_argnames=('allow_nan',))
def _log_likelihood(self: Model, U: UType, args=(), params=None, *, allow_nan: bool = True) -> FloatArray:
    apply_return = transform(self.prior_model).apply(
        rngs=None,
        collections={'params': params, 'U': U},
        *args
    )
    log_likelihood = apply_return.fn_val
    if allow_nan:
        return log_likelihood
    else:
        return jnp.where(jnp.isnan(log_likelihood), -jnp.inf, log_likelihood)


@partial(jax.jit, inline=True)
def _log_prior(self: Model, U: UType, args=(), params=None) -> FloatArray:
    apply_return = transform(self.prior_model).apply(
        rngs=None,
        collections={'params': params, 'U': U},
        *args
    )
    log_prior = jax.tree.leaves(apply_return.collections['log_prior'])
    if len(log_prior) == 0:
        raise ValueError("No log prior found in the model. Ensure the prior model is correctly defined.")
    else:
        return sum(log_prior[1:], log_prior[0])


@partial(jax.jit, inline=True, static_argnames=('allow_nan',))
def _log_joint(self: Model, U: UType, args=(), params=None, *, allow_nan: bool = True) -> FloatArray:
    return self.log_prior(U, args, params) + self.log_likelihood(U, args, params, allow_nan=allow_nan)
