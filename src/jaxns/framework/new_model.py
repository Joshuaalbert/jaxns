from abc import abstractmethod, ABC

import jax
from jax import numpy as jnp
from jaxctx import transform, CtxParams

from jaxns.framework.bases import PriorModelType
from jaxns.internals.logging import logger
from jaxns.internals.pytree import Pytree
from jaxns.internals.types import PRNGKey, MeasureType

__all__ = [
    'Model'
]

UType = CtxParams
XType = CtxParams


class AbstractModel(ABC):
    """
    Represents a Bayesian model in terms of a generative prior, and likelihood function.
    """

    @abstractmethod
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
        ...

    @abstractmethod
    def log_likelihood(self, U: UType, args=(), params=None, *, allow_nan: bool = True) -> MeasureType:
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
        ...

    @abstractmethod
    def log_prior(self, U: UType, args=(), params=None) -> MeasureType:
        """
        Computes the log-probability of the prior.

        Args:
            U: The U-space sample
            args: additional arguments for the prior function
            params: parameters of the model

        Returns:
            the log probability of prior
        """
        ...

    @abstractmethod
    def log_joint(self, U: UType, args=(), params=None, *, allow_nan: bool = True) -> MeasureType:
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
        ...

    @abstractmethod
    def sanity_check(self, key: PRNGKey, args=(), params=None, num_samples: int = 100) -> None:
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
        ...


class Model(AbstractModel, Pytree):
    """
    Represents a Bayesian model in terms of a generative prior, and likelihood function.
    """

    def __init__(self, prior_model: PriorModelType):
        self.transformed_model = transform(prior_model)

    def sample_U(self, key: PRNGKey, args=(), params=None) -> UType:
        init_return = self.transformed_model.init(
            rngs={'params': key},
            collections={'params': params},
            *args
        )
        return init_return.collections['U']

    def transform_to_X(self, U: UType, args=(), params=None) -> XType:
        apply_return = self.transformed_model.apply(
            rngs=None,
            collections={'params': params, 'U': U},
            *args
        )
        return apply_return.collections['X']

    def log_likelihood(self, U: UType, args=(), params=None, *, allow_nan: bool = True) -> MeasureType:
        apply_return = self.transformed_model.apply(
            rngs=None,
            collections={'params': params},
            *args
        )
        return apply_return.fn_val

    def log_prior(self, U: UType, args=(), params=None) -> MeasureType:
        apply_return = self.transformed_model.apply(
            rngs=None,
            collections={'params': params},
            *args
        )
        log_prior = jax.tree.leaves(apply_return.collections['log_prior'])
        if len(log_prior) == 0:
            raise ValueError("No log prior found in the model. Ensure the prior model is correctly defined.")
        else:
            return sum(log_prior[1:], log_prior[0])

    def log_joint(self, U: UType, args=(), params=None, *, allow_nan: bool = True) -> MeasureType:
        return self.log_prior(U, args, params) + self.log_likelihood(U, args, params, allow_nan=allow_nan)

    def sanity_check(self, key: PRNGKey, args=(), params=None, num_samples: int = 100):
        logger.info("Sanity check...")
        for key in jax.random.split(key, num_samples):
            u_sample = self.sample_U(key, args=args, params=params)

            log_likelihood = self.log_likelihood(u_sample, args=args, params=params, allow_nan=True)
            if not jnp.isfinite(log_likelihood):
                logger.info(f"Found bad point:"
                            f"\n{u_sample} -> {self.transform_to_X(u_sample, args=args, params=params)}"
                            f"\nlog_likelihood: {log_likelihood}")
        logger.info("Sanity check passed")
