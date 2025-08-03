from abc import abstractmethod, ABC

import jax
import numpy as np
from jax import random, vmap, jit, numpy as jnp
from jaxctx import transform, CtxParams

from jaxns.framework.bases import PriorModelType
from jaxns.internals.logging import logger
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


class Model(AbstractModel):
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
        pass

    def sanity_check(self, key: PRNGKey, S: int):
        U = jit(vmap(self.sample_U))(random.split(key, S))
        log_L = jit(vmap(lambda u: self.forward(u, allow_nan=True)))(U)
        logger.info("Sanity check...")
        for _U, _log_L in zip(U, log_L):
            if jnp.isnan(_log_L):
                logger.info(f"Found bad point:"
                            f"\n{_U} -> {self.transform(_U)}"
                            f"\n -> {self.transform_parametrised(_U)}")
        assert not any(np.isnan(log_L))
        logger.info("Sanity check passed")
        if 'parsed_prior' in self.__dict__:
            del self.__dict__['parsed_prior']
