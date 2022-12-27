import logging
from uuid import uuid4

import numpy as np
from etils.array_types import PRNGKey, FloatArray
from jax import random, vmap, jit, numpy as jnp

from jaxns.internals.types import float_type
from jaxns.new_code.prior import PriorModelType, LikelihoodType, parse_prior, compute_log_likelihood, UType, \
    transform, log_prob_prior, XType

logger = logging.getLogger('jaxns')

__all__ = ['Model']


class Model:
    """
    Represents a Bayesian model in terms of a generative prior, and likelihood function.
    """

    def __init__(self, prior_model: PriorModelType, log_likelihood: LikelihoodType):
        self.id = str(uuid4())
        self.prior_model = prior_model
        self.log_likelihood = log_likelihood
        self._U_placeholder, self._X_placeholder = parse_prior(prior_model=prior_model)
        self._U_ndims = self.U_placeholder.size

    @property
    def U_placeholder(self) -> UType:
        """
        A placeholder for U-space sample.
        """
        return self._U_placeholder

    @property
    def X_placeholder(self) -> XType:
        """
        A placeholder for X-space sample.
        """
        return self._X_placeholder

    @property
    def U_ndims(self) -> int:
        """
        The prior dimensionality.
        """
        return self._U_ndims

    def __hash__(self):
        return hash(self.id)

    def sample_U(self, key: PRNGKey) -> FloatArray:
        """
        Sample uniformly from the prior in U-space.

        Args:
            key: PRNGKey

        Returns:
            U-space sample
        """
        return random.uniform(key=key, shape=(self.U_ndims,), dtype=float_type)

    def transform(self, U: UType) -> XType:
        """
        Compute the prior sample.

        Args:
            U: U-space sample

        Returns:
            prior sample
        """
        return transform(U=U, prior_model=self.prior_model)

    def forward(self, U: UType, allow_nan: bool = False) -> FloatArray:
        """
        Compute the log-likelihood.

        Args:
            U: U-space sample
            allow_nan: whether to allow nans in likelihood

        Returns:
            log likelihood at the sample
        """
        return compute_log_likelihood(U=U, prior_model=self.prior_model, log_likelihood=self.log_likelihood,
                                      allow_nan=allow_nan)

    def log_prob_prior(self, U: UType) -> FloatArray:
        """
        Computes the log-probability of the prior.

        Args:
            U: The U-space sample

        Returns:
            the log probability of prior
        """
        return log_prob_prior(U=U, prior_model=self.prior_model)

    def sanity_check(self, key: PRNGKey, S: int):
        U = jit(vmap(self.sample_U))(random.split(key, S))
        log_L = jit(vmap(lambda u: self.forward(u, allow_nan=True)))(U)
        logger.info("Sanity check...")
        for _U, _log_L in zip(U, log_L):
            if jnp.isnan(_log_L):
                logger.info(f"Found bad point: {_U} -> {self.transform(_U)}")
        assert not any(np.isnan(log_L))
        logger.info("Sanity check passed")
