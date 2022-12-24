from typing import Tuple
from uuid import uuid4

from etils.array_types import PRNGKey, FloatArray
from jax import random

from jaxns.internals.types import float_type
from jaxns.new_code.prior import PriorModelType, LikelihoodType, parse_prior, XType, compute_log_likelihood, UType, \
    transform, log_prob_prior


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
        Compute the X-space sample.

        Args:
            U: U-space sample

        Returns:
            X-space sample
        """
        return transform(U=U, prior_model=self.prior_model)

    def forward(self, U: UType) -> Tuple[XType, FloatArray]:
        """
        Compute the X-space sample, and log-likelihood.

        Args:
            U: U-space sample

        Returns:
            X-space sample, and log likelihood at the sample
        """
        return compute_log_likelihood(U=U, prior_model=self.prior_model, log_likelihood=self.log_likelihood)

    def log_prob_prior(self, U: UType) -> FloatArray:
        """
        Computes the log-probability of the prior.

        Args:
            U: The U-space sample

        Returns:
            the log probability of prior
        """
        return log_prob_prior(U=U, prior_model=self.prior_model)
