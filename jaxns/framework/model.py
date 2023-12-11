import logging
from typing import Tuple
from uuid import uuid4

import numpy as np
from jax import random, vmap, jit, numpy as jnp

from jaxns.framework.bases import BaseAbstractModel, PriorModelType
from jaxns.framework.ops import parse_prior, transform, prepare_input, log_prob_prior, compute_log_likelihood
from jaxns.internals.types import PRNGKey, FloatArray, float_type, LikelihoodType, UType, XType, LikelihoodInputType

__all__ = [
    'Model'
]

logger = logging.getLogger('jaxns')


class Model(BaseAbstractModel):
    """
    Represents a Bayesian model in terms of a generative prior, and likelihood function.
    """

    def __init__(self, prior_model: PriorModelType, log_likelihood: LikelihoodType):
        super().__init__(prior_model=prior_model, log_likelihood=log_likelihood)
        self._id = str(uuid4())  # Used for making sure it's hashable, so it can be used as a key in a dict.

    def _parsed_prior(self) -> Tuple[UType, XType]:
        return parse_prior(prior_model=self.prior_model)

    def __hash__(self):
        return hash(self._id)

    def __repr__(self):
        # TODO(Joshuaalbert): Pretty print the model
        return f""

    def sample_U(self, key: PRNGKey) -> FloatArray:
        return random.uniform(key=key, shape=(self.U_ndims,), dtype=float_type)

    def transform(self, U: UType) -> XType:
        return transform(U=U, prior_model=self.prior_model)

    def forward(self, U: UType, allow_nan: bool = False) -> FloatArray:
        return compute_log_likelihood(U=U, prior_model=self.prior_model, log_likelihood=self.log_likelihood,
                                      allow_nan=allow_nan)

    def log_prob_prior(self, U: UType) -> FloatArray:
        return log_prob_prior(U=U, prior_model=self.prior_model)

    def prepare_input(self, U: UType) -> LikelihoodInputType:
        return prepare_input(U=U, prior_model=self.prior_model)

    def sanity_check(self, key: PRNGKey, S: int):
        U = jit(vmap(self.sample_U))(random.split(key, S))
        log_L = jit(vmap(lambda u: self.forward(u, allow_nan=True)))(U)
        logger.info("Sanity check...")
        for _U, _log_L in zip(U, log_L):
            if jnp.isnan(_log_L):
                logger.info(f"Found bad point: {_U} -> {self.transform(_U)}")
        assert not any(np.isnan(log_L))
        logger.info("Sanity check passed")
        if 'parsed_prior' in self.__dict__:
            del self.__dict__['parsed_prior']
