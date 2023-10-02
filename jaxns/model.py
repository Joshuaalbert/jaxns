import logging
from typing import Tuple
from uuid import uuid4

import numpy as np
from jax import random, vmap, jit, numpy as jnp

from jaxns.abc import AbstractModel, PriorModelType
from jaxns.prior import parse_prior, compute_log_likelihood, transform, log_prob_prior, prepare_input
from jaxns.types import PRNGKey, FloatArray, float_type, LikelihoodType, UType, XType, LikelihoodInputType

try:
    import haiku as hk
except ImportError:
    print("You must `pip install dm-haiku` first.")
    raise

__all__ = [
    'Model'
]

logger = logging.getLogger('jaxns')


class Model(AbstractModel):
    """
    Represents a Bayesian model in terms of a generative prior, and likelihood function.
    """

    def __init__(self, prior_model: PriorModelType, log_likelihood: LikelihoodType):
        self._id = str(uuid4())
        self._prior_model = prior_model
        self._log_likelihood = log_likelihood

    def _prior_model(self) -> PriorModelType:
        return self._prior_model

    def _log_likelihood(self) -> LikelihoodType:
        return self._log_likelihood

    def _parsed_prior(self) -> Tuple[UType, XType]:
        return parse_prior(prior_model=self.prior_model)

    def __hash__(self):
        return hash(self._id)

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
