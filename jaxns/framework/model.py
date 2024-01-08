import logging
from typing import Optional
from uuid import uuid4

import numpy as np
from jax import random, vmap, jit, numpy as jnp

try:
    import haiku as hk
except ImportError:
    print("You must `pip install dm-haiku` first.")
    raise

from jaxns.framework.bases import BaseAbstractModel, PriorModelType
from jaxns.framework.ops import transform, prepare_input, compute_log_prob_prior, compute_log_likelihood, parse_prior, \
    parse_joint
from jaxns.internals.types import PRNGKey, FloatArray, float_type, LikelihoodType, UType, XType, LikelihoodInputType

__all__ = [
    'Model'
]

logger = logging.getLogger('jaxns')


class Model(BaseAbstractModel):
    """
    Represents a Bayesian model in terms of a generative prior, and likelihood function.
    """

    def __init__(self, prior_model: PriorModelType, log_likelihood: LikelihoodType,
                 params: Optional[hk.MutableParams] = None):
        super().__init__(prior_model=prior_model, log_likelihood=log_likelihood)
        # hk.transform(
        #     lambda: parse_prior(prior_model=self.prior_model)
        # ).apply(params=self._params, rng=None)
        if params is None:
            params = self.init_params(rng=random.PRNGKey(0))
        self._params = params
        # Parse the prior model to get place holders
        self.__U_placeholder, self.__X_placeholder = hk.transform(
            lambda: parse_prior(prior_model=self.prior_model)
        ).apply(params=self._params, rng=None)
        self._id = str(uuid4())  # Used for making sure it's hashable, so it can be used as a key in a dict.

    @property
    def num_params(self) -> int:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")
        return hk.data_structures.tree_size(self._params)

    @property
    def params(self):
        if self._params is None:
            raise RuntimeError("Model has not been initialised")
        return self._params

    def set_params(self, params: hk.MutableParams) -> 'Model':
        """
        Create a new parametrised model with the given parameters.

        Args:
            params: The parameters to use.

        Returns:
            A model with set parameters.
        """
        return Model(prior_model=self.prior_model, log_likelihood=self.log_likelihood, params=params)

    def __call__(self, params: hk.MutableParams) -> 'Model':
        """
        Create a new parametrised model with the given parameters.

        **This is (and must be) a pure function.**

        Args:
            params: The parameters to use.

        Returns:
            A model with set parameters.
        """
        return self.set_params(params=params)

    def _U_placeholder(self) -> UType:
        return self.__U_placeholder

    def _X_placeholder(self) -> XType:
        return self.__X_placeholder

    def init_params(self, rng: PRNGKey) -> hk.MutableParams:
        """
        Initialise the parameters of the model.

        Args:
            rng: PRNGkey to initialise the parameters.

        Returns:
            The initialised parameters.
        """

        def _log_prob_joint():
            U_placeholder, X_placeholder, likelihood_input_placeholder, log_L_placeholder = parse_joint(
                prior_model=self.prior_model,
                log_likelihood=self.log_likelihood
            )
            log_prob_prior = compute_log_prob_prior(
                U=U_placeholder,
                prior_model=self.prior_model)
            log_prob_likelihood = compute_log_likelihood(
                U=U_placeholder,
                prior_model=self.prior_model,
                log_likelihood=self.log_likelihood,
                allow_nan=True
            )
            return log_prob_prior + log_prob_likelihood

        params = hk.transform(_log_prob_joint).init(rng)
        return params

    def __hash__(self):
        return hash(self._id)

    def __repr__(self):
        # TODO(Joshuaalbert): Pretty print the model
        return f""

    def sample_U(self, key: PRNGKey) -> FloatArray:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")

        def _sample_U():
            return random.uniform(key=hk.next_rng_key(), shape=(self.U_ndims,), dtype=float_type)

        return hk.transform(_sample_U).apply(params=self._params, rng=None)

    def transform(self, U: UType) -> XType:
        def _transform():
            return transform(U=U, prior_model=self.prior_model)

        return hk.transform(_transform).apply(params=self._params, rng=None)

    def forward(self, U: UType, allow_nan: bool = False) -> FloatArray:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")

        def _forward():
            return compute_log_likelihood(U=U, prior_model=self.prior_model, log_likelihood=self.log_likelihood,
                                          allow_nan=allow_nan)

        return hk.transform(_forward).apply(params=self._params, rng=None)

    def log_prob_prior(self, U: UType) -> FloatArray:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")

        def _log_prob_prior():
            return compute_log_prob_prior(U=U, prior_model=self.prior_model)

        return hk.transform(_log_prob_prior).apply(params=self._params, rng=None)

    def prepare_input(self, U: UType) -> LikelihoodInputType:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")

        def _prepare_input():
            return prepare_input(U=U, prior_model=self.prior_model)

        return hk.transform(_prepare_input).apply(params=self._params, rng=None)

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
