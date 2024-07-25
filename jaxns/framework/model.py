import warnings
from typing import Optional
from uuid import uuid4

import numpy as np
from jax import random, vmap, jit, numpy as jnp

from jaxns.internals.logging import logger
from jaxns.internals.maps import pytree_unravel

try:
    import haiku as hk
except ImportError:
    warnings.warn("You must `pip install dm-haiku` first.")
    raise

from jaxns.framework.bases import BaseAbstractModel, PriorModelType
from jaxns.framework.ops import transform, prepare_input, compute_log_prob_prior, compute_log_likelihood, parse_prior, \
    parse_joint, transform_parametrised
from jaxns.internals.types import PRNGKey, FloatArray, float_type, LikelihoodType, UType, XType, LikelihoodInputType, \
    WType

__all__ = [
    'Model'
]


class Model(BaseAbstractModel):
    """
    Represents a Bayesian model in terms of a generative prior, and likelihood function.
    """

    def __init__(self, prior_model: PriorModelType, log_likelihood: LikelihoodType,
                 params: Optional[hk.MutableParams] = None):
        super().__init__(prior_model=prior_model, log_likelihood=log_likelihood)
        if params is None:
            params = self.init_params(rng=random.PRNGKey(0))
        self._params = params
        # Parse the prior model to get place holders
        self.__U_placeholder, self.__X_placeholder, self.__W_placeholder = hk.transform(
            lambda: parse_prior(prior_model=self.prior_model)
        ).apply(params=self._params, rng=random.PRNGKey(0))
        self._id = str(uuid4())  # Used for making sure it's hashable, so it can be used as a key in a dict.
        self.ravel_fn, self.unravel_fn = pytree_unravel(self.__W_placeholder)

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

    def _W_placeholder(self) -> WType:
        return self.__W_placeholder

    def init_params(self, rng: PRNGKey) -> hk.MutableParams:
        """
        Initialise the parameters of the model.

        Args:
            rng: PRNGkey to initialise the parameters.

        Returns:
            The initialised parameters.
        """

        def _log_prob_joint():
            U_placeholder, X_placeholder, W_placeholder, likelihood_input_placeholder, log_L_placeholder = parse_joint(
                prior_model=self.prior_model,
                log_likelihood=self.log_likelihood
            )
            log_prob_prior = compute_log_prob_prior(W=W_placeholder, prior_model=self.prior_model)
            log_prob_likelihood = compute_log_likelihood(W=W_placeholder, prior_model=self.prior_model,
                                                         log_likelihood=self.log_likelihood, allow_nan=True)
            return log_prob_prior + log_prob_likelihood

        params = hk.transform(_log_prob_joint).init(rng)
        return params

    def __hash__(self):
        return hash(self._id)

    def sample_U(self, key: PRNGKey) -> UType:
        """
        Sample from the prior model.

        Args:
            key: PRNGKey to use.

        Returns:
            The sampled U.
        """
        if self._params is None:
            raise RuntimeError("Model has not been initialised")

        def _sample_U():
            return random.uniform(key=hk.next_rng_key(), shape=(self.U_ndims,), dtype=float_type)

        return hk.transform(_sample_U).apply(params=self._params, rng=key)

    def sample_W(self, key: PRNGKey) -> WType:
        """
        Sample from the prior model.

        Args:
            key: PRNGKey to use.

        Returns:
            The sampled W.
        """
        if self._params is None:
            raise RuntimeError("Model has not been initialised")

        return self.unravel_fn(self.sample_U(key=key))

    def transform(self, U: UType) -> XType:
        def _transform():
            return transform(W=self.unravel_fn(U), prior_model=self.prior_model)

        return hk.transform(_transform).apply(params=self._params, rng=random.PRNGKey(0))

    def transform_parametrised(self, U: UType) -> XType:
        def _transform():
            return transform_parametrised(W=self.unravel_fn(U), prior_model=self.prior_model)

        return hk.transform(_transform).apply(params=self._params, rng=random.PRNGKey(0))

    def forward(self, U: UType, allow_nan: bool = False) -> FloatArray:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")

        def _forward():
            return compute_log_likelihood(W=self.unravel_fn(U), prior_model=self.prior_model, log_likelihood=self.log_likelihood,
                                          allow_nan=allow_nan)

        return hk.transform(_forward).apply(params=self._params, rng=random.PRNGKey(0))

    def log_prob_prior(self, U: UType) -> FloatArray:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")

        def _log_prob_prior():
            return compute_log_prob_prior(W=self.unravel_fn(U), prior_model=self.prior_model)

        return hk.transform(_log_prob_prior).apply(params=self._params, rng=random.PRNGKey(0))

    def prepare_input(self, U: UType) -> LikelihoodInputType:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")

        def _prepare_input():
            return prepare_input(W=self.unravel_fn(U), prior_model=self.prior_model)

        return hk.transform(_prepare_input).apply(params=self._params, rng=random.PRNGKey(0))

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
