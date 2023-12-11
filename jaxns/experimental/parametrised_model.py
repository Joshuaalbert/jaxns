import logging
from typing import Tuple, Optional

import numpy as np
from jax import random, jit, vmap, numpy as jnp

try:
    import haiku as hk
except ImportError:
    print("You must `pip install dm-haiku` first.")
    raise

from jaxns.internals.types import UType, XType, PRNGKey, FloatArray, LikelihoodInputType
from jaxns.framework.bases import BaseAbstractModel

__all__ = [
    'ParametrisedModel'
]

logger = logging.getLogger('jaxns')


class ParametrisedModel(BaseAbstractModel):
    """
    A parametrised model, which is a wrapper around a model and its parameters.
    """

    def __init__(self, base_model: BaseAbstractModel, params: Optional[hk.MutableParams] = None):
        """
        Initialise the parametrised model. This means you can use hk.get_parameter anywhere within the prior or
        likelihood definitions.

        Args:
            base_model: The base model to wrap.
            params: The parameters to use. If None, then you must call init_params and set params
                before using the model.
        """
        super().__init__(prior_model=base_model.prior_model, log_likelihood=base_model.log_likelihood)
        self.base_model = base_model
        if params is None:
            params = self.init_params(rng=random.PRNGKey(0))
        self._params = params

    @property
    def params(self):
        if self._params is None:
            raise RuntimeError("Model has not been initialised")
        return self._params

    def new(self, params: hk.MutableParams) -> 'ParametrisedModel':
        """
        Create a new parametrised model with the given parameters.

        **This is (and must be) a pure function.**

        Args:
            params: The parameters to use.

        Returns:
            The new parametrised model.
        """
        return ParametrisedModel(base_model=self.base_model, params=params)

    def __hash__(self):
        return self.base_model.__hash__()

    def _parsed_prior(self) -> Tuple[UType, XType]:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")
        return hk.transform(self.base_model._parsed_prior).apply(params=self._params, rng=None)

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

    def sample_U(self, key: PRNGKey) -> FloatArray:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")
        return hk.transform(self.base_model.sample_U).apply(params=self._params, rng=None, key=key)

    def transform(self, U: UType) -> XType:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")
        return hk.transform(self.base_model.transform).apply(params=self._params, rng=None, U=U)

    def forward(self, U: UType, allow_nan: bool = False) -> FloatArray:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")
        return hk.transform(self.base_model.forward).apply(params=self._params, rng=None, U=U,
                                                           allow_nan=allow_nan)

    def log_prob_prior(self, U: UType) -> FloatArray:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")
        return hk.transform(self.base_model.log_prob_prior).apply(params=self._params, rng=None, U=U)

    def prepare_input(self, U: UType) -> LikelihoodInputType:
        if self._params is None:
            raise RuntimeError("Model has not been initialised")
        return hk.transform(self.base_model.prepare_input).apply(params=self._params, rng=None, U=U)

    def init_params(self, rng: PRNGKey) -> hk.MutableParams:
        """
        Initialise the parameters of the model.

        Args:
            rng: PRNGkey to initialise the parameters.

        Returns:
            The initialised parameters.
        """

        def log_joint_prob():
            # A pure function that returns the log joint of model.
            U = self.base_model.sample_U(key=hk.next_rng_key())
            log_L = self.base_model.forward(U=U)
            log_prior_prob = self.base_model.log_prob_prior(U=U)
            return log_L + log_prior_prob

        params = hk.transform(log_joint_prob).init(rng)
        return params
