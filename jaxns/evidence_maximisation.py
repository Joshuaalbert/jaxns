import logging
from typing import Tuple

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp, jit
from jax import random
from jax import vmap, tree_map
from jax._src.scipy.special import logsumexp, ndtr
from tqdm import tqdm

from jaxns import ExactNestedSampler, TerminationCondition, NestedSamplerResults, UType, XType, PRNGKey, FloatArray, \
    float_type, LikelihoodType, LikelihoodInputType
from jaxns.abc import AbstractModel, PriorModelType
from jaxns.prior import Prior, SingularDistribution

try:
    import haiku as hk
except ImportError:
    print("You must `pip install dm-haiku` first.")
    raise

try:
    import optax
except ImportError:
    print("You must `pip install optax` first.")
    raise

__all__ = [
    'EM',
    'ParametrisedModel',
    'prior_to_parametrised_singular'
]

tfpd = tfp.distributions
tfpk = tfp.math.psd_kernels

logger = logging.getLogger('jaxns')


def prior_to_parametrised_singular(prior: Prior) -> Prior:
    """
    We use a Normal parameter with centre on unit cube, and scale covering the whole cube, as the base representation.

    Args:
        prior: any prior

    Returns:
        A parameter representing the prior.
    """
    name = f"{prior.name}_param"
    init_value = 0.5 * jnp.ones(prior.base_shape, dtype=float_type)
    norm_U_base_param = hk.get_parameter(
        name=name,
        shape=prior.base_shape,
        dtype=float_type,
        init=hk.initializers.Constant(init_value)
    )
    # transform norm_base_param with normal cdf.
    mu = 0.5
    scale = 0.5
    U_base_param = ndtr((norm_U_base_param - mu) / scale)
    param = prior.forward(U_base_param)
    dist = SingularDistribution(value=param, dist=prior.dist)
    return Prior(dist_or_value=dist, name=prior.name)


class ParametrisedModel(AbstractModel):
    """
    A parametrised model, which is a wrapper around a model and its parameters.
    """

    def __init__(self, base_model: AbstractModel, params: hk.MutableParams | None = None):
        """
        Initialise the parametrised model. This means you can use hk.get_parameter anywhere within the prior or
        likelihood definitions.

        Args:
            base_model: The base model to wrap.
            params: The parameters to use. If None, then you must call init_params and set params
                before using the model.
        """
        self.base_model = base_model
        if params is None:
            params = self.init_params(rng=random.PRNGKey(0))
        self._params = params

    def _prior_model(self) -> PriorModelType:
        return self.base_model.prior_model

    def _log_likelihood(self) -> LikelihoodType:
        return self.base_model.log_likelihood

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
        return hk.transform(self.base_model.prepare_input).apply(params=self._params, rng=None, U=U,
                                                                 prior_model=self.prior_model)

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
            U = self.base_model.sample_U(key=random.PRNGKey(0))
            log_L = self.base_model.forward(U=U)
            log_prior_prob = self.base_model.log_prob_prior(U=U)
            return log_L + log_prior_prob

        params = hk.transform(log_joint_prob).init(rng)
        return params


class EM:
    """
    Evidence Maximisation class, that implements the E and M steps. Iteratively computes the evidence and maximises it.
    """

    def __init__(self, model: ParametrisedModel, learning_rate: float = 1e-2, max_num_epochs: int = 100, gtol=1e-4,
                 log_Z_ftol=1., log_Z_atol=1e-4):
        """
        Initialise the EM class.

        Args:
            model: The model to train.
            learning_rate: The learning rate for the M-step.
            max_num_epochs: The maximum number of epochs to train for.
            gtol: The gradient tolerance for the M-step.
            log_Z_ftol: The tolerance for the change in the evidence as function of log_Z_uncert.
            log_Z_atol: The absolute tolerance for the change in the evidence.
        """
        if not isinstance(model, ParametrisedModel):
            raise ValueError("model must be an instance of ParametrisedModel")
        self.model = model
        self.learning_rate = learning_rate
        self.max_num_epochs = max_num_epochs
        self.gtol = gtol
        self.log_Z_ftol = log_Z_ftol
        self.log_Z_atol = log_Z_atol

    def e_step(self, params: hk.MutableParams) -> NestedSamplerResults:
        """
        The E-step is just nested sampling.

        Args:
            params: The parameters to use.

        Returns:
            The nested sampling results.
        """

        # The E-step is just nested sampling
        model = self.model.new(params=params)
        # Create the nested sampler class. In this case without any tuning.
        exact_ns = ExactNestedSampler(model=model, num_live_points=model.U_ndims * 20, max_samples=1e6)

        termination_reason, state = exact_ns(random.PRNGKey(42),
                                             term_cond=TerminationCondition(live_evidence_frac=1e-5))
        results = exact_ns.to_results(state, termination_reason)
        # exact_ns.summary(results)
        # exact_ns.plot_diagnostics(results)
        # exact_ns.plot_cornerplot(results)

        return results

    def m_step(self, results: NestedSamplerResults, params: hk.MutableParams) -> hk.MutableParams:
        """
        The M-step is just evidence maximisation.

        Args:
            results: The nested sampling results.
            params: The parameters to use.

        Returns:
            The updated parameters.
        """

        # The M-step is just evidence maximisation

        def neg_log_evidence(params: hk.MutableParams):
            # Compute the log evidence
            model = self.model.new(params=params)
            # To make manageable, we could do chunked_pmap
            log_L_samples = vmap(model.forward)(results.U_samples)
            # We add the log_Z_mean because log_dp_mean is normalised
            log_weights = results.log_dp_mean - results.log_L_samples + results.log_Z_mean
            log_Z = logsumexp(log_L_samples + log_weights)
            return -log_Z

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(params, opt_state):
            val, grads = jax.value_and_grad(neg_log_evidence)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            # Get L2 of grads
            l_oo_grad = tree_map(lambda x: jnp.max(jnp.abs(x)), grads)
            return val, l_oo_grad, new_params, new_opt_state

        p_bar = tqdm(range(self.max_num_epochs), desc="Training Progress", dynamic_ncols=True)
        for epoch in p_bar:
            neg_log_Z, l_oo, params, opt_state = train_step(params, opt_state)
            p_bar.set_description(f"Step {epoch}: log Z = {-neg_log_Z:.4f}, l_oo = {l_oo}")
            p_bar.refresh()
            # Flatten l_oo pytree and get term condition
            l_oo_small = jnp.all(jnp.asarray(jax.tree_leaves(l_oo)) < self.gtol)
            if l_oo_small:
                print(f"Terminating at step {epoch} due to l(inf) small enough.")
                break

        return params

    def train(self, num_steps: int = 10, params: hk.MutableParams | None = None) -> Tuple[
        NestedSamplerResults, hk.MutableParams]:
        """
        Train the model using EM for num_steps.

        Args:
            num_steps: The number of steps to train for, or until convergence.
            params: The initial parameters to use. If None, then the model's params are used.

        Returns:
            The trained parameters.
        """
        if params is None:
            params = self.model.params
        log_Z = -jnp.inf

        # Initialize the progress bar with a description
        p_bar = tqdm(range(num_steps), desc="Processing Steps", dynamic_ncols=True)

        results = None
        for step in p_bar:
            # Execute the e_step
            results = self.e_step(params=params)
            # Update progress bar description
            p_bar.set_description(f"Step {step}: log Z = {results.log_Z_mean:.4f} +- {results.log_Z_uncert:.4f}")
            p_bar.refresh()

            # Check termination condition
            log_Z_change = jnp.abs(results.log_Z_mean - log_Z)
            if log_Z_change < max(self.log_Z_ftol * results.log_Z_uncert, self.log_Z_atol):
                print(f"Convergence achieved at step {step}.")
                break

            # Update log_Z and log_Z_uncert values
            log_Z = results.log_Z_mean

            # Execute the m_step
            params = self.m_step(results, params)

        if results is None:
            raise RuntimeError("No results were computed.")
        return results, params
