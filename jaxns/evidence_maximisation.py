import logging
from typing import Tuple, Dict, Any

import jax
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp, random, vmap, tree_map
from jax._src.scipy.special import logsumexp, ndtr
from tqdm import tqdm

from jaxns.nested_sampler.standard_static import StandardStaticNestedSampler

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

from jaxns.types import TerminationCondition, NestedSamplerResults, float_type
from jaxns.model.parametrised_model import ParametrisedModel
from jaxns.model.prior import Prior
from jaxns.model.distribution import SingularDistribution

__all__ = [
    'EM',
    'prior_to_parametrised_singular'
]

tfpd = tfp.distributions
tfpk = tfp.math.psd_kernels

logger = logging.getLogger('jaxns')


def prior_to_parametrised_singular(prior: Prior) -> Prior:
    """
    Convert a prior into a non-Bayesian parameter, that takes a single value in the model, but still has an associated
    log_prob. The parameter is registered as a `hk.Parameter` with added `_param` name suffix.

    To constrain the parameter we use a Normal parameter with centre on unit cube, and scale covering the whole cube,
    as the base representation. This base representation covers the whole real line and be reliably used with SGD, etc.

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
    # TODO(Joshuaalbert): consider something faster than ndtr to save FLOPs
    U_base_param = ndtr((norm_U_base_param - mu) / scale)
    param = prior.forward(U_base_param)
    dist = SingularDistribution(value=param, dist=prior.dist)
    return Prior(dist_or_value=dist, name=prior.name)


class EM:
    """
    Evidence Maximisation class, that implements the E and M steps. Iteratively computes the evidence and maximises it.
    """

    def __init__(self, model: ParametrisedModel, learning_rate: float = 1e-2, max_num_epochs: int = 100, gtol=1e-4,
                 log_Z_ftol=1., log_Z_atol=1e-4, ns_kwargs: Dict[str, Any] | None = None,
                 termination_cond: TerminationCondition | None = None):

        """
        Initialise the EM class.

        Args:
            model: The model to train.
            learning_rate: The learning rate for the M-step.
            max_num_epochs: The maximum number of epochs to train for.
            gtol: The gradient tolerance for the M-step.
            log_Z_ftol: The tolerance for the change in the evidence as function of log_Z_uncert.
            log_Z_atol: The absolute tolerance for the change in the evidence.
            ns_kwargs: The keyword arguments to pass to the nested sampler.
        """
        if not isinstance(model, ParametrisedModel):
            raise ValueError("model must be an instance of ParametrisedModel")
        self.model = model
        self.learning_rate = learning_rate
        self.max_num_epochs = max_num_epochs
        self.gtol = gtol
        self.log_Z_ftol = log_Z_ftol
        self.log_Z_atol = log_Z_atol
        self.ns_kwargs = ns_kwargs
        self.termination_cond = termination_cond

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
        if self.ns_kwargs is None:
            kwargs = dict()
        else:
            kwargs = self.ns_kwargs.copy()
        kwargs['num_live_points'] = kwargs.get('num_live_points', model.U_ndims * 20)
        kwargs['max_samples'] = kwargs.get('max_samples', 1e5)
        if self.termination_cond is None:
            termination_cond = TerminationCondition(live_evidence_frac=1e-5)
        else:
            termination_cond = self.termination_cond
        exact_ns = StandardStaticNestedSampler(model=model, **kwargs)
        termination_reason, state = exact_ns._run(random.PRNGKey(42),
                                             term_cond=termination_cond)
        results = exact_ns._to_results(termination_reason, state, trim=True)
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

    def train(self, num_steps: int = 10, params: hk.MutableParams | None = None, do_final_e_step: bool = False) -> \
            Tuple[
                NestedSamplerResults, hk.MutableParams]:
        """
        Train the model using EM for num_steps.

        Args:
            num_steps: The number of steps to train for, or until convergence.
            params: The initial parameters to use. If None, then the model's params are used.
            do_final_e_step: Whether to do a final E-step after training, for updated evidence.

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
            p_bar.set_description(
                f"Step {step}: log Z = {results.log_Z_mean:.4f} +- {results.log_Z_uncert:.4f}"
            )
            p_bar.refresh()

            # Check termination condition
            log_Z_change = jnp.abs(results.log_Z_mean - log_Z)
            if log_Z_change < max(self.log_Z_ftol * results.log_Z_uncert, self.log_Z_atol):
                print(f"Convergence achieved at step {step}.")
                if do_final_e_step:
                    results = self.e_step(params=params)
                    p_bar.set_description(
                        f"Step {step}: log Z = {results.log_Z_mean:.4f} +- {results.log_Z_uncert:.4f}"
                    )
                    p_bar.refresh()
                break

            # Update log_Z and log_Z_uncert values
            log_Z = results.log_Z_mean

            # Execute the m_step
            params = self.m_step(results, params)

        if results is None:
            raise RuntimeError("No results were computed.")
        return results, params
