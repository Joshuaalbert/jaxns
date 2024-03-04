import logging
from functools import partial
from typing import Tuple, Dict, Any, Optional, NamedTuple

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp, random
from jax._src.scipy.special import logsumexp
from jaxopt import NonlinearCG, ArmijoSGD
from tqdm import tqdm

from jaxns.internals.cumulative_ops import cumulative_op_static
from jaxns.internals.log_semiring import LogSpace

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

from jaxns import DefaultNestedSampler, Model
from jaxns.internals.types import TerminationCondition, NestedSamplerResults, StaticStandardNestedSamplerState, \
    IntArray, PRNGKey, float_type

__all__ = [
    'EvidenceMaximisation'
]

tfpd = tfp.distributions
tfpk = tfp.math.psd_kernels

logger = logging.getLogger('jaxns')


class MStepData(NamedTuple):
    U_samples: jnp.ndarray
    log_weights: jnp.ndarray
    # log_dp_mean: jnp.ndarray
    # log_L_samples: jnp.ndarray
    # log_Z_mean: jnp.ndarray


def next_power_2(x: int) -> int:
    """
    Next largest power of 2.

    Args:
        x:  int

    Returns:
        next largest n**2
    """
    return int(2 ** np.ceil(np.log2(x)))


class EvidenceMaximisation:
    """
    Evidence Maximisation class, that implements the E and M steps. Iteratively computes the evidence and maximises it.
    """

    def __init__(self, model: Model, ns_kwargs: Dict[str, Any],
                 max_num_epochs: int = 50, gtol=1e-2,
                 log_Z_ftol=1., log_Z_atol=1e-4,
                 batch_size: int = 128,
                 termination_cond: Optional[TerminationCondition] = None,
                 verbose: bool = False):

        """
        Initialise the EM class.

        Args:
            model: The model to train.
            max_num_epochs: The maximum number of epochs to run M-step for.
            gtol: The parameter tolerance for the M-step. End when all parameters change by less than gtol.
            log_Z_ftol, log_Z_atol: The tolerances for the change in the evidence as function of log_Z_uncert.
                Terminate if the change in log_Z is less than max(log_Z_ftol * log_Z_uncert, log_Z_atol).
            ns_kwargs: The keyword arguments to pass to the nested sampler. Needs at least `max_samples`.
            verbose: Whether to print progress verbosely.
        """
        if not isinstance(model, Model):
            raise ValueError("model must be an instance of ParametrisedModel")
        self.model = model
        self.max_num_epochs = max_num_epochs
        self.gtol = gtol
        self.log_Z_ftol = log_Z_ftol
        self.log_Z_atol = log_Z_atol
        self._verbose = bool(verbose)
        self._ns_kwargs = ns_kwargs
        self._batch_size = batch_size
        self._termination_cond = termination_cond
        self._e_step = self._create_e_step()
        self._m_step = self._create_m_step_stochastic()

    def _create_e_step(self):
        """
        Create a compiled function that runs nested sampling and returns trimmed results.

        Returns:
            A compiled function that runs nested sampling and returns trimmed results.
        """

        def _ns_solve(params: hk.MutableParams, key: random.PRNGKey) -> Tuple[
            IntArray, StaticStandardNestedSamplerState]:
            model = self.model(params=params)
            ns = DefaultNestedSampler(model=model, **self._ns_kwargs)
            termination_reason, state = ns(key, self._termination_cond)
            return termination_reason, state

        # Ahead of time compile the function
        ns_compiled = jax.jit(_ns_solve).lower(self.model.params, random.PRNGKey(42)).compile()

        def _e_step(key: PRNGKey, params: hk.MutableParams, p_bar: tqdm) -> NestedSamplerResults:
            p_bar.set_description(f"Running E-step... {p_bar.desc}")
            termination_reason, state = ns_compiled(params, key)
            ns = DefaultNestedSampler(model=self.model(params=params), **self._ns_kwargs)
            # Trim now
            return ns.to_results(termination_reason=termination_reason, state=state, trim=True)

        return _e_step

    def e_step(self, key: PRNGKey, params: hk.MutableParams, p_bar: tqdm) -> NestedSamplerResults:
        """
        The E-step is just nested sampling.

        Args:
            key: The random number generator key.
            params: The parameters to use.
            p_bar: progress bar

        Returns:
            The nested sampling results.
        """

        # The E-step is just nested sampling
        return self._e_step(key, params, p_bar)

    def _m_step_iterator(self, key: PRNGKey, data: MStepData):
        num_samples = int(data.U_samples.shape[0])
        permutation = jax.random.permutation(key, num_samples)
        num_batches = num_samples // self._batch_size
        if num_batches == 0:
            raise RuntimeError("Batch size is too large for number of samples.")
        for i in range(num_batches):
            perm = permutation[i * self._batch_size:(i + 1) * self._batch_size]
            batch = MStepData(
                U_samples=data.U_samples[perm],
                log_weights=data.log_weights[perm]
            )
            yield batch

    def _create_m_step_stochastic(self):
        def log_evidence(params: hk.MutableParams, data: MStepData):
            # Compute the log evidence
            model = self.model(params=params)
            # To make manageable, we could do chunked_pmap
            log_dZ = jax.vmap(
                lambda U, log_weight: model.forward(U) + log_weight
            )(data.U_samples, data.log_weights)
            # We add the log_Z_mean because log_dp_mean is normalised
            log_Z = logsumexp(log_dZ)
            return log_Z

        def loss(params: hk.MutableParams, data: MStepData):
            log_Z, grad = jax.value_and_grad(log_evidence, argnums=0)(params, data)
            obj = -log_Z
            grad = jax.tree_map(jnp.negative, grad)
            aux = (log_Z,)
            if self._verbose:
                jax.debug.print("log_Z={log_Z}", log_Z=log_Z)
            return (obj, aux), grad

        solver = ArmijoSGD(
            fun=loss,
            has_aux=True,
            value_and_grad=True,
            jit=True,
            unroll=False,
            verbose=self._verbose,
            momentum=self.momentum
        )

        def _m_step_stochastic(key: PRNGKey, params: hk.MutableParams, data: MStepData) -> Tuple[hk.MutableParams, Any]:
            """
            The M-step is just evidence maximisation.

            Args:
                key: The random number generator key.
                params: The parameters to use.
                data: The data to use.

            Returns:
                The updated parameters.
            """

            # The M-step is just evidence maximisation

            iterator = self._m_step_iterator(key, data)
            opt_results = solver.run_iterator(init_params=params, iterator=iterator)
            return opt_results.params, opt_results.state.aux

        return _m_step_stochastic

    def _create_m_step(self):

        def log_evidence(params: hk.MutableParams, data: MStepData):
            # Compute the log evidence
            model = self.model(params=params)

            def op(log_Z, data):
                log_dZ = model.forward(data.U_samples) + data.log_weights
                return (LogSpace(log_Z) + LogSpace(log_dZ)).log_abs_val

            log_Z, _ = cumulative_op_static(op=op, init=jnp.asarray(-jnp.inf, float_type), xs=data)
            return log_Z

        def loss(params: hk.MutableParams, data: MStepData):
            log_Z, grad = jax.value_and_grad(log_evidence, argnums=0)(params, data)
            obj = -log_Z
            grad = jax.tree_map(jnp.negative, grad)
            aux = (log_Z,)
            if self._verbose:
                jax.debug.print("log_Z={log_Z}", log_Z=log_Z)
            return (obj, aux), grad

        solver = NonlinearCG(
            fun=loss,
            has_aux=True,
            value_and_grad=True,
            jit=True,
            unroll=False,
            verbose=self._verbose
        )

        @partial(jax.jit, static_argnums=(0,))
        def _m_step(key: PRNGKey, params: hk.MutableParams, data: MStepData) -> Tuple[hk.MutableParams, Any]:
            """
            The M-step is just evidence maximisation.

            Args:
                params: The parameters to use.
                data: The data to use.

            Returns:
                The updated parameters and the negative log evidence.
            """
            opt_results = solver.run(init_params=params, data=data)
            return opt_results.params, opt_results.state.aux

        return _m_step

    def m_step(self, key: PRNGKey, params: hk.MutableParams, ns_results: NestedSamplerResults, p_bar: tqdm) -> Tuple[
        hk.MutableParams, Any]:
        """
        The M-step is just evidence maximisation. We pad the data to the next power of 2, to make JIT compilation
        happen less frequently.

        Args:
            key: The random number generator key.
            params: The parameters to use.
            ns_results: The nested sampling results to use.
            p_bar: progress bar

        Returns:
            The updated parameters
        """
        # next_power_2 pad
        num_samples = int(ns_results.total_num_samples)
        n = next_power_2(num_samples)

        p_bar.set_description(f"Running M-step ({num_samples} samples padded to {n})... {p_bar.desc}")

        def _pad_to_n(x, fill_value, dtype):
            return jnp.concatenate([x, jnp.full((n - x.shape[0],) + x.shape[1:], fill_value, dtype)], axis=0)

        log_weights = ns_results.log_dp_mean - ns_results.log_L_samples + ns_results.log_Z_mean
        data = MStepData(
            U_samples=_pad_to_n(ns_results.U_samples, 0.5, float_type),
            log_weights=_pad_to_n(log_weights, -jnp.inf, float_type)
        )
        desc = p_bar.desc
        last_params = params
        epoch = 0
        log_Z = None
        while epoch < self.max_num_epochs:
            params, (log_Z,) = self._m_step(key=key, params=params, data=data)
            l_oo = jax.tree_map(lambda x, y: jnp.max(jnp.abs(x - y)), last_params, params)
            last_params = params
            p_bar.set_description(f"{desc}: Epoch {epoch}: log_Z={log_Z}, l_oo={l_oo}")
            if all(_l_oo < self.gtol for _l_oo in jax.tree_leaves(l_oo)):
                break
            epoch += 1

        return params, log_Z

    def train(self, num_steps: int = 10, params: Optional[hk.MutableParams] = None) -> \
            Tuple[
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

        ns_results = None
        for step in p_bar:
            key_e_stek, key_m_step = random.split(random.PRNGKey(step), 2)

            # Execute the e_step
            if ns_results is None:
                p_bar.set_description(f"Step {step}: Initial run")
            else:
                p_bar.set_description(
                    f"Step {step}: log Z = {ns_results.log_Z_mean:.4f} +- {ns_results.log_Z_uncert:.4f}"
                )
            ns_results = self.e_step(key=key_e_stek, params=params, p_bar=p_bar)
            # Update progress bar description

            # Check termination condition
            log_Z_change = jnp.abs(ns_results.log_Z_mean - log_Z)
            if log_Z_change < max(self.log_Z_ftol * ns_results.log_Z_uncert, self.log_Z_atol):
                p_bar.set_description(f"Convergence achieved at step {step}.")
                break

            # Update log_Z and log_Z_uncert values
            log_Z = ns_results.log_Z_mean

            # Execute the m_step
            p_bar.set_description(
                f"Step {step}: log Z = {ns_results.log_Z_mean:.4f} +- {ns_results.log_Z_uncert:.4f}"
            )
            params, log_Z_opt = self.m_step(key=key_m_step, params=params, ns_results=ns_results, p_bar=p_bar)

        if ns_results is None:
            raise RuntimeError("No results were computed.")
        return ns_results, params
