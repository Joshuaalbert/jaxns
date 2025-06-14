import dataclasses
import time
from functools import partial
from typing import Tuple, Dict, Any, Optional, NamedTuple

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp, random

from jaxns import NestedSampler, Model
from jaxns.experimental.solvers.gauss_newton_cg import newton_cg_solver
from jaxns.framework.context import MutableParams
from jaxns.internals.cumulative_ops import cumulative_op_static
from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.logging import logger
from jaxns.internals.mixed_precision import mp_policy
from jaxns.internals.types import IntArray, PRNGKey
from jaxns.nested_samplers.common.types import TerminationCondition, NestedSamplerResults, \
    StaticStandardNestedSamplerState

__all__ = [
    'EvidenceMaximisation'
]

tfpd = tfp.distributions
tfpk = tfp.math.psd_kernels


class MStepData(NamedTuple):
    U_samples: jax.Array
    log_weights: jax.Array
    # log_dp_mean: jax.Array
    # log_L_samples: jax.Array
    # log_Z_mean: jax.Array


def next_power_2(x: int) -> int:
    """
    Next largest power of 2.

    Args:
        x:  int

    Returns:
        next largest n**2
    """
    return int(2 ** np.ceil(np.log2(x)))


@dataclasses.dataclass(eq=False)
class EvidenceMaximisation:
    """
    Evidence Maximisation class, that implements the E and M steps. Iteratively computes the evidence and maximises it
    using stochastic minibatching over samples from E-step.

    Args:
        model: The model to train.
        ns_kwargs: The keyword arguments to pass to the nested sampler. Needs at least `max_samples`.
        max_num_epochs: The maximum number of epochs to run M-step for.
        gtol: The parameter tolerance for the M-step. End when all parameters change by less than gtol.
        log_Z_ftol, log_Z_atol: The tolerances for the change in the evidence as function of log_Z_uncert.
            Terminate if the change in log_Z is less than max(log_Z_ftol * log_Z_uncert, log_Z_atol).
        batch_size: The batch size to use for the M-step.
        termination_cond: The termination condition to use for the nested sampler.
        verbose: Whether to print progress verbosely.
    """
    model: Model
    ns_kwargs: Optional[Dict[str, Any]] = None
    max_num_epochs: int = 50
    gtol: float = 1e-2
    log_Z_ftol: float = 1.
    log_Z_atol: float = 1e-4
    batch_size: Optional[int] = 128
    termination_cond: Optional[TerminationCondition] = None
    verbose: bool = False

    def __post_init__(self):
        if self.ns_kwargs is None:
            self.ns_kwargs = {}
        self._e_step = self._create_e_step()
        self._m_step = self._create_m_step()

    def _create_e_step(self):
        """
        Create a compiled function that runs nested sampling and returns trimmed results.

        Returns:
            A compiled function that runs nested sampling and returns trimmed results.
        """

        def _ns_solve(params: MutableParams, key: random.PRNGKey) -> Tuple[
            IntArray, StaticStandardNestedSamplerState]:
            model = self.model(params=params)
            ns = NestedSampler(model=model, **self.ns_kwargs)
            termination_reason, state = ns(key, self.termination_cond)
            return termination_reason, state

        # Ahead of time compile the function
        t0 = time.time()
        ns_solve_compiled = jax.jit(_ns_solve).lower(self.model.params, random.PRNGKey(42)).compile()
        if self.verbose:
            logger.info(f"E-step compilation time: {time.time() - t0:.2f}s")
        ns = NestedSampler(model=self.model(params=self.model.params), **self.ns_kwargs)

        def _e_step(key: PRNGKey, params: MutableParams, desc: str) -> NestedSamplerResults:
            print(f"Running E-step... {desc}")
            termination_reason, state = ns_solve_compiled(params, key)
            # Trim results
            return ns.to_results(termination_reason=termination_reason, state=state, trim=True)

        return _e_step

    def e_step(self, key: PRNGKey, params: MutableParams, desc) -> NestedSamplerResults:
        """
        The E-step is just nested sampling.

        Args:
            key: The random number generator key.
            params: The parameters to use.
            desc: progress bar desc

        Returns:
            The nested sampling results.
        """

        # The E-step is just nested sampling
        return self._e_step(key, params, desc)

    def _m_step_iterator(self, key: PRNGKey, data: MStepData):
        num_samples = int(data.U_samples.shape[0])
        permutation = jax.random.permutation(key, num_samples)
        if self.batch_size is None:
            batch_size = num_samples
        else:
            batch_size = self.batch_size
        num_batches = num_samples // batch_size
        if num_batches == 0:
            raise RuntimeError(
                f"Batch size {batch_size} is too large for number of samples, number of samples {num_samples}.")
        for i in range(num_batches):
            perm = permutation[i * batch_size:(i + 1) * batch_size]
            batch = MStepData(
                U_samples=data.U_samples[perm],
                log_weights=data.log_weights[perm]
            )
            yield batch

    def _create_m_step(self):

        def log_evidence(params: MutableParams, data: MStepData):
            # Compute the log evidence
            model = self.model(params=params)

            def op(log_Z, data):
                log_dZ = model.forward(data.U_samples) + data.log_weights
                return (LogSpace(log_Z) + LogSpace(log_dZ)).log_abs_val

            log_Z, _ = cumulative_op_static(op=op, init=jnp.asarray(-jnp.inf, mp_policy.measure_dtype), xs=data)
            return log_Z

        def loss(params: MutableParams, data: MStepData):
            log_Z = log_evidence(params, data)
            obj = -log_Z
            if self.verbose:
                jax.debug.print("log_Z={log_Z}", log_Z=log_Z)
            return obj

        @partial(jax.jit)
        def _m_step(key: PRNGKey, params: MutableParams, data: MStepData) -> Tuple[MutableParams, Any]:
            """
            The M-step is just evidence maximisation.

            Args:
                params: The parameters to use.
                data: The data to use.

            Returns:
                The updated parameters and the negative log evidence.
            """
            params, diagnostics = newton_cg_solver(loss, params, args=(data,))
            i = jnp.max(diagnostics.iteration)
            return params, -diagnostics.f[i]

        return _m_step

    def m_step(self, key: PRNGKey, params: MutableParams, ns_results: NestedSamplerResults, desc: str) -> Tuple[
        MutableParams, Any]:
        """
        The M-step is just evidence maximisation. We pad the data to the next power of 2, to make JIT compilation
        happen less frequently.

        Args:
            key: The random number generator key.
            params: The parameters to use.
            ns_results: The nested sampling results to use.
            desc: progress bar description

        Returns:
            The updated parameters
        """
        # next_power_2 pad
        num_samples = int(ns_results.total_num_samples)
        n = next_power_2(num_samples)

        print(f"Running M-step ({num_samples} samples padded to {n})... {desc}")

        def _pad_to_n(x, fill_value, dtype):
            if x.shape[0] == n:
                return x
            return jnp.concatenate([x, jnp.full((n - x.shape[0],) + x.shape[1:], fill_value, dtype)], axis=0)

        log_weights = ns_results.log_dp_mean - ns_results.log_L_samples + ns_results.log_Z_mean
        data = MStepData(
            U_samples=_pad_to_n(ns_results.U_samples, 0.5, mp_policy.measure_dtype),
            log_weights=_pad_to_n(log_weights, -jnp.inf, mp_policy.measure_dtype)
        )
        last_params = params
        epoch = 0
        log_Z = None
        while epoch < self.max_num_epochs:
            params, log_Z = self._m_step(key=key, params=params, data=data)
            l_oo = jax.tree.map(lambda x, y: float(jnp.max(jnp.abs(x - y))) if np.size(x) > 0 else 0.,
                                last_params, params)
            last_params = params
            print(f"{desc}: Epoch {epoch}: log_Z={log_Z}, l_oo={l_oo}")

            if all(_l_oo < self.gtol for _l_oo in jax.tree.leaves(l_oo)):
                break
            epoch += 1

        return params, log_Z

    def train(self, num_steps: int = 10, params: Optional[MutableParams] = None) -> \
            Tuple[
                NestedSamplerResults, MutableParams]:
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

        ns_results = None
        for step in range(num_steps):
            key_e_step, key_m_step = random.split(random.PRNGKey(step), 2)

            # Execute the e_step
            if ns_results is None:
                desc = f"Step {step}: Initial run"

            else:
                desc = f"Step {step}: log Z = {ns_results.log_Z_mean:.4f} +- {ns_results.log_Z_uncert:.4f}"

            ns_results = self.e_step(key=key_e_step, params=params, desc=desc)
            # Update progress bar description

            # Check termination condition
            log_Z_change = jnp.abs(ns_results.log_Z_mean - log_Z)

            if log_Z_change < self.log_Z_atol:
                desc = (
                    f"Convergence achieved at step {step}, "
                    f"due to delta log_Z {log_Z_change} < log_Z_atol {self.log_Z_atol}."
                )

                break

            relative_atol = float(self.log_Z_ftol * ns_results.log_Z_uncert)
            if log_Z_change < relative_atol:
                desc = (
                    f"Convergence achieved at step {step}, "
                    f"due to log_Z {log_Z_change} < log_Z_ftol * log_Z_uncert {relative_atol}."
                )

                break

            # Update log_Z and log_Z_uncert values
            log_Z = ns_results.log_Z_mean

            # Execute the m_step
            desc = (
                f"Step {step}: log Z = {ns_results.log_Z_mean:.4f} +- {ns_results.log_Z_uncert:.4f}"
            )

            params, log_Z_opt = self.m_step(key=key_m_step, params=params, ns_results=ns_results, desc=desc)

        if ns_results is None:
            raise RuntimeError("No results were computed.")
        return ns_results, params
