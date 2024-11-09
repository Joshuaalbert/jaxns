import dataclasses
from typing import Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import core
from jaxlib import xla_client

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.logging import logger
from jaxns.internals.mixed_precision import mp_policy
from jaxns.internals.types import PRNGKey, IntArray
from jaxns.nested_samplers.abc import AbstractNestedSampler
from jaxns.nested_samplers.common.types import TerminationCondition, NestedSamplerResults, \
    NestedSamplerState
from jaxns.nested_samplers.sharded import ShardedStaticNestedSampler
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.samplers.uni_slice_sampler import UniDimSliceSampler
from jaxns.utils import summary, save_results, load_results

tfpd = tfp.distributions

__all__ = [
    'NestedSampler'
]


@dataclasses.dataclass(eq=False)
class NestedSampler:
    """
    A static nested sampler that uses 1-dimensional slice sampler for the sampling step.
    Uses the phantom-powered algorithm. A robust default choice is provided for all parameters.

    s,k,c are defined in the paper: https://arxiv.org/abs/2312.11330

    Args:
        model: a model to perform nested sampling on
        max_samples: maximum number of samples to take
        num_live_points: approximate number of live points to use. Defaults is c * (k + 1).
        s: number of slices to use per dimension. Defaults to 4.
        k: number of phantom samples to use. Defaults to 0.
        c: number of parallel Markov-chains to use. Defaults to 20 * D.
        devices: devices to use. Defaults to all available devices.
        difficult_model: if True, uses more robust default settings. Defaults to False.
        parameter_estimation: if True, uses more robust default settings for parameter estimation. Defaults to False.
        shell_fraction: fraction of the shell to use for the slice sampler. Defaults to 0.5.
        gradient_guided: if True, uses gradient guided sampling. Defaults to False.
        init_efficiency_threshold: if > 0 then use uniform sampling first down to this acceptance efficiency.
            0 turns it off.
        verbose: whether to log progress.
    """
    model: BaseAbstractModel
    max_samples: Optional[Union[int, float]] = None
    num_live_points: Optional[int] = None
    num_slices: Optional[int] = None
    s: Optional[int] = None
    k: Optional[int] = None
    c: Optional[int] = None
    devices: Optional[List[xla_client.Device]] = None
    difficult_model: bool = False
    parameter_estimation: bool = False
    shell_fraction: float = 0.5
    gradient_guided: bool = False
    init_efficiency_threshold: float = 0.1
    verbose: bool = False

    def __post_init__(self):
        # Determine number of slices per acceptance
        if self.num_slices is None:
            if self.difficult_model:
                self.s = 10 if self.s is None else int(self.s)
            else:
                self.s = 5 if self.s is None else int(self.s)
            if self.s <= 0:
                raise ValueError(f"Expected s > 0, got s={self.s}")
            self.num_slices = self.model.U_ndims * self.s
        self.num_slices = int(self.num_slices)

        # Determine number of phantom samples
        if self.parameter_estimation:
            max_k = self.s * self.model.U_ndims - 1
            self.k = min(self.model.U_ndims, max_k) if self.k is None else int(self.k)
        else:
            self.k = 0 if self.k is None else int(self.k)
        if not (0 <= self.k < self.num_slices):
            raise ValueError(
                f"Expected 0 <= k < num_slices, got k={self.k}, num_slices={self.num_slices}, U_ndims={self.model.U_ndims}")

        # Determine number of parallel Markov-chains
        if self.num_live_points is not None:
            self.c = max(1, int(self.num_live_points / (self.k + 1)))
            logger.info(f"Number of Markov-chains set to: {self.c}")
        else:
            if self.difficult_model:
                self.c = 100 * self.model.U_ndims if self.c is None else int(self.c)
            else:
                self.c = 30 * self.model.U_ndims if self.c is None else int(self.c)
            if self.c <= 0:
                raise ValueError(f"Expected c > 0, got c={self.c}")

        # Sanity check for max_samples (should be able to at least do one shrinkage)
        if self.max_samples is None:
            # Default to 100 shrinkages.
            self.max_samples = self.c * (self.k + 1) * 100
        self.max_samples = int(self.max_samples)

        self._nested_sampler = ShardedStaticNestedSampler(
            model=self.model,
            num_live_points=self.c,
            max_samples=self.max_samples,
            sampler=UniDimSliceSampler(
                model=self.model,
                num_slices=self.num_slices,
                num_phantom_save=self.k,
                midpoint_shrink=not self.difficult_model,
                gradient_guided=self.gradient_guided,
                perfect=True
            ),
            init_efficiency_threshold=self.init_efficiency_threshold,
            shell_fraction=self.shell_fraction,
            devices=self.devices,
            verbose=self.verbose,

        )
        # Back propagate any updates here
        self.num_live_points = self._nested_sampler.num_live_points

        # Post-analysis utilities
        self.summary = summary
        self.plot_cornerplot = plot_cornerplot
        self.plot_diagnostics = plot_diagnostics
        self.save_results = save_results
        self.load_results = load_results

    @property
    def nested_sampler(self) -> AbstractNestedSampler:
        return self._nested_sampler

    def __call__(self, key: PRNGKey, term_cond: Optional[TerminationCondition] = None) -> Tuple[
        IntArray, NestedSamplerState]:
        """
        Performs nested sampling with the given termination conditions.

        Args:
            key: PRNGKey
            term_cond: termination conditions. If not given, see `TerminationCondition` for defaults.

        Returns:
            termination reason, state
        """
        if term_cond is None:
            if self.parameter_estimation:
                term_cond = TerminationCondition(
                    peak_XL_frac=jnp.asarray(0.1, mp_policy.measure_dtype),
                    max_samples=jnp.asarray(jnp.iinfo(mp_policy.count_dtype).max, mp_policy.count_dtype)
                )
            else:
                term_cond = TerminationCondition(
                    dlogZ=jnp.asarray(np.log(1. + 1e-3), mp_policy.measure_dtype),
                    max_samples=jnp.asarray(jnp.iinfo(mp_policy.count_dtype).max, mp_policy.count_dtype)
                )
        term_cond = term_cond._replace(
            max_samples=(
                jnp.minimum(term_cond.max_samples, jnp.asarray(self._nested_sampler.max_samples, mp_policy.count_dtype))
                if term_cond.max_samples is not None
                else jnp.asarray(self._nested_sampler.max_samples, mp_policy.count_dtype)
            )
        )
        termination_reason, termination_register, state = self._nested_sampler._run(
            key=key,
            term_cond=term_cond
        )
        return termination_reason, state

    def to_results(self, termination_reason: IntArray, state: NestedSamplerState,
                   trim: bool = True) -> NestedSamplerResults:
        """
        Convert the state to results.

        Note: Requires static context.

        Args:
            termination_reason: termination reason
            state: state to convert
            trim: if True, trims the results to the number of samples taken, requires static context.

        Returns:
            results
        """
        return self._nested_sampler._to_results(
            termination_reason=termination_reason,
            state=state,
            trim=trim
        )

    @staticmethod
    def trim_results(results: NestedSamplerResults) -> NestedSamplerResults:
        """
        Trims the results to the number of samples taken. Requires static context.

        Args:
            results: results to trim

        Returns:
            trimmed results
        """

        if isinstance(results.total_num_samples, core.Tracer):
            raise RuntimeError("Tracer detected, but expected imperative context.")

        def trim(x):
            if x.size > 1:
                return x[:results.total_num_samples]
            return x

        results = jax.tree.map(trim, results)
        return results


DefaultNestedSampler = NestedSampler
