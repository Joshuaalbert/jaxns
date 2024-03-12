import warnings
from typing import Optional, Tuple, Union

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import tree_map, core

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.logging import logger
from jaxns.internals.types import PRNGKey, IntArray, StaticStandardNestedSamplerState, TerminationCondition, \
    NestedSamplerResults
from jaxns.nested_sampler.bases import BaseAbstractNestedSampler
from jaxns.nested_sampler.standard_static import StandardStaticNestedSampler
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.samplers.uni_slice_sampler import UniDimSliceSampler
from jaxns.utils import summary, save_results, load_results

tfpd = tfp.distributions

__all__ = [
    'DefaultNestedSampler',
    'ApproximateNestedSampler',
    'ExactNestedSampler',
    'TerminationCondition'
]


class DefaultNestedSampler:
    """
    A static nested sampler that uses 1-dimensional slice sampler for the sampling step.
    Uses the phantom-powered algorithm. A robust default choice is provided for all parameters.
    """

    def __init__(self, model: BaseAbstractModel,
                 max_samples: Union[int, float],
                 num_live_points: Optional[int] = None,
                 s: Optional[int] = None,
                 k: Optional[int] = None,
                 c: Optional[int] = None,
                 num_parallel_workers: int = 1,
                 difficult_model: bool = False,
                 parameter_estimation: bool = False,
                 init_efficiency_threshold: float = 0.1,
                 verbose: bool = False):
        """
        Initialises the nested sampler.

        s,k,c are defined in the paper: https://arxiv.org/abs/2312.11330

        Args:
            model: a model to perform nested sampling on
            max_samples: maximum number of samples to take
            num_live_points: approximate number of live points to use. Defaults is c * (k + 1).
            s: number of slices to use per dimension. Defaults to 4.
            k: number of phantom samples to use. Defaults to 0.
            c: number of parallel Markov-chains to use. Defaults to 20 * D.
            num_parallel_workers: number of parallel workers to use. Defaults to 1. Experimental feature.
            difficult_model: if True, uses more robust default settings. Defaults to False.
            parameter_estimation: if True, uses more robust default settings for parameter estimation. Defaults to False.
            init_efficiency_threshold: if > 0 then use uniform sampling first down to this acceptance efficiency.
                0 turns it off.
            verbose: whether to use JAX printing
        """
        if difficult_model:
            self._s = 10 if s is None else int(s)
        else:
            self._s = 5 if s is None else int(s)
        if self._s <= 0:
            raise ValueError(f"Expected s > 0, got s={self._s}")
        if parameter_estimation:
            self._k = model.U_ndims if k is None else int(k)
        else:
            self._k = 0 if k is None else int(k)
        if not (0 <= self._k < self._s * model.U_ndims):
            raise ValueError(f"Expected 0 <= k < s * U_ndims, got k={self._k}, s={self._s}, U_ndims={model.U_ndims}")
        if num_live_points is not None:
            self._c = max(1, int(num_live_points / (self._k + 1)))
            logger.info(f"Number of parallel Markov-chains set to: {self._c}")
        else:
            if difficult_model:
                self._c = 50 * model.U_ndims if c is None else int(c)
            else:
                self._c = 30 * model.U_ndims if c is None else int(c)
        if self._c <= 0:
            raise ValueError(f"Expected c > 0, got c={self._c}")
        # Sanity check for max_samples (should be able to at least do one shrinkage)
        if max_samples < self._c * (self._k + 1):
            warnings.warn(f"max_samples={max_samples} is likely too small!")
        self._nested_sampler = StandardStaticNestedSampler(
            model=model,
            num_live_points=self._c,
            max_samples=max_samples,
            sampler=UniDimSliceSampler(
                model=model,
                num_slices=model.U_ndims * self._s,
                num_phantom_save=self._k,
                midpoint_shrink=True,
                perfect=True
            ),
            init_efficiency_threshold=init_efficiency_threshold,
            num_parallel_workers=num_parallel_workers,
            verbose=verbose
        )

        # Post-analysis utilities
        self.summary = summary
        self.plot_cornerplot = plot_cornerplot
        self.plot_diagnostics = plot_diagnostics
        self.save_results = save_results
        self.load_results = load_results

    def __repr__(self):
        return f"DefaultNestedSampler(s={self._s}, c={self._c},  k={self._k})"

    @property
    def num_live_points(self) -> int:
        return self._nested_sampler.num_live_points

    @property
    def nested_sampler(self) -> BaseAbstractNestedSampler:
        return self._nested_sampler

    def __call__(self, key: PRNGKey, term_cond: Optional[TerminationCondition] = None) -> Tuple[
        IntArray, StaticStandardNestedSamplerState]:
        """
        Performs nested sampling with the given termination conditions.

        Args:
            key: PRNGKey
            term_cond: termination conditions. If not given, see `TerminationCondition` for defaults.

        Returns:
            termination reason, state
        """
        if term_cond is None:
            term_cond = TerminationCondition()
        term_cond = term_cond._replace(
            max_samples=jnp.minimum(term_cond.max_samples, self._nested_sampler.max_samples)
        )
        return self._nested_sampler._run(
            key=key,
            term_cond=term_cond
        )

    def to_results(self, termination_reason: IntArray, state: StaticStandardNestedSamplerState,
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

        results = tree_map(trim, results)
        return results


class ApproximateNestedSampler(DefaultNestedSampler):
    def __init__(self, *args, **kwargs):
        warnings.warn(f"ApproximateNestedSampler is deprecated. Use DefaultNestedSampler instead.")
        super().__init__(*args, **kwargs)


class ExactNestedSampler(ApproximateNestedSampler):
    def __init__(self, *args, **kwargs):
        warnings.warn(f"ExactNestedSampler is deprecated. Use DefaultNestedSampler instead.")
        super().__init__(*args, **kwargs)
