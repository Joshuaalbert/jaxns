import logging
from typing import Optional, Tuple, Union, List

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.types import PRNGKey, IntArray, StaticStandardNestedSamplerState, TerminationCondition, \
    NestedSamplerResults
from jaxns.nested_sampler.bases import BaseAbstractNestedSampler
from jaxns.nested_sampler.standard_static import StandardStaticNestedSampler
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.samplers.uni_slice_sampler import UniDimSliceSampler
from jaxns.utils import summary, save_results, load_results

tfpd = tfp.distributions

logger = logging.getLogger('jaxns')

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

    def __init__(self, model: BaseAbstractModel, max_samples: Union[int, float], num_live_points: Optional[int] = None,
                 num_parallel_workers: int = 1):
        """
        Initialises the nested sampler.

        Args:
            model: a model to perform nested sampling on
            max_samples: maximum number of samples to take
            num_live_points: number of live points to use. If not given, defaults to `50 * model.U_ndims`.
            num_parallel_workers: number of parallel workers to use. Defaults to 1.
        """
        s = 4
        k = model.U_ndims//2
        num_live_points = num_live_points or model.U_ndims * 64
        c = max(1, int(num_live_points / (k + 1) / num_parallel_workers))
        self._nested_sampler = StandardStaticNestedSampler(
            model=model,
            num_live_points=c,
            max_samples=max_samples,
            sampler=UniDimSliceSampler(
                model=model,
                num_slices=model.U_ndims * s,
                num_phantom_save=k,
                midpoint_shrink=True,
                perfect=True
            ),
            init_efficiency_threshold=0.1,
            num_parallel_workers=num_parallel_workers
        )

    @property
    def num_live_points(self) -> int:
        return self._nested_sampler.num_live_points

    @property
    def nested_sampler(self) -> BaseAbstractNestedSampler:
        return self._nested_sampler

    def summary(self, results: NestedSamplerResults) -> str:
        """
        Prints a summary of the results of the nested sampling run.

        Args:
            results: results of the nested sampling run

        Returns:
            the summary as a string
        """
        return summary(results)

    def plot_cornerplot(self, results: NestedSamplerResults, vars: Optional[List[str]] = None,
                        save_name: Optional[str] = None):
        """
        Plots a corner plot of the samples.

        Args:
            results: results of the nested sampling run
            vars: variables to plot. If not given, defaults to all variables.
            save_name: if given, saves the plot to the given file name
        """
        plot_cornerplot(results, vars=vars, save_name=save_name)

    def plot_diagnostics(self, results: NestedSamplerResults, save_name: Optional[str] = None):
        """
        Plots diagnostic plots of the results of the nested sampling run.

        Args:
            results: results of the nested sampling run
            save_name: if given, saves the plot to the given file name
        """
        plot_diagnostics(results, save_name=save_name)

    def save_results(self, results: NestedSamplerResults, save_file: str):
        """
        Saves the results of the nested sampling run to a file.

        Args:
            results: results of the nested sampling run
            save_file: file to save the results to
        """
        save_results(results, save_file)

    def load_results(self, save_file: str) -> NestedSamplerResults:
        """
        Loads the results of a nested sampling run from a file.

        Args:
            save_file: file to load the results from

        Returns:
            results
        """
        return load_results(save_file)

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


class ApproximateNestedSampler(DefaultNestedSampler):
    def __init__(self, *args, **kwargs):
        logger.warning(f"ApproximateNestedSampler is deprecated. Use DefaultNestedSampler instead.")
        super().__init__(*args, **kwargs)


class ExactNestedSampler(ApproximateNestedSampler):
    def __init__(self, *args, **kwargs):
        logger.warning(f"ExactNestedSampler is deprecated. Use DefaultNestedSampler instead.")
        super().__init__(*args, **kwargs)
