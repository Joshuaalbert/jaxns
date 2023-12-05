import logging
from typing import Optional, Tuple, Union, List

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from jaxns.model.bases import BaseAbstractModel
from jaxns.nested_sampler.bases import BaseAbstractNestedSampler
from jaxns.nested_sampler.standard_static import StandardStaticNestedSampler
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.samplers.uni_slice_sampler import UniDimSliceSampler
from jaxns.types import PRNGKey, IntArray, StaticStandardNestedSamplerState, TerminationCondition, NestedSamplerResults
from jaxns.utils import summary, save_results, load_results

tfpd = tfp.distributions

logger = logging.getLogger('jaxns')

__all__ = [
    'BaseNestedSampler',
    'ApproximateNestedSampler',
    'ExactNestedSampler'
]


class BaseNestedSampler:
    def __init__(self, model: BaseAbstractModel, max_samples: Union[int, float], num_live_points: Optional[int] = None,
                 num_parallel_workers: int = 1):
        self._nested_sampler = StandardStaticNestedSampler(
            model=model,
            num_live_points=num_live_points or model.U_ndims * 20,
            max_samples=max_samples,
            sampler=UniDimSliceSampler(
                model=model,
                num_slices=model.U_ndims * 3,
                num_phantom_save=0,
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
        return summary(results)

    def plot_cornerplot(self, results: NestedSamplerResults, vars: Optional[List[str]] = None):
        plot_cornerplot(results, vars=vars)

    def plot_diagnostics(self, results: NestedSamplerResults):
        plot_diagnostics(results)

    def save_results(self, results: NestedSamplerResults, save_file: str):
        save_results(results, save_file)

    def load_results(self, save_file: str):
        return load_results(save_file)

    def __call__(self, key: PRNGKey, term_cond: TerminationCondition) -> Tuple[
        IntArray, StaticStandardNestedSamplerState]:
        """
        Performs nested sampling with the given termination conditions.

        Args:
            key: PRNGKey
            term_cond: termination conditions

        Returns:
            termination reason, state
        """
        term_cond = term_cond._replace(
            max_samples=jnp.minimum(term_cond.max_samples, self._nested_sampler.max_samples)
        )
        return self._nested_sampler._run(
            key=key,
            term_cond=term_cond
        )

    def to_results(self, termination_reason: IntArray, state: StaticStandardNestedSamplerState) -> NestedSamplerResults:
        """
        Convert the state to results. Requires static context.

        Args:
            termination_reason: termination reason
            state: state to convert

        Returns:
            results
        """
        return self._nested_sampler._to_results(
            termination_reason=termination_reason,
            state=state,
            trim=True
        )


class ApproximateNestedSampler(BaseNestedSampler):
    pass


class ExactNestedSampler(BaseNestedSampler):
    pass
