import logging
from typing import Optional, Tuple, Union, List

import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp, core, tree_map, vmap
from jaxns.initial_state import init_sample_collection, get_uniform_init_live_points
from jaxns.statistics import analyse_sample_collection

from jaxns.internals.log_semiring import LogSpace, normalise_log_space
from jaxns.internals.stats import linear_to_log_stats, effective_sample_size
from jaxns.samplers.uni_slice_sampler import UniDimSliceSampler
from jaxns.model.abc import AbstractModel
from jaxns.model.bases import BaseAbstractModel
from jaxns.nested_sampler.bases import BaseAbstractNestedSampler
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.samplers.abc import AbstractSampler
from jaxns.types import PRNGKey, IntArray, StaticStandardNestedSamplerState
from jaxns.nested_sampler.standard_static import StandardStaticNestedSampler
from jaxns.types import TerminationCondition, NestedSamplerResults
from jaxns.utils import summary, save_results, load_results
from jaxns.warnings import deprecated

tfpd = tfp.distributions

logger = logging.getLogger('jaxns')

__all__ = [
    'ApproximateNestedSampler',
    'ExactNestedSampler'
]


class BaseNestedSampler:
    def __init__(self, model: AbstractModel, max_samples: Union[int, float]):
        self.model = model
        self.max_samples = int(max_samples)

    def initialise(self, key: PRNGKey, num_live_points: int) -> Tuple[NestedSamplerState, LivePoints]:
        """
        Creates initial live points from -inf contour and state.

        Args:
            key: PRNGKey

        Returns:
            initial state, and live points
        """
        state_key, live_points_key = random.split(key, 2)
        init_state = NestedSamplerState(
            key=state_key,
            sample_collection=init_sample_collection(size=self.max_samples, model=self.model)
        )
        live_points = get_uniform_init_live_points(live_points_key,
                                                   num_live_points=num_live_points,
                                                   model=self.model)
        return init_state, live_points

    def resize_state(self, state: NestedSamplerState, max_num_samples: int) -> NestedSamplerState:
        """
        Grow the state to the size of max_num_samples.

        Args:
            state: state to enlarge
            max_num_samples: size of resulting sample collection

        Returns:
            resized state
        """
        reservoir = state.sample_collection.reservoir
        if max_num_samples <= reservoir.log_L.size:
            logger.warning(
                f"expected max_num_samples larger than the current size {reservoir.log_L.size}")
            return state
        diff_size = max_num_samples - reservoir.log_L.size
        if diff_size <= 0:
            return state
        reservoir_extra = init_sample_collection(size=diff_size, model=self.model).reservoir
        reservoir = tree_map(lambda old, update: jnp.concatenate([old, update], axis=0),
                             reservoir,
                             reservoir_extra)
        return state._replace(sample_collection=state.sample_collection._replace(reservoir=reservoir))

    def to_results(self, state: NestedSamplerState, termination_reason: IntArray) -> NestedSamplerResults:
        """
        Converts a state into a result.

        Args:
            state: state
            termination_reason: the termination reason

        Returns:
            results object
        """

        if isinstance(state.sample_collection.reservoir.iid, core.Tracer):
            raise RuntimeError("Tracer detected, but expected imperative context.")

        num_samples = jnp.minimum(state.sample_collection.sample_idx, state.sample_collection.reservoir.log_L.size)

        sample_collection = state.sample_collection._replace(
            reservoir=tree_map(lambda x: x[:num_samples], state.sample_collection.reservoir)
        )

        evidence_calculation, sample_stats = analyse_sample_collection(
            sample_collection=sample_collection,
            sorted_collection=True,
            dual=False
        )

        log_Z_mean, log_Z_var = linear_to_log_stats(
            log_f_mean=evidence_calculation.log_Z_mean,
            log_f2_mean=evidence_calculation.log_Z2_mean
        )
        log_Z_uncert = jnp.sqrt(log_Z_var)

        # Kish's ESS = [sum dZ]^2 / [sum dZ^2]
        ESS = effective_sample_size(evidence_calculation.log_Z_mean, evidence_calculation.log_dZ2_mean)

        samples = vmap(self.model.transform)(sample_collection.reservoir.point_U)

        log_L_samples = sample_collection.reservoir.log_L
        dp_mean = LogSpace(sample_stats.log_dZ_mean)
        dp_mean = normalise_log_space(dp_mean)
        H_mean = LogSpace(jnp.where(jnp.isneginf(dp_mean.log_abs_val),
                                    -jnp.inf,
                                    dp_mean.log_abs_val + log_L_samples)).sum().value - log_Z_mean
        X_mean = LogSpace(sample_stats.log_X_mean)
        num_likelihood_evaluations_per_sample = sample_collection.reservoir.num_likelihood_evaluations
        total_num_likelihood_evaluations = jnp.sum(num_likelihood_evaluations_per_sample)
        num_live_points_per_sample = sample_stats.num_live_points
        efficiency = LogSpace(jnp.log(num_samples) - jnp.log(total_num_likelihood_evaluations))

        log_posterior_density = sample_collection.reservoir.log_L + vmap(self.model.log_prob_prior)(
            sample_collection.reservoir.point_U)

        total_num_slices = jnp.sum(sample_collection.reservoir.num_slices)

        return NestedSamplerResults(
            log_Z_mean=log_Z_mean,  # estimate of log(E[Z])
            log_Z_uncert=log_Z_uncert,  # estimate of log(StdDev[Z])
            ESS=ESS,  # estimate of Kish's effective sample size
            H_mean=H_mean,  # estimate of E[int log(L) L dp/Z]
            total_num_samples=num_samples,  # int, the total number of samples collected.
            log_L_samples=log_L_samples,  # log(L) of each sample
            log_dp_mean=dp_mean.log_abs_val,
            log_posterior_density=log_posterior_density,
            # log(E[dZ]) of each sample, where dZ is how much it contributes to the total evidence.
            # log(StdDev[dZ]) of each sample, where dZ is how much it contributes to the total evidence.
            log_X_mean=X_mean.log_abs_val,  # log(E[U]) of each sample
            num_likelihood_evaluations_per_sample=num_likelihood_evaluations_per_sample,
            # how many likelihood evaluations were made per sample.
            num_live_points_per_sample=num_live_points_per_sample,
            # how many live points were taken for the samples.
            total_num_likelihood_evaluations=total_num_likelihood_evaluations,
            # how many likelihood evaluations were made in total,
            # sum of num_likelihood_evaluations_per_sample.
            total_num_slices=total_num_slices,
            log_efficiency=efficiency.log_abs_val,
            # total_num_samples / total_num_likelihood_evaluations
            termination_reason=termination_reason,  # termination condition as bit mask
            num_slices=sample_collection.reservoir.num_slices,
            samples=samples,
            U_samples=sample_collection.reservoir.point_U
        )


@deprecated(StandardStaticNestedSampler)
class ApproximateNestedSampler:
    """
    Performs nested sampling, using a chain of nested samplers, producing only weakly i.i.d. samples during the
    shrinkage process.
    """

    def __init__(self, model: BaseAbstractModel, num_live_points: Union[int, float], num_parallel_samplers: int,
                 max_samples: Union[int, float],
                 sampler_chain: Optional[List[AbstractSampler]] = None):

        if sampler_chain is None:
            sampler_chain = [
                UniDimSliceSampler(model=model, num_slices=model.U_ndims * 3, midpoint_shrink=True, perfect=True)
            ]

        self._nested_sampler = StandardStaticNestedSampler(
            model=model,
            num_live_points=num_live_points,
            max_samples=max_samples,
            sampler=sampler_chain[-1],
            init_efficiency_threshold=0.1,
            num_parallel_workers=num_parallel_samplers
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

    def __call__(self, key: PRNGKey, term_cond: TerminationCondition, *,
                 init_state: Optional[StaticStandardNestedSamplerState] = None) -> Tuple[
        IntArray, StaticStandardNestedSamplerState]:
        """
        Performs approximate nested sampling.

        Args:
            key: PRNGKey
            term_cond: termination conditions
            init_state: optional initial state, resume from here if provided

        Returns:
            termination reason, state
        """
        if term_cond.max_samples is None:  # add a convenience termination condition for user
            term_cond = term_cond._replace(
                max_samples=jnp.asarray(self._nested_sampler.max_samples)
            )
        else:  # truncate for sanity
            term_cond = term_cond._replace(
                max_samples=jnp.minimum(self._nested_sampler.max_samples, term_cond.max_samples)
            )
        return self._nested_sampler._run(key=key, term_cond=term_cond)


@deprecated(StandardStaticNestedSampler)
class ExactNestedSampler(ApproximateNestedSampler):
    pass
