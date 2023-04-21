import logging
from functools import partial
from typing import Optional, Tuple, Union, List

import tensorflow_probability.substrates.jax as tfp
from etils.array_types import PRNGKey, IntArray
from jax import random, numpy as jnp, core, tree_map, vmap, jit

from jaxns.adaptive_refinement import AdaptiveRefinement
from jaxns.initial_state import init_sample_collection, get_uniform_init_live_points
from jaxns.internals.log_semiring import LogSpace, normalise_log_space
from jaxns.internals.stats import linear_to_log_stats, effective_sample_size
from jaxns.model import Model
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.static_nested_sampler import StaticNestedSampler
from jaxns.static_slice import UniDimSliceSampler
from jaxns.static_uniform import UniformSampler
from jaxns.statistics import analyse_sample_collection
from jaxns.types import TerminationCondition, NestedSamplerState, NestedSamplerResults, LivePoints
from jaxns.utils import collect_samples
from jaxns.utils import summary, save_results, load_results

tfpd = tfp.distributions

logger = logging.getLogger('jaxns')

__all__ = ['NestedSampler',
           'ApproximateNestedSampler',
           'ExactNestedSampler']


class NestedSampler:
    def __init__(self, model: Model, max_samples: int | float):
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
            sorted_collection=True
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
            samples=samples)


class ApproximateNestedSampler(NestedSampler):
    """
    Performs nested sampling, using a chain of nested samplers, producing only weakly i.i.d. samples during the
    shrinkage process.
    """

    def __init__(self, model: Model, num_live_points: Union[int, float], num_parallel_samplers: int,
                 max_samples: Union[int, float]):
        super().__init__(model=model, max_samples=max_samples)

        sampler_chain = [
            UniformSampler(model=model, efficiency_threshold=0.1),
            UniDimSliceSampler(model=model, num_slices=model.U_ndims, midpoint_shrink=True, perfect=True)
        ]

        self.nested_sampler_chain = list(map(
            lambda sampler: StaticNestedSampler(sampler=sampler,
                                                num_live_points=num_live_points,
                                                num_parallel_samplers=num_parallel_samplers),
            sampler_chain
        ))

    @property
    def num_live_points(self) -> int:
        return self.nested_sampler_chain[0].num_live_points

    def _run_chain(self, state: NestedSamplerState, live_points: LivePoints, term_cond: TerminationCondition) -> Tuple[
        IntArray, NestedSamplerState, LivePoints]:
        if len(self.nested_sampler_chain) == 0:
            raise ValueError(f"Expected at least one sampler in chain.")

        for nested_sampler in self.nested_sampler_chain:
            termination_reason, state, live_points = nested_sampler(
                state=state, live_points=live_points, termination_cond=term_cond)
        return termination_reason, state, live_points

    def approximate_shrinkage(self, state: NestedSamplerState, live_points: LivePoints,
                              term_cond: TerminationCondition) -> Tuple[IntArray, NestedSamplerState]:
        """
        Performs approximate shrinkage using a fast but inaccurate static nested sampler.

        Note: samples are not sure to be i.i.d. from within the likelihood constraint.

        Args:
            state: initial state
            live_points: initial live points
            term_cond: termination condition

        Returns:
            termination reason, and state
        """
        termination_reason, state, live_points = self._run_chain(state=state, live_points=live_points,
                                                                 term_cond=term_cond)
        state = collect_samples(state=state, new_reservoir=live_points.reservoir)
        return termination_reason, state

    @partial(jit, static_argnums=0)
    def fresh_run(self, key: PRNGKey, term_cond: TerminationCondition) -> Tuple[IntArray, NestedSamplerState]:
        """
        Creates a new initial state and live points from -inf contour and samples until termination.

        Args:
            key: PRNGKey
            term_cond: termination condition

        Returns:
            termination reason, and state
        """
        state, live_points = self.initialise(key=key, num_live_points=self.num_live_points)
        return self.approximate_shrinkage(state=state, live_points=live_points, term_cond=term_cond)

    @partial(jit, static_argnums=0)
    def concat_run(self, state: NestedSamplerState, live_points: LivePoints,
                   term_cond: TerminationCondition) -> Tuple[IntArray, NestedSamplerState, LivePoints]:
        """
        Uses a given live points reservoir and performs approximate sampling until termination with a given initial
        state.

        Args:
            state: state to augment onto
            live_points: live points to use
            term_cond: termination condition

        Returns:
            termination reason, state, and live points
        """
        termination_reason, state, live_points = self._run_chain(state=state, live_points=live_points,
                                                                 term_cond=term_cond)
        return termination_reason, state, live_points

    @partial(jit, static_argnums=0)
    def augment_run(self, state: NestedSamplerState, term_cond: TerminationCondition) -> Tuple[
        IntArray, NestedSamplerState]:
        """
        Creates a new live points reservoir from -inf contour and performs approximate sampling until termination with
        a given initial state.

        Args:
            state: state to augment onto
            term_cond: termination condition

        Returns:
            termination reason, and state
        """
        key, live_points_key = random.split(state.key)
        state = state._replace(key=key)
        live_points = get_uniform_init_live_points(live_points_key,
                                                   num_live_points=self.num_live_points,
                                                   model=self.model)
        return self.approximate_shrinkage(state=state, live_points=live_points, term_cond=term_cond)

    def __call__(self, key: PRNGKey, term_cond: TerminationCondition, *,
                 init_state: Optional[NestedSamplerState] = None) -> Tuple[IntArray, NestedSamplerState]:
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
            term_cond = term_cond._replace(max_samples=self.max_samples)
        else:  # truncate for sanity
            term_cond = term_cond._replace(max_samples=jnp.minimum(self.max_samples, term_cond.max_samples))
        if init_state is None:
            # We create fresh live points and state
            termination_reason, state = self.fresh_run(key=key, term_cond=term_cond)
        else:
            # We use the input live points and run on top of init_state
            init_state = self.resize_state(init_state, max_num_samples=self.max_samples)
            termination_reason, state = self.augment_run(state=init_state, term_cond=term_cond)
        return termination_reason, state


class ExactNestedSampler(NestedSampler):
    """
    A two stage nested sampler, where the first stage produces sample that are only weakly i.i.d., and the second stage
    adaptively refines these samples until they are strongly i.i.d. (according to a stopping criterion).
    """

    def __init__(self, model: Model, num_live_points: Union[int, float],
                 max_samples: Union[int, float],
                 num_parallel_samplers: int = 1,
                 patience: int = 1):
        """
        A two stage nested sampler, where the first stage produces sample that are only weakly i.i.d., and the
        second stage adaptively refines these samples until they are strongly i.i.d. (according to a stopping
        criterion).

        Args:
            model: model
            num_live_points: the number of static live points
            max_samples: maximum number of (non equally weighted) samples to allocate space for.
                Note: to control stopping criterion related to a maximum number of samples, use TerminationCondition.
            num_parallel_samplers: the number of parallel instances to run. This must be <= len(jax.devices())
            patience: how many rounds of converged refinement to wait before stopping.
                Values > 1 mean refinement will continue until `patience` consecutive rounds of convergence are
                observed. Similar to famous patience parameter of early stopping in deep learning community.
        """
        super(ExactNestedSampler, self).__init__(model=model, max_samples=max_samples)

        self.approximate_sampler = ApproximateNestedSampler(
            model=model,
            num_live_points=num_live_points,
            num_parallel_samplers=num_parallel_samplers,
            max_samples=max_samples
        )

        self.adaptive_refinement = AdaptiveRefinement(
            model=self.model,
            patience=patience
        )

    def __call__(self, key: PRNGKey, term_cond: TerminationCondition, *,
                 init_state: Optional[NestedSamplerState] = None) -> Tuple[IntArray, NestedSamplerState]:
        """
        Performs approximate nested sampling followed by adaptive refinement.

        Args:
            key: PRNGKey
            term_cond: termination condition
            init_state: optional initial state

        Returns:
            termination reason, and exact state
        """
        if isinstance(key, core.Tracer):
            raise RuntimeError("Tracer detected, but expected imperative context.")

        termination_reason, state = self.approximate_sampler(key=key, term_cond=term_cond, init_state=init_state)
        state = self.adaptive_refinement(state=state)

        return termination_reason, state
