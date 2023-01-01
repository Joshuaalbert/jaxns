import logging
from functools import partial
from typing import Optional, Tuple

import tensorflow_probability.substrates.jax as tfp
from etils.array_types import PRNGKey, IntArray
from jax import random, numpy as jnp, core, tree_map, vmap, jit

from jaxns.slice_sampler import UniDimSliceSampler
from jaxns.utils import summary, save_results, load_results
from jaxns.adaptive_refinement import AdaptiveRefinement
from jaxns.initial_state import init_sample_collection, get_uniform_init_live_points
from jaxns.internals.log_semiring import LogSpace, normalise_log_space
from jaxns.internals.stats import linear_to_log_stats, effective_sample_size
from jaxns.model import Model
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.static_slice import StaticSlice
from jaxns.static_uniform import StaticUniform
from jaxns.statistics import analyse_sample_collection
from jaxns.types import TerminationCondition, NestedSamplerState, NestedSamplerResults, LivePoints
from jaxns.utils import collect_samples

tfpd = tfp.distributions

logger = logging.getLogger('jaxns')

__all__ = ['NestedSampler',
           'ApproximateNestedSampler',
           'ExactNestedSampler']


class NestedSampler:
    def __init__(self, model: Model, num_live_points: int, num_parallel_samplers: int, max_samples: int):
        remainder = num_live_points % num_parallel_samplers
        extra = (num_parallel_samplers - remainder) % num_parallel_samplers
        if extra > 0:
            logger.warning(
                f"Increasing max_samples ({num_live_points}) by {extra} to closest multiple of num_parallel_samplers.")
        self.num_live_points = int(num_live_points + extra)
        self.num_parallel_samplers = int(num_parallel_samplers)
        self.model = model

        self.max_samples = int(max_samples)

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
        reservoir_extra = init_sample_collection(size=diff_size, model=self.model).reservoir
        reservoir = tree_map(lambda old, update: jnp.concatenate([old, update], axis=0),
                             reservoir,
                             reservoir_extra)
        return state._replace(sample_collection=state.sample_collection._replace(reservoir=reservoir))

    def summary(self, results: NestedSamplerResults) -> str:
        return summary(results)

    def plot_cornerplot(self, results: NestedSamplerResults):
        plot_cornerplot(results)

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

        num_samples = state.sample_collection.sample_idx

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

        num_likelihood_evaluations_per_slice = jnp.nanmean(
            jnp.where(sample_collection.reservoir.num_slices > 0,
                      sample_collection.reservoir.num_likelihood_evaluations / sample_collection.reservoir.num_slices,
                      jnp.nan)
        )

        log_posterior_density = sample_collection.reservoir.log_L + vmap(self.model.log_prob_prior)(
            sample_collection.reservoir.point_U)

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
            log_efficiency=efficiency.log_abs_val,
            # total_num_samples / total_num_likelihood_evaluations
            termination_reason=termination_reason,  # termination condition as bit mask
            num_likelihood_evaluations_per_slice=num_likelihood_evaluations_per_slice,
            num_slices=sample_collection.reservoir.num_slices,
            samples=samples)


class ApproximateNestedSampler(NestedSampler):
    def __init__(self, model: Model, num_live_points: int, num_parallel_samplers: int, max_samples: int):
        super(ApproximateNestedSampler, self).__init__(model=model, num_live_points=num_live_points,
                                                       num_parallel_samplers=num_parallel_samplers,
                                                       max_samples=max_samples)

        self.static_uniform = StaticUniform(model=self.model,
                                            num_live_points=self.num_live_points,
                                            efficiency_threshold=0.1)
        slice_sampler = UniDimSliceSampler(model=model,
                                           midpoint_shrink=True,
                                           multi_ellipse_bound=False)
        self.static_slice = StaticSlice(model=self.model,
                                        num_live_points=self.num_live_points,
                                        num_parallel_samplers=self.num_parallel_samplers,
                                        slice_sampler=slice_sampler)

    def initialise(self, key: PRNGKey) -> Tuple[NestedSamplerState, LivePoints]:
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
                                                   num_live_points=self.num_live_points,
                                                   model=self.model)
        return init_state, live_points

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
        # Perform uniform "perfect" sampling down to a given efficiency <==> Up to a given X(L)
        state, live_points = self.static_uniform(state=state, live_points=live_points)
        # Low accuracy sampling
        termination_reason, state, live_points = self.static_slice(state=state,
                                                                   live_points=live_points,
                                                                   num_slices=self.model.U_ndims,
                                                                   termination_cond=term_cond)
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
        state, live_points = self.initialise(key=key)
        return self.approximate_shrinkage(state=state, live_points=live_points, term_cond=term_cond)

    @partial(jit, static_argnums=0)
    def concat_run(self, state: NestedSamplerState, live_points: LivePoints,
                   term_cond: TerminationCondition) -> Tuple[IntArray, NestedSamplerState]:
        """
        Uses a given live points reservoir and performs approximate sampling until termination with a given initial
        state.

        Args:
            state: state to augment onto
            term_cond: termination condition

        Returns:
            termination reason, and state
        """
        return self.approximate_shrinkage(state=state, live_points=live_points, term_cond=term_cond)

    @jit
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
                 init_state: Optional[NestedSamplerState] = None,
                 live_points: Optional[LivePoints] = None) -> Tuple[IntArray, NestedSamplerState]:
        """
        Performs approximate nested sampling.

        Args:
            key: PRNGKey
            term_cond: termination conditions
            init_state: optional initial state
            live_points: optional initial live points

        Returns:
            termination reason, state
        """
        if init_state is None:
            # We create fresh live points and state
            termination_reason, state = self.fresh_run(key=key, term_cond=term_cond)
        else:
            if live_points is None:
                # We create a fresh set of live points from -inf contour and run on top of init_state
                init_state = self.resize_state(init_state, max_num_samples=self.max_samples)
                termination_reason, state = self.augment_run(state=init_state, term_cond=term_cond)
            else:
                # We use the input live points and run on top of init_state
                init_state = self.resize_state(init_state, max_num_samples=self.max_samples)
                termination_reason, state = self.concat_run(state=init_state, live_points=live_points,
                                                            term_cond=term_cond)
        return termination_reason, state


class ExactNestedSampler(NestedSampler):
    def __init__(self, model: Model, num_live_points: int, num_parallel_samplers: int, max_samples: int):
        super(ExactNestedSampler, self).__init__(model=model, num_live_points=num_live_points,
                                                 num_parallel_samplers=num_parallel_samplers, max_samples=max_samples)

        self.approximate_sampler = ApproximateNestedSampler(model=model, num_live_points=num_live_points,
                                                            num_parallel_samplers=num_parallel_samplers,
                                                            max_samples=max_samples)

        self.adaptive_refinement = AdaptiveRefinement(model=self.model,
                                                      uncert_improvement_patience=2,
                                                      num_slices=self.model.U_ndims,
                                                      num_parallel_samplers=self.num_parallel_samplers)

        self.max_samples = max_samples

    def improvement(self, state: NestedSamplerState) -> NestedSamplerState:
        """
        Run adaptive refinement until evidence converges.

        Args:
            state: state (may be approximate, i.e. non-iid samples.)

        Returns:
            state with samples considered iid sampled from the likelihood constraint
        """
        return self.adaptive_refinement(state)

    def __call__(self, key: PRNGKey, term_cond: TerminationCondition, *,
                 init_state: Optional[NestedSamplerState] = None,
                 live_points: Optional[LivePoints] = None) -> Tuple[IntArray, NestedSamplerState]:
        """
        Performs approximate nested sampling followed by adaptive refinement.

        Args:
            key: PRNGKey
            term_cond: termination condition
            init_state: optional initial state
            live_points: optional initial live points

        Returns:
            termination reason, and exact state
        """
        if isinstance(key, core.Tracer):
            raise RuntimeError("Tracer detected, but expected imperative context.")

        termination_reason, state = self.approximate_sampler(key=key, term_cond=term_cond, init_state=init_state,
                                                             live_points=live_points)
        state = self.improvement(state=state)

        return termination_reason, state
