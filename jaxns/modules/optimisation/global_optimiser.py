import logging
from typing import Tuple, Union, Callable, Any

from jax import numpy as jnp, random, tree_map, pmap
from jax.lax import scan
from jax.lax import while_loop

from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.maps import replace_index
from jaxns.internals.maps import prepare_func_args
from jaxns.nested_sampler.nested_sampling import build_get_sample, sample_goal_distribution
from jaxns.modules.optimisation.global_optimisation import sort_reservoir
from jaxns.modules.optimisation.utils import summary
from jaxns.modules.optimisation.termination import termination_condition
from jaxns.prior_transforms import PriorChain
from jaxns.internals.types import Reservoir, float_type, int_type
from jaxns.modules.optimisation.types import GlobalOptimiserState, GlobalOptimiserResults

logger = logging.getLogger(__name__)


class GlobalOptimiser(object):
    """
    Applies the same technology of nested sampling to perform global optimisation.
    """
    _available_samplers = ['slice']

    def __init__(self,
                 loglikelihood: Callable,
                 prior_chain: PriorChain,
                 sampler_name='slice',
                 num_parallel_samplers: int = 1,
                 samples_per_step: int = None,
                 sampler_kwargs=None):
        """
        The global optimiser class.
        """
        if sampler_name not in self._available_samplers:
            raise ValueError("sampler_name {} should be one of {}.".format(sampler_name, self._available_samplers))
        self.sampler_name = sampler_name
        if samples_per_step is None:
            samples_per_step = prior_chain.U_ndims * 20
        samples_per_step = int(samples_per_step)
        if samples_per_step < 1:
            raise ValueError(f"samples_per_step {samples_per_step} should be >= 1.")
        self.samples_per_step = samples_per_step
        num_parallel_samplers = int(num_parallel_samplers)
        if num_parallel_samplers < 1:
            raise ValueError(f"num_parallel_samplers {num_parallel_samplers} should be >= 1.")
        self.num_parallel_samplers = num_parallel_samplers
        if sampler_kwargs is None:
            sampler_kwargs = dict()
        if sampler_name == 'multi_ellipsoid':
            logger.warning(f"Sampler multi_ellipsoid is being deprecated from JAXNS.")
            sampler_kwargs['depth'] = int(sampler_kwargs.get('depth', 5))
            if sampler_kwargs['depth'] < 1:
                raise ValueError(f"depth {sampler_kwargs['depth']} should be >= 1.")
        elif sampler_name == 'slice':
            sampler_kwargs['num_slices'] = sampler_kwargs.get('num_slices', prior_chain.U_ndims * 1)
            if sampler_kwargs['num_slices'] < 1:
                raise ValueError(f"num_slices {sampler_kwargs['num_slices']} should be >= 1.")
            sampler_kwargs['midpoint_shrink'] = bool(sampler_kwargs.get('midpoint_shrink', False))
            sampler_kwargs['gradient_boost'] = bool(sampler_kwargs.get('gradient_boost', False))
            sampler_kwargs['destructive_shrink'] = bool(sampler_kwargs.get('destructive_shrink', False))
            assert not (sampler_kwargs['destructive_shrink'] and sampler_kwargs['midpoint_shrink']), \
                "Only midpoint_shrink or destructive_shrink should be used."
            sampler_kwargs['num_parallel_samplers'] = int(sampler_kwargs.get('num_parallel_samplers', 1))
            if sampler_kwargs['num_parallel_samplers'] < 1:
                raise ValueError(f"num_parallel_samplers {sampler_kwargs['num_parallel_samplers']} should be >= 1.")
        else:
            raise ValueError(f"sampler_name {sampler_name} is invalid.")
        self.sampler_kwargs = sampler_kwargs

        def corrected_likelihood(**x):
            """
            Adds the log-homogeneous measure to the log-likelihood to account for the transform from a PriorBase.

            Args:
                **x: dict of priors in U domain.

            Returns:
                log-likelihood plus the log-homogeneous measure, and -inf if it is a nan.
            """
            log_L = prepare_func_args(loglikelihood)(**x)
            log_homogeneous_measure = prior_chain.log_homogeneous_measure(**x)
            if log_homogeneous_measure is not None:
                log_L += log_homogeneous_measure
            log_L = jnp.asarray(log_L, float_type)
            if log_L.shape != ():
                raise ValueError("Shape of likelihood should be scalar, got {}".format(log_L.shape))
            return jnp.asarray(jnp.where(jnp.isnan(log_L), -jnp.inf, log_L), dtype=float_type)

        self.loglikelihood = corrected_likelihood
        self.prior_chain = prior_chain
        self.prior_chain.build()

        def loglikelihood_from_U(U_flat):
            """
            Computes the log-likelihood from flattened U-domain representation.

            Args:
                U_flat: vector of U-domain prior variables, to be transformed.

            Returns:
                log-likelihood (with log-homogeneous meaure added)
            """
            return corrected_likelihood(**prior_chain(U_flat))

        self.loglikelihood_from_U = loglikelihood_from_U

    def summary(self, results: GlobalOptimiserResults) -> str:
        return summary(results)

    @property
    def dtype(self):
        return float_type

    def _filter_prior_chain(self, d):
        """
        Filters a dict's keys to only those where prior variable of same name is tracked.
        Used for removing untracked priors from a dict.

        Args:
            d: dict

        Returns:
            dict with only keys that correspond to names being tracked.
        """
        return self.prior_chain.filter_sample(d)

    def initial_state(self, key) -> GlobalOptimiserState:
        """
        Initialises the state of samplers.
        """
        key, init_key_reservoir = random.split(key, 2)

        state = GlobalOptimiserState(
            key=key,
            done=jnp.asarray(False, jnp.bool_),
            num_steps=jnp.asarray(0, int_type),
            reservoir=None,
            num_samples=jnp.asarray(self.samples_per_step, int_type),
            termination_reason=jnp.asarray(0, int_type),
            patience_steps=jnp.asarray(0, int_type),
            num_likelihood_evaluations=jnp.asarray(0, int_type)
        )

        return state

    def _init_reservoir(self, key, size: int) -> Reservoir:
        # Some of the points might have log(L)=-inf, so we need to filter those out. Otherwise we could do:
        # N = self.num_live_points + self.reservoir_size
        # samples = vmap(lambda key: random.permutation(key, N) + 0.5)(random.split(key2, D)).T
        # samples /= N

        def single_sample(unused_state, key):
            """
            Produces a single sample from the joint-prior and computes the likelihood.

            Args:
                key: PRNG key

            Returns:
                U, U, log_L_samples
            """

            def body(state):
                (_, key, _, _, _, num_likelihood_evals) = state
                key, sample_key, break_plateau_key = random.split(key, 3)
                U = self.prior_chain.sample_U_flat(sample_key)
                X = self.prior_chain(U)
                log_L = self.loglikelihood(**X)
                done = ~jnp.isinf(log_L)
                num_likelihood_evals += jnp.asarray(1, int_type)
                return (done, key, U, self._filter_prior_chain(X), log_L, num_likelihood_evals)

            (_, _, U, X, log_L, num_likelihood_evals) = while_loop(lambda s: jnp.bitwise_not(s[0]),
                                                                   body,
                                                                   (jnp.asarray(False), key,
                                                                    self.prior_chain.U_flat_placeholder,
                                                                    self._filter_prior_chain(
                                                                        self.prior_chain.sample_placeholder),
                                                                    jnp.zeros((), float_type),
                                                                    jnp.asarray(0, int_type)))
            log_L_constraint = -jnp.inf
            sample = Reservoir(points_U=U,
                               points_X=X,
                               log_L_constraint=log_L_constraint,
                               log_L_samples=log_L,
                               num_likelihood_evaluations=num_likelihood_evals,
                               num_slices=jnp.inf)
            return (), sample

        # generate initial reservoir of points, filtering out those -inf (forbidden zones)
        (), reservoir = scan(single_sample, (), random.split(key, size))

        return reservoir

    def _new_maximise_likelihood_loop(self,
                                      init_state: GlobalOptimiserState,
                                      num_slices: jnp.ndarray,
                                      static_num_live_points: int,
                                      num_parallel_samplers: int = 1,
                                      termination_max_num_steps=None,
                                      termination_likelihood_contour=None,
                                      termination_patience=None,
                                      termination_max_num_likelihood_evaluations=None,
                                      termination_frac_likelihood_improvement=None
                                      ) -> GlobalOptimiserState:

        get_sample = build_get_sample(prior_chain=self.prior_chain,
                                      loglikelihood_from_U=self.loglikelihood_from_U,
                                      midpoint_shrink=self.sampler_kwargs.get(
                                          'midpoint_shrink'),
                                      destructive_shrink=self.sampler_kwargs.get(
                                          'destructive_shrink'),
                                      gradient_boost=self.sampler_kwargs.get(
                                          'gradient_boost'))

        extra = static_num_live_points % num_parallel_samplers
        if extra > 0:
            logger.info(
                f"Extending num_live_points by {extra} to evenly divide num_live_points {static_num_live_points} by of parallel samplers {num_parallel_samplers}.")
            static_num_live_points += (num_parallel_samplers - extra)

        key, init_reservoir_key = random.split(init_state.key, 2)

        if init_state.reservoir is None:
            # initial uniformly sampled reservoir
            init_reservoir = self._init_reservoir(init_reservoir_key, static_num_live_points)
            init_state = init_state._replace(key=key, reservoir=init_reservoir)
        else:
            init_reservoir = init_state.reservoir

        def single_thread_sampler(key: jnp.ndarray, init_reservoir: Reservoir) -> Tuple[Reservoir, Reservoir]:
            """
            Performs a single thread of nested sampling with a reduced number of live_points.
            Args:
                init_reservoir:

            Returns:

            """

            def body(body_state: Tuple[jnp.ndarray, Reservoir], X: Any) -> Tuple[
                Tuple[jnp.ndarray, Reservoir], Reservoir]:
                (key, reservoir) = body_state

                idx_min = jnp.argmin(reservoir.log_L_samples)
                dead_point = tree_map(lambda x: x[idx_min], reservoir)
                log_L_dead = dead_point.log_L_samples

                # contour becomes log_L_dead if log_L_dead is not supremum of live-points, else we choose the original constraint of dead point
                # Note: we are at liberty to choose any log_L level as a contour so long as we can sample within it uniformly.
                on_supremum = jnp.equal(log_L_dead, jnp.max(reservoir.log_L_samples))
                log_L_contour = jnp.where(on_supremum, dead_point.log_L_constraint, log_L_dead)

                # replace dead point with a new sample about contour

                key, seed_key, sample_key = random.split(key, 3)

                log_seed_goal = jnp.where(reservoir.log_L_samples > log_L_contour,
                                          jnp.asarray(0., float_type), -jnp.inf)

                seed_idx = sample_goal_distribution(seed_key, log_seed_goal, 1, replace=True)[0]

                point_U_seed = reservoir.points_U[seed_idx]
                log_L_seed = reservoir.log_L_samples[seed_idx]

                new_reservoir_point = get_sample(sample_key,
                                                 point_U_seed,
                                                 log_L_seed,
                                                 log_L_contour,
                                                 num_slices)

                new_reservoir_point = new_reservoir_point._replace(
                    points_X=self._filter_prior_chain(new_reservoir_point.points_X))

                reservoir = tree_map(lambda old, update: replace_index(old, update, idx_min),
                                     reservoir, new_reservoir_point)

                return (key, reservoir), dead_point

            (_, live_reservoir), dead_reservoir = scan(body, (key, init_reservoir), init_reservoir.log_L_samples)
            return (live_reservoir, dead_reservoir)

        parallel_single_thread_sampler = pmap(single_thread_sampler)

        def body(body_state: Tuple[GlobalOptimiserState, Reservoir]) -> Tuple[GlobalOptimiserState, Reservoir]:
            (state, live_reservoir) = body_state

            key, key_sample = random.split(state.key, 2)
            state = state._replace(key=key)

            if self.num_parallel_samplers > 1:
                # split up reservoir into number of samplers
                def _add_chunk_dim(a):
                    shape = list(a.shape)
                    shape = [num_parallel_samplers, shape[0] // num_parallel_samplers] + shape[1:]
                    return jnp.reshape(a, shape)

                live_reservoir = tree_map(_add_chunk_dim, live_reservoir)
                keys = random.split(key_sample, num_parallel_samplers)
                # parallel sampling
                live_reservoir, dead_reservoir = parallel_single_thread_sampler(keys, live_reservoir)

                # concatenate reservoir samples
                def _remove_chunk_dim(a):
                    shape = list(a.shape)
                    shape = [shape[0] * shape[1]] + shape[2:]
                    return jnp.reshape(a, shape)

                live_reservoir = tree_map(_remove_chunk_dim, live_reservoir)
                dead_reservoir = tree_map(_remove_chunk_dim, dead_reservoir)
            else:
                live_reservoir, dead_reservoir = single_thread_sampler(key_sample, live_reservoir)

            # update the state with sampled points from last round

            #     key: jnp.ndarray
            #     done: jnp.ndarray
            #     reservoir: Reservoir  # Arrays to hold samples taken from the reservoir.
            #     num_steps: jnp.ndarray  # the step index of the algorithm, where one step is a single consumption step.
            #     num_samples: jnp.ndarray  # how many samples have been drawn
            #     num_likelihood_evaluations: jnp.ndarray  # how many times the likelihood has been evaluated
            #     patience_steps: jnp.ndarray  # how many steps since goal incremental improvement
            #     termination_reason: jnp.ndarray  # this will be an int reflecting the reason for termination
            new_state = state._replace(reservoir=dead_reservoir,
                                       num_steps=state.num_steps + jnp.ones_like(state.num_steps),
                                       num_samples=state.num_samples + jnp.asarray(dead_reservoir.log_L_samples.size,
                                                                                   state.num_samples.dtype),
                                       num_likelihood_evaluations=state.num_likelihood_evaluations + jnp.sum(
                                           dead_reservoir.num_likelihood_evaluations))

            # terminate if all plateau, or if likelihood chagne from min/max is not big enough
            new_log_L_min = jnp.max(dead_reservoir.log_L_samples)
            new_log_L_max = jnp.max(live_reservoir.log_L_samples)

            if termination_frac_likelihood_improvement is not None:
                # L_max/L_min - 1 <= delta
                not_enough_improvement = new_log_L_max - new_log_L_min <= jnp.log1p(
                    termination_frac_likelihood_improvement)
            else:
                not_enough_improvement = jnp.asarray(False)
            patience_steps = jnp.where(not_enough_improvement,
                                       state.patience_steps + jnp.ones_like(state.patience_steps),
                                       jnp.zeros_like(state.patience_steps))

            done, termination_reason = termination_condition(
                new_log_L_min=new_log_L_min,
                new_log_L_max=new_log_L_max,
                patience_steps=patience_steps,
                num_likelihood_evaluations=state.num_likelihood_evaluations,
                num_steps=state.num_steps,
                termination_patience=termination_patience,
                termination_frac_likelihood_improvement=termination_frac_likelihood_improvement,
                termination_likelihood_contour=termination_likelihood_contour,
                termination_max_num_steps=termination_max_num_steps,
                termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations)

            new_state = new_state._replace(done=done,
                                           patience_steps=patience_steps,
                                           termination_reason=termination_reason)

            return new_state, live_reservoir

        (state, live_reservoir) = while_loop(lambda body_state: jnp.bitwise_not(body_state[0].done),
                                             body,
                                             (init_state, init_reservoir))

        # live reservoir has the high-value log-likelihood
        state = state._replace(reservoir=live_reservoir,
                               num_steps=state.num_steps + jnp.ones_like(state.num_steps),
                               num_samples=state.num_samples + jnp.asarray(live_reservoir.log_L_samples.size,
                                                                           state.num_samples.dtype),
                               num_likelihood_evaluations=state.num_likelihood_evaluations + jnp.sum(
                                   live_reservoir.num_likelihood_evaluations))

        return state

    def __call__(self, key,
                 termination_patience=1,
                 termination_frac_likelihood_improvement=1e-4,
                 termination_likelihood_contour=None,
                 termination_max_num_steps=None,
                 termination_max_num_likelihood_evaluations=None,
                 *,
                 num_live_points: int = None,
                 return_state: bool = False,
                 refine_state: GlobalOptimiserState = None
                 ) -> Union[GlobalOptimiserResults, Tuple[GlobalOptimiserResults, GlobalOptimiserState]]:
        """
        Performs global optimisation of the model, where the likelihood is maximised with a search that is guided by
        the prior.

        Args:
            key: PRNG key
            termination_patience: Terminate after this many termination conditions being true in a row
            termination_frac_likelihood_improvement: Terminate if likelihood log-difference between max/min is
                less that this.
            termination_likelihood_contour: Terminate if likelihood gets above this.
            termination_max_num_steps: Terminate if this many steps taken.
            termination_max_num_likelihood_evaluations: Terminate if this many likelihood evaluations made.
            return_state: If true, then return the state with result, which can be used for refinement.
            refine_state: GlobalOptimiserState, If given, then refines the provided state.

        Returns:
            if return_state is true, the a tuple (GlobalOptimiserResults, GlobalOptimiserState)
            otherwise GlobalOptimiserResults
        """

        if num_live_points is None:
            num_live_points = self.samples_per_step

        assert any([termination_patience is not None,
                    termination_frac_likelihood_improvement is not None,
                    termination_likelihood_contour is not None,
                    termination_max_num_steps is not None,
                    termination_max_num_likelihood_evaluations is not None]), "Need at least one termination criterion."
        if refine_state is not None:
            state = refine_state
            state = state._replace(done=jnp.asarray(False))
        else:
            state = self.initial_state(key)

        state = self._new_maximise_likelihood_loop(
            init_state=state,
            num_slices=self.sampler_kwargs.get('num_slices'),
            static_num_live_points=num_live_points,
            num_parallel_samplers=self.num_parallel_samplers,
            termination_max_num_steps=termination_max_num_steps,
            termination_likelihood_contour=termination_likelihood_contour,
            termination_patience=termination_patience,
            termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations,
            termination_frac_likelihood_improvement=termination_frac_likelihood_improvement
        )
        results = self._finalise_results(state)
        if return_state:
            return results, state
        return results

    def _finalise_results(self, state: GlobalOptimiserState) -> GlobalOptimiserResults:
        samples = state.reservoir.points_X
        total_num_samples = state.num_samples
        log_L_samples = state.reservoir.log_L_samples
        max_L_idx = jnp.argmax(log_L_samples)
        log_L_max = log_L_samples[max_L_idx]
        sample_L_max = tree_map(lambda x: x[max_L_idx], samples)
        total_num_likelihood_evaluations = state.num_likelihood_evaluations
        efficiency = LogSpace(jnp.log(total_num_samples) - jnp.log(total_num_likelihood_evaluations))
        termination_reason = state.termination_reason

        results = GlobalOptimiserResults(samples=samples,
                                         total_num_samples=total_num_samples,
                                         log_L_samples=log_L_samples,
                                         total_num_likelihood_evaluations=total_num_likelihood_evaluations,
                                         log_efficiency=efficiency.log_abs_val,
                                         termination_reason=termination_reason,
                                         log_L_max=log_L_max,
                                         sample_L_max=sample_L_max)
        return results
