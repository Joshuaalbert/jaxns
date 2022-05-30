import logging
from typing import Tuple, Union, Callable

from jax import numpy as jnp, random, tree_map
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
from jaxns.internals.types import Reservoir, GlobalOptimiserState, GlobalOptimiserResults, float_type, int_type

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
            samples_per_step = prior_chain.U_ndims * 10
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
            sampler_kwargs['num_slices'] = sampler_kwargs.get('num_slices', prior_chain.U_ndims * 10)
            if sampler_kwargs['num_slices'] < 1:
                raise ValueError(f"num_slices {sampler_kwargs['num_slices']} should be >= 1.")
            sampler_kwargs['midpoint_shrink'] = bool(sampler_kwargs.get('midpoint_shrink', False))
            sampler_kwargs['gradient_boost'] = bool(sampler_kwargs.get('gradient_boost', True))
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
        return {name: d[name] for name, prior in self.prior_chain._prior_chain.items() if prior.tracked}

    def initial_state(self, key) -> GlobalOptimiserState:
        """
        Initialises the state of samplers.
        """

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
        key, init_key_reservoir = random.split(key, 2)

        (), reservoir = scan(single_sample, (), random.split(init_key_reservoir, self.samples_per_step))
        reservoir = sort_reservoir(reservoir)

        state = GlobalOptimiserState(
            key=key,
            done=jnp.asarray(False, jnp.bool_),
            num_steps=jnp.asarray(1, int_type),
            reservoir=reservoir,
            num_samples=jnp.asarray(self.samples_per_step, int_type),
            termination_reason=jnp.asarray(0, int_type),
            patience_steps=jnp.asarray(0, int_type),
            num_likelihood_evaluations=jnp.sum(reservoir.num_likelihood_evaluations)
        )

        return state

    def _maximise_likelihood_loop(self,
                                  init_state: GlobalOptimiserState,
                                  num_slices: jnp.ndarray,
                                  num_parallel_samplers: int = 1,
                                  *,
                                  termination_patience=None,
                                  termination_frac_likelihood_improvement=None,
                                  termination_likelihood_contour=None,
                                  termination_max_num_steps=None,
                                  termination_max_num_likelihood_evaluations=None
                                  ) -> GlobalOptimiserState:
        assert num_parallel_samplers == 1, "Optimisation is currently a sequential process."

        get_sample = build_get_sample(prior_chain=self.prior_chain,
                                     loglikelihood_from_U=self.loglikelihood_from_U,
                                     midpoint_shrink=self.sampler_kwargs.get(
                                         'midpoint_shrink'),
                                     gradient_boost=self.sampler_kwargs.get(
                                         'gradient_boost'),
                                     destructive_shrink=self.sampler_kwargs.get(
                                         'destructive_shrink')
                                     )

        def body(state: GlobalOptimiserState) -> GlobalOptimiserState:
            # Note: state enters with consistent definition, and exits with consistent definition.
            key, sample_key, seed_key, alpha_key = random.split(state.key, 4)
            state = state._replace(key=key)

            idx_min = jnp.argmin(state.reservoir.log_L_samples)
            log_L_constraint = state.reservoir.log_L_samples[idx_min]

            log_seed_goal = jnp.where(state.reservoir.log_L_samples > log_L_constraint,
                                      jnp.asarray(0., float_type), -jnp.inf)

            seed_idx = sample_goal_distribution(seed_key, log_seed_goal, 1, replace=True)[0]


            point_U_seed = state.reservoir.points_U[seed_idx]
            log_L_seed = state.reservoir.log_L_samples[seed_idx]

            # We replace every point in the sample collection with a new point sampled from that point.
            # expects: key, point_U0, log_L0, log_L_constraint, num_slices
            new_reservoir_point = get_sample(sample_key,
                                     point_U_seed,
                                     log_L_seed,
                                     log_L_constraint,
                                     num_slices)
            new_reservoir_point = new_reservoir_point._replace(points_X=self._filter_prior_chain(new_reservoir_point.points_X))
            new_reservoir = tree_map(lambda old, new: replace_index(old, new, idx_min),
                     state.reservoir,
                     new_reservoir_point)

            # terminate if all plateau, or if likelihood chagne from min/max is not big enough
            new_log_L_min = jnp.min(new_reservoir.log_L_samples)
            new_log_L_max = jnp.max(new_reservoir.log_L_samples)

            if termination_frac_likelihood_improvement is not None:
                # L_max/L_min - 1 <= delta
                not_enough_improvement = new_log_L_max - new_log_L_min <= jnp.log1p(
                    termination_frac_likelihood_improvement)
            else:
                not_enough_improvement = jnp.asarray(False)
            patience_steps = jnp.where(not_enough_improvement,
                                       state.patience_steps + jnp.ones_like(state.patience_steps),
                                       jnp.zeros_like(state.patience_steps))

            num_likelihood_evaluations = state.num_likelihood_evaluations + new_reservoir_point.num_likelihood_evaluations
            num_steps = state.num_steps + jnp.ones_like(state.num_steps)
            num_samples = state.num_samples + jnp.ones_like(state.num_samples)

            done, termination_reason = termination_condition(
                new_log_L_min=new_log_L_min,
                new_log_L_max=new_log_L_max,
                patience_steps=patience_steps,
                num_likelihood_evaluations=num_likelihood_evaluations,
                num_steps=num_steps,
                termination_patience=termination_patience,
                termination_frac_likelihood_improvement=termination_frac_likelihood_improvement,
                termination_likelihood_contour=termination_likelihood_contour,
                termination_max_num_steps=termination_max_num_steps,
                termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations)
            state = state._replace(done=done,
                                   termination_reason=termination_reason,
                                   reservoir=new_reservoir,
                                   num_steps=num_steps,
                                   num_samples=num_samples,
                                   num_likelihood_evaluations=num_likelihood_evaluations,
                                   patience_steps=patience_steps)
            return state

        state = while_loop(lambda state: jnp.bitwise_not(state.done),
                           body,
                           init_state)

        return state

    def __call__(self, key,
                 termination_patience=3,
                 termination_frac_likelihood_improvement=1e-3,
                 termination_likelihood_contour=None,
                 termination_max_num_steps=None,
                 termination_max_num_likelihood_evaluations=None,
                 *,
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

        state = self._maximise_likelihood_loop(
            init_state=state,
            num_slices=self.sampler_kwargs.get('num_slices'),
            num_parallel_samplers=self.num_parallel_samplers,
            termination_patience=termination_patience,
            termination_frac_likelihood_improvement=termination_frac_likelihood_improvement,
            termination_likelihood_contour=termination_likelihood_contour,
            termination_max_num_steps=termination_max_num_steps,
            termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations)
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
