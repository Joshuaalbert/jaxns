import logging
from typing import Tuple, Union, Callable

from jax import numpy as jnp, random, tree_map
from jax.lax import scan
from jax.lax import while_loop

from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.maps import chunked_pmap
from jaxns.internals.maps import prepare_func_args
from jaxns.nested_sampler.nested_sampling import build_get_sample
from jaxns.optimisation.global_optimisation import sort_reservoir
from jaxns.optimisation.utils import summary
from jaxns.optimisation.termination import termination_condition
from jaxns.prior_transforms import PriorChain
from jaxns.internals.types import Reservoir, GlobalOptimiserState, GlobalOptimiserResults

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
                 sampler_kwargs=None,
                 dtype=jnp.float_):
        """
        The global optimiser class.

        :param loglikelihood:
        :param prior_chain:
        :param sampler_name:
        :param num_parallel_samplers:
        :param samples_per_step:
        :param sampler_kwargs:
        :param dtype:
        """
        if sampler_name not in self._available_samplers:
            raise ValueError("sampler_name {} should be one of {}.".format(sampler_name, self._available_samplers))
        self.sampler_name = sampler_name
        if samples_per_step is None:
            samples_per_step = prior_chain.U_ndims * 50
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
            sampler_kwargs['num_slices'] = jnp.asarray(
                sampler_kwargs.get('num_slices', prior_chain.U_ndims * 10))
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
        self._dtype = dtype

        def corrected_likelihood(**x):
            """
            Adds the log-homogeneous measure to the log-likelihood to account for the transform from a PriorBase.

            Args:
                **x: dict of priors in X domain.

            Returns:
                log-likelihood plus the log-homogeneous measure, and -inf if it is a nan.
            """
            log_L = prepare_func_args(loglikelihood)(**x)
            log_homogeneous_measure = prior_chain.log_homogeneous_measure(**x)
            if log_homogeneous_measure is not None:
                log_L += log_homogeneous_measure
            if log_L.shape != ():
                raise ValueError("Shape of likelihood should be scalar, got {}".format(log_L.shape))
            return jnp.asarray(jnp.where(jnp.isnan(log_L), -jnp.inf, log_L), dtype=self.dtype)

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
        return self._dtype

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
                U, X, log_L_samples
            """

            def body(state):
                (_, key, _, _, _, num_likelihood_evals) = state
                key, sample_key, break_plateau_key = random.split(key, 3)
                U = self.prior_chain.sample_U_flat(sample_key)
                X = self.prior_chain(U)
                log_L = self.loglikelihood(**X)
                done = ~jnp.isinf(log_L)
                num_likelihood_evals += jnp.asarray(1, jnp.int_)
                return (done, key, U, self._filter_prior_chain(X), log_L, num_likelihood_evals)

            (_, _, U, X, log_L, num_likelihood_evals) = while_loop(lambda s: jnp.bitwise_not(s[0]),
                                                                   body,
                                                                   (jnp.asarray(False), key,
                                                                    self.prior_chain.U_flat_placeholder,
                                                                    self._filter_prior_chain(
                                                                        self.prior_chain.sample_placeholder),
                                                                    jnp.zeros((), self.dtype),
                                                                    jnp.asarray(0, jnp.int_)))
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
            num_steps=jnp.asarray(1, jnp.int_),
            reservoir=reservoir,
            num_samples=jnp.asarray(self.samples_per_step, jnp.int_),
            termination_reason=jnp.asarray(0, jnp.int_),
            patience_steps=jnp.asarray(0, jnp.int_),
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
        # construct the parallel version of get_sample (if num_parallel_samplers > 1)
        get_samples_parallel = chunked_pmap(build_get_sample(prior_chain=self.prior_chain,
                                                             loglikelihood_from_U=self.loglikelihood_from_U,
                                                             midpoint_shrink=self.sampler_kwargs.get(
                                                                 'midpoint_shrink'),
                                                             gradient_boost=self.sampler_kwargs.get(
                                                                 'gradient_boost'),
                                                             destructive_shrink=self.sampler_kwargs.get('destructive_shrink')
                                                             ),
                                            chunksize=num_parallel_samplers)

        def body(state: GlobalOptimiserState) -> GlobalOptimiserState:
            # Note: state enters with consistent definition, and exits with consistent definition.
            key, sample_key, seed_key, alpha_key = random.split(state.key, 4)
            state = state._replace(key=key)

            contours = jnp.concatenate([state.reservoir.log_L_constraint[0:1],
                                        state.reservoir.log_L_samples])
            alpha = 0.5  # random.uniform(alpha_key, state.reservoir.log_L_samples.shape)
            L_constraint_reinforce = LogSpace(state.reservoir.log_L_constraint[0]) + LogSpace(jnp.log(alpha)) * (
                    LogSpace(state.reservoir.log_L_samples[-1]) - LogSpace(state.reservoir.log_L_constraint[0]))
            log_L_constraint_reinforce = L_constraint_reinforce.log_abs_val
            log_L_constraints_reinforce = log_L_constraint_reinforce * jnp.ones_like(state.reservoir.log_L_samples)

            constraint_supremum_idx = jnp.clip(jnp.searchsorted(contours, log_L_constraint_reinforce, side='right'),
                                               1, state.reservoir.log_L_samples.size) - 1
            seed_idx = random.randint(seed_key,
                                      shape=state.reservoir.log_L_samples.shape,
                                      minval=constraint_supremum_idx,
                                      maxval=state.reservoir.log_L_samples.size)
            points_U_seed = state.reservoir.points_U[seed_idx]
            log_L_seed = state.reservoir.log_L_samples[seed_idx]

            # We replace every point in the sample collection with a new point sampled from that point.
            sample_keys = random.split(sample_key, seed_idx.size)
            # expects: key, point_U0, log_L0, log_L_constraint, num_slices
            new_reservoir = get_samples_parallel(sample_keys,
                                                 points_U_seed,
                                                 log_L_seed,
                                                 log_L_constraints_reinforce,
                                                 num_slices * jnp.ones_like(state.reservoir.num_slices))
            new_reservoir = sort_reservoir(new_reservoir)
            new_reservoir = new_reservoir._replace(points_X=self._filter_prior_chain(new_reservoir.points_X),
                                                   num_slices=new_reservoir.num_slices.astype(
                                                       state.reservoir.num_slices.dtype))

            prev_log_L_max = jnp.max(state.reservoir.log_L_samples)
            new_log_L_max = jnp.max(new_reservoir.log_L_samples)
            if termination_frac_likelihood_improvement is not None:
                not_enough_improvement = new_log_L_max - prev_log_L_max <= jnp.log1p(
                    termination_frac_likelihood_improvement)
            else:
                not_enough_improvement = jnp.asarray(False)
            patience_steps = jnp.where(not_enough_improvement,
                                       state.patience_steps + jnp.ones_like(state.patience_steps),
                                       jnp.zeros_like(state.patience_steps))

            num_likelihood_evaluations = state.num_likelihood_evaluations + jnp.sum(
                new_reservoir.num_likelihood_evaluations)
            num_steps = state.num_steps + jnp.ones_like(state.num_steps)
            num_samples = state.num_samples + jnp.asarray(self.samples_per_step, state.num_samples.dtype)

            done, termination_reason = termination_condition(
                prev_log_L_max=prev_log_L_max,
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
                 termination_patience=None,
                 termination_frac_likelihood_improvement=None,
                 termination_likelihood_contour=None,
                 termination_max_num_steps=None,
                 termination_max_num_likelihood_evaluations=None,
                 *,
                 return_state: bool = False,
                 refine_state: GlobalOptimiserState = None
                 ) -> Union[GlobalOptimiserResults, Tuple[GlobalOptimiserResults, GlobalOptimiserState]]:

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
