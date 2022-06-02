import logging
from typing import Tuple, Union, Callable, Any

from jax import numpy as jnp, random, tree_map, pmap
from jax.lax import scan
from jax.lax import while_loop

from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.maps import chunked_pmap, replace_index, get_index, prepare_func_args
from jaxns.internals.stats import linear_to_log_stats, effective_sample_size
from jaxns.internals.types import NestedSamplerState, Reservoir, SampleCollection, NestedSamplerResults, ThreadStats, \
    float_type, int_type
from jaxns.nested_sampler.nested_sampling import build_get_sample, get_dynamic_goal, \
    collect_samples, compute_evidence, _update_thread_stats, _sample_collection_to_reservoir, sample_goal_distribution, \
    _update_evidence_calculation, _init_evidence_calculation
from jaxns.nested_sampler.termination import termination_condition
from jaxns.nested_sampler.utils import summary, save_results, load_results
from jaxns.nested_sampler.plotting import plot_diagnostics, plot_cornerplot
from jaxns.prior_transforms import PriorChain

logger = logging.getLogger(__name__)


class NestedSampler(object):
    """
    Runs the dynamic nested sampler algorithm.

    The algorithm consists of two loops:

    1. control _dynamic_loop -- determine a likelihood range and number of live points to use.
    2. thread _dynamic_loop -- perform sampling and shrinkage over the likelihood range, and collect the threads samples using
        the sample merging method proposed in [1]. We handle plateaus by assigning an equal weight to samples in the
        same contour as discussed in [2].

    References:
        [1] "Dynamic nested sampling: an improved algorithm for parameter estimation and evidence calculation"
            E. Higson et al. (2017), https://arxiv.org/pdf/1704.03459.pdf
        [2] "Nested sampling with plateaus"
            A. Fowlie et al. (2021), https://arxiv.org/abs/2010.13884
    """
    _available_samplers = ['slice']

    def __init__(self,
                 loglikelihood: Callable,
                 prior_chain: PriorChain,
                 sampler_name='slice',
                 num_parallel_samplers: int = 1,
                 samples_per_step: int = None,
                 sampler_kwargs=None,
                 max_samples=1e5,
                 dynamic: bool = False):
        """
        The nested sampler class. Does both static and dynamic nested sampling with flexible termination criteria, and
        advanced likelihood samplers.

        Args:
            loglikelihood: callable(**priors) evaluates the log-likelihood given kwarg-provided priors in U-domain.
            prior_chain: PriorChain, the prior chain describing prior formulation
            sampler_name: str, the likelihood sampler to use
            num_parallel_samplers: int, how many parallel likelihood evaluations to make using vmap.
            max_num_live_points: int, how big is the reservoir
            num_live_points: int, how many live-points in the static run.
            sampler_kwargs:
            max_samples:
            dynamic: bool, whether to apply the dynamic version.
            dtype: dtype of the likelihood, which is casted.
        """
        if sampler_name not in self._available_samplers:
            raise ValueError("sampler_name {} should be one of {}.".format(sampler_name, self._available_samplers))
        self.sampler_name = sampler_name
        if samples_per_step is None:
            samples_per_step = prior_chain.U_ndims * 100
        samples_per_step = int(samples_per_step)
        if samples_per_step < 1:
            raise ValueError(f"samples_per_step {samples_per_step} should be >= 1.")
        self.samples_per_step = samples_per_step
        max_samples = int(max_samples)
        if max_samples < 1:
            raise ValueError(f"max_samples {max_samples} should be >= 1.")
        # ensure samples_per_step evenly divides max_samples, rounding up
        remainder = max_samples % self.samples_per_step
        extra = self.samples_per_step - remainder
        if extra > 0:
            logger.warning(f"Increasing max_samples ({max_samples}) by {extra} to evenly divide samples_per_step")
        self.max_samples = max_samples + extra
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
            sampler_kwargs['min_num_slices'] = sampler_kwargs.get('min_num_slices', prior_chain.U_ndims * 1)
            sampler_kwargs['max_num_slices'] = sampler_kwargs.get('max_num_slices', prior_chain.U_ndims * 25)
            if sampler_kwargs['min_num_slices'] < 1:
                raise ValueError(f"min_num_slices {sampler_kwargs['min_num_slices']} should be >= 1.")
            if sampler_kwargs['max_num_slices'] < 1:
                raise ValueError(f"max_num_slices {sampler_kwargs['max_num_slices']} should be >= 1.")
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
        self.dynamic = bool(dynamic)

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

    def _init_sample_collection(self, size: int):
        sample_collection = SampleCollection(
            points_U=jnp.zeros((size, self.prior_chain.U_ndims),
                               self.prior_chain.U_flat_placeholder.dtype),
            points_X=dict((name, jnp.full((size,) + self.prior_chain.shapes[name],
                                          0., self.prior_chain.dtypes[name]))
                          for name in self._filter_prior_chain(self.prior_chain.shapes)),
            log_L_samples=jnp.full((size,), jnp.inf, float_type),
            log_L_constraint=jnp.full((size,), jnp.inf, float_type),
            num_likelihood_evaluations=jnp.full((size,), 0, int_type),
            log_dZ_mean=jnp.full((size,), -jnp.inf, float_type),
            log_X_mean=jnp.full((size,), -jnp.inf, float_type),
            num_live_points=jnp.full((size,), 0., float_type),
            num_slices=jnp.full((size,), 0., float_type),
        )
        return sample_collection

    def _initial_state(self, key):
        """
        Initialises the state of samplers.
        """

        # Allocate for collection of points.
        # We collect sums (along paths), and then at the end we use path_counts to compute statistics
        init_sample_collection = self._init_sample_collection(self.max_samples)

        # This contains the required information to compute Z and ZH
        init_evidence_calculation = _init_evidence_calculation()

        max_num_steps = self.max_samples // self.samples_per_step
        init_thread_stats = ThreadStats(evidence_uncert=jnp.zeros(max_num_steps),
                                        ess=jnp.zeros(max_num_steps),
                                        evidence=jnp.zeros(max_num_steps),
                                        log_L_max=jnp.zeros(max_num_steps),
                                        num_likelihood_evaluations=jnp.zeros(max_num_steps)
                                        )

        state = NestedSamplerState(
            key=key,
            done=jnp.asarray(False, jnp.bool_),
            step_idx=jnp.asarray(0, int_type),
            sample_collection=init_sample_collection,
            evidence_calculation=init_evidence_calculation,
            log_L_contour=jnp.asarray(-jnp.inf, float_type),
            sample_idx=jnp.asarray(0, int_type),
            termination_reason=jnp.asarray(0, int_type),
            thread_stats=init_thread_stats,
            num_likelihood_evaluations=jnp.asarray(0, int_type)
        )

        return state

    def _new_static_loop(self,
                         init_state: NestedSamplerState,
                         num_slices: jnp.ndarray,
                         static_num_live_points: int,
                         num_parallel_samplers: int = 1,
                         termination_ess=None,
                         termination_evidence_uncert=None,
                         termination_live_evidence_frac=None,
                         termination_max_num_steps=None,
                         termination_max_samples=None,
                         termination_max_num_likelihood_evaluations=None,
                         termination_likelihood_contour=None,
                         log_L_constraint=None
                         ) -> NestedSamplerState:

        get_sample = build_get_sample(prior_chain=self.prior_chain,
                                      loglikelihood_from_U=self.loglikelihood_from_U,
                                      midpoint_shrink=self.sampler_kwargs.get(
                                          'midpoint_shrink'),
                                      destructive_shrink=self.sampler_kwargs.get(
                                          'destructive_shrink'),
                                      gradient_boost=self.sampler_kwargs.get(
                                          'gradient_boost'))

        # construct the parallel version of get_sample (only used for live reservoir construction, other pmap used)
        get_samples_parallel = chunked_pmap(get_sample, chunksize=num_parallel_samplers)
        extra = static_num_live_points % num_parallel_samplers
        if extra > 0:
            logger.info(
                f"Extending num_live_points by {extra} to evenly divide num_live_points {static_num_live_points} by of parallel samplers {num_parallel_samplers}.")
            static_num_live_points += (num_parallel_samplers - extra)
        if log_L_constraint is not None:
            # create live reservoir by sampling from and above this index (which should be a supremum to the
            # contour to sample above so that randint gives good results).
            contours = jnp.concatenate([init_state.sample_collection.log_L_constraint[0:1],
                                        init_state.sample_collection.log_L_samples])
            constraint_supremum_idx = jnp.clip(jnp.searchsorted(contours, log_L_constraint, side='right'),
                                               1, init_state.sample_idx) - 1

            key, sample_key, seed_key = random.split(init_state.key, 3)
            init_state = init_state._replace(key=key)
            seed_idx = random.randint(seed_key,
                                      shape=(static_num_live_points,),
                                      minval=constraint_supremum_idx,
                                      maxval=init_state.sample_idx)

            points_U0_seed = init_state.sample_collection.points_U[seed_idx]
            log_L0_seed = init_state.sample_collection.log_L_samples[seed_idx]
            log_L_constraint = log_L_constraint * jnp.ones_like(log_L0_seed)

            # Sample from those seed locations, optionally in parallel.
            sample_keys = random.split(sample_key, seed_idx.size)
            # expects: key, point_U0, log_L0, log_L_constraint, num_slices
            init_reservoir = get_samples_parallel(sample_keys,
                                                  points_U0_seed,
                                                  log_L0_seed,
                                                  log_L_constraint,
                                                  num_slices * jnp.ones_like(log_L0_seed))
            init_reservoir = init_reservoir._replace(points_X=self._filter_prior_chain(init_reservoir.points_X))
        else:
            # Uniformly sample the whole prior
            key, init_reservoir_key = random.split(init_state.key, 2)
            init_state = init_state._replace(key=key)
            # initial uniformly sampled reservoir
            init_reservoir = self._init_reservoir(init_reservoir_key, static_num_live_points)

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

        def body(body_state: Tuple[NestedSamplerState, Reservoir]) -> Tuple[NestedSamplerState, Reservoir]:
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
            new_state = collect_samples(state, dead_reservoir)

            # determine if done

            remaining_evidence_upper_bound = LogSpace(live_reservoir.log_L_samples).max() * LogSpace(
                new_state.evidence_calculation.log_X_mean)
            evidence_upper_bound = remaining_evidence_upper_bound + LogSpace(new_state.evidence_calculation.log_Z_mean)
            likelihood_contour = jnp.max(dead_reservoir.log_L_samples)
            _, log_Z_var = linear_to_log_stats(
                log_f_mean=new_state.evidence_calculation.log_Z_mean,
                log_f2_mean=new_state.evidence_calculation.log_Z2_mean)
            ess = effective_sample_size(new_state.evidence_calculation.log_Z_mean,
                                        new_state.evidence_calculation.log_dZ2_mean)
            num_samples = new_state.sample_idx
            num_steps = new_state.step_idx
            num_likelihood_evaluations = new_state.num_likelihood_evaluations
            done, termination_reason = termination_condition(
                num_samples=num_samples,
                log_Z_var=log_Z_var,
                log_Z_remaining_upper=remaining_evidence_upper_bound.log_abs_val,
                log_Z_upper=evidence_upper_bound.log_abs_val,
                ess=ess,
                num_likelihood_evaluations=num_likelihood_evaluations,
                num_steps=num_steps,
                all_plateau=None,  # We replace a dead point from it's last constraint if it's a supremum point.
                likelihood_contour=likelihood_contour,
                termination_ess=termination_ess,
                termination_evidence_uncert=termination_evidence_uncert,
                termination_live_evidence_frac=termination_live_evidence_frac,
                termination_max_num_steps=termination_max_num_steps,
                termination_max_samples=termination_max_samples,
                termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations,
                termination_likelihood_contour=termination_likelihood_contour)
            new_state = new_state._replace(done=done, termination_reason=termination_reason)

            return new_state, live_reservoir

        (state, live_reservoir) = while_loop(lambda body_state: jnp.bitwise_not(body_state[0].done),
                                             body,
                                             (init_state, init_reservoir))

        state = collect_samples(state, live_reservoir)

        new_thread_stats = _update_thread_stats(state)
        state = state._replace(thread_stats=new_thread_stats,
                               step_idx=state.step_idx + jnp.ones_like(state.step_idx))

        return state

    def _dynamic_loop(self,
                      init_state: NestedSamplerState,
                      num_slices: jnp.ndarray,
                      G=None,
                      num_parallel_samplers: int = 1,
                      termination_ess=None,
                      termination_evidence_uncert=None,
                      termination_max_num_steps=None,
                      termination_max_samples=None,
                      termination_max_num_likelihood_evaluations=None
                      ) -> NestedSamplerState:
        """
        Performs nested sampling.

        If static then we sample from any contour that doesn't have enough live points, until the amount of evidence in
        next reservoir is less than termination fraction, calculated as the fractional increase in evidence was small enough.

        If dynamic, we are refining a state until evidence uncertainty is low enough or enough effective samples.

        Args:
            key:
            num_samples:
            goal_type:
            G:
            search_top_n:
            num_parallel_samplers:
            termination_ess:
            termination_likelihood_contour:
            termination_evidence_uncert:
            termination_max_num_steps:

        Returns:

        """

        def body(prev_state: NestedSamplerState):
            # Note: state enters with consistent definition, and exits with consistent definition.
            key, seed_key, sample_key = random.split(prev_state.key, 3)
            prev_state = prev_state._replace(key=key)
            # Get the likelihood range from dynamic goal.
            ## Get the goal distribution from sample population.
            log_L_constaint_start, log_L_constraint_end = get_dynamic_goal(
                key=seed_key,
                state=prev_state,
                num_samples=self.samples_per_step,
                G=G)

            new_state = self._new_static_loop(prev_state, num_slices, self.prior_chain.U_ndims * 5,
                                              num_parallel_samplers=num_parallel_samplers,
                                              termination_ess=termination_ess,
                                              termination_evidence_uncert=termination_evidence_uncert,
                                              termination_live_evidence_frac=None,
                                              termination_max_num_steps=termination_max_num_steps,
                                              termination_max_samples=termination_max_samples,
                                              termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations,
                                              termination_likelihood_contour=log_L_constraint_end,
                                              log_L_constraint=log_L_constaint_start
                                              )

            # determine if done

            _, log_Z_var = linear_to_log_stats(
                log_f_mean=new_state.evidence_calculation.log_Z_mean,
                log_f2_mean=new_state.evidence_calculation.log_Z2_mean)
            ess = effective_sample_size(new_state.evidence_calculation.log_Z_mean,
                                        new_state.evidence_calculation.log_dZ2_mean)
            num_samples = new_state.sample_idx
            num_steps = new_state.step_idx
            num_likelihood_evaluations = new_state.num_likelihood_evaluations
            done, termination_reason = termination_condition(
                num_samples=num_samples,
                log_Z_var=log_Z_var,
                log_Z_remaining_upper=None,
                log_Z_upper=None,
                ess=ess,
                num_likelihood_evaluations=num_likelihood_evaluations,
                num_steps=num_steps,
                all_plateau=None,  # We replace a dead point from it's last constraint if it's a supremum point.
                likelihood_contour=None,
                termination_ess=termination_ess,
                termination_evidence_uncert=termination_evidence_uncert,
                termination_live_evidence_frac=None,
                termination_max_num_steps=termination_max_num_steps,
                termination_max_samples=termination_max_samples,
                termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations,
                termination_likelihood_contour=None)
            new_state = new_state._replace(done=done, termination_reason=termination_reason)
            return new_state

        state = while_loop(lambda state: jnp.bitwise_not(state.done),
                           body,
                           init_state)

        return state

    def _adaptive_refinement(self,
                             init_state: NestedSamplerState,
                             min_num_slices: jnp.ndarray,
                             max_num_slices: jnp.ndarray,
                             adaptive_evidence_stopping_threshold,
                             adaptive_evidence_patience,
                             num_parallel_samplers: int = 1
                             ) -> NestedSamplerState:
        """
        Refine the likelihood constraint sampling to decrease auto-correlation.

        :param init_state:
        :param min_num_slices:
        :param max_num_slices:
        :param num_parallel_samplers:
        :return:
        """
        # construct the parallel version of get_sample (if num_parallel_samplers > 1)
        get_samples_parallel = chunked_pmap(build_get_sample(prior_chain=self.prior_chain,
                                                             loglikelihood_from_U=self.loglikelihood_from_U,
                                                             midpoint_shrink=True,
                                                             destructive_shrink=False,
                                                             gradient_boost=False
                                                             ),
                                            chunksize=num_parallel_samplers)

        def body(body_state: Tuple[NestedSamplerState, jnp.ndarray, jnp.ndarray]) -> Tuple[
            NestedSamplerState, jnp.ndarray, jnp.ndarray]:
            # Note: state enters with consistent definition, and exits with consistent definition.
            (prev_state, num_slices_goal, patience_steps) = body_state

            key, sample_key = random.split(prev_state.key, 2)

            diff_from_goal = jnp.maximum(num_slices_goal - prev_state.sample_collection.num_slices,
                                         jnp.zeros_like(num_slices_goal))
            diff_from_goal = jnp.where(jnp.arange(diff_from_goal.size) >= prev_state.sample_idx,
                                       jnp.zeros_like(diff_from_goal), diff_from_goal)
            # Sample from those seed locations, optionally in parallel.
            sample_keys = random.split(sample_key, diff_from_goal.size)

            def sample_body(sample_body_state: Tuple[jnp.ndarray, SampleCollection]) -> Tuple[
                jnp.ndarray, SampleCollection]:
                (sample_idx, sample_collection) = sample_body_state

                log_L_constraints_reinforce = get_index(sample_collection.log_L_constraint,
                                                        sample_idx, self.samples_per_step)

                _sample_keys = get_index(sample_keys,
                                         sample_idx, self.samples_per_step)
                points_U0_seed = get_index(sample_collection.points_U,
                                           sample_idx, self.samples_per_step)
                log_L0_seed = get_index(sample_collection.log_L_samples,
                                        sample_idx, self.samples_per_step)
                extra_num_slices = get_index(diff_from_goal,
                                             sample_idx, self.samples_per_step)

                # expects: key, point_U0, log_L0, log_L_constraint, num_slices
                new_reservoir = get_samples_parallel(_sample_keys, points_U0_seed, log_L0_seed,
                                                     log_L_constraints_reinforce, extra_num_slices)
                new_reservoir = new_reservoir._replace(points_X=self._filter_prior_chain(new_reservoir.points_X))

                # aggregate statistics
                slice_num_slices = get_index(sample_collection.num_slices,
                                             sample_idx, self.samples_per_step)
                slice_num_likelihood_evaluations = get_index(sample_collection.num_likelihood_evaluations,
                                                             sample_idx, self.samples_per_step)
                new_reservoir = new_reservoir._replace(
                    points_X=self._filter_prior_chain(new_reservoir.points_X),
                    num_slices=slice_num_slices + new_reservoir.num_slices,
                    num_likelihood_evaluations=slice_num_likelihood_evaluations + new_reservoir.num_likelihood_evaluations)

                old_reservoir = Reservoir(points_U=sample_collection.points_U,
                                          points_X=sample_collection.points_X,
                                          log_L_constraint=sample_collection.log_L_constraint,
                                          log_L_samples=sample_collection.log_L_samples,
                                          num_likelihood_evaluations=sample_collection.num_likelihood_evaluations,
                                          num_slices=sample_collection.num_slices)
                # Insert the new samples with a slice update
                reservoir = tree_map(lambda old, new: replace_index(old, new, sample_idx), old_reservoir, new_reservoir)
                # Update the sample collection
                sample_collection = sample_collection._replace(**reservoir._asdict())
                return (sample_idx + jnp.asarray(self.samples_per_step, int_type), sample_collection)

            _, sample_collection = while_loop(lambda sample_body_state: sample_body_state[0] < prev_state.sample_idx,
                                              sample_body,
                                              (jnp.asarray(0, int_type), prev_state.sample_collection))

            new_state = prev_state._replace(sample_collection=sample_collection,
                                            num_likelihood_evaluations=jnp.sum(
                                                sample_collection.num_likelihood_evaluations))
            new_state = compute_evidence(new_state)

            log_Z_mean, log_Z_var = linear_to_log_stats(
                log_f_mean=new_state.evidence_calculation.log_Z_mean,
                log_f2_mean=new_state.evidence_calculation.log_Z2_mean)

            prev_log_Z_mean, _ = linear_to_log_stats(
                log_f_mean=prev_state.evidence_calculation.log_Z_mean,
                log_f2_mean=prev_state.evidence_calculation.log_Z2_mean)
            # next slice goal is incremented by U_ndims, but could be less or more.
            num_slices_goal = num_slices_goal + jnp.asarray(self.prior_chain.U_ndims, num_slices_goal.dtype)
            # stop when evidence changes very little between iterations.

            if adaptive_evidence_stopping_threshold is None:
                # force closure
                _adaptive_evidence_stopping_threshold = 0.5 * jnp.sqrt(log_Z_var)
            else:
                _adaptive_evidence_stopping_threshold = adaptive_evidence_stopping_threshold

            small_enough_change = jnp.abs(log_Z_mean - prev_log_Z_mean) <= _adaptive_evidence_stopping_threshold
            too_many_slices = num_slices_goal > max_num_slices
            if adaptive_evidence_patience is not None:
                patience_steps = jnp.where(small_enough_change,
                                           patience_steps + jnp.asarray(1, int_type),
                                           jnp.asarray(0, int_type))
                done = (patience_steps > adaptive_evidence_patience) | too_many_slices
            else:
                done = small_enough_change | too_many_slices
            new_state = new_state._replace(key=key, done=done)
            return (new_state, num_slices_goal, patience_steps)

        (state, _, _) = while_loop(lambda body_state: jnp.bitwise_not(body_state[0].done),
                                   body,
                                   (
                                       init_state,
                                       min_num_slices + jnp.asarray(self.prior_chain.U_ndims, jnp.int_),
                                       jnp.asarray(0, int_type)
                                   ))

        return state

    def __call__(self, key,
                 termination_ess: Union[float, int, jnp.ndarray] = None,
                 termination_evidence_uncert: Union[float, jnp.ndarray] = None,
                 termination_live_evidence_frac: Union[float, jnp.ndarray] = 1e-4,
                 termination_max_num_steps: Union[float, int, jnp.ndarray] = None,
                 termination_max_samples: Union[float, int, jnp.ndarray] = None,
                 termination_max_num_likelihood_evaluations: Union[float, int, jnp.ndarray] = None,
                 adaptive_evidence_patience: Union[int, jnp.ndarray] = 0,
                 adaptive_evidence_stopping_threshold: Union[float, jnp.ndarray] = None,
                 G: Union[float, jnp.ndarray] = 0.,
                 *,
                 num_live_points: int = None,
                 return_state: bool = False,
                 refine_state: NestedSamplerState = None,
                 ) -> Union[NestedSamplerResults, Tuple[NestedSamplerResults, NestedSamplerState]]:
        """
        Applies static nested sampling, and optionally also dynamic improvement, with adaptive refinement.

        Args:
            key: PRNG key
            num_live_points: approximate number of live points to use in static case.
            termination_ess: terminate when this many effective samples taken.
            termination_evidence_uncert: terminate when evidence uncertainty falls below this point.
            termination_live_evidence_frac: terminate when reduction in evidence from step less than this.
                This applies only to the static case.
            termination_max_num_steps: terminate when this many steps taken.
            termination_max_samples: terminate when this many samples taken.
            termination_max_num_likelihood_evaluations: terinate when this many likelihood evalations made.
            adaptive_evidence_patience: If set then adaptive stopping condition should occur this many times in a row before stopping.
            G: dynamic goal parameter, interpolate between evidence accuracy goal (0.) and posterior accuracy goal (1.)
            return_state: bool, whether to return state with result
            refine_state: optional, if given then refine the provided state rather than initialising.

        Returns:
            if return_state returns NestedSamplerResults and NestedSamplerState, else just NestedSamplerResults


        """

        if num_live_points is None:
            num_live_points = self.samples_per_step

        assert isinstance(num_live_points, int), "num_live_points must be a static int"

        # if termination_evidence_uncert is not None:
        #     adaptive_evidence_stopping_threshold = termination_evidence_uncert / 3.

        # Establish the state that we need to carry through iterations, and to facilitate post-analysis.
        if refine_state is not None:
            state = refine_state
            # TODO: maybe other things to prepare a provided state
            state = state._replace(done=jnp.asarray(False))
            assert state.sample_collection.log_L_samples.size != self.max_samples, \
                "Resizing sample collection not yet implemented."
        else:
            state = self._initial_state(key)

        if termination_max_samples is None:
            termination_max_samples = state.sample_collection.log_L_samples.size
        termination_max_samples = jnp.minimum(termination_max_samples,
                                              state.sample_collection.log_L_samples.size)
        max_num_steps = state.sample_collection.log_L_samples.size // self.samples_per_step
        if termination_max_num_steps is None:
            termination_max_num_steps = max_num_steps
        termination_max_num_steps = jnp.minimum(termination_max_num_steps,
                                                max_num_steps)

        if refine_state is None:
            # initialise with static sampling
            assert termination_live_evidence_frac is not None, \
                "Need the static stopping condition"
            # TODO: maybe turn off the other termination criteria since they are really for dynamic
            state = self._new_static_loop(
                init_state=state,
                num_slices=self.sampler_kwargs.get('min_num_slices'),
                static_num_live_points=num_live_points,
                num_parallel_samplers=self.num_parallel_samplers,
                termination_ess=termination_ess,
                termination_evidence_uncert=termination_evidence_uncert,
                termination_live_evidence_frac=termination_live_evidence_frac,
                termination_max_num_steps=termination_max_num_steps,
                termination_max_samples=termination_max_samples,
                termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations,
                termination_likelihood_contour=None,
                log_L_constraint=None)
        if self.dynamic:
            if not any([termination_ess is not None,
                        termination_evidence_uncert is not None,
                        termination_max_num_steps is not None,
                        termination_max_samples is not None,
                        termination_max_num_likelihood_evaluations is not None]):
                # reduce uncertainty by half
                _, log_Z_var = linear_to_log_stats(
                    log_f_mean=state.evidence_calculation.log_Z_mean,
                    log_f2_mean=state.evidence_calculation.log_Z2_mean)
                termination_evidence_uncert = 0.5 * jnp.sqrt(log_Z_var)

            assert any([termination_ess is not None,
                        termination_evidence_uncert is not None,
                        termination_max_num_steps is not None,
                        termination_max_samples is not None,
                        termination_max_num_likelihood_evaluations is not None]), \
                'Dynamic run needs at least one termination criterion'

            state = state._replace(done=jnp.asarray(False))
            state = self._dynamic_loop(
                init_state=state,
                num_slices=self.sampler_kwargs.get('min_num_slices'),
                G=jnp.clip(G, 0., 1.),
                num_parallel_samplers=self.num_parallel_samplers,
                termination_ess=termination_ess,
                termination_evidence_uncert=termination_evidence_uncert,
                termination_max_num_steps=termination_max_num_steps,
                termination_max_samples=termination_max_samples,
                termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations)

        if adaptive_evidence_patience is not None:
            # adaptively decrease auto-correlation of samples until evidence converges
            state = state._replace(done=jnp.asarray(False))
            state = self._adaptive_refinement(
                init_state=state,
                min_num_slices=self.sampler_kwargs.get('min_num_slices'),
                max_num_slices=self.sampler_kwargs.get('max_num_slices'),
                adaptive_evidence_stopping_threshold=adaptive_evidence_stopping_threshold,
                adaptive_evidence_patience=adaptive_evidence_patience,
                num_parallel_samplers=self.num_parallel_samplers)

        # collect live-points, and to post-analysis
        results = self._finalise_results(state)
        if return_state:
            return results, state
        return results

    def _finalise_results(self, state: NestedSamplerState) -> NestedSamplerResults:
        """
        Produces the NestedSamplingResult.
        """

        log_Z_mean, log_Z_var = linear_to_log_stats(log_f_mean=state.evidence_calculation.log_Z_mean,
                                                    log_f2_mean=state.evidence_calculation.log_Z2_mean)
        log_Z_uncert = jnp.sqrt(log_Z_var)

        # Kish's ESS = [sum dZ]^2 / [sum dZ^2]
        ESS = effective_sample_size(state.evidence_calculation.log_Z_mean,
                                    state.evidence_calculation.log_dZ2_mean)
        samples = state.sample_collection.points_X
        total_num_samples = state.sample_idx
        log_L_samples = state.sample_collection.log_L_samples
        dp_mean = LogSpace(state.sample_collection.log_dZ_mean)
        dp_mean = dp_mean / dp_mean.sum()
        H_mean = LogSpace(jnp.where(jnp.isneginf(dp_mean.log_abs_val),
                                    -jnp.inf,
                                    dp_mean.log_abs_val + log_L_samples)).sum().value - log_Z_mean
        X_mean = LogSpace(state.sample_collection.log_X_mean)
        num_likelihood_evaluations_per_sample = state.sample_collection.num_likelihood_evaluations
        total_num_likelihood_evaluations = jnp.sum(num_likelihood_evaluations_per_sample)
        num_live_points_per_sample = state.sample_collection.num_live_points
        num_slices_per_sample = state.sample_collection.num_slices
        efficiency = LogSpace(jnp.log(total_num_samples) - jnp.log(total_num_likelihood_evaluations))
        termination_reason = state.termination_reason

        return NestedSamplerResults(
            log_Z_mean=log_Z_mean,  # estimate of log(E[Z])
            log_Z_uncert=log_Z_uncert,  # estimate of log(StdDev[Z])
            ESS=ESS,  # estimate of Kish's effective sample size
            H_mean=H_mean,  # estimate of E[int log(L) L dp/Z]
            total_num_samples=total_num_samples,  # int, the total number of samples collected.
            log_L_samples=log_L_samples,  # log(L) of each sample
            log_dp_mean=dp_mean.log_abs_val,
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
            thread_stats=state.thread_stats,
            num_slices_per_sample=num_slices_per_sample,
            samples=samples)

    def resize_state(self, state: NestedSamplerState, max_num_samples: int) -> NestedSamplerState:
        if max_num_samples <= state.sample_collection.log_L_samples.size:
            logger.warning(
                f"resize_state expects max_num_samples larger than the current size {state.sample_collection.log_L_samples.size}")
            return state
        diff_size = max_num_samples - state.sample_collection.log_L_samples.size
        sample_collection_extra = self._init_sample_collection(diff_size)
        sample_collection = tree_map(lambda old, update: jnp.concatenate([old, update], axis=0),
                                     state.sample_collection,
                                     sample_collection_extra)
        # todo: resize ThreadStats too
        return state._replace(sample_collection=sample_collection)

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
