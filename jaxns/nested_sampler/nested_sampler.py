import logging
from typing import Dict, Any, Optional, Tuple, Union, Callable

from jax import numpy as jnp, random, tree_map
from jax.lax import scan
from jax.lax import while_loop

from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.maps import chunked_pmap, replace_index
from jaxns.internals.maps import prepare_func_args
from jaxns.internals.stats import linear_to_log_stats
from jaxns.nested_sampler.live_points import infimum_constraint
from jaxns.nested_sampler.nested_sampling import build_get_sample, get_seed_goal, sample_goal_distribution, \
    collect_samples
from jaxns.nested_sampler.termination import termination_condition
from jaxns.prior_transforms import PriorChain
from jaxns.internals.types import NestedSamplerState, EvidenceCalculation, Reservoir
from jaxns.internals.types import SampleCollection, NestedSamplerResults, ThreadStats, TerminationStats

logger = logging.getLogger(__name__)


class NestedSampler(object):
    """
    Runs the dynamic nested sampler algorithm.

    The algorithm consists of two loops:

    1. control _loop -- determine a likelihood range and number of live points to use.
    2. thread _loop -- perform sampling and shrinkage over the likelihood range, and collect the threads samples using
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
                 collect_samples: bool = True,
                 max_samples=1e5,
                 dynamic: bool = False,
                 dtype=jnp.float_):
        """
        The nested sampler class. Does both static and dynamic nested sampling with flexible termination criteria, and
        advanced likelihood samplers.

        Args:
            loglikelihood: callable(**priors) evaluates the log-likelihood given kwarg-provided priors in X-domain.
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
            sampler_kwargs['depth'] = int(sampler_kwargs.get('depth', 5))
            if sampler_kwargs['depth'] < 1:
                raise ValueError(f"depth {sampler_kwargs['depth']} should be >= 1.")
        elif sampler_name == 'slice':
            sampler_kwargs['num_slices'] = int(sampler_kwargs.get('num_slices', prior_chain.U_ndims * 5))
            if sampler_kwargs['num_slices'] < 1:
                raise ValueError(f"num_slices {sampler_kwargs['num_slices']} should be >= 1.")
            sampler_kwargs['midpoint_shrink'] = bool(sampler_kwargs.get('midpoint_shrink', False))
            sampler_kwargs['gradient_boost'] = bool(sampler_kwargs.get('gradient_boost', False))
            sampler_kwargs['num_parallel_samplers'] = int(sampler_kwargs.get('num_parallel_samplers', 1))
        else:
            raise ValueError(f"sampler_name {sampler_name} is invalid.")
        self.collect_samples = bool(collect_samples)
        self.sampler_kwargs = sampler_kwargs
        self._dtype = dtype
        self.dynamic = bool(dynamic)

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
            if not isinstance(log_L, jnp.ndarray):
                log_L = jnp.asarray(log_L, dtype=self.dtype)
            if log_L.dtype.type != self.dtype:
                log_L = log_L.astype(self.dtype)
            if log_L.shape != ():
                raise ValueError("Shape of likelihood should be scalar, got {}".format(log_L.shape))
            return jnp.where(jnp.isnan(log_L), -jnp.inf, log_L)

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

    def initial_state(self, key):
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
            return (), (U, X, log_L, log_L_constraint, num_likelihood_evals, jnp.asarray(True))

        # generate initial reservoir of points, filtering out those -inf (forbidden zones)
        key, init_key_reservoir = random.split(key, 2)

        (), (reservoir_points_U, reservoir_points_X, log_L_reservoir, log_L_constraint, reservoir_num_likelihood_evals,
             reservoir_available) = scan(single_sample, (), random.split(init_key_reservoir, self.samples_per_step))

        reservoir = Reservoir(points_U=reservoir_points_U,
                              points_X=reservoir_points_X,
                              log_L_constraint=log_L_constraint,
                              log_L_samples=log_L_reservoir,
                              num_likelihood_evaluations=reservoir_num_likelihood_evals)

        # Allocate for collection of points.
        # We collect sums (along paths), and then at the end we use path_counts to compute statistics
        sample_collection = SampleCollection(
            points_U=jnp.zeros((self.max_samples, self.prior_chain.U_ndims),
                               self.prior_chain.U_flat_placeholder.dtype),
            points_X=dict((name, jnp.full((self.max_samples,) + self.prior_chain.shapes[name],
                                          0., self.prior_chain.dtypes[name]))
                          for name in self._filter_prior_chain(self.prior_chain.shapes)),
            # The X-valued samples
            log_L_samples=jnp.full((self.max_samples,), jnp.inf, self.dtype),
            log_L_constraint=jnp.full((self.max_samples,), jnp.inf, self.dtype),
            # The log-likelihood of sample, defaulting to inf to sort them after rest.
            num_likelihood_evaluations=jnp.full((self.max_samples,), 0, jnp.int_),
            # How many likelihood evaluations were required to obtain sample
            log_dZ_mean=jnp.full((self.max_samples,), -jnp.inf, self.dtype),
            # The log mean evidence difference of the sample (averaged across chains)
            log_X_mean=jnp.full((self.max_samples,), -jnp.inf, self.dtype),
            # The log mean enclosed prior volume of sample
            num_live_points=jnp.full((self.max_samples,), jnp.inf, jnp.float_)
            # How many live points were taken for the samples.
        )

        # This contains the required information to compute Z and ZH
        evidence_calculation = EvidenceCalculation(
            log_X_mean=jnp.asarray(0., self.dtype),
            log_X2_mean=jnp.asarray(0., self.dtype),
            log_Z_mean=jnp.asarray(-jnp.inf, self.dtype),
            log_ZX_mean=jnp.asarray(-jnp.inf, self.dtype),
            log_Z2_mean=jnp.asarray(-jnp.inf, self.dtype),
            log_dZ2_mean=jnp.asarray(-jnp.inf, self.dtype)
        )

        init_thread_stats = ThreadStats(evidence_uncert_diff=jnp.zeros(20),
                                        ess_diff=jnp.zeros(20),
                                        evidence_diff=jnp.zeros(20),
                                        log_L_max=jnp.zeros(20),
                                        num_likelihood_evaluations=jnp.zeros(20)
                                        )

        nested_sampler_state = NestedSamplerState(
            key=key,
            done=jnp.asarray(False, jnp.bool_),
            step_idx=jnp.asarray(0, jnp.int_),
            sample_collection=sample_collection,
            evidence_calculation=evidence_calculation,
            log_L_contour=jnp.asarray(-jnp.inf, self.dtype),
            sample_idx=jnp.asarray(0, jnp.int_),
            termination_reason=jnp.asarray(0, jnp.int_),
            thread_stats=init_thread_stats
        )
        nested_sampler_state = collect_samples(nested_sampler_state, reservoir)

        return nested_sampler_state

    def refine_state(self, state: NestedSamplerState,
                     delta_num_live_points: jnp.ndarray = None,
                     termination_ess=None,
                     termination_evidence_uncert=None,
                     termination_max_num_steps=None,
                     termination_likelihood_contour=None,
                     termination_likelihood_frac_increase=None,
                     dynamic_kwargs=None,
                     resize_max_samples: int = None,
                     maximise_likelihood: bool = False,
                     *, return_state: bool = False) \
            -> Union[NestedSamplerResults, Tuple[NestedSamplerResults, NestedSamplerState]]:
        """
        Incrementally improve the nested sampling result, according to a specific goal.

        Args:
            state: NestedSamplerState from a previous run.
            delta_num_live_points: a number of live points to increment by in each thread.
            termination_ess: terminate after reaching this many effective samples.
            termination_evidence_uncert: terminate after reaching this uncertainty in the evidence.
            dynamic_kwargs: Dict with keys:
                f = float, fraction of peak to select thread likelihood range.
                G = float, interpolates between pure evidence accuracy (0.) to pure posterior accuracy (1.)
            resize_max_samples: int, if given then grow the SampleCollection to this new total size.
            termination_max_num_steps: terminate when this many threads has run.
            return_state: bool, whether to return state with result

        Returns:
            if return_state returns NestedSamplerResults and NestedSamplerState, else just NestedSamplerResults
        """

        if resize_max_samples is not None:
            assert resize_max_samples > state.sample_collection.log_L_samples.size
            grow_by = resize_max_samples - state.sample_collection.log_L_samples.size

            def _extend(operand, fill_value):
                extra = jnp.full((grow_by,) + operand.shape[1:],
                                 fill_value,
                                 operand.dtype)
                return jnp.concatenate([operand, extra], axis=0)

            fill_values = SampleCollection(points_U=jnp.asarray(0.),
                                           points_X=jnp.asarray(0.),
                                           log_L_samples=-jnp.inf,
                                           log_L_constraint=-jnp.inf,
                                           num_likelihood_evaluations=jnp.asarray(0.),
                                           log_dZ_mean=-jnp.inf,
                                           log_X_mean=-jnp.inf,
                                           num_live_points=jnp.inf)
            resized_sample_collection = tree_map(_extend,
                                                 state.sample_collection,
                                                 fill_values
                                                 )
            state = state._replace(sample_collection=resized_sample_collection)

        state = self._loop(init_state=state,
                           num_samples=self.samples_per_step,
                           goal_type='static',
                           G=jnp.clip(dynamic_kwargs.get('G', 0.), 0., 1.),
                           num_parallel_samplers=self.num_parallel_samplers,
                           termination_ess=termination_ess,
                           termination_likelihood_contour=termination_likelihood_contour,
                           termination_evidence_uncert=termination_evidence_uncert,
                           termination_live_evidence_frac=None,
                           termination_max_num_steps=termination_max_num_steps)
        # collect live-points, and to post-analysis
        results = self._finalise_results(state)
        if return_state:
            return results, state

    def _loop(self,
              init_state: NestedSamplerState,
              num_samples: int,
              goal_type: str,
              G=None,
              search_top_n=None,
              static_num_live_points=None,
              num_parallel_samplers: int = 1,
              termination_ess=None,
              termination_likelihood_contour=None,
              termination_evidence_uncert=None,
              termination_live_evidence_frac=None,
              termination_max_num_steps=None
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
        # construct the paralle version of get_sample (if num_parallel_samplers > 1)
        get_samples_parallel = chunked_pmap(build_get_sample(prior_chain=self.prior_chain,
                                                             loglikelihood_from_U=self.loglikelihood_from_U,
                                                             num_slices=self.sampler_kwargs.get('num_slices'),
                                                             midpoint_shrink=self.sampler_kwargs.get(
                                                                 'midpoint_shrink'),
                                                             gradient_boost=self.sampler_kwargs.get(
                                                                 'gradient_boost')
                                                             ),
                                            chunksize=num_parallel_samplers)

        def body(state: NestedSamplerState):
            # Note: state enters with consistent definition, and exits with consistent definition.

            # Get the seed samples and constraints, over the shrinkage constraints to sample from
            log_goal_weights = get_seed_goal(state,
                                             goal_type=goal_type,
                                             G=G,
                                             search_top_n=search_top_n,
                                             static_num_live_points=static_num_live_points)
            key, seed_key = random.split(state.key, 2)
            state = state._replace(key=key)
            seed_indices = sample_goal_distribution(seed_key, log_goal_weights, num_samples,
                                                    replace=False)
            # Note: sample_collection already sorted so sort_idx not needed
            _, log_L_constraints_seed = infimum_constraint(log_L_constraints=state.sample_collection.log_L_constraint,
                                                           log_L_samples=state.sample_collection.log_L_samples,
                                                           sort_idx=None,
                                                           return_contours=True)
            log_L_constraints_seed = log_L_constraints_seed[seed_indices]
            points_U0_seed = state.sample_collection.points_U[seed_indices]
            log_L0_seed = state.sample_collection.log_L_samples[seed_indices]
            # Sample from those seed locations, optionally in parallel.
            key, sample_key = random.split(state.key, 2)
            sample_keys = random.split(sample_key, seed_indices.size)
            # expects: key, point_U0, log_L0, log_L_constraint
            new_reservoir = get_samples_parallel(sample_keys, points_U0_seed, log_L0_seed, log_L_constraints_seed)
            new_reservoir = new_reservoir._replace(points_X=self._filter_prior_chain(new_reservoir.points_X))
            # Merge samples, and recalculate statistics
            prev_evidence_calculation = state.evidence_calculation
            state = collect_samples(state, new_reservoir)
            # Calculate diagnostic stats
            prev_log_Z_mean, prev_log_Z_uncert = linear_to_log_stats(
                log_f_mean=prev_evidence_calculation.log_Z_mean,
                log_f2_mean=prev_evidence_calculation.log_Z2_mean)
            log_Z_mean, log_Z_uncert = linear_to_log_stats(
                log_f_mean=state.evidence_calculation.log_Z_mean,
                log_f2_mean=state.evidence_calculation.log_Z2_mean)
            prev_ess = LogSpace(prev_evidence_calculation.log_Z_mean).square() \
                       / LogSpace(prev_evidence_calculation.log_dZ2_mean)
            ess = LogSpace(state.evidence_calculation.log_Z_mean).square() \
                  / LogSpace(state.evidence_calculation.log_dZ2_mean)
            thread_stats_update = \
                ThreadStats(evidence_uncert_diff=log_Z_uncert - prev_log_Z_uncert,
                            ess_diff=ess.value - prev_ess.value,
                            evidence_diff=log_Z_mean - prev_log_Z_mean,
                            log_L_max=jnp.max(jnp.where(
                                jnp.arange(state.sample_collection.log_L_samples.size) >= state.sample_idx,
                                -jnp.inf,
                                state.sample_collection.log_L_samples)),
                            num_likelihood_evaluations=jnp.sum(state.sample_collection.num_likelihood_evaluations)
                            )
            thread_stats = tree_map(lambda operand, update: replace_index(operand, update, state.step_idx),
                                    state.thread_stats, thread_stats_update)
            # Determine if we are done
            state = state._replace(step_idx=state.step_idx + jnp.ones_like(state.step_idx),
                                   thread_stats=thread_stats
                                   )
            termination_stats = TerminationStats(previous_evidence_calculation=prev_evidence_calculation,
                                                 current_evidence_calculation=state.evidence_calculation,
                                                 num_samples=state.sample_idx,
                                                 num_steps=state.step_idx,
                                                 log_L_contour_max=state.sample_collection.log_L_samples[
                                                     state.sample_idx - 1])
            done, termination_reason = termination_condition(termination_stats,
                                                             termination_ess=termination_ess,
                                                             termination_likelihood_contour=termination_likelihood_contour,
                                                             termination_evidence_uncert=termination_evidence_uncert,
                                                             termination_live_evidence_frac=termination_live_evidence_frac,
                                                             termination_max_num_steps=termination_max_num_steps,
                                                             termination_max_samples=state.sample_collection.log_L_samples.size)
            state = state._replace(done=done, termination_reason=termination_reason)
            return state

        state = while_loop(lambda state: jnp.bitwise_not(state.done),
                           body,
                           init_state)
        return state

    def __call__(self, key,
                 num_live_points: Union[float, int, jnp.ndarray] = None,
                 termination_live_evidence_frac: Union[float, jnp.ndarray] = 0.01,
                 termination_likelihood_contour: Union[float, jnp.ndarray] = None,
                 termination_ess: Union[float, int, jnp.ndarray] = None,
                 termination_evidence_uncert: Union[float, jnp.ndarray] = None,
                 termination_max_num_steps: Union[float, int, jnp.ndarray] = None,
                 dynamic_kwargs: Optional[Dict[str, Any]] = None,
                 maximise_likelihood: bool = False,
                 *,
                 return_state: bool = False,
                 refine_state: NestedSamplerState = None,
                 resize_max_samples: int = None
                 ):
        """
        Applies static nested sampling, and optionally also dynamic improvement.

        Args:
            key: PRNG Key
            num_live_points: number of live points in static run.
            termination_live_evidence_frac: terminate static run after this much evidence left in live-points.
            termination_likelihood_contour: halt when likelihood contour passes this level.
            termination_ess: terminate when this many effective samples taken.
            termination_evidence_uncert: terminate when evidence falls below this.
            termination_max_num_steps: terminate when this many threads used.
            delta_num_live_points: refine with this many live points in each thread.
            dynamic_kwargs: Dict with items:
                G - interpolate between evidence goal (0.) and posterior goal (1.)
            return_state: bool, whether to return state with result
            refine_state: optional, if given then only refine the state
            resize_max_samples: optional, if given then extend sample collection size before refinement

        Returns:
            if return_state returns NestedSamplerResults and NestedSamplerState, else just NestedSamplerResults

        """

        if num_live_points is None:
            num_live_points = self.prior_chain.U_ndims * 50
        num_live_points = jnp.asarray(num_live_points, self.dtype)

        if dynamic_kwargs is None:
            dynamic_kwargs = dict()

        # Establish the state that we need to carry through iterations, and to facilitate post-analysis.
        state = self.initial_state(key)

        if maximise_likelihood:
            # todo: for max likelihood goal refactor
            search_top_n = num_live_points
            state = self._loop(init_state=state,
                               num_samples=self.samples_per_step,
                               goal_type='likelihood_maximisation',
                               search_top_n=search_top_n,
                               num_parallel_samplers=self.num_parallel_samplers,
                               termination_ess=None,
                               termination_live_evidence_frac=None,
                               termination_evidence_uncert=None,
                               termination_likelihood_contour=termination_likelihood_contour,
                               termination_max_num_steps=termination_max_num_steps)
        else:
            # cover space with static space
            state = self._loop(init_state=state,
                               num_samples=self.samples_per_step,
                               goal_type='static',
                               static_num_live_points=num_live_points,
                               num_parallel_samplers=self.num_parallel_samplers,
                               termination_ess=termination_ess,
                               termination_likelihood_contour=termination_likelihood_contour,
                               termination_evidence_uncert=termination_evidence_uncert,
                               termination_live_evidence_frac=termination_live_evidence_frac,
                               termination_max_num_steps=termination_max_num_steps)

            if self.dynamic:
                assert (termination_evidence_uncert is not None) \
                       or (termination_ess is not None) \
                       or (termination_max_num_steps is not None), 'Dynamic run needs at least one termination criterion'
                state = state._replace(done=jnp.asarray(False))
                state = self._loop(init_state=state,
                                   num_samples=self.samples_per_step,
                                   goal_type='dynamic',
                                   G=jnp.clip(dynamic_kwargs.get('G', 0.), 0., 1.),
                                   num_parallel_samplers=self.num_parallel_samplers,
                                   termination_ess=termination_ess,
                                   termination_likelihood_contour=None,
                                   termination_evidence_uncert=termination_evidence_uncert,
                                   termination_live_evidence_frac=None,
                                   termination_max_num_steps=termination_max_num_steps)

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
        ESS = LogSpace(state.evidence_calculation.log_Z_mean).square() / LogSpace(
            state.evidence_calculation.log_dZ2_mean)
        samples = state.sample_collection.points_X
        total_num_samples = state.sample_idx
        log_L_samples = state.sample_collection.log_L_samples
        empty_mask = jnp.arange(state.sample_collection.log_L_samples.size) >= state.sample_idx
        max_L_idx = jnp.argmax(jnp.where(empty_mask, -jnp.inf, log_L_samples))
        log_L_max = log_L_samples[max_L_idx]
        max_L_sample = tree_map(lambda x: x[max_L_idx], samples)
        dp_mean = LogSpace(state.sample_collection.log_dZ_mean)
        dp_mean /= dp_mean.sum()
        H_mean = (LogSpace(log_L_samples) * dp_mean).sum() / LogSpace(log_Z_mean)
        X_mean = LogSpace(state.sample_collection.log_X_mean)
        num_likelihood_evaluations_per_sample = state.sample_collection.num_likelihood_evaluations
        total_num_likelihood_evaluations = jnp.sum(num_likelihood_evaluations_per_sample)
        num_live_points_per_sample = state.sample_collection.num_live_points
        efficiency = LogSpace(jnp.log(total_num_samples) - jnp.log(total_num_likelihood_evaluations))
        termination_reason = state.termination_reason

        data = dict(
            log_Z_mean=log_Z_mean,  # estimate of log(E[Z])
            log_Z_uncert=log_Z_uncert,  # estimate of log(StdDev[Z])
            ESS=ESS.value,  # estimate of Kish's effective sample size
            H_mean=H_mean.value,  # estimate of E[int log(L) L dp/Z]
            total_num_samples=total_num_samples,  # int, the total number of samples collected.
            log_L_samples=log_L_samples,  # log(L) of each sample
            log_dp_mean=dp_mean.log_abs_val,
            # log(E[dZ]) of each sample, where dZ is how much it contributes to the total evidence.
            # log(StdDev[dZ]) of each sample, where dZ is how much it contributes to the total evidence.
            log_X_mean=X_mean.log_abs_val,  # log(E[X]) of each sample
            num_likelihood_evaluations_per_sample=num_likelihood_evaluations_per_sample,
            # how many likelihood evaluations were made per sample.
            num_live_points_per_sample=num_live_points_per_sample,  # how many live points were taken for the samples.
            total_num_likelihood_evaluations=total_num_likelihood_evaluations,
            # how many likelihood evaluations were made in total,
            # sum of num_likelihood_evaluations_per_sample.
            log_efficiency=efficiency.log_abs_val,  # total_num_samples / total_num_likelihood_evaluations
            termination_reason=termination_reason,  # termination condition as bit mask
            thread_stats=state.thread_stats,
            log_L_max=log_L_max,
            sample_L_max=max_L_sample
        )
        if self.collect_samples:
            data['samples'] = samples
        else:
            data['samples'] = None

        return NestedSamplerResults(**data)
