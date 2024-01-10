import logging
from typing import NamedTuple, Optional

import jax.numpy as jnp
from jax import lax, random, pmap
from jax._src.lax import parallel

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.maps import remove_chunk_dim
from jaxns.internals.types import PRNGKey, StaticStandardNestedSamplerState, BoolArray, StaticStandardSampleCollection, \
    int_type, Sample, IntArray, UType, XType, FloatArray, LikelihoodInputType
from jaxns.nested_sampler.standard_static import draw_uniform_samples, _inter_sync_shrinkage_process
from jaxns.samplers import UniDimSliceSampler
from jaxns.samplers.bases import BaseAbstractSampler

logger = logging.getLogger('jaxns')

__all__ = [
    'GlobalOptimisationResults',
    'GlobalOptimisationTerminationCondition',
    'GlobalOptimisationState',
    'SimpleGlobalOptimisation',
    'DefaultGlobalOptimisation'
]


class GlobalOptimisationState(NamedTuple):
    key: PRNGKey
    samples: Sample
    num_likelihood_evaluations: IntArray
    num_samples: IntArray


class GlobalOptimisationResults(NamedTuple):
    U_solution: UType
    solution: LikelihoodInputType
    log_L_solution: FloatArray
    num_likelihood_evaluations: IntArray
    num_samples: IntArray


class GlobalOptimisationTerminationCondition(NamedTuple):
    max_likelihood_evaluations: int = jnp.inf
    log_likelihood_contour: float = jnp.inf
    rtol: float = 0.
    atol: float = 0.
    min_efficiency: float = 1e-2


def _single_thread_global_optimisation(init_state: GlobalOptimisationState,
                                       termination_cond: GlobalOptimisationTerminationCondition,
                                       sampler: BaseAbstractSampler) -> GlobalOptimisationState:
    """
    Runs a single thread of global optimisation. Sequentially samples until termination condition is met,
    replacing the worst sample with a new one in groups of num_samples.

    Args:
        init_state: initial state of the global optimisation
        termination_cond: termination condition
        sampler: sampler

    Returns:
        final state of the global optimisation
    """

    class CarryType(NamedTuple):
        state: GlobalOptimisationState

    def cond(carry: CarryType) -> BoolArray:
        done_1 = carry.state.num_likelihood_evaluations >= termination_cond.max_likelihood_evaluations
        max_log_L = jnp.max(carry.state.samples.log_L)
        min_log_L = jnp.min(carry.state.samples.log_L)
        diff_log_L = jnp.abs(max_log_L - min_log_L)
        done_2 = max_log_L >= termination_cond.log_likelihood_contour
        done_3 = diff_log_L <= 0.5 * termination_cond.rtol * jnp.abs(max_log_L + min_log_L)
        done_4 = diff_log_L <= termination_cond.atol
        efficiency = carry.state.samples.log_L.shape[0] / jnp.sum(carry.state.samples.num_likelihood_evaluations)
        done_5 = efficiency <= termination_cond.min_efficiency
        done = done_1 | done_2 | done_3 | done_4 | done_5
        return jnp.bitwise_not(done)

    def body(carry: CarryType) -> CarryType:
        # Devices are independent, i.e. expect no communication between them in sampler.
        key, sample_key = random.split(carry.state.key, 2)

        num_samples = carry.state.samples.log_L.shape[0]

        fake_state = StaticStandardNestedSamplerState(
            key=sample_key,
            next_sample_idx=jnp.asarray(0, int_type),
            sample_collection=StaticStandardSampleCollection(
                sender_node_idx=jnp.zeros((num_samples,), int_type),
                log_L=carry.state.samples.log_L,
                U_samples=carry.state.samples.U_sample,
                num_likelihood_evaluations=carry.state.samples.num_likelihood_evaluations,
                phantom=jnp.zeros((num_samples,), jnp.bool_)
            ),
            front_idx=jnp.arange(num_samples, dtype=int_type)
        )

        fake_state = _inter_sync_shrinkage_process(
            init_state=fake_state,
            sampler=sampler,
            num_samples=num_samples
        )

        samples = Sample(
            U_sample=fake_state.sample_collection.U_samples,
            log_L_constraint=fake_state.sample_collection.log_L,
            log_L=fake_state.sample_collection.log_L,
            num_likelihood_evaluations=fake_state.sample_collection.num_likelihood_evaluations
        )

        num_likelihood_evaluations = carry.state.num_likelihood_evaluations + jnp.sum(
            samples.num_likelihood_evaluations)

        state = GlobalOptimisationState(
            key=key,
            samples=samples,
            num_likelihood_evaluations=num_likelihood_evaluations,
            num_samples=carry.state.num_samples + jnp.asarray(num_samples, int_type)
        )

        return CarryType(state=state)

    init_carry_state = CarryType(
        state=init_state
    )

    carry_state: CarryType = lax.while_loop(
        cond_fun=cond,
        body_fun=body,
        init_val=init_carry_state
    )

    return carry_state.state


def create_init_state(key: PRNGKey, num_search_chains: int,
                      model: BaseAbstractModel) -> GlobalOptimisationState:
    """
    Creates the initial state of the global optimisation.

    Args:
        key: PRNGKey
        num_search_chains: number of search chains
        model: model

    Returns:
        initial state of the global optimisation
    """
    key, sample_key = random.split(key, 2)
    init_samples = draw_uniform_samples(key=sample_key, num_live_points=num_search_chains, model=model)

    return GlobalOptimisationState(
        key=key,
        samples=init_samples,
        num_likelihood_evaluations=jnp.sum(init_samples.num_likelihood_evaluations),
        num_samples=jnp.asarray(num_search_chains, int_type)
    )


class SimpleGlobalOptimisation:
    """
    Simple global optimisation leveraging building blocks of nested sampling.
    """

    def __init__(self, sampler: BaseAbstractSampler, num_search_chains: int,
                 model: BaseAbstractModel, num_parallel_workers: int = 1):
        self.sampler = sampler
        self.num_search_chains = int(num_search_chains)
        self.num_parallel_workers = int(num_parallel_workers)

        if self.num_parallel_workers > 1:
            logger.info(f"Using {self.num_parallel_workers} parallel workers, each running identical samplers.")
        self.model = model
        self.num_search_chains = num_search_chains

    def _to_results(self, state: GlobalOptimisationState) -> GlobalOptimisationResults:
        """
        Converts the final state of the global optimisation to results.

        Args:
            state: final state of the global optimisation

        Returns:
            results of the global optimisation
        """
        best_idx = jnp.argmax(state.samples.log_L)
        U_solution = state.samples.U_sample[best_idx]
        solution = self.model.prepare_input(U_solution)
        return GlobalOptimisationResults(
            U_solution=state.samples.U_sample[best_idx],
            solution=solution,
            log_L_solution=state.samples.log_L[best_idx],
            num_likelihood_evaluations=state.num_likelihood_evaluations,
            num_samples=state.num_samples
        )

    def _run(self, key: PRNGKey, term_cond: GlobalOptimisationTerminationCondition) -> GlobalOptimisationState:
        """
        Runs the global optimisation.

        Args:
            key: PRNGKey
            term_cond: termination condition

        Returns:
            the final state of the global optimisation
        """

        def replica(key: PRNGKey) -> GlobalOptimisationState:
            state = create_init_state(
                key=key,
                num_search_chains=self.num_search_chains,
                model=self.model
            )

            # Continue sampling with provided sampler until user-defined termination condition is met.
            state = _single_thread_global_optimisation(
                init_state=state,
                termination_cond=term_cond,
                sampler=self.sampler
            )
            if self.num_parallel_workers > 1:
                target_log_L_contour = jnp.max(
                    parallel.all_gather(jnp.max(state.samples.log_L), 'i')
                )
                state = _single_thread_global_optimisation(
                    init_state=state,
                    termination_cond=term_cond._replace(log_likelihood_contour=target_log_L_contour),
                    sampler=self.sampler
                )

            return state

        if self.num_parallel_workers > 1:
            parallel_ns = pmap(replica, axis_name='i')
            keys = random.split(key, self.num_parallel_workers)
            batched_state = parallel_ns(keys)
            state = remove_chunk_dim(batched_state)
            state = state._replace(key=state.key[0])
        else:
            state = replica(key)

        return state


class DefaultGlobalOptimisation:
    """
    Default global optimisation class.
    """

    def __init__(self, model: BaseAbstractModel,
                 num_search_chains: Optional[int] = None,
                 num_parallel_workers: int = 1,
                 s: Optional[int] = None):
        if num_search_chains is None:
            num_search_chains = model.U_ndims * 20
        if s is None:
            s = 1

        sampler = UniDimSliceSampler(
            model=model,
            num_slices=model.U_ndims * int(s),
            num_phantom_save=0,
            midpoint_shrink=True,
            perfect=True
        )

        self._global_optimiser = SimpleGlobalOptimisation(
            sampler=sampler,
            num_search_chains=int(num_search_chains),
            model=model,
            num_parallel_workers=num_parallel_workers
        )

    def __call__(self, key: PRNGKey, term_cond: GlobalOptimisationTerminationCondition) -> GlobalOptimisationResults:
        """
        Runs the global optimisation.

        Args:
            key: PRNGKey
            term_cond: termination condition

        Returns:
            results of the global optimisation
        """
        state = self._global_optimiser._run(key, term_cond)
        return self._global_optimiser._to_results(state)
