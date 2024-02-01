import io
import logging
from typing import NamedTuple, Optional, Union, TextIO, Tuple, List

import jax.nn
import jax.numpy as jnp
import numpy as np
from jax import lax, random, pmap, tree_map
from jax._src.lax import parallel
from jax._src.scipy.special import logit
from jaxopt import NonlinearCG

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.maps import remove_chunk_dim
from jaxns.internals.types import PRNGKey, StaticStandardNestedSamplerState, BoolArray, StaticStandardSampleCollection, \
    int_type, Sample, IntArray, UType, FloatArray, LikelihoodInputType, XType
from jaxns.nested_sampler.standard_static import draw_uniform_samples, _inter_sync_shrinkage_process, \
    create_init_termination_register
from jaxns.samplers.bases import BaseAbstractSampler
from jaxns.utils import _bit_mask

logger = logging.getLogger('jaxns')

__all__ = [
    'GlobalOptimisationResults',
    'GlobalOptimisationTerminationCondition',
    'GlobalOptimisationState',
    'SimpleGlobalOptimisation'
]


class GlobalOptimisationState(NamedTuple):
    key: PRNGKey
    samples: Sample
    num_likelihood_evaluations: IntArray
    num_samples: IntArray


class GlobalOptimisationResults(NamedTuple):
    U_solution: UType
    X_solution: XType
    solution: LikelihoodInputType
    log_L_solution: FloatArray
    num_likelihood_evaluations: IntArray
    num_samples: IntArray
    termination_reason: IntArray
    relative_spread: FloatArray
    absolute_spread: FloatArray


class GlobalOptimisationTerminationCondition(NamedTuple):
    max_likelihood_evaluations: Optional[Union[IntArray, int]] = None  # jnp.asarray(jnp.iinfo(int_type).max, int_type)
    log_likelihood_contour: Optional[
        Union[FloatArray, float]] = None  # jnp.asarray(jnp.finfo(float_type).max, float_type)
    rtol: Optional[Union[FloatArray, float]] = None  # jnp.asarray(0., float_type)
    atol: Optional[Union[FloatArray, float]] = None  # jnp.asarray(0., float_type)
    min_efficiency: Optional[Union[FloatArray, float]] = None  # jnp.asarray(0., float_type)

    def __and__(self, other):
        return TerminationConditionConjunction(conds=[self, other])

    def __or__(self, other):
        return TerminationConditionDisjunction(conds=[self, other])


class TerminationConditionConjunction(NamedTuple):
    conds: List[Union[
        'TerminationConditionDisjunction', 'TerminationConditionConjunction', GlobalOptimisationTerminationCondition]]


class TerminationConditionDisjunction(NamedTuple):
    conds: List[Union[
        'TerminationConditionDisjunction', TerminationConditionConjunction, GlobalOptimisationTerminationCondition]]


def determine_termination(term_cond: GlobalOptimisationTerminationCondition,
                          state: GlobalOptimisationState) -> Tuple[BoolArray, IntArray]:
    """
    Determine if termination should happen. Termination Flags are bits:
        0-bit -> 1: used maximum allowed number of likelihood evaluations
        1-bit -> 2: reached goal log-likelihood contour
        2-bit -> 4: relative spread of log-likelihood values below threshold
        3-bit -> 8: absolute spread of log-likelihood values below threshold
        4-bit -> 16: efficiency below threshold
        5-bit -> 32: on a plateau (possibly local minimum, or due to numerical issues)

    Multiple flags are summed together

    Args:
        term_cond: termination condition
        state: global optimisation state

    Returns:
        boolean done signal, and termination reason
    """

    termination_reason = jnp.asarray(0, int_type)
    done = jnp.asarray(False, jnp.bool_)

    def _set_done_bit(bit_done, bit_reason, done, termination_reason):
        if bit_done.size > 1:
            raise RuntimeError("bit_done must be a scalar.")
        done = jnp.bitwise_or(bit_done, done)
        termination_reason += jnp.where(bit_done,
                                        jnp.asarray(2 ** bit_reason, int_type),
                                        jnp.asarray(0, int_type))
        return done, termination_reason

    if isinstance(term_cond, TerminationConditionConjunction):
        for c in term_cond.conds:
            _done, _reason = determine_termination(term_cond=c, state=state)
            done = jnp.bitwise_and(_done, done)
            termination_reason = jnp.bitwise_and(_reason, termination_reason)
        return done, termination_reason

    if isinstance(term_cond, TerminationConditionDisjunction):
        for c in term_cond.conds:
            _done, _reason = determine_termination(term_cond=c, state=state)
            done = jnp.bitwise_or(_done, done)
            termination_reason = jnp.bitwise_or(_reason, termination_reason)
        return done, termination_reason

    if term_cond.max_likelihood_evaluations is not None:
        # used all points
        reached_max_samples = state.num_likelihood_evaluations >= term_cond.max_likelihood_evaluations
        done, termination_reason = _set_done_bit(reached_max_samples, 0,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.log_likelihood_contour is not None:
        # reached goal log-likelihood contour
        reached_log_L_contour = jnp.max(state.samples.log_L) >= term_cond.log_likelihood_contour
        done, termination_reason = _set_done_bit(reached_log_L_contour, 1,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.rtol is not None:
        # relative spread of log-likelihood values below threshold
        max_log_L = jnp.max(state.samples.log_L)
        min_log_L = jnp.min(state.samples.log_L)
        diff_log_L = jnp.abs(max_log_L - min_log_L)
        reached_rtol = diff_log_L <= 0.5 * term_cond.rtol * jnp.abs(max_log_L + min_log_L)
        done, termination_reason = _set_done_bit(reached_rtol, 2,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.atol is not None:
        # absolute spread of log-likelihood values below threshold
        max_log_L = jnp.max(state.samples.log_L)
        min_log_L = jnp.min(state.samples.log_L)
        diff_log_L = jnp.abs(max_log_L - min_log_L)
        reached_atol = diff_log_L <= term_cond.atol
        done, termination_reason = _set_done_bit(reached_atol, 3,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.min_efficiency is not None:
        # efficiency below threshold
        efficiency = state.samples.log_L.shape[0] / jnp.sum(state.samples.num_likelihood_evaluations)
        reached_min_efficiency = efficiency <= term_cond.min_efficiency
        done, termination_reason = _set_done_bit(reached_min_efficiency, 4,
                                                 done=done, termination_reason=termination_reason)

    # on plateau
    on_plateau = jnp.max(state.samples.log_L) == jnp.min(state.samples.log_L)
    done, termination_reason = _set_done_bit(on_plateau, 5,
                                             done=done, termination_reason=termination_reason)

    return done, termination_reason


def gradient_based_optimisation(model: BaseAbstractModel, init_U_point: UType) -> Tuple[UType, FloatArray, IntArray]:
    def loss(U_unconstrained: UType):
        U = jax.nn.sigmoid(U_unconstrained)
        return -model.log_prob_likelihood(U, allow_nan=False)

    solver = NonlinearCG(
        fun=loss,
        jit=True,
        unroll=False,
        verbose=False
    )

    results = solver.run(init_params=logit(init_U_point))
    return jax.nn.sigmoid(results.params), -results.state.value, results.state.num_fun_eval


def _single_thread_global_optimisation(init_state: GlobalOptimisationState,
                                       termination_cond: GlobalOptimisationTerminationCondition,
                                       sampler: BaseAbstractSampler) -> Tuple[GlobalOptimisationState, IntArray]:
    """
    Runs a single thread of global optimisation. Sequentially samples until termination condition is met,
    replacing the worst sample with a new one in groups of num_samples.

    Args:
        init_state: initial state of the global optimisation
        termination_cond: termination condition
        sampler: sampler

    Returns:
        final state of the global optimisation and termination reason
    """

    class CarryType(NamedTuple):
        state: GlobalOptimisationState

    def cond(carry: CarryType) -> BoolArray:
        done, _ = determine_termination(term_cond=termination_cond, state=carry.state)
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

        k = sampler.num_phantom()
        if k > 0:
            def _repeat(x):
                return jnp.repeat(x, (k + 1), axis=0)

            fake_state = fake_state._replace(
                sample_collection=tree_map(_repeat, fake_state.sample_collection)
            )

        fake_state, fake_termination_register = _inter_sync_shrinkage_process(
            init_state=fake_state,
            sampler=sampler,
            num_samples=num_samples * (1 + k),
            init_termination_register=create_init_termination_register()
        )

        num_likelihood_evaluations = carry.state.num_likelihood_evaluations + jnp.sum(
            fake_state.sample_collection.num_likelihood_evaluations)

        if k > 0:
            # Choose the maximum likelihood sample from the k+1 samples (requires reshape first to unstack)

            choose_idx = jnp.argmax(
                jnp.reshape(fake_state.sample_collection.log_L, ((k + 1), num_samples)),
                axis=0
            )

            def _select(x):
                x = jnp.reshape(x, ((k + 1), num_samples) + x.shape[1:])  # [k+1, N, ...]
                return x[choose_idx, jnp.arange(num_samples)]  # [N, ...]

            fake_state = fake_state._replace(
                sample_collection=tree_map(
                    _select,
                    fake_state.sample_collection
                )
            )

        samples = Sample(
            U_sample=fake_state.sample_collection.U_samples,
            log_L_constraint=fake_state.sample_collection.log_L,
            log_L=fake_state.sample_collection.log_L,
            num_likelihood_evaluations=fake_state.sample_collection.num_likelihood_evaluations
        )

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

    _, termination_reason = determine_termination(term_cond=termination_cond, state=carry_state.state)

    return carry_state.state, termination_reason


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
        if num_search_chains < 1:
            raise ValueError("num_search_chains must be >= 1.")
        self.num_search_chains = int(num_search_chains)
        self.num_parallel_workers = int(num_parallel_workers)

        if self.num_parallel_workers > 1:
            logger.info(f"Using {self.num_parallel_workers} parallel workers, each running identical samplers.")
        self.model = model
        self.num_search_chains = num_search_chains

    def _gradient_descent(self, results: GlobalOptimisationResults) -> GlobalOptimisationResults:
        U_solution, log_L_solution, _num_likelihood_evals = gradient_based_optimisation(self.model,
                                                                                        init_U_point=results.U_solution)
        X_solution = self.model.transform(U_solution)
        solution = self.model.prepare_input(U_solution)
        num_likelihood_evals = results.num_likelihood_evaluations + _num_likelihood_evals
        return results._replace(
            U_solution=U_solution,
            log_L_solution=log_L_solution,
            X_solution=X_solution,
            solution=solution,
            num_likelihood_evaluations=num_likelihood_evals
        )

    def _to_results(self, termination_reason: IntArray, state: GlobalOptimisationState) -> GlobalOptimisationResults:
        """
        Converts the final state of the global optimisation to results.

        Args:
            state: final state of the global optimisation

        Returns:
            results of the global optimisation
        """
        best_idx = jnp.argmax(state.samples.log_L)
        U_solution = state.samples.U_sample[best_idx]
        X_solution = self.model.transform(U_solution)
        solution = self.model.prepare_input(U_solution)
        max_log_L = state.samples.log_L[best_idx]
        min_log_L = jnp.min(state.samples.log_L)
        relative_spread = 2. * jnp.abs(max_log_L - min_log_L) / jnp.abs(max_log_L + min_log_L)
        absolute_spread = jnp.abs(max_log_L - min_log_L)
        return GlobalOptimisationResults(
            U_solution=state.samples.U_sample[best_idx],
            X_solution=X_solution,
            solution=solution,
            log_L_solution=state.samples.log_L[best_idx],
            num_likelihood_evaluations=state.num_likelihood_evaluations,
            num_samples=state.num_samples,
            relative_spread=relative_spread,
            absolute_spread=absolute_spread,
            termination_reason=termination_reason
        )

    def _run(self, key: PRNGKey, term_cond: GlobalOptimisationTerminationCondition) -> Tuple[
        IntArray, GlobalOptimisationState]:
        """
        Runs the global optimisation.

        Args:
            key: PRNGKey
            term_cond: termination condition

        Returns:
            the termination reason and final state of the global optimisation
        """

        def replica(key: PRNGKey) -> Tuple[GlobalOptimisationState, IntArray]:
            state = create_init_state(
                key=key,
                num_search_chains=self.num_search_chains,
                model=self.model
            )

            # Continue sampling with provided sampler until user-defined termination condition is met.
            state, termination_reason = _single_thread_global_optimisation(
                init_state=state,
                termination_cond=term_cond,
                sampler=self.sampler
            )
            if self.num_parallel_workers > 1:
                target_log_L_contour = jnp.max(
                    parallel.all_gather(jnp.max(state.samples.log_L), 'i')
                )
                state, termination_reason = _single_thread_global_optimisation(
                    init_state=state,
                    termination_cond=term_cond._replace(log_likelihood_contour=target_log_L_contour),
                    sampler=self.sampler
                )

            return state, termination_reason

        if self.num_parallel_workers > 1:
            parallel_ns = pmap(replica, axis_name='i')
            keys = random.split(key, self.num_parallel_workers)
            batched_state, termination_reason = parallel_ns(keys)
            state = remove_chunk_dim(batched_state)
            state = state._replace(key=state.key[0])
        else:
            state, termination_reason = replica(key)

        return termination_reason, state


def summary(results: GlobalOptimisationResults, f_obj: Optional[Union[str, TextIO]] = None):
    """
    Gives a summary of the results of a global optimisation.

    Args:
        results (GlobalOptimisationResults): Nested sampler result
    """
    main_s = []

    def _print(s):
        print(s)
        main_s.append(s)

    def _round(v, uncert_v):
        v = float(v)
        uncert_v = float(uncert_v)
        try:
            sig_figs = -int("{:e}".format(uncert_v).split('e')[1]) + 1
            return round(float(v), sig_figs)
        except:
            return float(v)

    def _print_termination_condition(_termination_reason: int):
        termination_bit_mask = _bit_mask(int(_termination_reason), width=8)
        # 0-bit -> 1: used maximum allowed number of likelihood evaluations
        #         1-bit -> 2: reached goal log-likelihood contour
        #         2-bit -> 4: relative spread of log-likelihood values below threshold
        #         3-bit -> 8: absolute spread of log-likelihood values below threshold
        #         4-bit -> 16: efficiency below threshold
        #         5-bit -> 32: on a plateau (possibly local minimum, or due to numerical issues)
        for bit, condition in zip(termination_bit_mask, [
            'Reached max num likelihood evaluations',
            'Reached goal log-likelihood contour',
            'Small relative spread of log-likelihood values',
            'Small absolute spread of log-likelihood values',
            'Sampler efficiency too low',
            'On plateau (possibly local minimum, or due to numerical issues)'
        ]):
            if bit == 1:
                _print(condition)

    _print("--------")
    _print("Termination Conditions:")
    if np.size(results.termination_reason) > 1:
        for replica_idx in range(np.size(results.termination_reason)):
            _print(f"Replica {replica_idx}:")
            _print_termination_condition(int(results.termination_reason[replica_idx]))
    else:
        _print_termination_condition(int(results.termination_reason))
    _print("--------")
    _print(f"likelihood evals: {int(results.num_likelihood_evaluations):d}")
    _print(f"samples: {int(results.num_samples):d}")
    _print(
        f"likelihood evals / sample: {float(results.num_likelihood_evaluations / results.num_samples):.1f}"
    )
    _print("--------")
    _print(
        f"max(log_L)={_round(results.log_L_solution, results.log_L_solution)}"
    )
    _print(
        f"relative spread: {_round(results.relative_spread, results.relative_spread)}"
    )
    _print(
        f"absolute spread: {_round(results.absolute_spread, results.absolute_spread)}"
    )

    X_solution = results.X_solution
    for name in X_solution.keys():
        _sample = X_solution[name].reshape((-1,))
        ndims = _sample.size
        _print("--------")
        var_name = name if ndims == 1 else "{}[#]".format(name)
        _print(
            f"{var_name}: max(L) est."
        )
        for dim in range(ndims):
            _max_like_point = _sample[dim]
            # two sig-figs based on uncert
            sig_figs = -int("{:e}".format(_max_like_point * 0.1).split('e')[1]) + 1

            def _round(ar):
                return round(float(ar), sig_figs)

            _print("{}: {}".format(
                name if ndims == 1 else "{}[{}]".format(name, dim),
                _round(_max_like_point)
            ))
    _print("--------")
    if f_obj is not None:
        out = "\n".join(main_s)
        if isinstance(f_obj, str):
            with open(f_obj, 'w') as f:
                f.write(out)
        elif isinstance(f_obj, io.TextIOBase):
            f_obj.write(out)
        else:
            raise TypeError(f"Invalid f_obj: {type(f_obj)}")
