from typing import Callable, Dict, Any, Tuple

from jax.lax import while_loop, dynamic_update_slice
from jax import numpy as jnp, tree_map, random

from jaxns.internals.log_semiring import LogSpace
from jaxns.nested_sampler.evidence_calculation import _update_evidence_calculation
from jaxns.nested_sampler.reservoir_refiller import ReservoirRefiller
from jaxns.prior_transforms import PriorChain
from jaxns.types import NestedSamplerState, SampleCollection, EvidenceCalculation, Reservoir, ThreadStats
from jaxns.nested_sampler.termination import termination_condition
from jaxns.internals.stats import linear_to_log_stats


def normalise_log_space(x: LogSpace):
    norm = x.sum()
    x /= norm
    x = LogSpace(jnp.where(jnp.isneginf(norm.log_abs_val), -jnp.inf, x.log_abs_val))
    return x

def _get_dynamic_goal(state: NestedSamplerState, G: jnp.ndarray, f: jnp.ndarray):
    """
    Get likelihood range to run a thread over.
    """
    n_i = state.sample_collection.num_live_points
    dZ_mean = LogSpace(state.sample_collection.log_dZ_mean)

    def logsumexp_cumsum_body(_state):
        (log_abs_val, idx) = _state
        next_val = LogSpace(log_abs_val[idx]) + LogSpace(log_abs_val[idx - 1])
        next_idx = idx - jnp.ones_like(idx)
        next_log_abs_val = dynamic_update_slice(log_abs_val, next_val.log_abs_val[None], [next_idx])
        return (next_log_abs_val, next_idx)

    # Calculate remaining evidence, doing only the amount of work necessary.
    # Z_remaining = dZ_mean.cumsum(reverse=True)
    # [a,b,-inf], 2 -> [a+b, b, -inf]
    # [-inf, -inf,-inf], 0 -> [-inf, -inf, -inf]
    
    (log_Z_remaining, _) = while_loop(lambda _state: _state[1] > 0,
                                      logsumexp_cumsum_body,
                                      (state.sample_collection.log_dZ_mean,
                                       state.sample_idx - jnp.ones_like(state.sample_idx)))
    
    Z_remaining = LogSpace(log_Z_remaining)
    # I_evidence = ((LogSpace(jnp.log(n_i + 1.)) * Z_remaining + LogSpace(jnp.log(n_i)) * dZ_mean) / (
    #         LogSpace(jnp.log(n_i)).sqrt() * LogSpace(jnp.log(n_i + 2.)) ** (1.5)))
    # I_evidence = ((LogSpace(jnp.log(1 + 1. / n_i)) * Z_remaining + dZ_mean) / (
    #         LogSpace(jnp.log(n_i)) * LogSpace(jnp.log(1. + 2. / n_i)) ** (1.5)))
    I_evidence = ((LogSpace(jnp.logaddexp(0., -jnp.log(n_i))) * Z_remaining + dZ_mean)
                  / (LogSpace(jnp.log(n_i)) * LogSpace(jnp.logaddexp(0., jnp.log(2.) - jnp.log(n_i))) ** (1.5)))
    # I_evidence = Z_remaining / LogSpace(jnp.log(n_i))
    I_evidence = normalise_log_space(I_evidence)
    I_posterior = dZ_mean
    I_posterior = normalise_log_space(I_posterior)
    I_goal = LogSpace(jnp.log(1. - G)) * I_evidence + LogSpace(jnp.log(G)) * I_posterior
    I_goal = normalise_log_space(I_goal)

    mask = jnp.arange(I_goal.size) >= state.sample_idx
    I_goal = LogSpace(jnp.where(mask, -jnp.inf, I_goal.log_abs_val))
    important = I_goal >= LogSpace(jnp.log(f)) * (I_goal.max() - I_goal.min()).abs() + I_goal.min()

    # first instance - 1
    # [False, True, False] -> 0
    # [True, True, False] -> -1
    # [False, False, False] -> -1
    start_idx = jnp.clip(jnp.argmax(important) - 1, 0,  I_goal.size - 1)
    log_L_start = state.sample_collection.log_L[start_idx]
    # last instance + 1
    # [False, True, False] -> 2
    # [False, True, True] -> 2
    # [False, False, False] -> 2
    end_idx = jnp.clip(I_goal.size - jnp.argmax(important[::-1]), 0, I_goal.size - 1)
    log_L_end = state.sample_collection.log_L[end_idx]
    
    # import pylab as plt
    #
    # plt.plot(I_goal.cumsum().value, label='goal')
    #
    # plt.plot(I_evidence.cumsum().value, label='Evidence')
    #
    # plt.plot(I_posterior.cumsum().value, label='post')
    #
    # plt.scatter(start_idx, I_goal.cumsum().value[start_idx])
    #
    # plt.scatter(end_idx, I_goal.cumsum().value[end_idx])
    #
    # plt.legend()
    # plt.show()
    #
    #
    # plt.plot(I_goal.value, label='goal')
    #
    # plt.scatter(start_idx, I_goal.value[start_idx])
    #
    # plt.scatter(end_idx, I_goal.value[end_idx])
    #
    # plt.plot(I_evidence.value, label='evidence')
    #
    # plt.plot(I_posterior.value, label='post.')
    #
    # plt.legend()
    # plt.show()

    return log_L_start, log_L_end


def create_initial_thread_state(state, log_L_entry):
    
    # Fill reservoir with points suitable for seeding sampler
    key, g_key = random.split(state.key)
    state = state._replace(key=key)
    # there should always be at least one satisfying point that makes its way into the reservoir.
    above_contour = state.sample_collection.log_L > log_L_entry
    on_or_above_contour = state.sample_collection.log_L >= log_L_entry
    g = -random.gumbel(g_key, shape=state.sample_collection.log_L.shape)
    # above contour will be exp(1000) more likely that on or above contour
    g = jnp.where(above_contour, g, jnp.where(on_or_above_contour, g + jnp.asarray(1000., g.dtype), jnp.inf))
    reservoir_selection = jnp.argsort(g)[:state.reservoir.log_L.size]
    # only available, points_U, and log_L are needed for seeding. The rest can be any value (so we reuse).
    initial_reservoir = Reservoir(
        points_U=state.sample_collection.points_U[reservoir_selection],
        # will be only used to seed if static_run
        points_X=state.reservoir.points_X,
        log_L=state.sample_collection.log_L[reservoir_selection],
        # will be only used to seed if static_run
        num_likelihood_evaluations=state.reservoir.num_likelihood_evaluations,
        available=jnp.zeros(reservoir_selection.shape, jnp.bool_)
    )
    # This contains the required information to compute Z and ZH
    initial_evidence_calculation = EvidenceCalculation(
        log_X_mean=jnp.asarray(0., state.log_L_contour.dtype),
        log_X2_mean=jnp.asarray(0., state.log_L_contour.dtype),
        log_Z_mean=jnp.asarray(-jnp.inf, state.log_L_contour.dtype),
        log_ZX_mean=jnp.asarray(-jnp.inf, state.log_L_contour.dtype),
        log_Z2_mean=jnp.asarray(-jnp.inf, state.log_L_contour.dtype),
        log_dZ2_mean=jnp.asarray(-jnp.inf, state.log_L_contour.dtype)
    )

    state = state._replace(
        done=jnp.asarray(False, jnp.bool_),
        termination_reason=jnp.zeros_like(state.termination_reason),
        sample_idx=jnp.zeros_like(state.sample_idx),
        log_L_contour=-jnp.inf,
        reservoir=initial_reservoir,
        evidence_calculation=initial_evidence_calculation)

    def fast_forward_body(state: NestedSamplerState) -> NestedSamplerState:
        num_live_points = state.sample_collection.num_live_points[state.sample_idx]
        log_L = state.sample_collection.log_L[state.sample_idx]

        next_evidence_calculation = _update_evidence_calculation(num_live_points=num_live_points,
                                                                 log_L=state.log_L_contour,
                                                                 next_log_L_contour=log_L,
                                                                 evidence_calculation=state.evidence_calculation)
        state = state._replace(evidence_calculation=next_evidence_calculation,
                               sample_idx=state.sample_idx + jnp.ones_like(state.sample_idx),
                               step_idx=state.step_idx + jnp.ones_like(state.step_idx),
                               log_L_contour=log_L
                               )
        return state

    def fast_forward_cond(state: NestedSamplerState):
        return (state.log_L_contour <= log_L_entry) & (state.sample_idx < state.sample_collection.log_L.size)

    state = while_loop(fast_forward_cond,
                       fast_forward_body,
                       state)

    return state


def _collect_sample(state: NestedSamplerState, prev_state: NestedSamplerState, i_main: jnp.ndarray) \
        -> Tuple[NestedSamplerState, jnp.ndarray]:
    n_points_main = prev_state.sample_idx
    sample_collection_main = prev_state.sample_collection

    # get the next dead point, which will be used if lower than the next point in main samples.
    # This will be ignored if reservoir is empty.
    idx_min_thread = jnp.argmin(jnp.where(state.reservoir.available, state.reservoir.log_L, jnp.inf))
    num_live_points_thread = jnp.sum(state.reservoir.available)
    log_L_thread = state.reservoir.log_L[idx_min_thread]
    num_live_points_main = sample_collection_main.num_live_points[i_main]
    log_L_main = sample_collection_main.log_L[i_main]

    # if all samples merged then nothing to select, rest must come from reservoir
    all_main_samples_merged = i_main == n_points_main
    next_num_live_points = jnp.where(all_main_samples_merged,
                                     num_live_points_thread,
                                     num_live_points_main + num_live_points_thread)

    reservoir_empty = jnp.bitwise_not(jnp.any(state.reservoir.available))
    # We select main collection if log_L < log_L_thread or the reservoir is empty, but not if all samples are merged.
    # It is not possible that all main samples are merged and the reservoir is empty (see next_done def.)
    select_main = (jnp.less(log_L_main, log_L_thread) | reservoir_empty) & jnp.bitwise_not(all_main_samples_merged)
    # when all main samples are merged, then the number of num_live_points are only from thread
    next_log_L = jnp.where(select_main,
                           log_L_main,
                           log_L_thread)
    next_point_X = tree_map(lambda x, y: jnp.where(select_main, x[i_main], y[idx_min_thread]),
                            sample_collection_main.points_X,
                            state.reservoir.points_X)
    next_point_U = jnp.where(select_main,
                             sample_collection_main.points_U[i_main],
                             state.reservoir.points_U[idx_min_thread])
    next_num_likelihood_evaluations = jnp.where(select_main,
                                                sample_collection_main.num_likelihood_evaluations[i_main],
                                                state.reservoir.num_likelihood_evaluations[idx_min_thread])

    # Get log_dZ_mean, and log_X_mean
    next_evidence_calculation = _update_evidence_calculation(num_live_points=next_num_live_points,
                                                             log_L=state.log_L_contour,
                                                             next_log_L_contour=next_log_L,
                                                             evidence_calculation=state.evidence_calculation)

    dZ_mean = (LogSpace(next_evidence_calculation.log_Z_mean)
               - LogSpace(state.evidence_calculation.log_Z_mean)).abs()

    def _update(operand, update):
        start_indices = [state.sample_idx] + [jnp.asarray(0, jnp.int_)] * len(update.shape)
        return dynamic_update_slice(operand, update[None].astype(operand.dtype), start_indices)

    next_sample_collection = \
        tree_map(_update,
                 state.sample_collection,
                 SampleCollection(
                     # The XUvalued sample
                     points_U=next_point_U,
                     # The X-valued sample
                     points_X=next_point_X,
                     # The log-likelihood of sample.
                     log_L=next_log_L,
                     # How many likelihood evaluations were required to obtain sample
                     num_likelihood_evaluations=next_num_likelihood_evaluations,
                     # The log mean evidence difference of the sample
                     log_dZ_mean=dZ_mean.log_abs_val,
                     # The log mean enclosed prior volume of sample
                     log_X_mean=next_evidence_calculation.log_X_mean,
                     # How many live points were taken for the sample.
                     num_live_points=next_num_live_points
                 ))

    next_reservoir = state.reservoir._replace(
        available=dynamic_update_slice(state.reservoir.available,
                                       jnp.where(select_main,
                                                 state.reservoir.available[idx_min_thread][None],
                                                 jnp.asarray([False])),
                                       idx_min_thread[None]))

    state = state._replace(sample_collection=next_sample_collection,
                           evidence_calculation=next_evidence_calculation,
                           reservoir=next_reservoir,
                           log_L_contour=next_log_L,
                           sample_idx=state.sample_idx + jnp.ones_like(state.sample_idx),
                           step_idx=state.step_idx + jnp.ones_like(state.step_idx))

    next_i_main = jnp.where(select_main,
                            i_main + jnp.ones_like(i_main),
                            i_main)

    return state, next_i_main


def _consume_remaining_samples(state: NestedSamplerState, prev_state: NestedSamplerState, i_main:jnp.ndarray) -> NestedSamplerState:
    def thread_cond(body_state: Tuple[NestedSamplerState, jnp.ndarray]):
        (state, i_main) = body_state
        return jnp.bitwise_not(state.done)

    def thread_body(body_state: Tuple[NestedSamplerState, jnp.ndarray]) -> Tuple[NestedSamplerState, jnp.ndarray]:
        (state, i_main) = body_state
        next_state, i_main = _collect_sample(state, prev_state, i_main)
        next_reservoir_empty = jnp.bitwise_not(jnp.any(next_state.reservoir.available))
        next_out_of_space = next_state.sample_idx == state.sample_collection.log_L.size
        all_samples_merged = i_main == prev_state.sample_idx
        done = (next_reservoir_empty & all_samples_merged) | next_out_of_space
        next_state = next_state._replace(done=done)
        return (next_state, i_main)

    # on each thread loop, we require that we collect some new point, or else we can get in an infinite loop.
    reservoir_empty = jnp.bitwise_not(jnp.any(state.reservoir.available))
    state = state._replace(done=reservoir_empty)
    (final_state, _) = while_loop(thread_cond,
                             thread_body,
                                  (state, i_main))

    return final_state

def _run_thread(prev_state: NestedSamplerState,
                init_thread_state: NestedSamplerState,
                log_L_exit: jnp.ndarray,
                reservoir_refiller: ReservoirRefiller) -> NestedSamplerState:

    def thread_cond(thread_state: Tuple[NestedSamplerState, jnp.ndarray, jnp.ndarray]):
        # Keep going while reservoir keeps getting filled, or while there are still old points to pick up.
        # We ensure that if we are only filling old points, that we're not trying to fill from the empty reservoir.
        (state, _, _) = thread_state
        return jnp.bitwise_not(state.done)

    def thread_body(thread_state: Tuple[NestedSamplerState, jnp.ndarray, jnp.ndarray]) \
            -> Tuple[NestedSamplerState, jnp.ndarray, jnp.ndarray]:
        (thread_state, i_main, on_plateau) = thread_state

        # We refill if we are not moving toward empty (plateau or thread goal reached), or new thread
        thread_reached_goal = thread_state.log_L_contour >= log_L_exit
        refill_thread = jnp.bitwise_not(thread_reached_goal) & jnp.bitwise_not(on_plateau)
        # We accept points on contour if there are not any satisfying points (plateau) and we demand refill.
        reservoir_has_any_satisfying = jnp.any(thread_state.reservoir.log_L > thread_state.log_L_contour)
        strict_contour = reservoir_has_any_satisfying
        # draw samples to at least goal_num_live_points, else no (not much) work
        thread_state = reservoir_refiller(
            state=thread_state,
            refill_thread=refill_thread,
            strict_contour=strict_contour)

        next_state, next_i_main = _collect_sample(thread_state, prev_state, i_main)
        next_on_plateau = thread_state.log_L_contour == next_state.log_L_contour
        next_out_of_space = next_state.sample_idx == thread_state.sample_collection.log_L.size
        all_samples_merged = next_i_main == prev_state.sample_idx
        reservoir_empty = jnp.bitwise_not(jnp.any(next_state.reservoir.available))
        thread_reached_exit_goal = next_state.log_L_contour >= log_L_exit
        done = thread_reached_exit_goal | (reservoir_empty & all_samples_merged) | next_out_of_space
        # set this, so that when we are done it's set correctly
        next_state = next_state._replace(done=done)
        return (next_state, next_i_main, next_on_plateau)

    # on each thread loop, we require that we collect some new point, or else we can get in an infinite loop.
    init_i_main = init_thread_state.sample_idx
    (final_state, i_main, _) = while_loop(thread_cond,
                                     thread_body,
                                     (init_thread_state, init_i_main, jnp.asarray(False, jnp.bool_)))

    final_state = _consume_remaining_samples(final_state, prev_state, i_main)

    return final_state


def set_thread_stats(prev_state, merged_state, log_L_entry, log_L_exit, init_i_main):
    _, prev_evidence_var = linear_to_log_stats(prev_state.evidence_calculation.log_Z_mean,
                                               log_f2_mean=prev_state.evidence_calculation.log_Z2_mean)
    _, merged_evidence_var = linear_to_log_stats(merged_state.evidence_calculation.log_Z_mean,
                                                 log_f2_mean=merged_state.evidence_calculation.log_Z2_mean)
    prev_ess = LogSpace(prev_state.evidence_calculation.log_Z_mean).square() / LogSpace(
        prev_state.evidence_calculation.log_dZ2_mean)
    merged_ess = LogSpace(merged_state.evidence_calculation.log_Z_mean).square() / LogSpace(
        merged_state.evidence_calculation.log_dZ2_mean)
    num_samples = merged_state.sample_idx - prev_state.sample_idx
    num_likelihood_evaluations = jnp.sum(merged_state.sample_collection.num_likelihood_evaluations) \
                                 - jnp.sum(prev_state.sample_collection.num_likelihood_evaluations)
    thread_stat_update = ThreadStats(evidence_uncert_diff=jnp.sqrt(prev_evidence_var)-jnp.sqrt(merged_evidence_var),
                                     ess_diff=(merged_ess - prev_ess).value,
                                     init_i_main=init_i_main,
                                     log_L_entry=log_L_entry,
                                     log_L_exit=log_L_exit,
                                     num_samples=num_samples,
                                     num_likelihood_evaluations=num_likelihood_evaluations)
    thread_stats = tree_map(lambda operand, update: dynamic_update_slice(operand,
                                                                         update.astype(operand.dtype)[None],
                                                                         [merged_state.thread_idx] + [
                                                                             jnp.asarray(0, jnp.int_)] * len(
                                                                             update.shape)),
                            merged_state.thread_stats,
                            thread_stat_update)
    merged_state = merged_state._replace(thread_idx=merged_state.thread_idx + jnp.ones_like(prev_state.thread_idx),
                                         thread_stats=thread_stats)
    return merged_state


def _dynamic_run(prior_chain: PriorChain,
                 loglikelihood_from_U: Callable[[Dict[str, jnp.ndarray]], jnp.ndarray],
                 state: NestedSamplerState,
                 delta_num_live_points: jnp.ndarray,
                 sampler_name: str,
                 sampler_kwargs: Dict[str, Any],
                 termination_ess=None,
                 termination_likelihood_contour=None,
                 termination_evidence_uncert=None,
                 termination_max_num_threads=None,
                 f: jnp.ndarray = None,
                 G: jnp.ndarray = None):
    if f is None:
        raise ValueError("When `dynamic`=True `f` can't be None.")
    if G is None:
        raise ValueError("When `dynamic`=True `G` can't be None.")
    
    def control_cond(control_state: NestedSamplerState):
        # while the termination conditions have not been met, keep going
        state = control_state
        return jnp.bitwise_not(state.done)

    def control_body(control_state: NestedSamplerState) -> NestedSamplerState:
        prev_state = control_state
        # get likelihood range parameters, based on the type of iteration
        # init_i_main - the index of starting contour in samples,
        # [a, b, l*, c] -> 2 -> 3 merged samples already, next sample replaced at 3, first old sample idx at 3
        # l*, [a, b, c] -> -1 -> 0 merged samples already, next sample replaced at 0, first old sample idx at Nan
        
        log_L_entry, log_L_exit = _get_dynamic_goal(prev_state,
                                                    f=f,
                                                    G=G)

        init_thread_state = create_initial_thread_state(prev_state, log_L_entry)

        reservoir_refiller = ReservoirRefiller(prior_chain=prior_chain,
                                               loglikelihood_from_U=loglikelihood_from_U,
                                               goal_num_live_points=delta_num_live_points,
                                               sampler_name=sampler_name,
                                               sampler_kwargs=sampler_kwargs)
        # TODO: Suitable to use more parallel operation here, given the large number of iterations to do.
        # refill the reservoir from empty to full
        strict_contour = jnp.any(init_thread_state.reservoir.log_L > log_L_entry)
        init_thread_state = init_thread_state._replace(log_L_contour=log_L_entry)
        init_thread_state = reservoir_refiller(state=init_thread_state,
                                               refill_thread=jnp.asarray(True),
                                               strict_contour=strict_contour)
        init_thread_state = init_thread_state._replace(log_L_contour=-jnp.inf)

        merged_state = _run_thread(prev_state=prev_state,
                                   init_thread_state=init_thread_state,
                                   log_L_exit=log_L_exit,
                                   reservoir_refiller=reservoir_refiller)

        merged_state = set_thread_stats(prev_state, merged_state, log_L_entry, log_L_exit,init_thread_state.sample_idx)

        # Determine if we are done.
        done, termination_reason = termination_condition(merged_state,
                                                         termination_ess=termination_ess,
                                                         termination_likelihood_contour=termination_likelihood_contour,
                                                         termination_evidence_uncert=termination_evidence_uncert,
                                                         termination_max_num_threads=termination_max_num_threads)

        merged_state = merged_state._replace(done=done, termination_reason=termination_reason)
        return merged_state

    state = state._replace(done=jnp.asarray(False, jnp.bool_),
                           termination_reason=jnp.zeros_like(state.termination_reason))

    final_state = while_loop(control_cond,
                             control_body,
                             state)

    return final_state
