from functools import partial

import jax.numpy as jnp
import jax.random

from jaxns.nested_samplers.model import Model
from jaxns.nested_samplers.mixed_precision import mp_policy
from jaxns.nested_samplers.constrained_sampler import AbstractSampler
from jaxns.nested_samplers.samples import Samples
from jaxns.nested_samplers.state import State
from jaxns.nested_samplers.termination_condition import TerminationCondition, TerminationRegister
from jaxns.nested_samplers.types import FloatArray, IntArray

"""
We formulate nested sampling as an exponential race of lineages.

All node indices are in sorted order, so i < j implies L(i) >= L(j) (where = only for plateaus).
We handle pleateaus by marginalising the race over permutations within plateaus.

A parent graph is defined,

    p(i) -> i means that i is a child of p(i), which mans i was sampled from {L > L(p(i))}.
    
A lineage is maximal chain of nodes where p(i_{k}}) = i_{k-1}.

A dummy root node, 0, is defined with L(0) = 0.

Suppose we have N samples, indexed 1,...,N, via argsort(L) + 1 (stable sort so within plateau ordering preserved).

define s_i = -log(X_i) where X_i is the prior volume associated with sample i. 
Set s_0 = 0.

Define the number of active children of node i at index j>i, as C(i,j) = |{ k : p(k) = i, k >= j}|.

Define the active parent set A(i-1) = {j : j=1..i-1, C(j,i) > 0} which is the set of nodes before i which have active children at or after i.

Each parent in A(i-1) defines a lineage, which at least one active child at or after i.

For node i, sample ds_i ~ Exponential(K(i)) where K(i)=sum_{j in A(i-1)} C(j,i) is the total number of active children over all active lineages A(i-1).

The principle property of exponential races is that the minimum of independent exponentials is itself an exponential with rate equal to the sum of the rates.

So we set s_i = s_{i-1} + ds_i, and assign the race winner to the observed winning lineage (unconditionally we would need to sample which lineage won, but nested sampling conditions on this).
In the case of plateaus we marginalise over the permutations within the plateau, which is equivalent to assigning winners in every possible order and averaging the results.
We rely on permutations and stable sorting to ensure that the order of nodes within plateaus is consistent across all computations.

Now, we don't need to actually compute C(j,i), nor maintain A(i-1) explicitly.
We can maintain a running count of the number of active children for each node, and update this as we move through the samples.
Define out-degree d(i) = |{ j : p(j) = i }|, and set d(0) = K, which K is the number of live points.

When we process node i, we have K(i) = sum_{j in A(i-1)} C(j,i) as the total number of active children over all active lineages.
We then decrement C(p(i),i+1) = C(p(i),i) - 1, removing one active child from the parent lineage of i, and then add d(i) active children to C(i+1) = C(i) + d(i).

So in total after consuming node i, perform the update:

    K(i) = K(i-1) - 1 + d(i)
    
Thus, we only need to maintain d(i) for each node, and not actually the graph {p(i)->i}, since the only thing we do with that graph is compute the out-degrees d(i).

We are thus free to arbitrarily add samples to the graph from any node in any lineage, including the dummy node. 
This implies creating new lineages is possibly by creating d(i) > 1 for some i.

We have two modalities of evidence estimation: expectation-based and sampling-based.

In the sampling based approach, we compute the trajectory of s_i many times, and then x_i = exp(-s_i) for each sample, and use these to sample evidence.
When doing this for each sample we permutate the samples first, after which stable sorting preserves the that permuatation within plateaus.
Thus each evidence sample corresponds to a different trajectory of s_i handling plateaus correctly.

For the expectation-based approach, we track several sufficient statistics to compute E[logZ], Var[logZ], E[logH] etc, marginalising over permutations within plateaus.
These are ...

Now the above tells us how to compute the prior volumes X_i associated with each sample i, but we now turn to how to sample.

We begin with a dummy node 0, and K children sampled iid from the prior (constrained to L>L(0)), forming the initial live point set.
We then apply the above tracking of K(i) in an online manner starting from node 0 with K(0) = K.
We introduce parallelism by discarding m in (1, K/2) points at a time, and outsourcing the sampling of their replacements to different devices.
That is we apply the above logic, to sequentially select nodes 1..m, each time updating K(i) = K(i-1) - 1 + d(i) as above, and updating the sufficient statistics for evidence estimation after each discard.
Once we have selected m nodes to discard, we send their replacement sampling to different devices, sampling from within the last discarded point {L>L(m)}.
"""

@partial(jax.jit, inline=True, static_argnames=['num_live_points', 'max_samples', 'batch_size'])
def sample_init_state(key, num_live_points: int, max_samples: int, model: Model, args=(),
                      params=None,
                      batch_size: int | None = None) -> State:
    def single_sample(key):
        key, subkey = jax.random.split(key)
        U_sample = model.sample_U(subkey)
        log_L = model.log_likelihood(U_sample, args=args, params=params, allow_nan=False)
        num_likelihood_evaluations = jnp.array(1, dtype=mp_policy.count_dtype)
        carry = (key, U_sample, log_L, num_likelihood_evaluations)

        def cond_fn(carry):
            _, _, log_L, _ = carry
            return log_L <= -jnp.inf

        def body_fn(carry):
            key, _, _, num_likelihood_evaluations = carry
            key, subkey = jax.random.split(key)
            U_sample = model.sample_U(subkey)
            log_L = model.log_likelihood(U_sample, args=args, params=params, allow_nan=False)
            num_likelihood_evaluations += jnp.ones_like(num_likelihood_evaluations)
            return (key, U_sample, log_L, num_likelihood_evaluations)

        key, U_sample, log_L, num_likelihood_evaluations = jax.lax.while_loop(cond_fn, body_fn, carry)
        return U_sample, log_L, num_likelihood_evaluations

    U_samples, log_likelihoods, num_likelihood_evaluations = jax.lax.map(
        single_sample,
        jax.random.split(key, num_live_points),
        batch_size=batch_size
    )

    # extend each to max_samples
    def _concat(x, fill_value):
        return jax.tree.map(
            lambda x: jnp.concatenate([
                x,
                jnp.full((max_samples - num_live_points,) + x.shape[1:], fill_value, dtype=x.dtype)
            ], axis=0),
            x
        )

    U_samples = _concat(U_samples, 0.0)
    log_likelihoods = _concat(log_likelihoods, jnp.inf)
    num_likelihood_evaluations = _concat(num_likelihood_evaluations, 0)
    out_degree = jnp.zeros((max_samples,), dtype=mp_policy.count_dtype)
    samples = Samples(
        log_likelihoods=log_likelihoods,
        U_samples=U_samples,
        out_degree=out_degree,
        num_likelihood_evaluations=num_likelihood_evaluations
    )
    state = State(
        root_out_degree=jnp.array(num_live_points, dtype=mp_policy.count_dtype),
        samples=samples,
        num_samples=jnp.array(num_live_points, dtype=mp_policy.count_dtype)
    )
    # Sort samples into increasing log-likelihood order
    return State(
        root_out_degree=state.root_out_degree,
        samples=state.samples.sort(),
        num_samples=state.num_samples
    )


def single_ns_run(key, root_out_degree: int, max_samples: int, shell_size: int, model: Model, args=(),
                  sampler: AbstractSampler | None = None,
                  params=None,
                  termination_condition: TerminationCondition | None = None,
                  batch_size: int | None = None) -> State:
    """
    Perform a single nested sampling run, using a shell-based parallel nested sampling algorithm.

    The branching strategy is as follows:
    0 -> {r(1),...,r(K)}
    S -> {r(1) ,..., r(S)}
    2S -> {r(S), ..., r(2S)}
    ...
    nS -> {r((n-1)S), ..., r(nS)}

    where r(i) is the likelihood rank of the i-th sample.


    Args:
        key: PRNG key
        root_out_degree: the number of live points to use off root
        max_samples: the maximum number of samples to store
        shell_size: the number of samples to discard and replenish per iteration
        model: the model to sample from
        args: arguments to pass to the model
        sampler: the sampler to use to produce i.i.d. samples within likelihood constraints
        params: parameters to pass to the model
        termination_condition: the termination condition to use
        batch_size: how many likelihood evaluations to batch together

    Returns:
        A final state object containing all samples and relevant information for resuming, or evidence calculation.
    """
    # Initialise the state
    key, init_key = jax.random.split(key)
    state = sample_init_state(
        key=init_key,
        num_live_points=root_out_degree,
        max_samples=max_samples,
        model=model,
        args=args,
        params=params,
        batch_size=batch_size
    )

    # Initialize register
    register = TerminationRegister.initialise()
    register.num_samples_used = root_out_degree
    register.num_likelihood_evaluations = jnp.sum(state.samples.num_likelihood_evaluations)
    # register.log_L_contour = register.evidence_calc.L.log_abs_val
    register.efficiency = register.num_samples_used / register.num_likelihood_evaluations
    register.plateau = (log_L0 := state.samples.log_likelihoods[0]) == (
        log_L1 := state.samples.log_likelihoods[root_out_degree - 1])
    register.no_seed_points = register.plateau
    register.absolute_spread = jnp.abs(log_L1 - log_L0)
    register.relative_spread = 2. * register.absolute_spread / jnp.abs(log_L0 + log_L1)
    register.cummax_log_XL = jnp.maximum(register.cummax_log_XL,
                                         (register.evidence_calc.X_mean * register.evidence_calc.L).log_abs_val)
    register.evidence_calc_with_remaining = register.evidence_calc
    K_total_tmp = state.root_out_degree
    for idx in range(root_out_degree):
        register.evidence_calc_with_remaining = register.evidence_calc_with_remaining.update_evidence(
            K_total_tmp,
            state.samples.log_likelihoods[idx]
        )
        K_total_tmp = K_total_tmp - 1 + state.samples.out_degree[idx]

    # Sequentially discard shell, and replenish until termination condition
    discard_idx = 0
    K_total = state.root_out_degree
    done = register.is_done(termination_condition)
    while not done:
        for _ in range(shell_size):
            # Partial register update per discard
            register.evidence_calc = register.evidence_calc.update_evidence(
                K_total, state.samples.log_likelihoods[discard_idx])
            K_total = K_total - 1 + state.samples.out_degree[discard_idx]
            discard_idx += 1

        # Last discarded sample sets the likelihood constraint
        parent_idx = discard_idx - 1
        insert_idx = discard_idx

        # Replenish discarded samples, by merging and sorting with active samples
        kept_size = root_out_degree - shell_size
        active_samples = state.samples.slice(insert_idx, kept_size)

        log_L_constraint = state.samples.log_likelihoods[parent_idx]
        # When there are no seeds, we reparent off the root.
        # However, since this retroactively changes the out-degree of the root,
        # the evidence calculation needs to be recalculated.
        # It also becomes unclear where to continue sampling from, since there were no seeds in the active set.
        # We therefore stop the run if there are no seeds.
        # The user can then use adaptive refinement to continue improving if desired.
        no_seeds = jnp.all(active_samples.log_likelihoods <= log_L_constraint)
        log_L_constraint = jnp.where(no_seeds, jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype), log_L_constraint)
        delta_root_out_degree = jnp.where(no_seeds, shell_size, 0).astype(mp_policy.count_dtype)
        delta_parent_out_degree = jnp.where(no_seeds, 0, shell_size).astype(mp_policy.count_dtype)

        # TODO: modularise out the sampling distribution strategy.

        def get_sample(key, log_L_constraint, active_samples: Samples):
            seed_key, sample_key = jax.random.split(key)
            seed_select_idx = jax.random.randint(seed_key, (), 0, len(active_samples))
            seed_point = active_samples.U_samples[seed_select_idx]
            return sampler.get_sample(
                sample_key, log_L_constraint, seed_point
            )

        key, subkey = jax.random.split(key)
        U_samples, log_likelihoods, num_likelihood_evaluations = jax.lax.map(
            lambda key: get_sample(key, log_L_constraint, active_samples),
            jax.random.split(subkey, shell_size),
            batch_size=batch_size
        )

        new_samples = Samples(
            log_likelihoods=log_likelihoods,
            U_samples=U_samples,
            out_degree=jnp.zeros((shell_size,), dtype=mp_policy.count_dtype),
            num_likelihood_evaluations=num_likelihood_evaluations
        )
        # If no reparenting off root, then this is sufficient to maintain ordering.
        joint_samples = active_samples.concat(new_samples).sort()
        assert len(joint_samples) == root_out_degree
        state.samples = state.samples.append_samples(
            insert_idx=insert_idx,
            parent_idx=parent_idx,
            samples=new_samples,
            delta_parent_out_degree=delta_parent_out_degree
        )
        state.num_samples += shell_size
        state.root_out_degree += delta_root_out_degree
        K_total += delta_root_out_degree

        # Update register now rest of the way (post sampling)
        register.num_samples_used = state.num_samples
        register.num_likelihood_evaluations += delta_num_like_evals := jnp.sum(new_samples.num_likelihood_evaluations)
        register.log_L_contour = register.evidence_calc.L.log_abs_val
        register.efficiency = 0.1 * register.efficiency + 0.9 * (shell_size / delta_num_like_evals)
        register.plateau = (log_L0 := joint_samples.log_likelihoods[0]) == (log_L1 := joint_samples.log_likelihoods[-1])
        register.no_seed_points = no_seeds
        register.absolute_spread = jnp.abs(log_L1 - log_L0)
        register.relative_spread = 2. * register.absolute_spread / jnp.abs(log_L0 + log_L1)
        register.cummax_log_XL = jnp.maximum(register.cummax_log_XL,
                                             (register.evidence_calc.X_mean * register.evidence_calc.L).log_abs_val)

        K_total_tmp = K_total
        register.evidence_calc_with_remaining = register.evidence_calc
        for idx in range(len(joint_samples)):
            register.evidence_calc_with_remaining = register.evidence_calc_with_remaining.update_evidence(
                K_total_tmp,
                joint_samples.log_likelihoods[idx]
            )
            K_total_tmp = K_total_tmp - 1 + joint_samples.out_degree[idx]

        # Check termination condition
        done = register.is_done(termination_condition)

    state.samples = state.samples.sort()
    return state


def resume_ns_run(key, state: State, log_L_start: FloatArray, log_L_end: FloatArray, model: Model, args=(),
                  sampler: AbstractSampler | None = None,
                  params=None,
                  max_samples: int | None = None) -> State:
    ...


def refine_ns_run(
        key, state: State,
        log_L_start: FloatArray, log_L_end: FloatArray,
        target_live_points: IntArray,
        model: Model, args=(), sampler: AbstractSampler | None = None,
        params=None,
        max_samples: int | None = None,
        batch_size: int | None = None
) -> State:
    """
    Adds children off parents in the specified log-likelihood range, until each contour within that range has at least
    the target number of live points, or until there are no seed points.
    Greedily adds children off the lowest live point parents first.

    Args:
        key: PRNG key
        state: the current state of the nested sampling run
        log_L_start: the starting log-likelihood contour to refine from
        log_L_end: the ending log-likelihood contour to refine to
        target_live_points: the target number of live points per contour
        model: the model to sample from
        args: arguments to pass to the model
        sampler: the sampler to use to produce i.i.d. samples within likelihood constraints
        params: parameters to pass to the model
        max_samples: the maximum number of samples to store
        batch_size: how many likelihood evaluations to batch together

    Returns:
        A final state object containing all samples and relevant information for resuming, or evidence calculation.
    """
    if max_samples is not None:
        state.samples = state.samples.resize(max_samples)
    done = ...
    while not done:
        state.samples = state.samples.sort()
        K_total_per_sample = state.samples.compute_num_live_points_per_sample(
            root_out_degree=state.root_out_degree,
            num_samples=state.num_samples
        )  # [max_samples]
        sample_mask = jnp.logical_and(
            K_total_per_sample < target_live_points,
            (state.samples.log_likelihoods >= log_L_start) &
            (state.samples.log_likelihoods <= log_L_end)
        )
        key, select_key = jax.random.split(key)
        parent_idxs = jax.random.choice(
            select_key, len(state.samples), replace=True,
            shape=(batch_size,),
            p=jnp.where(
                sample_mask.astype(jnp.float32),
                target_live_points - K_total_per_sample,
                0
            )
        )
        empty_sample_mask = ~jnp.any(sample_mask)
        log_L_constraints = jnp.where(
            empty_sample_mask,
            state.samples.log_likelihoods[parent_idxs],
            -jnp.inf
        )

        def get_sample(key, log_L_constraint, active_samples: Samples):
            seed_key, sample_key = jax.random.split(key)
            seed_select_idx = jax.random.randint(seed_key, (), 0, len(active_samples))
            seed_point = active_samples.U_samples[seed_select_idx]
            return sampler.get_sample(
                sample_key, log_L_constraint, seed_point
            )

        key, subkey = jax.random.split(key)
        U_samples, log_likelihoods, num_likelihood_evaluations = jax.lax.map(
            lambda key, log_L_constraint: get_sample(key, log_L_constraint, active_samples),
            jax.random.split(subkey, shell_size),
            batch_size=batch_size
        )

        new_samples = Samples(
            log_likelihoods=log_likelihoods,
            U_samples=U_samples,
            out_degree=jnp.zeros((shell_size,), dtype=mp_policy.count_dtype),
            num_likelihood_evaluations=num_likelihood_evaluations
        )
