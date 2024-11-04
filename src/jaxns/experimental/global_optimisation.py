import dataclasses
import io
from typing import NamedTuple, Optional, Union, TextIO, Tuple

import jax.numpy as jnp
import numpy as np
import pylab as plt
from jaxlib import xla_client
from jaxopt import NonlinearCG

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.constraint_bijections import quick_unit, quick_unit_inverse
from jaxns.internals.mixed_precision import mp_policy
from jaxns.internals.types import PRNGKey, IntArray, UType, FloatArray, LikelihoodInputType, XType
from jaxns.nested_samplers import ShardedStaticNestedSampler
from jaxns.nested_samplers.common.types import TerminationCondition, SampleCollection
from jaxns.samplers.abc import AbstractSampler
from jaxns.utils import _bit_mask

__all__ = [
    'GlobalOptimisationResults',
    'GlobalOptimisationTerminationCondition',
    'GlobalOptimisationState',
    'SimpleGlobalOptimisation'
]


class GlobalOptimisationState(NamedTuple):
    key: PRNGKey
    samples: SampleCollection
    num_samples: IntArray
    relative_spread: FloatArray
    absolute_spread: FloatArray
    num_likelihood_evaluations: IntArray


class GlobalOptimisationResults(NamedTuple):
    U_solution: UType
    X_solution: XType
    solution: LikelihoodInputType
    log_L_solution: FloatArray
    log_L_progress: FloatArray
    num_likelihood_evaluations: IntArray
    num_samples: IntArray
    termination_reason: IntArray
    relative_spread: FloatArray
    absolute_spread: FloatArray


class GlobalOptimisationTerminationCondition(NamedTuple):
    max_likelihood_evaluations: Optional[IntArray] = None  # jnp.asarray(jnp.iinfo(int_type).max, int_type)
    log_likelihood_contour: Optional[FloatArray] = None  # jnp.asarray(jnp.finfo(float_type).max, float_type)
    rtol: Optional[FloatArray] = None  # jnp.asarray(0., float_type)
    atol: Optional[FloatArray] = None  # jnp.asarray(0., float_type)
    min_efficiency: Optional[FloatArray] = None  # jnp.asarray(0., float_type)


def gradient_based_optimisation(model: BaseAbstractModel, init_U_point: UType) -> Tuple[UType, FloatArray, IntArray]:
    def loss(U_unconstrained: UType):
        U = quick_unit(U_unconstrained)
        return -model.log_prob_likelihood(U, allow_nan=False)

    solver = NonlinearCG(
        fun=loss,
        jit=True,
        unroll=False,
        verbose=False
    )

    results = solver.run(init_params=quick_unit_inverse(init_U_point))
    return quick_unit(results.params), -results.state.value, results.state.num_fun_eval


@dataclasses.dataclass(eq=False)
class SimpleGlobalOptimisation:
    """
    Simple global optimisation leveraging building blocks of nested sampling.
    """
    sampler: AbstractSampler
    num_search_chains: int
    model: BaseAbstractModel
    shell_frac: float = 0.5
    devices: Optional[xla_client.Device] = None
    verbose: bool = False

    def __post_init__(self):
        if self.num_search_chains < 1:
            raise ValueError("num_search_chains must be >= 1.")
        self.num_search_chains = int(self.num_search_chains)

        self._nested_sampler = ShardedStaticNestedSampler(
            model=self.model,
            max_samples=self.num_search_chains * 10,
            init_efficiency_threshold=0.1,
            sampler=self.sampler,
            num_live_points=self.num_search_chains,
            shell_fraction=self.shell_frac,
            devices=self.devices,
            verbose=self.verbose
        )

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
        is_sample_mask = jnp.arange(np.shape(state.samples.log_L)[0], dtype=mp_policy.index_dtype) < state.num_samples
        log_L_masked = jnp.where(is_sample_mask, state.samples.log_L, jnp.asarray(jnp.nan, mp_policy.measure_dtype))
        best_idx = jnp.nanargmax(log_L_masked)
        U_solution = state.samples.U_samples[best_idx]
        X_solution = self.model.transform(U_solution)
        solution = self.model.prepare_input(U_solution)  # The output of prior_model is solution
        log_L_solution = state.samples.log_L[best_idx]
        num_likelihood_evaluations = state.num_likelihood_evaluations
        log_L_progress = jnp.sort(log_L_masked)  # Low to high likelihoods, nan's at the end.
        return GlobalOptimisationResults(
            U_solution=U_solution,
            X_solution=X_solution,
            solution=solution,
            log_L_solution=log_L_solution,
            log_L_progress=log_L_progress,
            num_likelihood_evaluations=num_likelihood_evaluations,
            num_samples=state.num_samples,
            relative_spread=state.relative_spread,
            absolute_spread=state.absolute_spread,
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

        termination_reason, termination_register, state = self._nested_sampler._run(
            key=key,
            term_cond=TerminationCondition(
                max_num_likelihood_evaluations=term_cond.max_likelihood_evaluations,
                log_L_contour=term_cond.log_likelihood_contour,
                efficiency_threshold=term_cond.min_efficiency,
                atol=term_cond.atol,
                rtol=term_cond.rtol,
                max_samples=None  # Turn off max samples for global optimisation, wraps index
            )
        )

        go_state = GlobalOptimisationState(
            key=state.key,
            samples=state.sample_collection,
            num_samples=state.num_samples,
            absolute_spread=termination_register.absolute_spread,
            relative_spread=termination_register.relative_spread,
            num_likelihood_evaluations=termination_register.num_likelihood_evaluations
        )
        return termination_reason, go_state


def plot_progress(results: GlobalOptimisationResults, save_file: Optional[str] = None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    start_sample_idx = max(0, results.num_samples - int(np.shape(results.log_L_progress)[0]))
    end_sample_idx = int(results.num_samples)

    ax.plot(
        np.arange(start_sample_idx, end_sample_idx),
        results.log_L_progress
    )
    ax.set_title(f"Optimisation progress (samples {start_sample_idx} to {end_sample_idx})")
    ax.set_xlabel("Sample index")
    ax.set_ylabel(r"Objective ($\log L$)")
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def go_summary(results: GlobalOptimisationResults, f_obj: Optional[Union[str, TextIO]] = None):
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
        termination_bit_mask = _bit_mask(int(_termination_reason), width=11)
        for bit, condition in zip(termination_bit_mask, [
            'Reached max samples',
            'Evidence uncertainty low enough',
            'Small remaining evidence',
            'Reached ESS',
            "Used max num likelihood evaluations",
            'Likelihood contour reached',
            'Sampler efficiency too low',
            'All live-points are on a single plateau (sign of possible precision error)',
            'relative spread of live points < rtol',
            'absolute spread of live points < atol',
            'no seed points left (consider decreasing shell_fraction)'
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

        # For shaped arrays, we want to print "{name}[{i0, i1, ..., in}]" for all valid indices.

        if np.size(X_solution[name]) == 0:
            continue

        _print("--------")

        shape = np.shape(X_solution[name])
        num_dims = len(shape)
        is_shaped = num_dims > 0

        var_name = f"{name}[{','.join(['#'] * num_dims)}]" if is_shaped else name
        _print(
            f"{var_name}: max(L) est."
        )
        if is_shaped:
            indices = np.indices(shape).reshape((num_dims, -1)).T

            for inds in indices:
                _max_like_point = X_solution[name][tuple(inds)]
                _print(f"{name}[{','.join(str(i) for i in inds)}]: {_round(_max_like_point, 0.1 * _max_like_point)}")
        else:
            _max_like_point = X_solution[name]
            _print(f"{name}: {_round(_max_like_point, 0.1 * _max_like_point)}")

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
