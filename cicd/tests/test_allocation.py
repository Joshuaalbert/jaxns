from __future__ import annotations

import inspect

import numpy as np
import pytest
from jax import numpy as jnp

from cicd.tests.core_fixtures import make_state as _make_state
from jaxns import allocation
from jaxns.allocation import (
    VolumePath,
    evidence_improvement_utility,
    expected_volume_path,
    integer_allocation_targets,
    normalise_allocation_utility,
    posterior_improvement_utility,
    validate_allocation_target,
)
from jaxns.race_tree import build_block_state
from jaxns.shrinkage import (
    DirichletConcentrations,
    classic_dirichlet_concentrations,
)


@pytest.mark.parametrize(
    "allocation_target",
    [
        "uniform",
        "evidence_improving",
        "posterior_improving",
    ],
)
def test_validate_allocation_target_accepts_public_literals(allocation_target):
    assert validate_allocation_target(allocation_target) == allocation_target


def _reference_evidence_utility(
        likelihood: np.ndarray,
        X: np.ndarray,
        shell_mass: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
) -> np.ndarray:
    shell_dZ = likelihood * shell_mass
    Z = np.sum(shell_dZ)
    utilities = np.zeros_like(likelihood)
    reductions = np.zeros_like(likelihood)
    for h in range(likelihood.size):
        downstream = np.sum(shell_dZ[h + 1:])
        B_h = (likelihood[h] * X[h] - downstream) / Z
        reductions[h] = B_h ** 2 * (
                1.0 / alpha[h] ** 2
                - 1.0 / (alpha[h] + beta[h]) ** 2
        )
    for g in range(likelihood.size):
        if X[g] > 0.0:
            utilities[g] = np.sum(X[g + 1:] * reductions[g + 1:]) / X[g]
    return utilities


def _reference_posterior_delta(weights: np.ndarray, conservative: bool) -> np.ndarray:
    W = np.sum(weights)
    Q = np.sum(weights ** 2)
    delta = np.zeros_like(weights)
    for h, weight in enumerate(weights):
        if Q <= 0.0 or weight <= 0.0:
            continue
        if conservative:
            delta[h] = W ** 2 / (Q - weight ** 2 / 3.0) - W ** 2 / Q
            continue
        weight2 = weight ** 2
        delta[h] = W ** 2 * (
                np.arctan(np.sqrt(weight2 / (2.0 * Q - weight2)))
                / np.sqrt((Q - weight2 / 2.0) * (weight2 / 2.0))
                - 1.0 / Q
        )
    return delta


def _reference_posterior_utility(
        X: np.ndarray,
        shell_mass: np.ndarray,
        weights: np.ndarray,
        conservative: bool,
) -> np.ndarray:
    delta = _reference_posterior_delta(weights, conservative)
    utilities = np.zeros_like(weights)
    for g in range(weights.size):
        if X[g] > 0.0:
            utilities[g] = np.sum(shell_mass[g + 1:] * delta[g + 1:]) / X[g]
    return utilities


def test_expected_volume_path_uses_dirichlet_p_gt_means():
    concentrations = DirichletConcentrations(
        alpha_gt=jnp.asarray([1.0, 2.0, 3.0]),
        alpha_eq=jnp.asarray([1.0, 1.0, 1.0]),
        alpha_lt=jnp.asarray([2.0, 1.0, 0.0]),
        epsilon=jnp.asarray([1e-6, 1e-6, 1e-6]),
    )

    path = expected_volume_path(concentrations)

    expected_p_gt = np.asarray([1.0 / 4.0, 2.0 / 4.0, 3.0 / 4.0])
    expected_X = np.cumprod(expected_p_gt)
    expected_X_prev = np.concatenate([[1.0], expected_X[:-1]])
    np.testing.assert_allclose(np.asarray(path.X), expected_X)
    np.testing.assert_allclose(np.asarray(path.X_prev), expected_X_prev)
    np.testing.assert_allclose(
        np.asarray(path.shell_mass),
        expected_X_prev - expected_X,
    )


def test_evidence_improvement_utility_matches_reference_loop():
    likelihood = np.asarray([1.0, 2.0, 4.0])
    path = VolumePath(
        X_prev=jnp.asarray([1.0, 0.5, 0.25]),
        X=jnp.asarray([0.5, 0.25, 0.125]),
        shell_mass=jnp.asarray([0.5, 0.25, 0.125]),
    )
    alpha = np.asarray([2.0, 3.0, 4.0])
    beta = np.asarray([1.5, 2.0, 2.5])

    utility = evidence_improvement_utility(
        log_L_blocks=jnp.log(jnp.asarray(likelihood)),
        volume_path=path,
        alpha_gt=jnp.asarray(alpha),
        beta_gt=jnp.asarray(beta),
    )

    expected = _reference_evidence_utility(
        likelihood=likelihood,
        X=np.asarray(path.X),
        shell_mass=np.asarray(path.shell_mass),
        alpha=alpha,
        beta=beta,
    )
    np.testing.assert_allclose(np.asarray(utility), expected, rtol=1e-6)
    assert float(utility[-1]) == 0.0


def test_posterior_improvement_utility_matches_exact_and_conservative_references():
    path = VolumePath(
        X_prev=jnp.asarray([1.0, 0.6, 0.3]),
        X=jnp.asarray([0.6, 0.3, 0.1]),
        shell_mass=jnp.asarray([0.4, 0.3, 0.2]),
    )
    weights = np.asarray([0.2, 0.5, 0.3])

    exact = posterior_improvement_utility(
        volume_path=path,
        shell_weights=jnp.asarray(weights),
    )
    conservative = posterior_improvement_utility(
        volume_path=path,
        shell_weights=jnp.asarray(weights),
        conservative=True,
    )

    np.testing.assert_allclose(
        np.asarray(exact),
        _reference_posterior_utility(
            X=np.asarray(path.X),
            shell_mass=np.asarray(path.shell_mass),
            weights=weights,
            conservative=False,
        ),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(conservative),
        _reference_posterior_utility(
            X=np.asarray(path.X),
            shell_mass=np.asarray(path.shell_mass),
            weights=weights,
            conservative=True,
        ),
        rtol=1e-6,
    )
    assert float(exact[-1]) == 0.0
    assert float(conservative[-1]) == 0.0


def test_posterior_improvement_zero_weight_limit_is_zero():
    path = VolumePath(
        X_prev=jnp.asarray([1.0, 0.5]),
        X=jnp.asarray([0.5, 0.25]),
        shell_mass=jnp.asarray([0.5, 0.25]),
    )

    exact = posterior_improvement_utility(path, jnp.asarray([1.0, 0.0]))
    conservative = posterior_improvement_utility(
        path,
        jnp.asarray([1.0, 0.0]),
        conservative=True,
    )

    np.testing.assert_allclose(np.asarray(exact), np.asarray([0.0, 0.0]))
    np.testing.assert_allclose(
        np.asarray(conservative),
        np.asarray([0.0, 0.0]),
    )


def test_posterior_improvement_tiny_positive_weight_limit_is_continuous():
    path = VolumePath(
        X_prev=jnp.asarray([1.0, 0.5]),
        X=jnp.asarray([0.5, 0.25]),
        shell_mass=jnp.asarray([0.5, 0.25]),
    )
    tiny_positive = np.finfo(np.float32).tiny

    exact = posterior_improvement_utility(
        path,
        jnp.asarray([1.0, tiny_positive]),
    )
    conservative = posterior_improvement_utility(
        path,
        jnp.asarray([1.0, tiny_positive]),
        conservative=True,
    )

    assert np.all(np.isfinite(np.asarray(exact)))
    assert np.all(np.isfinite(np.asarray(conservative)))
    np.testing.assert_allclose(
        np.asarray(exact),
        np.asarray([0.0, 0.0]),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(conservative),
        np.asarray([0.0, 0.0]),
        atol=1e-7,
    )


def test_posterior_improvement_zero_shell_mass_does_not_create_work():
    path = VolumePath(
        X_prev=jnp.asarray([1.0, 0.5, 0.5]),
        X=jnp.asarray([0.5, 0.5, 0.25]),
        shell_mass=jnp.asarray([0.5, 0.0, 0.25]),
    )

    utility = posterior_improvement_utility(
        volume_path=path,
        shell_weights=jnp.asarray([0.2, 10.0, 0.2]),
    )

    assert np.all(np.isfinite(np.asarray(utility)))
    assert float(utility[0]) == pytest.approx(float(utility[1]))
    assert float(utility[-1]) == 0.0


def test_allocation_utilities_handle_tiny_positive_volume_without_inf():
    tiny_x = np.finfo(np.float32).tiny
    path = VolumePath(
        X_prev=jnp.asarray([1.0, tiny_x, tiny_x]),
        X=jnp.asarray([tiny_x, tiny_x, 0.0]),
        shell_mass=jnp.asarray([1.0 - tiny_x, 0.0, tiny_x]),
    )

    evidence = evidence_improvement_utility(
        log_L_blocks=jnp.log(jnp.asarray([1.0, 2.0, 3.0])),
        volume_path=path,
        alpha_gt=jnp.asarray([2.0, 2.0, 2.0]),
        beta_gt=jnp.asarray([1.0, 1.0, 1.0]),
    )
    posterior = posterior_improvement_utility(
        volume_path=path,
        shell_weights=jnp.asarray([1.0, 0.0, tiny_x]),
    )

    assert np.all(np.isfinite(np.asarray(evidence)))
    assert np.all(np.asarray(evidence) >= 0.0)
    assert np.all(np.isfinite(np.asarray(posterior)))
    assert np.all(np.asarray(posterior) >= 0.0)


def test_evidence_improvement_nonpositive_evidence_falls_back_to_zero():
    path = VolumePath(
        X_prev=jnp.asarray([1.0, 1.0]),
        X=jnp.asarray([1.0, 1.0]),
        shell_mass=jnp.asarray([0.0, 0.0]),
    )

    utility = evidence_improvement_utility(
        log_L_blocks=jnp.asarray([-jnp.inf, -jnp.inf]),
        volume_path=path,
        alpha_gt=jnp.asarray([2.0, 2.0]),
        beta_gt=jnp.asarray([1.0, 1.0]),
    )

    np.testing.assert_array_equal(np.asarray(utility), np.asarray([0.0, 0.0]))


def test_utility_normalisation_and_integer_targets_are_deterministic():
    utility = jnp.asarray([np.nan, -1.0, 2.0, 1.0])
    valid = jnp.asarray([True, True, True, False])

    unit_peak = normalise_allocation_utility(utility, valid=valid)
    targets = integer_allocation_targets(
        root_out_degree=5,
        iteration=2,
        delta_K=3,
        unit_peak_utility=unit_peak,
        valid=valid,
        max_target=9,
    )

    np.testing.assert_allclose(
        np.asarray(unit_peak),
        np.asarray([0.0, 0.0, 1.0, 0.0]),
    )
    np.testing.assert_array_equal(np.asarray(targets), np.asarray([5, 5, 9, 0]))


def test_depth_first_uniform_targets_follow_k_delta_k_formula():
    targets = integer_allocation_targets(
        root_out_degree=4,
        iteration=3,
        delta_K=2,
        unit_peak_utility=jnp.ones((4,)),
        valid=jnp.asarray([True, True, False, True]),
    )

    np.testing.assert_array_equal(
        np.asarray(targets),
        np.asarray([10, 10, 0, 10]),
    )


@pytest.mark.parametrize(
    ("iteration", "expected_targets"),
    [
        (0, [3, 3, 3]),
        (2, [9, 9, 9]),
    ],
)
def test_build_uniform_allocation_plan_uses_exact_outer_iteration_targets(
        iteration,
        expected_targets,
):
    state = _make_state(
        root_out_degree=3,
        log_likelihoods=(0.0, 1.0, 2.0),
        out_degree=(0, 0, 0),
        max_samples=3,
    )

    plan = allocation.build_allocation_plan(
        state=state,
        allocation_target="uniform",
        iteration=iteration,
        delta_K=3,
    )

    np.testing.assert_array_equal(
        np.asarray(plan.valid),
        np.ones((3,), dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(plan.log_L_blocks),
        np.asarray([0.0, 1.0, 2.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(plan.current_K),
        np.asarray([3, 2, 1]),
    )
    np.testing.assert_allclose(
        np.asarray(plan.unit_peak_utility),
        np.asarray([1.0, 1.0, 1.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(plan.target_K),
        np.asarray(expected_targets),
    )


def test_build_allocation_plan_can_use_fixed_initial_root_out_degree():
    state = _make_state(
        root_out_degree=7,
        log_likelihoods=(0.0, 1.0, 2.0),
        out_degree=(0, 0, 0),
        max_samples=3,
    )
    kwargs = {
        "state": state,
        "allocation_target": "uniform",
        "iteration": 1,
        "delta_K": 3,
    }
    signature = inspect.signature(allocation.build_allocation_plan)
    if "root_out_degree" in signature.parameters:
        kwargs["root_out_degree"] = 2
    elif "initial_root_out_degree" in signature.parameters:
        kwargs["initial_root_out_degree"] = 2
    else:
        pytest.fail(
            "build_allocation_plan must accept root_out_degree or "
            "initial_root_out_degree for fixed d_0 targets."
        )

    plan = allocation.build_allocation_plan(**kwargs)

    np.testing.assert_array_equal(
        np.asarray(plan.target_K),
        np.asarray([5, 5, 5]),
    )


def test_fractional_utility_targets_ceil_and_clip_after_scaling():
    targets = integer_allocation_targets(
        root_out_degree=4,
        iteration=3,
        delta_K=2,
        unit_peak_utility=jnp.asarray([0.0, 0.01, 0.5, 1.0]),
        max_target=8,
    )

    np.testing.assert_array_equal(np.asarray(targets), np.asarray([4, 5, 7, 8]))


@pytest.mark.parametrize(
    "allocation_target",
    ["evidence_improving", "posterior_improving"],
)
def test_build_utility_allocation_plan_uses_summaries_and_shared_targets(
        allocation_target,
):
    state = _make_state(
        root_out_degree=3,
        log_likelihoods=(
            float(np.log(1.0)),
            float(np.log(3.0)),
            float(np.log(4.0)),
            float(np.log(20.0)),
        ),
        out_degree=(1, 0, 0, 0),
        log_L_constraints=(-np.inf, -np.inf, -np.inf, float(np.log(1.0))),
        max_samples=4,
    )
    block_state = build_block_state(
        state.samples,
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
        validate=True,
    )
    concentrations = classic_dirichlet_concentrations(block_state)
    volume_path = expected_volume_path(concentrations, valid=block_state.valid)

    if allocation_target == "evidence_improving":
        expected_utility = evidence_improvement_utility(
            log_L_blocks=block_state.log_L_blocks,
            volume_path=volume_path,
            alpha_gt=concentrations.alpha_gt,
            beta_gt=concentrations.alpha_eq + concentrations.alpha_lt,
            valid=block_state.valid,
        )
    else:
        shell_weights = jnp.exp(block_state.log_L_blocks) * volume_path.shell_mass
        expected_utility = posterior_improvement_utility(
            volume_path=volume_path,
            shell_weights=shell_weights,
            valid=block_state.valid,
        )

    expected_unit_peak = normalise_allocation_utility(
        expected_utility,
        valid=block_state.valid,
    )
    expected_targets = integer_allocation_targets(
        root_out_degree=state.root_out_degree,
        iteration=2,
        delta_K=3,
        unit_peak_utility=expected_unit_peak,
        valid=block_state.valid,
    )

    plan = allocation.build_allocation_plan(
        state=state,
        allocation_target=allocation_target,
        iteration=2,
        delta_K=3,
    )

    np.testing.assert_array_equal(
        np.asarray(plan.valid),
        np.asarray(block_state.valid),
    )
    np.testing.assert_array_equal(
        np.asarray(plan.current_K),
        np.asarray(block_state.incoming_K),
    )
    np.testing.assert_allclose(
        np.asarray(plan.log_L_blocks),
        np.asarray(block_state.log_L_blocks),
    )
    np.testing.assert_allclose(
        np.asarray(plan.unit_peak_utility),
        np.asarray(expected_unit_peak),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(plan.target_K),
        np.asarray(expected_targets),
    )


def test_build_posterior_allocation_plan_selects_conservative_utility():
    state = _make_state(
        root_out_degree=3,
        log_likelihoods=(
            float(np.log(1.0)),
            float(np.log(3.0)),
            float(np.log(4.0)),
            float(np.log(20.0)),
        ),
        out_degree=(1, 0, 0, 0),
        log_L_constraints=(-np.inf, -np.inf, -np.inf, float(np.log(1.0))),
        max_samples=4,
    )
    block_state = build_block_state(
        state.samples,
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
        validate=True,
    )
    concentrations = classic_dirichlet_concentrations(block_state)
    volume_path = expected_volume_path(concentrations, valid=block_state.valid)
    shell_weights = jnp.exp(block_state.log_L_blocks) * volume_path.shell_mass
    expected_unit_peak = normalise_allocation_utility(
        posterior_improvement_utility(
            volume_path=volume_path,
            shell_weights=shell_weights,
            valid=block_state.valid,
            conservative=True,
        ),
        valid=block_state.valid,
    )
    expected_targets = integer_allocation_targets(
        root_out_degree=state.root_out_degree,
        iteration=2,
        delta_K=3,
        unit_peak_utility=expected_unit_peak,
        valid=block_state.valid,
    )

    plan = allocation.build_allocation_plan(
        state=state,
        allocation_target="posterior_improving",
        iteration=2,
        delta_K=3,
        posterior_utility="conservative",
    )

    np.testing.assert_allclose(
        np.asarray(plan.unit_peak_utility),
        np.asarray(expected_unit_peak),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(plan.target_K),
        np.asarray(expected_targets),
    )


def test_evidence_utility_shape_is_invariant_to_large_log_likelihood_shift():
    path = VolumePath(
        X_prev=jnp.asarray([1.0, 0.7, 0.4, 0.1]),
        X=jnp.asarray([0.7, 0.4, 0.1, 0.02]),
        shell_mass=jnp.asarray([0.3, 0.3, 0.3, 0.08]),
    )
    log_L_blocks = jnp.log(jnp.asarray([1.0, 3.0, 4.0, 20.0]))
    kwargs = {
        "volume_path": path,
        "alpha_gt": jnp.asarray([3.0, 2.0, 8.0, 4.0]),
        "beta_gt": jnp.asarray([2.0, 5.0, 1.0, 3.0]),
    }

    base = normalise_allocation_utility(
        evidence_improvement_utility(
            log_L_blocks=log_L_blocks,
            **kwargs,
        )
    )
    shifted = normalise_allocation_utility(
        evidence_improvement_utility(
            log_L_blocks=log_L_blocks + 1000.0,
            **kwargs,
        )
    )

    assert np.all(np.isfinite(np.asarray(shifted)))
    np.testing.assert_allclose(np.asarray(shifted), np.asarray(base), rtol=1e-6)


def test_posterior_plan_shape_is_invariant_to_large_log_likelihood_shift():
    base_state = _make_state(
        root_out_degree=3,
        log_likelihoods=(
            float(np.log(1.0)),
            float(np.log(3.0)),
            float(np.log(4.0)),
            float(np.log(20.0)),
        ),
        out_degree=(1, 0, 0, 0),
        log_L_constraints=(-np.inf, -np.inf, -np.inf, float(np.log(1.0))),
        max_samples=4,
    )
    shift = 1000.0
    shifted_state = _make_state(
        root_out_degree=3,
        log_likelihoods=tuple(
            float(value + shift)
            for value in (
                np.log(1.0),
                np.log(3.0),
                np.log(4.0),
                np.log(20.0),
            )
        ),
        out_degree=(1, 0, 0, 0),
        log_L_constraints=(
            -np.inf,
            -np.inf,
            -np.inf,
            float(np.log(1.0) + shift),
        ),
        max_samples=4,
    )

    base = allocation.build_allocation_plan(
        state=base_state,
        allocation_target="posterior_improving",
        iteration=2,
        delta_K=3,
    )
    shifted = allocation.build_allocation_plan(
        state=shifted_state,
        allocation_target="posterior_improving",
        iteration=2,
        delta_K=3,
    )

    assert np.all(np.isfinite(np.asarray(shifted.unit_peak_utility)))
    np.testing.assert_allclose(
        np.asarray(shifted.unit_peak_utility),
        np.asarray(base.unit_peak_utility),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(shifted.target_K),
        np.asarray(base.target_K),
    )


def test_evidence_and_posterior_utilities_can_drive_different_targets():
    path = VolumePath(
        X_prev=jnp.asarray([1.0, 0.7, 0.4, 0.1]),
        X=jnp.asarray([0.7, 0.4, 0.1, 0.02]),
        shell_mass=jnp.asarray([0.3, 0.3, 0.3, 0.08]),
    )

    evidence_unit_peak = normalise_allocation_utility(
        evidence_improvement_utility(
            log_L_blocks=jnp.log(jnp.asarray([1.0, 3.0, 4.0, 20.0])),
            volume_path=path,
            alpha_gt=jnp.asarray([3.0, 2.0, 8.0, 4.0]),
            beta_gt=jnp.asarray([2.0, 5.0, 1.0, 3.0]),
        )
    )
    posterior_unit_peak = normalise_allocation_utility(
        posterior_improvement_utility(
            volume_path=path,
            shell_weights=jnp.asarray([0.6, 0.1, 0.25, 0.05]),
        )
    )

    evidence_targets = integer_allocation_targets(
        root_out_degree=4,
        iteration=2,
        delta_K=3,
        unit_peak_utility=evidence_unit_peak,
    )
    posterior_targets = integer_allocation_targets(
        root_out_degree=4,
        iteration=2,
        delta_K=3,
        unit_peak_utility=posterior_unit_peak,
    )

    np.testing.assert_array_equal(
        np.asarray(evidence_targets),
        np.asarray([10, 5, 5, 4]),
    )
    np.testing.assert_array_equal(
        np.asarray(posterior_targets),
        np.asarray([8, 10, 5, 4]),
    )


def test_utility_fallbacks_are_deterministic_for_zero_tied_and_nonfinite_inputs():
    all_zero = normalise_allocation_utility(
        jnp.asarray([0.0, -1e-12, 0.0]),
        valid=jnp.asarray([True, False, True]),
    )
    tied = normalise_allocation_utility(
        jnp.asarray([2.0, 2.0, 1.0]),
        valid=jnp.asarray([True, True, True]),
    )
    nonfinite = normalise_allocation_utility(
        jnp.asarray([np.nan, np.inf, -np.inf]),
        valid=jnp.asarray([True, True, False]),
    )

    np.testing.assert_allclose(
        np.asarray(all_zero),
        np.asarray([1.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(np.asarray(tied), np.asarray([1.0, 1.0, 0.5]))
    np.testing.assert_allclose(
        np.asarray(nonfinite),
        np.asarray([1.0, 1.0, 0.0]),
    )


def test_utility_normalisation_falls_back_to_uniform_over_valid_blocks():
    unit_peak = normalise_allocation_utility(
        jnp.asarray([0.0, np.inf, -1.0]),
        valid=jnp.asarray([True, False, True]),
    )

    np.testing.assert_allclose(np.asarray(unit_peak), np.asarray([1.0, 0.0, 1.0]))


def test_invalid_allocation_target_and_delta_k_fail_explicitly():
    with pytest.raises(ValueError, match="allocation_target"):
        validate_allocation_target("not-a-mode")

    with pytest.raises(ValueError, match="delta_K"):
        integer_allocation_targets(
            root_out_degree=1,
            iteration=1,
            delta_K=0,
            unit_peak_utility=jnp.asarray([1.0]),
        )
