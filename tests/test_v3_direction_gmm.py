from __future__ import annotations

import dataclasses
import importlib
from typing import NamedTuple

import numpy as np
import pytest
from jax import numpy as jnp
from jax import random

from jaxns.constrained_sampler import EllipsoidalGaussianDirectionKernel
from jaxns.constrained_sampler import _sample_direction_from_kernel
from jaxns.pytree import TreeField


@dataclasses.dataclass(frozen=True, slots=True)
class PosteriorDrawFixture:
    samples_U: np.ndarray
    block_ids: np.ndarray
    block_likelihoods: np.ndarray
    x_prev_draws: np.ndarray
    p_gt_draws: np.ndarray
    p_eq_draws: np.ndarray
    expected_weights: np.ndarray


@dataclasses.dataclass(frozen=True, slots=True)
class WeightedFitFixture:
    rows: np.ndarray
    weights: np.ndarray
    expected_weighted_mean: np.ndarray
    unweighted_mean: np.ndarray


class DatasetValidationCase(NamedTuple):
    name: str
    rows: np.ndarray
    weights: np.ndarray
    expected_reasons: tuple[str, ...]


class ComponentCountCase(NamedTuple):
    name: str
    n_eff: float
    d_dim: int
    unique_rows: int
    expected_requested: int
    expected_final: int
    expected_fallback: bool


class ShellEvent(NamedTuple):
    log_likelihood: float
    dataset_ready: bool = True
    fit_success: bool = True


class AllocationTargetCase(NamedTuple):
    allocation_target: str
    shell_epoch: int


@dataclasses.dataclass(frozen=True, slots=True)
class SeparatedGaussianFixture:
    rows: np.ndarray
    weights: np.ndarray
    expected_means: np.ndarray


def _require_symbol(module_name: str, symbol_name: str):
    module = importlib.import_module(module_name)
    symbol = getattr(module, symbol_name, None)
    if symbol is None:
        pytest.fail(
            f"{module_name}.{symbol_name} is required by Ticket 0014 "
            "posterior-weighted GMM direction-kernel tests."
        )
    return symbol


def _as_array(container, *names: str) -> np.ndarray:
    for name in names:
        if hasattr(container, name):
            return np.asarray(getattr(container, name))
        if isinstance(container, dict) and name in container:
            return np.asarray(container[name])
    pytest.fail(f"Expected one of {names} on {type(container).__name__}.")


def _as_float(container, *names: str) -> float:
    return float(_as_array(container, *names))


def _as_int(container, *names: str) -> int:
    return int(_as_array(container, *names))


def _as_bool(container, *names: str) -> bool:
    return bool(_as_array(container, *names))


def _as_str(container, *names: str) -> str:
    for name in names:
        if hasattr(container, name):
            return str(getattr(container, name))
        if isinstance(container, dict) and name in container:
            return str(container[name])
    pytest.fail(f"Expected one of {names} on {type(container).__name__}.")


def _diagnostics(result):
    if hasattr(result, "diagnostics"):
        return result.diagnostics
    if isinstance(result, dict) and "diagnostics" in result:
        return result["diagnostics"]
    pytest.fail("Ticket 0014 helpers must return explicit diagnostics.")


def _kernel(result):
    if hasattr(result, "kernel"):
        return result.kernel
    if isinstance(result, dict) and "kernel" in result:
        return result["kernel"]
    pytest.fail("Direction fit result must expose the selected kernel.")


def _updated_coordinator(result):
    for name in ("coordinator", "updated_coordinator", "state"):
        if hasattr(result, name):
            return getattr(result, name)
        if isinstance(result, dict) and name in result:
            return result[name]
    pytest.fail("Coordinator fit result must expose the updated coordinator.")


def _as_optional_int(container, *names: str) -> int | None:
    for name in names:
        if hasattr(container, name):
            value = getattr(container, name)
            return None if value is None else int(value)
        if isinstance(container, dict) and name in container:
            value = container[name]
            return None if value is None else int(value)
    return None


def _as_optional_str(container, *names: str) -> str | None:
    for name in names:
        if hasattr(container, name):
            value = getattr(container, name)
            return None if value is None else str(value)
        if isinstance(container, dict) and name in container:
            value = container[name]
            return None if value is None else str(value)
    return None


def _sorted_components(kernel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = _as_array(kernel, "component_means")
    radii = _as_array(kernel, "component_radii")
    probabilities = _as_array(kernel, "component_probabilities")
    order = np.argsort(means[:, 0])
    return means[order], radii[order], probabilities[order]


def _separated_gaussian_fixture(seed: int = 104) -> SeparatedGaussianFixture:
    rng = np.random.default_rng(seed)
    left = rng.normal(
        loc=np.asarray([-3.0, -0.5]),
        scale=np.asarray([0.20, 0.10]),
        size=(64, 2),
    )
    right = rng.normal(
        loc=np.asarray([3.0, 0.5]),
        scale=np.asarray([0.25, 0.12]),
        size=(64, 2),
    )
    rows = np.vstack([left, right])
    left_weights = np.full(left.shape[0], 0.45 / left.shape[0], dtype=float)
    right_weights = np.full(right.shape[0], 0.55 / right.shape[0], dtype=float)
    weights = np.concatenate([left_weights, right_weights])
    expected_means = np.asarray([left_weights @ left, right_weights @ right])
    expected_means /= np.asarray([[0.45], [0.55]])
    return SeparatedGaussianFixture(
        rows=rows,
        weights=weights,
        expected_means=expected_means,
    )


def _posterior_draw_fixture() -> PosteriorDrawFixture:
    # Four classic samples in three likelihood blocks. Block 1 is a plateau
    # with two samples, so equality atom mass is split equally between them.
    #
    # Per-draw unnormalised masses:
    # draw 0: [0.50, 0.50, 0.50, 1.05], total 2.55
    # draw 1: [0.80, 0.20, 0.20, 1.05], total 2.25
    # Posterior fitting weights are the mean of per-draw normalized weights.
    return PosteriorDrawFixture(
        samples_U=np.asarray(
            [
                [0.10, 0.10],
                [0.40, 0.40],
                [0.42, 0.41],
                [0.90, 0.90],
            ],
            dtype=float,
        ),
        block_ids=np.asarray([0, 1, 1, 2], dtype=int),
        block_likelihoods=np.asarray([2.0, 5.0, 7.0], dtype=float),
        x_prev_draws=np.asarray(
            [
                [1.00, 0.50, 0.25],
                [0.80, 0.40, 0.20],
            ],
            dtype=float,
        ),
        p_gt_draws=np.asarray(
            [
                [0.75, 0.25, 0.40],
                [0.50, 0.10, 0.25],
            ],
            dtype=float,
        ),
        p_eq_draws=np.asarray(
            [
                [0.00, 0.40, 0.00],
                [0.00, 0.20, 0.00],
            ],
            dtype=float,
        ),
        expected_weights=np.asarray(
            [
                0.5 * (0.50 / 2.55 + 0.80 / 2.25),
                0.5 * (0.50 / 2.55 + 0.20 / 2.25),
                0.5 * (0.50 / 2.55 + 0.20 / 2.25),
                0.5 * (1.05 / 2.55 + 1.05 / 2.25),
            ],
            dtype=float,
        ),
    )


def _weighted_fit_fixture() -> WeightedFitFixture:
    low_rows = np.asarray(
        [[0.02 * i, 0.10 * ((i % 5) - 2)] for i in range(36)],
        dtype=float,
    )
    high_rows = np.asarray(
        [[8.0 + 0.04 * i, 8.0 - 0.03 * i] for i in range(8)],
        dtype=float,
    )
    rows = np.vstack([low_rows, high_rows])
    weights = np.concatenate(
        [
            np.full(low_rows.shape[0], 0.40 / low_rows.shape[0], dtype=float),
            np.full(high_rows.shape[0], 0.60 / high_rows.shape[0], dtype=float),
        ]
    )
    return WeightedFitFixture(
        rows=rows,
        weights=weights,
        expected_weighted_mean=weights @ rows,
        unweighted_mean=np.mean(rows, axis=0),
    )


def test_posterior_fitting_weights_use_plateau_and_mc_shrinkage_means():
    weights_from_draws = _require_symbol(
        "jaxns.em_gmm",
        "posterior_fitting_weights_from_shrinkage_draws",
    )
    fixture = _posterior_draw_fixture()

    result = weights_from_draws(
        block_ids=fixture.block_ids,
        block_likelihoods=fixture.block_likelihoods,
        x_prev_draws=fixture.x_prev_draws,
        p_gt_draws=fixture.p_gt_draws,
        p_eq_draws=fixture.p_eq_draws,
    )

    weights = _as_array(result, "weights", "posterior_weights")
    diagnostics = _diagnostics(result)
    raw_average_then_normalize = np.asarray(
        [0.65 / 2.40, 0.35 / 2.40, 0.35 / 2.40, 1.05 / 2.40],
        dtype=float,
    )
    np.testing.assert_allclose(weights, fixture.expected_weights, rtol=1e-12)
    assert not np.allclose(weights, raw_average_then_normalize, rtol=1e-3)
    np.testing.assert_allclose(np.sum(weights), 1.0, rtol=1e-12)
    assert _as_int(diagnostics, "num_shrinkage_draws") == 2
    assert _as_int(diagnostics, "num_plateau_samples") == 2
    assert _as_int(diagnostics, "num_plateau_blocks") == 1


def test_weighted_em_one_component_fit_is_driven_by_posterior_weights():
    fit_direction_gmm = _require_symbol(
        "jaxns.em_gmm",
        "fit_posterior_weighted_direction_gmm",
    )
    fixture = _weighted_fit_fixture()

    result = fit_direction_gmm(
        key=random.PRNGKey(4),
        rows=fixture.rows,
        posterior_weights=fixture.weights,
        n_components=1,
        max_fit_rows=64,
    )

    kernel = _kernel(result)
    means = _as_array(kernel, "component_means")
    diagnostics = _diagnostics(result)

    assert means.shape == (1, 2)
    np.testing.assert_allclose(
        means[0],
        fixture.expected_weighted_mean,
        atol=0.20,
    )
    assert np.linalg.norm(means[0] - fixture.expected_weighted_mean) < 0.25
    assert np.linalg.norm(means[0] - fixture.unweighted_mean) > 2.0
    assert _as_str(diagnostics, "fitting_path") == "weighted_em"
    assert _as_float(diagnostics, "N_eff", "n_eff") >= 20.0
    np.testing.assert_allclose(
        _as_float(diagnostics, "N_eff", "n_eff"),
        1.0 / np.sum(fixture.weights ** 2),
        rtol=1e-12,
    )


def test_build_fitting_dataset_excludes_bad_rows_and_respects_phantom_policy():
    build_dataset = _require_symbol(
        "jaxns.em_gmm",
        "build_direction_fitting_dataset",
    )
    fixture = _posterior_draw_fixture()
    samples_with_bad_rows = np.vstack(
        [
            fixture.samples_U,
            [[np.nan, 0.2], [0.2, np.inf], [0.3, 0.3]],
        ]
    )
    weights = np.asarray(
        [0.20, 0.20, 0.20, 0.20, 0.10, 0.10, 0.00],
        dtype=float,
    )

    dataset = build_dataset(
        samples_U=samples_with_bad_rows,
        posterior_weights=weights,
        valid_mask=np.ones(samples_with_bad_rows.shape[0], dtype=bool),
        retained_phantom_log_likelihoods=np.asarray([3.0, 4.0], dtype=float),
        retained_phantom_valid_mask=np.asarray([True, True], dtype=bool),
        max_fit_rows=32,
        resampling_seed=12,
    )

    rows = _as_array(dataset, "rows", "fit_rows")
    fit_weights = _as_array(dataset, "weights", "posterior_weights")
    diagnostics = _diagnostics(dataset)

    assert rows.shape == (4, 2)
    np.testing.assert_allclose(fit_weights, np.full(4, 0.25))
    assert _as_int(diagnostics, "excluded_nonfinite_rows") == 2
    assert _as_int(diagnostics, "excluded_nonpositive_weight_rows") == 1
    assert _as_int(diagnostics, "excluded_phantom_rows") == 0
    assert _as_str(diagnostics, "phantom_coordinate_policy") == "discarded"
    assert _as_int(diagnostics, "N_pos", "n_pos") == 4
    np.testing.assert_allclose(_as_float(diagnostics, "weight_sum"), 1.0)


def test_all_filtered_rows_fallback_preserves_exclusion_diagnostics():
    fit_direction_gmm = _require_symbol(
        "jaxns.em_gmm",
        "fit_posterior_weighted_direction_gmm",
    )
    rows = np.asarray(
        [[np.nan, 0.0], [0.0, np.inf], [1.0, 1.0]],
        dtype=float,
    )
    weights = np.asarray([0.2, 0.3, 0.0], dtype=float)

    result = fit_direction_gmm(
        key=random.PRNGKey(15),
        rows=rows,
        posterior_weights=weights,
        n_components=1,
        max_fit_rows=32,
    )

    diagnostics = _diagnostics(result)
    assert _as_bool(diagnostics, "fallback_active", "used_fallback") is True
    assert _as_str(diagnostics, "fallback_reason") == "nonfinite_coordinates"
    assert _as_int(diagnostics, "excluded_nonfinite_rows") == 2
    assert _as_int(diagnostics, "excluded_nonpositive_weight_rows") == 1
    assert _as_int(diagnostics, "N_pos", "n_pos") == 0
    assert _as_float(diagnostics, "N_eff", "n_eff") == 0.0


@pytest.mark.parametrize(
    "case",
    (
        DatasetValidationCase(
            name="all_zero_weights",
            rows=np.asarray(
                [[float(i), float(i % 3)] for i in range(24)],
                dtype=float,
            ),
            weights=np.zeros(24, dtype=float),
            expected_reasons=("no_positive_posterior_weights",),
        ),
        DatasetValidationCase(
            name="negative_weight_rejected",
            rows=np.asarray(
                [[float(i), float(i % 5)] for i in range(24)],
                dtype=float,
            ),
            weights=np.concatenate(
                [np.asarray([-0.01], dtype=float), np.full(23, 1.01 / 23)]
            ),
            expected_reasons=("negative_posterior_weights",),
        ),
        DatasetValidationCase(
            name="insufficient_positive_rows",
            rows=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
            weights=np.asarray([0.5, 0.5]),
            expected_reasons=("insufficient_N_pos", "insufficient_N_eff"),
        ),
        DatasetValidationCase(
            name="small_dataset_insufficient_effective_size",
            rows=np.asarray(
                [[float(i), float(i % 3)] for i in range(6)],
                dtype=float,
            ),
            weights=np.full(6, 1.0 / 6, dtype=float),
            expected_reasons=("insufficient_N_eff",),
        ),
        DatasetValidationCase(
            name="insufficient_unique_rows",
            rows=np.asarray(
                [[0.0, 0.0], [1.0, 1.0]] * 12,
                dtype=float,
            ),
            weights=np.full(24, 1.0 / 24, dtype=float),
            expected_reasons=("insufficient_unique_rows",),
        ),
        DatasetValidationCase(
            name="insufficient_effective_size",
            rows=np.asarray(
                [[float(i), float(i % 7)] for i in range(24)],
                dtype=float,
            ),
            weights=np.asarray(
                [0.90] + [0.10 / 23] * 23,
                dtype=float,
            ),
            expected_reasons=("insufficient_N_eff",),
        ),
    ),
    ids=lambda case: case.name,
)
def test_invalid_fitting_datasets_fall_back_with_explicit_reason(case):
    fit_direction_gmm = _require_symbol(
        "jaxns.em_gmm",
        "fit_posterior_weighted_direction_gmm",
    )

    result = fit_direction_gmm(
        key=random.PRNGKey(5),
        rows=case.rows,
        posterior_weights=case.weights,
        n_components=1,
        max_fit_rows=32,
    )

    diagnostics = _diagnostics(result)
    assert _as_bool(diagnostics, "fallback_active", "used_fallback") is True
    assert _as_str(diagnostics, "fallback_reason") in case.expected_reasons
    assert _as_str(diagnostics, "active_kernel_mode", "mode") in {
        "isotropic",
        "isotropic_gaussian",
    }
    if case.name == "all_zero_weights":
        assert _as_int(diagnostics, "excluded_nonpositive_weight_rows") == (
            case.rows.shape[0]
        )
        assert _as_int(diagnostics, "N_pos", "n_pos") == 0
        assert _as_float(diagnostics, "N_eff", "n_eff") == 0.0
    if case.name == "negative_weight_rejected":
        assert _as_int(diagnostics, "excluded_nonpositive_weight_rows") == 1


def test_dimension_one_valid_dataset_uses_random_sign_direction_kernel():
    fit_direction_gmm = _require_symbol(
        "jaxns.em_gmm",
        "fit_posterior_weighted_direction_gmm",
    )
    rows = np.linspace(-1.0, 1.0, 24, dtype=float)[:, None]
    weights = np.full(rows.shape[0], 1.0 / rows.shape[0], dtype=float)

    result = fit_direction_gmm(
        key=random.PRNGKey(44),
        rows=rows,
        posterior_weights=weights,
        n_components=1,
        max_fit_rows=32,
    )

    diagnostics = _diagnostics(result)
    assert _as_bool(diagnostics, "fallback_active", "used_fallback") is False
    assert _as_int(diagnostics, "D_dim", "d_dim") == 1
    kernel = _kernel(result)
    directions = np.asarray(
        [
            _sample_direction_from_kernel(
                key=key,
                direction_kernel=kernel,
                current_point=TreeField(jnp.asarray([0.0])),
            ).tree[0]
            for key in random.split(random.PRNGKey(45), 256)
        ],
        dtype=float,
    )
    assert set(np.unique(directions)) == {-1.0, 1.0}


def test_bounded_systematic_resampling_uses_configurable_limit_and_diagnostics():
    build_dataset = _require_symbol(
        "jaxns.em_gmm",
        "build_direction_fitting_dataset",
    )
    rows = np.asarray(
        [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        dtype=float,
    )
    weights = np.asarray([0.70, 0.10, 0.05, 0.05, 0.05, 0.05], dtype=float)

    first = build_dataset(
        samples_U=rows,
        posterior_weights=weights,
        max_fit_rows=4,
        resampling_seed=123,
    )
    second = build_dataset(
        samples_U=rows,
        posterior_weights=weights,
        max_fit_rows=4,
        resampling_seed=123,
    )

    first_rows = _as_array(first, "rows", "fit_rows")
    second_rows = _as_array(second, "rows", "fit_rows")
    diagnostics = _diagnostics(first)

    assert first_rows.shape == (4, 1)
    np.testing.assert_allclose(first_rows, second_rows)
    assert np.unique(first_rows, axis=0).shape[0] < first_rows.shape[0]
    assert _as_str(diagnostics, "fitting_path") == "bounded_resampling"
    assert _as_int(diagnostics, "M_D", "max_fit_rows") == 4
    assert _as_int(diagnostics, "original_N_pos", "original_n_pos") == 6
    assert _as_int(diagnostics, "unique_finite_row_count") == np.unique(
        first_rows,
        axis=0,
    ).shape[0]
    assert _as_int(diagnostics, "resampled_row_count") == 4
    assert _as_int(diagnostics, "resampling_seed") == 123
    np.testing.assert_allclose(
        _as_float(diagnostics, "original_N_eff", "original_n_eff"),
        1.0 / np.sum(weights ** 2),
        rtol=1e-12,
    )


def test_systematic_resampling_uses_exact_offset_index_fixture():
    systematic_resample = _require_symbol(
        "jaxns.em_gmm",
        "systematic_resample_indices",
    )
    weights = np.asarray([0.20, 0.10, 0.40, 0.30], dtype=float)

    indices = np.asarray(
        systematic_resample(
            weights=weights,
            num_samples=10,
            offset=0.04,
        ),
        dtype=int,
    )

    np.testing.assert_array_equal(
        indices,
        np.asarray([0, 0, 1, 2, 2, 2, 2, 3, 3, 3], dtype=int),
    )


def test_weighted_em_and_bounded_resampling_agree_on_separated_fixture():
    fit_direction_gmm = _require_symbol(
        "jaxns.em_gmm",
        "fit_posterior_weighted_direction_gmm",
    )
    fixture = _separated_gaussian_fixture()

    weighted = fit_direction_gmm(
        key=random.PRNGKey(120),
        rows=fixture.rows,
        posterior_weights=fixture.weights,
        n_components=2,
        max_fit_rows=512,
    )
    bounded = fit_direction_gmm(
        key=random.PRNGKey(120),
        rows=fixture.rows,
        posterior_weights=fixture.weights,
        n_components=2,
        max_fit_rows=80,
        resampling_seed=121,
    )

    weighted_means, weighted_radii, weighted_probabilities = _sorted_components(
        _kernel(weighted)
    )
    bounded_means, bounded_radii, bounded_probabilities = _sorted_components(
        _kernel(bounded)
    )

    assert _as_str(_diagnostics(weighted), "fitting_path") == "weighted_em"
    assert _as_str(_diagnostics(bounded), "fitting_path") == "bounded_resampling"
    np.testing.assert_allclose(weighted_means, fixture.expected_means, atol=0.30)
    np.testing.assert_allclose(bounded_means, weighted_means, atol=0.35)
    np.testing.assert_allclose(bounded_radii, weighted_radii, rtol=0.35, atol=0.10)
    np.testing.assert_allclose(
        bounded_probabilities,
        weighted_probabilities,
        atol=0.20,
    )


def test_anisotropic_fit_calibrates_direction_geometry_without_perf_win_claim():
    fit_direction_gmm = _require_symbol(
        "jaxns.em_gmm",
        "fit_posterior_weighted_direction_gmm",
    )
    rng = np.random.default_rng(135)
    rows = rng.normal(
        loc=np.asarray([0.0, 0.0]),
        scale=np.asarray([3.0, 0.25]),
        size=(96, 2),
    )
    weights = np.full(rows.shape[0], 1.0 / rows.shape[0], dtype=float)

    result = fit_direction_gmm(
        key=random.PRNGKey(135),
        rows=rows,
        posterior_weights=weights,
        n_components=1,
        max_fit_rows=128,
    )
    diagnostics = _diagnostics(result)
    kernel = _kernel(result)
    radii = _as_array(kernel, "component_radii")
    directions = np.asarray(
        [
            _sample_direction_from_kernel(
                key=key,
                direction_kernel=kernel,
                current_point=TreeField(jnp.asarray([0.0, 0.0])),
            ).tree
            for key in random.split(random.PRNGKey(136), 1024)
        ]
    )

    assert _as_bool(diagnostics, "fallback_active", "used_fallback") is False
    assert _as_str(diagnostics, "fitting_path") == "weighted_em"
    assert float(np.max(radii[0]) / np.min(radii[0])) > 6.0
    assert float(np.var(directions[:, 0]) / np.var(directions[:, 1])) > 2.0
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-5)


def test_bounded_rows_reach_em_backend_before_superlinear_fitting_work():
    fit_direction_gmm = _require_symbol(
        "jaxns.em_gmm",
        "fit_posterior_weighted_direction_gmm",
    )
    rows = np.asarray(
        [[float(i), float(i % 11)] for i in range(128)],
        dtype=float,
    )
    weights = np.full(rows.shape[0], 1.0 / rows.shape[0], dtype=float)
    backend_row_counts = []

    def recording_em_backend(*, rows, posterior_weights, **kwargs):
        backend_row_counts.append(np.asarray(rows).shape[0])
        return {
            "component_means": np.mean(rows, axis=0, keepdims=True),
            "responsibilities": np.ones((np.asarray(rows).shape[0], 1)),
            "em_iteration_count": 1,
            "em_converged": True,
        }

    result = fit_direction_gmm(
        key=random.PRNGKey(122),
        rows=rows,
        posterior_weights=weights,
        n_components=1,
        max_fit_rows=16,
        resampling_seed=123,
        em_backend=recording_em_backend,
    )

    diagnostics = _diagnostics(result)
    assert backend_row_counts == [16]
    assert _as_str(diagnostics, "fitting_path") == "bounded_resampling"
    assert _as_int(diagnostics, "original_N_pos", "original_n_pos") == 128
    assert _as_int(diagnostics, "resampled_row_count") == 16


def test_bounded_resampling_auto_k_caps_by_resampled_unique_rows():
    fit_direction_gmm = _require_symbol(
        "jaxns.em_gmm",
        "fit_posterior_weighted_direction_gmm",
    )
    rows = np.asarray(
        [[float(i), float(i % 13)] for i in range(128)],
        dtype=float,
    )
    weights = np.full(rows.shape[0], 1.0 / rows.shape[0], dtype=float)
    backend_component_counts = []

    def recording_em_backend(*, rows, posterior_weights, n_components, **kwargs):
        rows = np.asarray(rows)
        backend_component_counts.append(int(n_components))
        return {
            "component_means": np.repeat(
                np.mean(rows, axis=0, keepdims=True),
                int(n_components),
                axis=0,
            ),
            "responsibilities": np.full(
                (rows.shape[0], int(n_components)),
                1.0 / int(n_components),
                dtype=float,
            ),
            "em_iteration_count": 1,
            "em_converged": True,
        }

    result = fit_direction_gmm(
        key=random.PRNGKey(123),
        rows=rows,
        posterior_weights=weights,
        max_fit_rows=6,
        resampling_seed=124,
        em_backend=recording_em_backend,
    )

    diagnostics = _diagnostics(result)
    assert backend_component_counts == [6]
    assert _as_str(diagnostics, "fitting_path") == "bounded_resampling"
    assert _as_int(diagnostics, "requested_K_D") == 8
    assert _as_int(diagnostics, "final_K_D") == 6
    assert _as_int(diagnostics, "unique_finite_row_count") == 6


@pytest.mark.parametrize(
    "case",
    (
        ComponentCountCase(
            name="floor_formula",
            n_eff=24.0,
            d_dim=2,
            unique_rows=24,
            expected_requested=4,
            expected_final=4,
            expected_fallback=False,
        ),
        ComponentCountCase(
            name="cap_at_eight",
            n_eff=1000.0,
            d_dim=2,
            unique_rows=100,
            expected_requested=8,
            expected_final=8,
            expected_fallback=False,
        ),
        ComponentCountCase(
            name="cap_by_unique_rows",
            n_eff=1000.0,
            d_dim=2,
            unique_rows=3,
            expected_requested=8,
            expected_final=3,
            expected_fallback=False,
        ),
        ComponentCountCase(
            name="fallback_when_unique_cap_is_zero",
            n_eff=1000.0,
            d_dim=2,
            unique_rows=0,
            expected_requested=8,
            expected_final=0,
            expected_fallback=True,
        ),
    ),
    ids=lambda case: case.name,
)
def test_k_d_policy_formula_caps_and_fallback(case):
    choose_component_count = _require_symbol(
        "jaxns.em_gmm",
        "choose_direction_component_count",
    )

    result = choose_component_count(
        n_eff=case.n_eff,
        d_dim=case.d_dim,
        unique_row_count=case.unique_rows,
    )

    assert _as_int(result, "requested_K_D", "requested_k") == case.expected_requested
    assert _as_int(result, "final_K_D", "final_k") == case.expected_final
    assert _as_bool(result, "fallback_active", "requires_fallback") is case.expected_fallback


def test_covariance_floor_uses_global_weighted_trace_and_sets_positive_radii():
    covariance_components = _require_symbol(
        "jaxns.em_gmm",
        "direction_covariance_components",
    )
    rows = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
        dtype=float,
    )
    weights = np.full(4, 0.25, dtype=float)
    responsibilities = np.ones((4, 1), dtype=float)

    components = covariance_components(
        rows=rows,
        posterior_weights=weights,
        responsibilities=responsibilities,
    )

    radii = _as_array(components, "component_radii", "radii")
    rotations = _as_array(components, "component_rotations", "rotations")
    diagnostics = _diagnostics(components)

    np.testing.assert_allclose(
        _as_float(diagnostics, "sigma2_floor"),
        1e-6,
        rtol=1e-12,
    )
    assert radii.shape == (1, 2)
    assert rotations.shape == (1, 2, 2)
    assert np.all(radii >= np.sqrt(1e-6))

    degenerate = covariance_components(
        rows=np.ones((4, 2), dtype=float),
        posterior_weights=weights,
        responsibilities=responsibilities,
    )
    degenerate_radii = _as_array(degenerate, "component_radii", "radii")
    np.testing.assert_allclose(
        _as_float(_diagnostics(degenerate), "sigma2_floor"),
        1e-12,
        rtol=1e-12,
    )
    assert np.all(degenerate_radii >= np.sqrt(1e-12))


def test_nonfinite_covariance_falls_back_without_failing_run():
    fit_direction_gmm = _require_symbol(
        "jaxns.em_gmm",
        "fit_posterior_weighted_direction_gmm",
    )
    rows = np.asarray(
        [[float(i), float(i % 5)] for i in range(24)] + [[np.inf, 1.0]],
        dtype=float,
    )
    weights = np.full(rows.shape[0], 1.0 / rows.shape[0], dtype=float)

    def failing_backend(**kwargs):
        del kwargs
        raise np.linalg.LinAlgError("fixture covariance failure")

    result = fit_direction_gmm(
        key=random.PRNGKey(14),
        rows=rows,
        posterior_weights=weights,
        n_components=1,
        max_fit_rows=32,
        hard_adaptation_errors=False,
        em_backend=failing_backend,
    )

    diagnostics = _diagnostics(result)
    assert _as_bool(diagnostics, "fallback_active", "used_fallback") is True
    assert _as_str(diagnostics, "fallback_reason") in {
        "nonfinite_coordinates",
        "nonfinite_covariance",
        "covariance_failure",
    }


def test_component_volumes_define_probabilities_and_reject_zero_total_volume():
    probabilities_from_radii = _require_symbol(
        "jaxns.em_gmm",
        "direction_component_probabilities_from_radii",
    )

    result = probabilities_from_radii(
        component_radii=np.asarray([[2.0, 3.0], [1.0, 1.0]], dtype=float)
    )
    probabilities = _as_array(result, "component_probabilities", "probabilities")
    volumes = _as_array(result, "component_integrated_volumes", "volumes")

    np.testing.assert_allclose(volumes, np.asarray([6.0, 1.0]))
    np.testing.assert_allclose(probabilities, np.asarray([6.0 / 7.0, 1.0 / 7.0]))
    np.testing.assert_allclose(probabilities.sum(), 1.0)

    with pytest.raises(ValueError, match="volume|positive|normal"):
        probabilities_from_radii(
            component_radii=np.asarray([[0.0, 1.0], [0.0, 2.0]], dtype=float)
        )


def test_invalid_components_are_filtered_and_valid_volumes_renormalize():
    filter_components = _require_symbol(
        "jaxns.em_gmm",
        "filter_direction_components",
    )
    result = filter_components(
        component_means=np.asarray(
            [[0.0, 0.0], [np.nan, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype=float,
        ),
        component_radii=np.asarray(
            [[2.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
            dtype=float,
        ),
        component_rotations=np.stack([np.eye(2)] * 4),
        responsibility_effective_sizes=np.asarray([4.0, 4.0, 4.0, 2.0]),
        d_dim=2,
    )

    means = _as_array(result, "component_means")
    probabilities = _as_array(result, "component_probabilities", "probabilities")
    diagnostics = _diagnostics(result)

    assert means.shape == (1, 2)
    np.testing.assert_allclose(means[0], np.asarray([0.0, 0.0]))
    np.testing.assert_allclose(probabilities, np.asarray([1.0]))
    assert _as_int(diagnostics, "dropped_nonfinite") == 1
    assert _as_int(diagnostics, "dropped_nonpositive_volume") == 1
    assert _as_int(diagnostics, "dropped_low_responsibility_neff") == 1
    assert _as_int(diagnostics, "final_K_D", "final_k") == 1


def test_nonfinite_responsibility_neff_drops_component():
    filter_components = _require_symbol(
        "jaxns.em_gmm",
        "filter_direction_components",
    )
    result = filter_components(
        component_means=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        component_radii=np.asarray([[1.0, 1.0], [1.0, 0.5]], dtype=float),
        component_rotations=np.stack([np.eye(2)] * 2),
        responsibility_effective_sizes=np.asarray([4.0, np.nan]),
        d_dim=2,
    )

    means = _as_array(result, "component_means")
    diagnostics = _diagnostics(result)
    assert means.shape == (1, 2)
    np.testing.assert_allclose(means[0], np.asarray([0.0, 0.0]))
    assert _as_int(diagnostics, "dropped_low_responsibility_neff") == 1
    assert _as_int(diagnostics, "final_K_D", "final_k") == 1


def test_no_valid_component_uses_previous_kernel_fallback():
    filter_components = _require_symbol(
        "jaxns.em_gmm",
        "filter_direction_components",
    )
    previous_kernel = EllipsoidalGaussianDirectionKernel(
        component_means=jnp.asarray([[1.0, 1.0]]),
        component_radii=jnp.asarray([[0.25, 0.10]]),
        component_rotations=jnp.asarray([np.eye(2)]),
        component_probabilities=jnp.asarray([1.0]),
    )

    result = filter_components(
        component_means=np.asarray([[np.nan, 0.0]], dtype=float),
        component_radii=np.asarray([[1.0, 1.0]], dtype=float),
        component_rotations=np.asarray([np.eye(2)], dtype=float),
        responsibility_effective_sizes=np.asarray([4.0], dtype=float),
        d_dim=2,
        previous_kernel=previous_kernel,
    )

    diagnostics = _diagnostics(result)
    assert _kernel(result) is previous_kernel
    assert _as_bool(diagnostics, "fallback_active", "used_fallback") is True
    assert _as_str(diagnostics, "fallback_reason") == "no_valid_components"


def test_frozen_gmm_direction_draws_are_unit_symmetric_and_ignore_means():
    kernel = EllipsoidalGaussianDirectionKernel(
        component_means=jnp.asarray([[1.0e6, -1.0e6]]),
        component_radii=jnp.asarray([[0.25, 2.00]]),
        component_rotations=jnp.asarray([np.eye(2)]),
        component_probabilities=jnp.asarray([1.0]),
    )
    first_point = TreeField(jnp.asarray([0.10, 0.90]))
    second_point = TreeField(jnp.asarray([0.90, 0.10]))

    first = np.asarray(
        [
            _sample_direction_from_kernel(
                key=key,
                direction_kernel=kernel,
                current_point=first_point,
            ).tree
            for key in random.split(random.PRNGKey(30), 2048)
        ]
    )
    second = np.asarray(
        [
            _sample_direction_from_kernel(
                key=key,
                direction_kernel=kernel,
                current_point=second_point,
            ).tree
            for key in random.split(random.PRNGKey(30), 2048)
        ]
    )

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    assert abs(float(np.mean(first[:, 0]))) < 0.08
    assert abs(float(np.mean(first[:, 1]))) < 0.08
    assert np.all(np.abs(first) <= 1.0 + 1e-6)


def test_update_cadence_counts_distinct_plateau_shells_and_freezes_snapshots():
    state_cls = _require_symbol(
        "jaxns.em_gmm",
        "DirectionKernelAdaptationState",
    )
    initial = state_cls.initial(update_every_shells=5)

    assert _as_str(initial, "active_kernel_mode", "mode") in {
        "isotropic",
        "isotropic_gaussian",
    }
    assert _as_int(initial, "active_kernel_version", "version") == 0

    state = initial
    snapshots = []
    events = (
        ShellEvent(log_likelihood=1.0, dataset_ready=False),
        ShellEvent(log_likelihood=2.0, dataset_ready=True, fit_success=True),
        ShellEvent(log_likelihood=2.0, dataset_ready=True, fit_success=True),
        ShellEvent(log_likelihood=3.0, dataset_ready=True, fit_success=True),
        ShellEvent(log_likelihood=4.0, dataset_ready=True, fit_success=True),
        ShellEvent(log_likelihood=5.0, dataset_ready=True, fit_success=True),
        ShellEvent(log_likelihood=5.0, dataset_ready=True, fit_success=True),
        ShellEvent(log_likelihood=6.0, dataset_ready=True, fit_success=True),
        ShellEvent(log_likelihood=7.0, dataset_ready=True, fit_success=False),
        ShellEvent(log_likelihood=8.0, dataset_ready=True, fit_success=True),
        ShellEvent(log_likelihood=9.0, dataset_ready=True, fit_success=True),
        ShellEvent(log_likelihood=10.0, dataset_ready=True, fit_success=True),
        ShellEvent(log_likelihood=10.0, dataset_ready=True, fit_success=True),
        ShellEvent(log_likelihood=11.0, dataset_ready=True, fit_success=True),
        ShellEvent(log_likelihood=12.0, dataset_ready=True, fit_success=True),
    )

    states_after = []
    for event in events:
        snapshots.append(
            state.dispatch_snapshot(chain_id=f"before-{event.log_likelihood}")
        )
        state = state.observe_completed_shell(
            log_likelihood=event.log_likelihood,
            fit_dataset_ready=event.dataset_ready,
            fit_success=event.fit_success,
        )
        states_after.append(state)

    assert _as_int(state, "attempted_update_count") == 3
    assert _as_int(state, "successful_update_count") == 2
    assert _as_int(state, "active_kernel_version", "version") == 2
    assert _as_optional_str(state, "fallback_reason") in {None, ""}
    assert _as_int(states_after[7], "attempted_update_count") == 1
    assert _as_int(states_after[7], "active_kernel_version", "version") == 1
    assert _as_int(states_after[8], "attempted_update_count") == 2
    assert _as_int(states_after[8], "successful_update_count") == 1
    assert _as_int(states_after[8], "active_kernel_version", "version") == 1
    assert _as_int(states_after[9], "attempted_update_count") == 3
    assert _as_int(states_after[9], "successful_update_count") == 2
    assert _as_int(states_after[9], "active_kernel_version", "version") == 2

    # The duplicate likelihoods at 2.0 and 5.0 are plateau shells. After the
    # first success at shell 2.0, shells 3.0, 4.0, 5.0, and 6.0 are only four
    # newly completed distinct shells, so the failed second attempt waits for
    # 7.0. The failed attempt increments attempts but keeps success count and
    # version unchanged. Because the cadence is measured from the last
    # successful update, not the last attempted update, the next distinct shell
    # at 8.0 is still eligible and can advance future dispatches to version 2.
    versions = [
        _as_int(snapshot, "kernel_version", "version")
        for snapshot in snapshots
    ]
    assert versions == [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert _as_int(snapshots[2], "kernel_version", "version") == 1
    assert _as_int(snapshots[9], "kernel_version", "version") == 1
    assert _as_int(snapshots[10], "kernel_version", "version") == 2
    future_snapshot = state.dispatch_snapshot(chain_id="future")
    assert _as_int(future_snapshot, "kernel_version") == 2


def test_direction_adaptation_diagnostics_include_audit_fields():
    fit_direction_gmm = _require_symbol(
        "jaxns.em_gmm",
        "fit_posterior_weighted_direction_gmm",
    )
    fixture = _separated_gaussian_fixture(seed=808)

    result = fit_direction_gmm(
        key=random.PRNGKey(80),
        rows=fixture.rows,
        posterior_weights=fixture.weights,
        n_components=2,
        max_fit_rows=40,
        resampling_seed=808,
    )
    diagnostics = _diagnostics(result)
    kernel = _kernel(result)

    required_fields = {
        "active_kernel_mode",
        "active_kernel_version",
        "attempted_update_count",
        "successful_update_count",
        "fallback_active",
        "fallback_reason",
        "fitting_path",
        "D_dim",
        "N_pos",
        "N_eff",
        "unique_finite_row_count",
        "excluded_nonfinite_rows",
        "excluded_nonpositive_weight_rows",
        "weight_sum",
        "M_D",
        "resampling_seed",
        "original_N_pos",
        "original_N_eff",
        "resampled_row_count",
        "requested_K_D",
        "final_K_D",
        "component_means",
        "component_radii",
        "component_integrated_volumes",
        "component_probabilities",
        "dropped_component_reasons",
        "em_iteration_count",
        "em_converged",
        "sigma2_floor",
    }
    missing = [
        name
        for name in sorted(required_fields)
        if not hasattr(diagnostics, name)
        and not (isinstance(diagnostics, dict) and name in diagnostics)
    ]
    assert missing == []
    assert _as_str(diagnostics, "active_kernel_mode") in {
        "ellipsoidal",
        "gmm",
        "non_isotropic",
    }
    assert _as_int(diagnostics, "active_kernel_version") >= 1
    assert _as_bool(diagnostics, "fallback_active") is False
    assert _as_optional_str(diagnostics, "fallback_reason") in {None, ""}
    assert _as_str(diagnostics, "fitting_path") == "bounded_resampling"
    assert _as_int(diagnostics, "D_dim") == 2
    assert _as_int(diagnostics, "N_pos") == fixture.rows.shape[0]
    np.testing.assert_allclose(
        _as_float(diagnostics, "N_eff"),
        1.0 / np.sum(fixture.weights ** 2),
        rtol=1e-12,
    )
    assert 1 <= _as_int(diagnostics, "unique_finite_row_count") <= _as_int(
        diagnostics,
        "resampled_row_count",
    )
    assert _as_int(diagnostics, "excluded_nonfinite_rows") == 0
    assert _as_int(diagnostics, "excluded_nonpositive_weight_rows") == 0
    np.testing.assert_allclose(_as_float(diagnostics, "weight_sum"), 1.0)
    assert _as_int(diagnostics, "M_D") == 40
    assert _as_int(diagnostics, "resampling_seed") == 808
    assert _as_int(diagnostics, "original_N_pos") == fixture.rows.shape[0]
    assert _as_int(diagnostics, "resampled_row_count") == 40
    assert 1 <= _as_int(diagnostics, "final_K_D") <= _as_int(
        diagnostics,
        "requested_K_D",
    )
    assert _as_int(diagnostics, "em_iteration_count") > 0
    assert isinstance(_as_bool(diagnostics, "em_converged"), bool)
    assert _as_float(diagnostics, "sigma2_floor") > 0.0
    probabilities = _as_array(kernel, "component_probabilities")
    radii = _as_array(kernel, "component_radii")
    volumes = _as_array(diagnostics, "component_integrated_volumes")
    np.testing.assert_allclose(probabilities.sum(), 1.0, rtol=1e-12)
    assert np.all(np.isfinite(radii))
    assert np.all(radii > 0.0)
    assert np.all(np.isfinite(volumes))
    assert np.all(volumes > 0.0)


def test_coordinator_dispatch_snapshots_are_immutable_worker_inputs(monkeypatch):
    coordinator_cls = _require_symbol(
        "jaxns.em_gmm",
        "DirectionKernelAdaptationCoordinator",
    )
    dispatch_request_cls = _require_symbol(
        "jaxns.em_gmm",
        "DirectionKernelDispatchRequest",
    )
    calls = []

    def forbidden_fit(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("sampler must not refit direction GMMs in-chain")

    monkeypatch.setattr(
        importlib.import_module("jaxns.em_gmm"),
        "fit_posterior_weighted_direction_gmm",
        forbidden_fit,
        raising=False,
    )
    initial_kernel = EllipsoidalGaussianDirectionKernel(
        component_means=jnp.asarray([[0.0, 0.0]]),
        component_radii=jnp.asarray([[1.0, 0.5]]),
        component_rotations=jnp.asarray([np.eye(2)]),
        component_probabilities=jnp.asarray([1.0]),
    )
    updated_kernel = EllipsoidalGaussianDirectionKernel(
        component_means=jnp.asarray([[9.0, 9.0]]),
        component_radii=jnp.asarray([[0.2, 0.1]]),
        component_rotations=jnp.asarray([np.eye(2)]),
        component_probabilities=jnp.asarray([1.0]),
    )

    coordinator = coordinator_cls.initial(
        active_kernel=initial_kernel,
        active_kernel_version=7,
        update_every_shells=5,
    )
    request = dispatch_request_cls(
        chain_id="chain-a",
        worker_id="worker-a",
        shell_epoch=11,
    )
    snapshot = coordinator.prepare_dispatch_snapshot(request)
    worker_context = snapshot.direction_adaptation_context()
    updated = coordinator.replace_active_kernel(
        kernel=updated_kernel,
        shell_epoch=12,
        update_reason="test-success",
    )
    future_snapshot = updated.prepare_dispatch_snapshot(
        dispatch_request_cls(
            chain_id="chain-b",
            worker_id="worker-b",
            shell_epoch=12,
        )
    )

    assert calls == []
    assert _as_int(snapshot, "kernel_version") == 7
    assert _as_int(future_snapshot, "kernel_version") == 8
    np.testing.assert_allclose(
        _as_array(worker_context, "component_means"),
        np.asarray([[0.0, 0.0]]),
    )
    np.testing.assert_allclose(
        _as_array(worker_context, "component_radii"),
        np.asarray([[1.0, 0.5]]),
    )
    np.testing.assert_allclose(
        _as_array(future_snapshot.direction_adaptation_context(), "component_means"),
        np.asarray([[9.0, 9.0]]),
    )
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        snapshot.kernel_version = 999


@pytest.mark.parametrize(
    "case",
    (
        AllocationTargetCase(allocation_target="uniform", shell_epoch=21),
        AllocationTargetCase(
            allocation_target="evidence_improving",
            shell_epoch=22,
        ),
        AllocationTargetCase(
            allocation_target="posterior_improving",
            shell_epoch=23,
        ),
    ),
    ids=lambda case: case.allocation_target,
)
def test_coordinator_fit_flow_freezes_gmm_kernel_for_allocation_target(case):
    coordinator_cls = _require_symbol(
        "jaxns.em_gmm",
        "DirectionKernelAdaptationCoordinator",
    )
    dispatch_request_cls = _require_symbol(
        "jaxns.em_gmm",
        "DirectionKernelDispatchRequest",
    )
    fit_request_cls = _require_symbol(
        "jaxns.em_gmm",
        "DirectionKernelFitRequest",
    )
    fixture = _separated_gaussian_fixture(seed=case.shell_epoch)
    initial_kernel = EllipsoidalGaussianDirectionKernel(
        component_means=jnp.asarray([[0.0, 0.0]]),
        component_radii=jnp.asarray([[1.0, 1.0]]),
        component_rotations=jnp.asarray([np.eye(2)]),
        component_probabilities=jnp.asarray([1.0]),
    )
    coordinator = coordinator_cls.initial(
        active_kernel=initial_kernel,
        active_kernel_version=5,
        update_every_shells=5,
    )

    fit_update = coordinator.request_direction_kernel_fit(
        fit_request_cls(
            shell_epoch=case.shell_epoch,
            allocation_target=case.allocation_target,
            samples_U=fixture.rows,
            posterior_weights=fixture.weights,
            key=random.PRNGKey(case.shell_epoch),
            n_components=2,
            max_fit_rows=128,
            resampling_seed=case.shell_epoch + 1000,
        )
    )
    updated = _updated_coordinator(fit_update)
    fit_diagnostics = _diagnostics(fit_update)
    fitted_kernel = _kernel(fit_update)
    fitted_means, fitted_radii, fitted_probabilities = _sorted_components(
        fitted_kernel
    )

    assert _as_bool(fit_diagnostics, "fallback_active", "used_fallback") is False
    assert _as_str(fit_diagnostics, "allocation_target") == case.allocation_target
    assert _as_str(fit_diagnostics, "active_kernel_mode", "mode") in {
        "ellipsoidal",
        "gmm",
        "non_isotropic",
    }
    assert _as_str(fit_diagnostics, "fitting_path") == "weighted_em"
    assert _as_int(fit_diagnostics, "shell_epoch") == case.shell_epoch
    assert _as_int(fit_diagnostics, "attempted_update_count") == 1
    assert _as_int(fit_diagnostics, "successful_update_count") == 1
    assert _as_int(fit_diagnostics, "active_kernel_version", "kernel_version") == 6
    assert _as_int(updated, "active_kernel_version", "version") == 6

    assert fitted_means.shape == (2, 2)
    assert fitted_radii.shape == (2, 2)
    assert fitted_probabilities.shape == (2,)
    np.testing.assert_allclose(fitted_means, fixture.expected_means, atol=0.35)
    np.testing.assert_allclose(fitted_probabilities.sum(), 1.0, rtol=1e-12)
    assert np.all(np.isfinite(fitted_radii))
    assert np.all(fitted_radii > 0.0)
    assert np.all(fitted_probabilities > 0.0)
    assert np.any(
        np.max(fitted_radii, axis=1) / np.min(fitted_radii, axis=1) > 1.2
    )

    request = dispatch_request_cls(
        chain_id=f"{case.allocation_target}-chain",
        worker_id=f"{case.allocation_target}-worker",
        shell_epoch=case.shell_epoch + 1,
        allocation_target=case.allocation_target,
    )

    snapshot = updated.prepare_dispatch_snapshot(request)
    snapshot_diagnostics = _diagnostics(snapshot)
    worker_context = snapshot.direction_adaptation_context()

    assert _as_str(snapshot, "allocation_target") == case.allocation_target
    assert (
        _as_str(snapshot_diagnostics, "allocation_target")
        == case.allocation_target
    )
    assert _as_str(worker_context, "allocation_target") == case.allocation_target
    assert _as_int(snapshot, "kernel_version", "version") == 6
    assert (
        _as_int(snapshot_diagnostics, "kernel_version", "active_kernel_version")
        == 6
    )
    assert _as_int(snapshot, "shell_epoch") == case.shell_epoch + 1
    assert _as_int(snapshot_diagnostics, "shell_epoch") == case.shell_epoch + 1
    np.testing.assert_allclose(
        _as_array(worker_context, "component_means"),
        _as_array(fitted_kernel, "component_means"),
    )
    np.testing.assert_allclose(
        _as_array(worker_context, "component_radii"),
        _as_array(fitted_kernel, "component_radii"),
    )
    np.testing.assert_allclose(
        _as_array(worker_context, "component_probabilities"),
        _as_array(fitted_kernel, "component_probabilities"),
    )
