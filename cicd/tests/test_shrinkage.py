import dataclasses
from typing import NamedTuple

import numpy as np
import pytest
from jax import numpy as jnp
from jax import random

import jaxns.shrinkage.classic as shrinkage
from jaxns.algorithm.race_tree import BlockState
from jaxns.shrinkage.classic import (
    DirichletConcentrations,
    classic_dirichlet_concentrations,
    expected_evidence_summary,
    expected_log_posterior_weights,
    sample_evidence,
)
from jaxns.shrinkage.phantom import sample_mc_shrinkage


@dataclasses.dataclass(frozen=True, slots=True)
class ShrinkageBlockFixture:
    log_L_blocks: np.ndarray
    block_size: np.ndarray
    incoming_K: np.ndarray
    block_sample_indices: np.ndarray

    def to_block_state(self) -> BlockState:
        num_blocks = self.log_L_blocks.shape[0]
        return BlockState(
            log_L_blocks=jnp.asarray(self.log_L_blocks),
            block_first_idx=jnp.arange(num_blocks, dtype=jnp.int32),
            block_size=jnp.asarray(self.block_size),
            incoming_K=jnp.asarray(self.incoming_K),
            block_out_degree=jnp.zeros((num_blocks,), dtype=jnp.int32),
            valid=jnp.ones((num_blocks,), dtype=jnp.bool_),
            block_sample_indices=jnp.asarray(self.block_sample_indices),
        )


class ExpectedConcentrations(NamedTuple):
    epsilon: np.ndarray
    alpha_gt: np.ndarray
    alpha_eq: np.ndarray
    alpha_lt: np.ndarray


def _logdiffexp(log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    return log_a + np.log1p(-np.exp(log_b - log_a))


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(
        x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)),
        axis=axis,
    )


def test_dirichlet_concentrations_use_plateau_epsilon_policy() -> None:
    blocks = ShrinkageBlockFixture(
        log_L_blocks=np.log(np.array([2.0, 5.0])),
        block_size=np.array([1, 3], dtype=np.int32),
        incoming_K=np.array([5, 4], dtype=np.int32),
        block_sample_indices=np.array(
            [
                [0, -1, -1],
                [1, 2, 3],
            ],
            dtype=np.int32,
        ),
    )
    expected = ExpectedConcentrations(
        epsilon=np.array([0.0, 0.5]),
        alpha_gt=np.array([5.0, 2.0]),
        alpha_eq=np.array([0.0, 3.5]),
        alpha_lt=np.array([1.0, 0.5]),
    )

    concentrations = classic_dirichlet_concentrations(blocks.to_block_state())

    np.testing.assert_allclose(
        np.asarray(concentrations.epsilon),
        expected.epsilon,
    )
    np.testing.assert_allclose(
        np.asarray(concentrations.alpha_gt),
        expected.alpha_gt,
    )
    np.testing.assert_allclose(
        np.asarray(concentrations.alpha_eq),
        expected.alpha_eq,
    )
    np.testing.assert_allclose(
        np.asarray(concentrations.alpha_lt),
        expected.alpha_lt,
    )


def test_dirichlet_concentrations_reject_K_g_less_than_m_g() -> None:
    blocks = ShrinkageBlockFixture(
        log_L_blocks=np.array([1.0]),
        block_size=np.array([2], dtype=np.int32),
        incoming_K=np.array([1], dtype=np.int32),
        block_sample_indices=np.array([[0, 1]], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="K_g|m_g|incoming|lineage"):
        classic_dirichlet_concentrations(blocks.to_block_state())


def test_sample_evidence_uses_strict_endpoint_recurrence() -> None:
    blocks = ShrinkageBlockFixture(
        log_L_blocks=np.log(np.array([2.0, 5.0, 7.0])),
        block_size=np.array([1, 1, 1], dtype=np.int32),
        incoming_K=np.array([8, 6, 4], dtype=np.int32),
        block_sample_indices=np.array([[0], [1], [2]], dtype=np.int32),
    ).to_block_state()
    concentrations = DirichletConcentrations(
        alpha_gt=jnp.array([8.0, 6.0, 4.0]),
        alpha_eq=jnp.array([2.0, 2.0, 2.0]),
        alpha_lt=jnp.array([1.0, 1.0, 1.0]),
        epsilon=jnp.array([1e-6, 1e-6, 1e-6]),
    )

    evidence = sample_evidence(
        key=random.PRNGKey(11),
        block_state=blocks,
        concentrations=concentrations,
        num_samples=3,
    )

    p_gt = np.asarray(evidence.p_gt_samples)
    log_X_g = np.cumsum(np.log(p_gt), axis=-1)
    log_X_prev = np.concatenate(
        [np.zeros((p_gt.shape[0], 1)), log_X_g[:, :-1]],
        axis=-1,
    )
    expected_log_dZ = _logdiffexp(log_X_prev, log_X_g)
    expected_log_dZ = (
        expected_log_dZ
        + np.asarray(blocks.log_L_blocks)[None, :]
    )
    expected_log_Z = _logsumexp(expected_log_dZ, axis=-1)

    np.testing.assert_allclose(
        np.asarray(evidence.log_dZ_samples),
        expected_log_dZ,
    )
    np.testing.assert_allclose(
        np.asarray(evidence.log_Z_samples),
        expected_log_Z,
    )


def test_mc_linear_evidence_moments_match_deterministic_expectations() -> None:
    """Check MC error against an explicit standard-error criterion."""
    block_state = ShrinkageBlockFixture(
        log_L_blocks=np.log(np.array([1.5, 3.0, 8.0])),
        block_size=np.array([1, 2, 1], dtype=np.int32),
        incoming_K=np.array([10, 7, 3], dtype=np.int32),
        block_sample_indices=np.array(
            [[0, -1], [1, 2], [3, -1]],
            dtype=np.int32,
        ),
    ).to_block_state()
    concentrations = classic_dirichlet_concentrations(block_state)
    expected = expected_evidence_summary(block_state, concentrations)
    num_draws = 65_536
    sampled = sample_evidence(
        key=random.PRNGKey(20260823),
        block_state=block_state,
        concentrations=concentrations,
        num_samples=num_draws,
    )
    z_samples = np.exp(np.asarray(sampled.log_Z_samples))

    expected_z = np.exp(float(expected.log_Z_linear_mean))
    expected_z2 = np.exp(float(expected.log_Z2_linear_mean))
    z_standard_error = np.std(z_samples, ddof=1) / np.sqrt(num_draws)
    z2_standard_error = np.std(z_samples ** 2, ddof=1) / np.sqrt(num_draws)
    assert abs(np.mean(z_samples) - expected_z) <= 5.0 * z_standard_error
    assert abs(np.mean(z_samples ** 2) - expected_z2) <= 5.0 * z2_standard_error


def test_posterior_weights_split_plateau_equality_atom_mass() -> None:
    blocks = ShrinkageBlockFixture(
        log_L_blocks=np.log(np.array([5.0, 7.0])),
        block_size=np.array([2, 1], dtype=np.int32),
        incoming_K=np.array([4, 2], dtype=np.int32),
        block_sample_indices=np.array(
            [
                [0, 2],
                [1, -1],
            ],
            dtype=np.int32,
        ),
    )
    block_state = blocks.to_block_state()
    concentrations = classic_dirichlet_concentrations(block_state)
    p_gt = np.asarray(concentrations.alpha_gt)
    p_eq = np.asarray(concentrations.alpha_eq)
    alpha0 = (
        p_gt
        + p_eq
        + np.asarray(concentrations.alpha_lt)
    )
    p_gt = p_gt / alpha0
    p_eq = p_eq / alpha0
    X_prev = np.array([1.0, p_gt[0]])

    raw_sample_weights = np.array(
        [
            5.0 * X_prev[0] * p_eq[0] / 2.0,
            7.0 * X_prev[1] * (1.0 - p_gt[1]),
            5.0 * X_prev[0] * p_eq[0] / 2.0,
        ]
    )
    expected_sample_weights = raw_sample_weights / np.sum(raw_sample_weights)

    log_weights = expected_log_posterior_weights(
        block_state,
        concentrations,
    )

    np.testing.assert_allclose(
        np.exp(np.asarray(log_weights)),
        expected_sample_weights,
    )
    assert expected_sample_weights[0] == expected_sample_weights[2]


def test_no_phantom_mc_shrinkage_uses_canonical_plateau_blocks() -> None:
    log_l = np.log(np.array([2.0, 3.0, 3.0, 5.0]))
    evidence = sample_mc_shrinkage(
        key=random.PRNGKey(17),
        log_L_constraints=jnp.full((4,), -jnp.inf),
        log_L_classic=jnp.asarray(log_l),
        K_classic=jnp.asarray([3, 2, 2, 1], dtype=jnp.int32),
        valid_phantom=jnp.zeros((4,), dtype=jnp.bool_),
        log_L_phantom=jnp.zeros((4, 0)),
        num_samples=jnp.asarray(4, dtype=jnp.int32),
        num_Z_samples=4,
    )

    valid = np.asarray(evidence.log_L_blocks) < np.inf
    np.testing.assert_allclose(
        np.asarray(evidence.log_L_blocks)[valid],
        np.log(np.array([2.0, 3.0, 5.0])),
    )
    np.testing.assert_array_equal(
        np.asarray(evidence.block_first_idx)[valid],
        np.array([0, 1, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(evidence.block_size)[valid],
        np.array([1, 2, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(evidence.incoming_K)[valid],
        np.array([3, 2, 1], dtype=np.int32),
    )
    assert np.asarray(evidence.log_Z_samples).shape == (4,)


def test_public_mc_shrinkage_rejects_invalid_plateau_capacity() -> None:
    with pytest.raises(ValueError, match="K_g|m_g|incoming|plateau"):
        sample_mc_shrinkage(
            key=random.PRNGKey(19),
            log_L_constraints=jnp.full((2,), -jnp.inf),
            log_L_classic=jnp.asarray([0.0, 0.0]),
            K_classic=jnp.asarray([1, 1], dtype=jnp.int32),
            valid_phantom=jnp.zeros((2,), dtype=jnp.bool_),
            log_L_phantom=jnp.zeros((2, 0)),
            num_samples=jnp.asarray(2, dtype=jnp.int32),
            num_Z_samples=4,
        )


def test_public_mc_shrinkage_rejects_plateau_when_first_K_too_small() -> None:
    with pytest.raises(ValueError, match="K_g|m_g|incoming|plateau"):
        sample_mc_shrinkage(
            key=random.PRNGKey(29),
            log_L_constraints=jnp.full((3,), -jnp.inf),
            log_L_classic=jnp.asarray([0.0, 0.0, 1.0]),
            K_classic=jnp.asarray([1, 2, 1], dtype=jnp.int32),
            valid_phantom=jnp.zeros((3,), dtype=jnp.bool_),
            log_L_phantom=jnp.zeros((3, 0)),
            num_samples=jnp.asarray(3, dtype=jnp.int32),
            num_Z_samples=4,
        )


def test_no_phantom_mc_shrinkage_uses_supplied_block_incoming_K() -> None:
    block_state = BlockState(
        log_L_blocks=jnp.asarray([np.log(3.0), jnp.inf]),
        block_first_idx=jnp.asarray([1, -1], dtype=jnp.int32),
        block_size=jnp.asarray([2, 0], dtype=jnp.int32),
        incoming_K=jnp.asarray([2, 0], dtype=jnp.int32),
        block_out_degree=jnp.asarray([0, 0], dtype=jnp.int32),
        valid=jnp.asarray([True, False]),
        block_sample_indices=jnp.asarray(
            [
                [1, 0],
                [-1, -1],
            ],
            dtype=jnp.int32,
        ),
    )

    evidence = sample_mc_shrinkage(
        key=random.PRNGKey(23),
        log_L_constraints=jnp.full((2,), -jnp.inf),
        log_L_classic=jnp.log(jnp.asarray([3.0, 3.0])),
        K_classic=jnp.asarray([1, 1], dtype=jnp.int32),
        valid_phantom=jnp.zeros((2,), dtype=jnp.bool_),
        log_L_phantom=jnp.zeros((2, 0)),
        num_samples=jnp.asarray(2, dtype=jnp.int32),
        num_Z_samples=4,
        block_state=block_state,
    )

    valid = np.asarray(evidence.log_L_blocks) < np.inf
    np.testing.assert_array_equal(
        np.asarray(evidence.incoming_K)[valid],
        np.array([2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(evidence.block_size)[valid],
        np.array([2], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(evidence.log_L_blocks)[valid],
        np.array([np.log(3.0)]),
    )


def _require_callable(name: str):
    fn = getattr(shrinkage, name, None)
    assert callable(fn), f"jaxns.shrinkage.classic.{name} is required by Ticket 0013."
    return fn


def _single_block_public_phantom_case():
    log_L_classic = np.zeros((20,), dtype=float)
    log_L_constraints = np.full((20,), -np.inf, dtype=float)
    K_classic = np.full((20,), 40, dtype=np.int32)
    valid_phantom = np.ones((20,), dtype=bool)
    log_L_phantom = np.concatenate(
        [
            np.full((12, 1), 1.0, dtype=float),
            np.full((4, 1), 0.0, dtype=float),
            np.full((4, 1), -1.0, dtype=float),
        ],
        axis=0,
    )
    return (
        log_L_constraints,
        log_L_classic,
        K_classic,
        valid_phantom,
        log_L_phantom,
    )


def _single_block_public_equivalent_cluster_cases():
    log_L_classic = np.zeros((20,), dtype=float)
    log_L_constraints = np.full((20,), -np.inf, dtype=float)
    K_classic = np.full((20,), 40, dtype=np.int32)
    phantom_values = np.asarray(
        [1.0] * 8 + [0.0] * 4 + [-1.0] * 8,
        dtype=float,
    )

    singleton_valid = np.ones((20,), dtype=bool)
    singleton_log_L_phantom = phantom_values[:, None]

    correlated_valid = np.zeros((20,), dtype=bool)
    correlated_valid[0] = True
    correlated_log_L_phantom = np.full((20, 20), -1.0, dtype=float)
    correlated_log_L_phantom[0] = phantom_values

    return (
        log_L_constraints,
        log_L_classic,
        K_classic,
        singleton_valid,
        singleton_log_L_phantom,
        correlated_valid,
        correlated_log_L_phantom,
    )


def _assert_probability_sample_mean(samples, expected: np.ndarray, *, atol: float) -> None:
    for field_name in ("p_gt_samples", "p_eq_samples", "p_lt_samples"):
        assert hasattr(samples, field_name), (
            "MC shrinkage samples must expose block probability samples "
            f"{field_name} so tests can verify the active shrinkage target."
        )
    got = np.stack(
        [
            np.asarray(samples.p_gt_samples)[:, 0],
            np.asarray(samples.p_eq_samples)[:, 0],
            np.asarray(samples.p_lt_samples)[:, 0],
        ],
        axis=-1,
    )
    np.testing.assert_allclose(np.mean(got, axis=0), expected, atol=atol)


def _probability_matrix(samples) -> np.ndarray:
    return np.stack(
        [
            np.asarray(samples.p_gt_samples)[:, 0],
            np.asarray(samples.p_eq_samples)[:, 0],
            np.asarray(samples.p_lt_samples)[:, 0],
        ],
        axis=-1,
    )


def test_public_phantom_eval_sample_mc_shrinkage_uses_gamma_conditioning_when_gate_active():
    (
        log_L_constraints,
        log_L_classic,
        K_classic,
        valid_phantom,
        log_L_phantom,
    ) = _single_block_public_phantom_case()
    num_draws = 4096

    active = sample_mc_shrinkage(
        key=random.PRNGKey(101),
        log_L_constraints=jnp.asarray(log_L_constraints),
        log_L_classic=jnp.asarray(log_L_classic),
        K_classic=jnp.asarray(K_classic),
        valid_phantom=jnp.asarray(valid_phantom),
        log_L_phantom=jnp.asarray(log_L_phantom),
        num_samples=jnp.asarray(20, dtype=jnp.int32),
        num_Z_samples=num_draws,
        C_min=20,
    )
    inactive = sample_mc_shrinkage(
        key=random.PRNGKey(101),
        log_L_constraints=jnp.asarray(log_L_constraints),
        log_L_classic=jnp.asarray(log_L_classic),
        K_classic=jnp.asarray(K_classic),
        valid_phantom=jnp.asarray(valid_phantom),
        log_L_phantom=jnp.asarray(log_L_phantom),
        num_samples=jnp.asarray(20, dtype=jnp.int32),
        num_Z_samples=num_draws,
        C_min=21,
    )

    classic_alpha = np.asarray([21.0, 20.5, 0.5], dtype=float)
    active_alpha = np.asarray([33.0, 24.5, 4.5], dtype=float)
    _assert_probability_sample_mean(
        inactive,
        classic_alpha / np.sum(classic_alpha),
        atol=0.02,
    )
    _assert_probability_sample_mean(
        active,
        active_alpha / np.sum(active_alpha),
        atol=0.02,
    )
    assert abs(
        float(np.mean(np.asarray(active.p_gt_samples)[:, 0]))
        - float(np.mean(np.asarray(inactive.p_gt_samples)[:, 0]))
    ) > 0.03
    np.testing.assert_array_equal(
        np.asarray(active.phantom_gate_active, dtype=bool),
        np.asarray([True]),
    )
    np.testing.assert_array_equal(
        np.asarray(inactive.phantom_gate_active, dtype=bool),
        np.asarray([False]),
    )
    np.testing.assert_allclose(
        np.asarray(active.kish_participating_cluster_counts),
        np.asarray([20.0]),
    )


def test_public_sample_mc_shrinkage_preserves_means_and_inflates_correlated_cluster_variance():
    (
        log_L_constraints,
        log_L_classic,
        K_classic,
        singleton_valid,
        singleton_log_L_phantom,
        correlated_valid,
        correlated_log_L_phantom,
    ) = _single_block_public_equivalent_cluster_cases()
    num_draws = 12000

    singleton = sample_mc_shrinkage(
        key=random.PRNGKey(301),
        log_L_constraints=jnp.asarray(log_L_constraints),
        log_L_classic=jnp.asarray(log_L_classic),
        K_classic=jnp.asarray(K_classic),
        valid_phantom=jnp.asarray(singleton_valid),
        log_L_phantom=jnp.asarray(singleton_log_L_phantom),
        num_samples=jnp.asarray(20, dtype=jnp.int32),
        num_Z_samples=num_draws,
        C_min=1,
    )
    correlated = sample_mc_shrinkage(
        key=random.PRNGKey(301),
        log_L_constraints=jnp.asarray(log_L_constraints),
        log_L_classic=jnp.asarray(log_L_classic),
        K_classic=jnp.asarray(K_classic),
        valid_phantom=jnp.asarray(correlated_valid),
        log_L_phantom=jnp.asarray(correlated_log_L_phantom),
        num_samples=jnp.asarray(20, dtype=jnp.int32),
        num_Z_samples=num_draws,
        C_min=1,
    )

    np.testing.assert_allclose(
        np.asarray(singleton.kish_participating_cluster_counts),
        np.asarray([20.0]),
    )
    np.testing.assert_allclose(
        np.asarray(correlated.kish_participating_cluster_counts),
        np.asarray([1.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(singleton.phantom_gate_active, dtype=bool),
        np.asarray([True]),
    )
    np.testing.assert_array_equal(
        np.asarray(correlated.phantom_gate_active, dtype=bool),
        np.asarray([True]),
    )

    expected_phantom_additions = np.asarray([8.0, 4.0, 8.0])
    for samples in (singleton, correlated):
        for field_name in (
            "phantom_add_gt_samples",
            "phantom_add_eq_samples",
            "phantom_add_lt_samples",
        ):
            assert hasattr(samples, field_name), (
                "Public EvidenceSamples must expose sampled gamma-weighted "
                f"phantom additions via {field_name}."
            )
        sampled_means = np.asarray(
            [
                np.mean(np.asarray(samples.phantom_add_gt_samples)[:, 0]),
                np.mean(np.asarray(samples.phantom_add_eq_samples)[:, 0]),
                np.mean(np.asarray(samples.phantom_add_lt_samples)[:, 0]),
            ]
        )
        np.testing.assert_allclose(
            sampled_means,
            expected_phantom_additions,
            atol=0.35,
        )

    singleton_probabilities = _probability_matrix(singleton)
    correlated_probabilities = _probability_matrix(correlated)
    np.testing.assert_allclose(
        np.mean(correlated_probabilities, axis=0),
        np.mean(singleton_probabilities, axis=0),
        atol=0.04,
    )

    singleton_variance = np.var(singleton_probabilities, axis=0)
    correlated_variance = np.var(correlated_probabilities, axis=0)
    assert correlated_variance[1] > 1.2 * singleton_variance[1]
    assert correlated_variance[2] > 1.5 * singleton_variance[2]


def test_gamma_weighted_helper_keeps_plateau_equality_and_open_interval_separate() -> None:
    helper = _require_callable("gamma_weighted_phantom_probabilities_from_draws")
    block_state = ShrinkageBlockFixture(
        log_L_blocks=np.asarray([0.0]),
        block_size=np.asarray([2], dtype=np.int32),
        incoming_K=np.asarray([5], dtype=np.int32),
        block_sample_indices=np.asarray([[0, 1]], dtype=np.int32),
    ).to_block_state()

    draw = helper(
        block_state=block_state,
        A_cg=jnp.asarray([[5.0]]),
        B_cg=jnp.asarray([[2.0]]),
        E_cg=jnp.asarray([[1.0]]),
        race_gamma_gt=jnp.asarray([3.0]),
        race_gamma_eq=jnp.asarray([4.0]),
        race_gamma_lt=jnp.asarray([5.0]),
        cluster_weights=jnp.asarray([7.0]),
        C_min=1,
    )

    expected_gt = 3.0 + 7.0 * 2.0
    expected_eq = 4.0 + 7.0 * 1.0
    expected_lt = 5.0 + 7.0 * (5.0 - 2.0 - 1.0)
    expected_total = expected_gt + expected_eq + expected_lt
    np.testing.assert_allclose(np.asarray(draw.p_gt), [expected_gt / expected_total])
    np.testing.assert_allclose(np.asarray(draw.p_eq), [expected_eq / expected_total])
    np.testing.assert_allclose(np.asarray(draw.p_lt), [expected_lt / expected_total])
    np.testing.assert_allclose(np.asarray(draw.phantom_add_lt), [14.0])


def test_gamma_weighted_helper_keeps_singleton_block_two_class() -> None:
    helper = _require_callable("gamma_weighted_phantom_probabilities_from_draws")
    block_state = ShrinkageBlockFixture(
        log_L_blocks=np.asarray([0.0]),
        block_size=np.asarray([1], dtype=np.int32),
        incoming_K=np.asarray([5], dtype=np.int32),
        block_sample_indices=np.asarray([[0]], dtype=np.int32),
    ).to_block_state()

    draw = helper(
        block_state=block_state,
        A_cg=jnp.asarray([[5.0]]),
        B_cg=jnp.asarray([[2.0]]),
        E_cg=jnp.asarray([[1.0]]),
        race_gamma_gt=jnp.asarray([3.0]),
        race_gamma_eq=jnp.asarray([4.0]),
        race_gamma_lt=jnp.asarray([5.0]),
        cluster_weights=jnp.asarray([7.0]),
        C_min=1,
    )

    expected_gt = 3.0 + 7.0 * 2.0
    expected_lt = 5.0 + 7.0 * (5.0 - 2.0)
    expected_total = expected_gt + expected_lt
    np.testing.assert_allclose(np.asarray(draw.p_gt), [expected_gt / expected_total])
    np.testing.assert_allclose(np.asarray(draw.p_eq), [0.0])
    np.testing.assert_allclose(np.asarray(draw.p_lt), [expected_lt / expected_total])
    np.testing.assert_allclose(np.asarray(draw.phantom_add_eq), [0.0])
    np.testing.assert_allclose(np.asarray(draw.phantom_add_lt), [21.0])


def test_gamma_weighted_helper_inactive_gate_ignores_phantom_counts_exactly() -> None:
    helper = _require_callable("gamma_weighted_phantom_probabilities_from_draws")
    block_state = ShrinkageBlockFixture(
        log_L_blocks=np.asarray([0.0, 1.0]),
        block_size=np.asarray([2, 2], dtype=np.int32),
        incoming_K=np.asarray([5, 4], dtype=np.int32),
        block_sample_indices=np.asarray([[0, 1], [2, 3]], dtype=np.int32),
    ).to_block_state()
    race_gt = jnp.asarray([2.0, 3.0])
    race_eq = jnp.asarray([5.0, 7.0])
    race_lt = jnp.asarray([11.0, 13.0])

    draw = helper(
        block_state=block_state,
        A_cg=jnp.asarray([[100.0, 0.0]]),
        B_cg=jnp.asarray([[90.0, 0.0]]),
        E_cg=jnp.asarray([[5.0, 0.0]]),
        race_gamma_gt=race_gt,
        race_gamma_eq=race_eq,
        race_gamma_lt=race_lt,
        cluster_weights=jnp.asarray([1000.0]),
        C_min=20,
    )

    total = np.asarray(race_gt + race_eq + race_lt)
    np.testing.assert_allclose(np.asarray(draw.p_gt), np.asarray(race_gt) / total)
    np.testing.assert_allclose(np.asarray(draw.p_eq), np.asarray(race_eq) / total)
    np.testing.assert_allclose(np.asarray(draw.p_lt), np.asarray(race_lt) / total)
    np.testing.assert_array_equal(
        np.asarray(draw.phantom_gate_active, dtype=bool),
        np.asarray([False, False]),
    )


def test_phantom_gate_rejects_zero_participation_when_threshold_is_zero() -> None:
    gate_fn = _require_callable("compute_phantom_gate_active")

    gate = gate_fn(jnp.zeros((3, 2)), C_min=0)

    np.testing.assert_array_equal(
        np.asarray(gate, dtype=bool),
        np.asarray([False, False]),
    )


def test_sample_mc_shrinkage_no_phantom_path_does_not_require_old_rho_diagnostics() -> None:
    evidence = sample_mc_shrinkage(
        key=random.PRNGKey(31),
        log_L_constraints=jnp.full((3,), -jnp.inf),
        log_L_classic=jnp.asarray([0.0, 1.0, 2.0]),
        K_classic=jnp.asarray([5, 4, 3], dtype=jnp.int32),
        valid_phantom=jnp.zeros((3,), dtype=jnp.bool_),
        log_L_phantom=jnp.zeros((3, 0)),
        num_samples=jnp.asarray(3, dtype=jnp.int32),
        num_Z_samples=4,
        C_min=20,
    )

    assert hasattr(evidence, "kish_participating_cluster_counts")
    assert hasattr(evidence, "phantom_gate_active")
    np.testing.assert_allclose(
        np.asarray(evidence.kish_participating_cluster_counts),
        np.zeros((3,), dtype=float),
    )
    np.testing.assert_array_equal(
        np.asarray(evidence.phantom_gate_active, dtype=bool),
        np.asarray([False, False, False]),
    )
    for old_name in ("rho_samples", "rho_values", "rho_fit", "rho_eta_samples"):
        if hasattr(evidence, old_name):
            assert getattr(evidence, old_name) is None
            assert old_name in getattr(evidence, "deprecated_fields", ())
