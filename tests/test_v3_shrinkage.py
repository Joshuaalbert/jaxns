import dataclasses
from typing import NamedTuple

from jax import numpy as jnp
from jax import random
import numpy as np
import pytest

from jaxns.phantom_eval import sample_mc_shrinkage
from jaxns.race_tree import BlockState
from jaxns.v3_shrinkage import (
    DirichletConcentrations,
    classic_dirichlet_concentrations,
    expected_v3_log_posterior_weights,
    sample_v3_evidence,
)


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


def test_v3_dirichlet_concentrations_use_plateau_epsilon_policy() -> None:
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
        epsilon=np.array([1e-6, 0.5]),
        alpha_gt=np.array([5.0, 2.0]),
        alpha_eq=np.array([1.000001, 3.5]),
        alpha_lt=np.array([0.999999, 0.5]),
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


def test_v3_dirichlet_concentrations_reject_K_g_less_than_m_g() -> None:
    blocks = ShrinkageBlockFixture(
        log_L_blocks=np.array([1.0]),
        block_size=np.array([2], dtype=np.int32),
        incoming_K=np.array([1], dtype=np.int32),
        block_sample_indices=np.array([[0, 1]], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="K_g|m_g|incoming|lineage"):
        classic_dirichlet_concentrations(blocks.to_block_state())


def test_sample_v3_evidence_uses_strict_endpoint_recurrence() -> None:
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

    evidence = sample_v3_evidence(
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

    log_weights = expected_v3_log_posterior_weights(
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
