import dataclasses
import inspect
from typing import NamedTuple

import numpy as np
import pytest
from jax import numpy as jnp

from jaxns.algorithm.race_tree import build_block_state
from jaxns.samples import PhantomSamples, Samples


@dataclasses.dataclass(frozen=True, slots=True)
class RaceTreeFixture:
    root_out_degree: int
    sample_indices: np.ndarray
    log_likelihoods: np.ndarray
    log_L_constraints: np.ndarray
    out_degree: np.ndarray

    def permute(self, order: np.ndarray) -> "RaceTreeFixture":
        return dataclasses.replace(
            self,
            sample_indices=self.sample_indices[order],
            log_likelihoods=self.log_likelihoods[order],
            log_L_constraints=self.log_L_constraints[order],
            out_degree=self.out_degree[order],
        )


class ExpectedBlocks(NamedTuple):
    log_L_g: np.ndarray
    m_g: np.ndarray
    K_g: np.ndarray
    out_degree_sum: np.ndarray
    block_start: np.ndarray
    block_stop: np.ndarray
    block_sample_indices: np.ndarray


def _plateau_race_tree_fixture() -> RaceTreeFixture:
    return RaceTreeFixture(
        root_out_degree=3,
        sample_indices=np.array(
            [100, 101, 102, 103, 104, 105],
            dtype=np.int32,
        ),
        log_likelihoods=np.array([2.0, 1.0, 2.0, 4.0, 3.0, 3.0]),
        log_L_constraints=np.array(
            [1.0, -np.inf, -np.inf, 3.0, 2.0, 1.0],
        ),
        out_degree=np.array([1, 2, 0, 0, 0, 0], dtype=np.int32),
    )


def _expected_plateau_blocks() -> ExpectedBlocks:
    return ExpectedBlocks(
        log_L_g=np.array([1.0, 2.0, 3.0, 4.0]),
        m_g=np.array([1, 2, 2, 1], dtype=np.int32),
        K_g=np.array([3, 4, 3, 1], dtype=np.int32),
        out_degree_sum=np.array([2, 1, 0, 0], dtype=np.int32),
        block_start=np.array([0, 1, 3, 5], dtype=np.int32),
        block_stop=np.array([1, 3, 5, 6], dtype=np.int32),
        block_sample_indices=np.array(
            [
                [101, -1],
                [100, 102],
                [104, 105],
                [103, -1],
            ],
            dtype=np.int32,
        ),
    )


def _derive_blocks(fixture: RaceTreeFixture):
    num_samples = fixture.log_likelihoods.shape[0]
    samples = Samples(
        log_L_constraints=jnp.asarray(fixture.log_L_constraints),
        log_likelihoods=jnp.asarray(fixture.log_likelihoods),
        U_samples=jnp.zeros((num_samples, 1)),
        out_degree=jnp.asarray(fixture.out_degree),
        num_likelihood_evaluations=jnp.ones(
            (num_samples,),
            dtype=jnp.int32,
        ),
        phantom_samples=PhantomSamples(
            U_samples=jnp.zeros((num_samples, 0, 1)),
            valid_mask=jnp.zeros((num_samples, 0), dtype=jnp.bool_),
            log_L=jnp.zeros((num_samples, 0)),
        ),
    )
    return build_block_state(
        samples=samples,
        root_out_degree=jnp.asarray(fixture.root_out_degree, dtype=jnp.int32),
        num_samples=jnp.asarray(num_samples, dtype=jnp.int32),
        sample_indices=jnp.asarray(fixture.sample_indices, dtype=jnp.int32),
        validate=True,
    )


def _assert_blocks_match_expected(blocks, expected: ExpectedBlocks) -> None:
    valid = np.asarray(blocks.valid, dtype=bool)
    np.testing.assert_allclose(
        np.asarray(blocks.log_L_blocks)[valid],
        expected.log_L_g,
    )
    np.testing.assert_array_equal(
        np.asarray(blocks.block_size)[valid],
        expected.m_g,
    )
    np.testing.assert_array_equal(
        np.asarray(blocks.incoming_K)[valid],
        expected.K_g,
    )
    np.testing.assert_array_equal(
        np.asarray(blocks.block_out_degree)[valid],
        expected.out_degree_sum,
    )
    np.testing.assert_array_equal(
        np.asarray(blocks.block_start),
        expected.block_start,
    )
    np.testing.assert_array_equal(
        np.asarray(blocks.block_stop),
        expected.block_stop,
    )
    np.testing.assert_array_equal(
        np.asarray(blocks.block_sample_indices),
        expected.block_sample_indices,
    )


def test_canonical_block_state_groups_plateaus_and_incoming_K_g() -> None:
    fixture = _plateau_race_tree_fixture()
    expected = _expected_plateau_blocks()

    blocks = _derive_blocks(fixture)

    _assert_blocks_match_expected(blocks, expected)
    np.testing.assert_array_equal(
        expected.K_g[:-1] - expected.m_g[:-1] + expected.out_degree_sum[:-1],
        expected.K_g[1:],
    )
    assert (
        expected.K_g[-1]
        - expected.m_g[-1]
        + expected.out_degree_sum[-1]
        == 0
    )


def test_canonical_block_state_is_invariant_to_generation_order() -> None:
    fixture = _plateau_race_tree_fixture()
    permuted = fixture.permute(np.array([5, 3, 0, 2, 1, 4], dtype=np.int32))
    expected = _expected_plateau_blocks()

    blocks = _derive_blocks(permuted)

    _assert_blocks_match_expected(blocks, expected)


def test_canonical_block_state_has_no_persisted_parent_field() -> None:
    forbidden_names = {
        "parent",
        "parents",
        "parent_idx",
        "parent_idxs",
        "parent_index",
        "parent_indices",
    }
    signature_names = set(inspect.signature(build_block_state).parameters)
    assert signature_names.isdisjoint(forbidden_names)

    blocks = _derive_blocks(_plateau_race_tree_fixture())
    if dataclasses.is_dataclass(blocks):
        field_names = {field.name for field in dataclasses.fields(blocks)}
    else:
        field_names = set(getattr(blocks, "_fields", ()))

    assert field_names.isdisjoint(forbidden_names)
    for name in forbidden_names:
        assert not hasattr(blocks, name)


def test_canonical_block_state_rejects_plateau_with_too_few_lineages() -> None:
    fixture = RaceTreeFixture(
        root_out_degree=1,
        sample_indices=np.array([10, 11], dtype=np.int32),
        log_likelihoods=np.array([1.0, 1.0]),
        log_L_constraints=np.array([-np.inf, -np.inf]),
        out_degree=np.array([1, 0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="K_g|m_g|incoming|lineage"):
        _derive_blocks(fixture)


def test_default_block_state_keeps_negative_infinity_likelihood_block() -> None:
    samples = Samples(
        log_L_constraints=jnp.array([-jnp.inf, -jnp.inf]),
        log_likelihoods=jnp.array([-jnp.inf, 0.0]),
        U_samples=jnp.zeros((2, 1)),
        out_degree=jnp.array([1, 0], dtype=jnp.int32),
        num_likelihood_evaluations=jnp.ones((2,), dtype=jnp.int32),
        phantom_samples=PhantomSamples(
            U_samples=jnp.zeros((2, 0, 1)),
            valid_mask=jnp.zeros((2, 0), dtype=jnp.bool_),
            log_L=jnp.zeros((2, 0)),
        ),
    )

    blocks = build_block_state(
        samples=samples,
        root_out_degree=jnp.asarray(1, dtype=jnp.int32),
        num_samples=jnp.asarray(2, dtype=jnp.int32),
    )

    valid = np.asarray(blocks.valid, dtype=bool)
    np.testing.assert_array_equal(valid, np.array([True, True]))
    np.testing.assert_allclose(
        np.asarray(blocks.log_L_blocks)[valid],
        np.array([-np.inf, 0.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(blocks.block_size)[valid],
        np.array([1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(blocks.block_start)[valid],
        np.array([0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(blocks.block_stop)[valid],
        np.array([1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(blocks.block_sample_indices),
        np.array([0, 1], dtype=np.int32),
    )


def test_block_state_rejects_strict_contour_violation() -> None:
    fixture = RaceTreeFixture(
        root_out_degree=1,
        sample_indices=np.array([10], dtype=np.int32),
        log_likelihoods=np.array([1.0]),
        log_L_constraints=np.array([1.0]),
        out_degree=np.array([0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="Strict contour"):
        _derive_blocks(fixture)


def test_block_state_rejects_inconsistent_out_degree_total() -> None:
    fixture = RaceTreeFixture(
        root_out_degree=2,
        sample_indices=np.array([10], dtype=np.int32),
        log_likelihoods=np.array([1.0]),
        log_L_constraints=np.array([-np.inf]),
        out_degree=np.array([0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="out-degree total"):
        _derive_blocks(fixture)
