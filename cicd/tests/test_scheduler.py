"""Tests for the frozen maximal-thread scheduler."""

import jax
import numpy as np
from jax import numpy as jnp

from jaxns.algorithm.depth import _decompose_gap
from jaxns.algorithm.scheduler import decompose_gap_python


def _coverage(
        size: int,
        starts: np.ndarray,
        terminals: np.ndarray,
        multiplicities: np.ndarray,
) -> np.ndarray:
    """Reconstruct the integer gap covered by compressed T(a, b) runs."""
    difference = np.zeros((size + 1,), dtype=np.int64)
    for start, terminal, count in zip(
            starts,
            terminals,
            multiplicities,
            strict=True,
    ):
        difference[start] += count
        difference[terminal + 1] -= count
    return np.cumsum(difference[:-1])


def test_jax_decomposition_matches_python_reference_and_exact_coverage():
    rng = np.random.default_rng(290)
    compiled_decomposition = jax.jit(_decompose_gap)
    cases = [
        np.asarray([1, 2, 1]),
        np.asarray([2, 1, 2]),
        np.asarray([0, 3, 3, 1, 0]),
    ]
    cases.extend(
        rng.integers(0, 20, size=31, dtype=np.int32)
        for _ in range(100)
    )

    for gap in cases:
        expected = decompose_gap_python(gap)
        starts, terminals, multiplicities, num_runs = compiled_decomposition(
            jnp.asarray(gap),
        )
        count = int(num_runs)
        got = tuple(
            np.asarray(values[:count], dtype=np.int64)
            for values in (starts, terminals, multiplicities)
        )

        for got_values, expected_values in zip(got, expected, strict=True):
            np.testing.assert_array_equal(got_values, expected_values)
        np.testing.assert_array_equal(_coverage(gap.size, *got), gap)
        assert count <= gap.size


def test_decomposition_keeps_large_multiplicity_compressed():
    gap = np.asarray([0, 1_000_000_000, 1_000_000_000, 0])
    starts, terminals, multiplicities = decompose_gap_python(gap)

    np.testing.assert_array_equal(starts, np.asarray([1]))
    np.testing.assert_array_equal(terminals, np.asarray([2]))
    np.testing.assert_array_equal(
        multiplicities,
        np.asarray([1_000_000_000]),
    )
