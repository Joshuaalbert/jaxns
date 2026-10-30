"""Regression for variable reflection counts shadowed by likelihood block sizes."""

import numpy as np
import pytest

from benchmarks.issue_312.analyse_reflection_tuning import reflection_work_statistics


def test_work_diagnostics_use_actual_bounces_and_jittered_duration():
    bounces = np.array([10, 30])
    gradients = np.array([90, 270])
    steps = np.repeat([[0.0024], [0.0012]], 9, axis=1)  # [2, 9]
    result = reflection_work_statistics(bounces, gradients, steps)
    assert result['bounce_count_min'] == 10
    assert result['bounce_count_max'] == 30
    assert result['mean_bounce_count'] == 20
    assert result['duration_min'] == pytest.approx(0.024)
    assert result['duration_max'] == pytest.approx(0.036)
    assert result['mean_realised_duration'] == pytest.approx(0.03)


def test_work_diagnostics_reject_likelihood_multiplicities_as_bounces():
    # All likelihoods may be distinct: those block sizes cannot count reflections.
    block_sizes = np.ones(2, dtype=int)
    with pytest.raises(ValueError, match='independently recorded gradients'):
        reflection_work_statistics(block_sizes, np.array([90, 270]), np.full((2, 9), 0.003))
