"""Regression for constant columns in retained seed-72 movement diagnostics."""

import json

import numpy as np
import pytest

from benchmarks.issue_312.online_audit import compression_movement
from benchmarks.issue_312.tuning_common import diagnostic_correlation


@pytest.mark.parametrize('start, end', [
    (0.47555438706486874, [0.4755387889503847, 0.47214392640577796, 0.47556070035298637]),
    (0.4895718315962462, [0.48958615042424186, 0.5143854667175862, 0.4895725803222121]),
])
def test_identical_seed_values_and_their_squares_are_undefined(start, end):
    starts, ends = np.full(3, start), np.asarray(end)
    assert diagnostic_correlation(starts, ends) is None
    assert diagnostic_correlation(ends, starts) is None
    assert diagnostic_correlation(starts**2, ends**2) is None
    assert diagnostic_correlation(ends**2, starts**2) is None


def test_finite_correlation_and_undefined_underflow():
    values = np.array([.1, .2, .4])
    assert diagnostic_correlation(values, values * 2) == pytest.approx(1.)
    assert diagnostic_correlation(values, 1-values) == pytest.approx(-1.)
    assert diagnostic_correlation(values[:1], values[:1]) is None
    # Nonzero exact range alone does not guarantee finite floating-point variance.
    tiny = np.array([1e-300, 2e-300, 3e-300])
    assert np.ptp(tiny) > 0
    assert diagnostic_correlation(tiny, tiny) is None
    with pytest.raises(ValueError, match='matching one-dimensional'):
        diagnostic_correlation(values, values[:2])


def test_compression_record_with_constant_starts_is_strict_json():
    # Three children share one seed; repeated seed columns must not become NaN.
    points = np.full((4, 18), .47555438706486874)
    points[1:, 4] = [.4755387889503847, .47214392640577796, .47556070035298637]
    points[0, 12] = .4895718315962462
    points[1:, 12] = [.48958615042424186, .5143854667175862, .4895725803222121]
    ledger = dict(
        sample_indices=np.array([1, 2, 3]), seed_indices=np.zeros(3, dtype=int),
        compression=np.array([44.1, 44.2, 44.3]),
        work=np.array([120, 121, 122]), gradient_evaluations=np.full(3, 90),
        log_L_constraints=np.array([37., 37.1, 37.2]),
    )
    record, = compression_movement({'unit_samples': points}, ledger)
    assert record['pairs'] == 3
    assert record['total_calls'] == 363 and record['gradient_calls'] == 270
    for coordinate in (4, 12):
        assert record['correlations'][coordinate] is None
        assert record['squared_coordinate_correlations'][coordinate] is None
    assert json.loads(json.dumps(record, allow_nan=False)) == record
