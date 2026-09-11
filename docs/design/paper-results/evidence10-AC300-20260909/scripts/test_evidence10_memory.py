"""Check actual-data extrapolation and scheduler forecast edge cases."""

import json
from pathlib import Path

import pytest

from evidence10_memory import Observation, fit_rss, project_tree, sample_exponents

SNAPSHOT = json.loads(Path(__file__).with_name('MEMORY_CALIBRATION_BASELINE.json').read_text())
OBSERVATIONS = [Observation(**row) for row in SNAPSHOT['observations']]


@pytest.mark.parametrize('phase', ['core', 'analysis'])
def test_largest_capacity_is_covered_when_held_out(phase):
    observations = OBSERVATIONS
    largest = max(row.capacity for row in observations)
    train = [row for row in observations if row.capacity < largest]
    held_out = [row for row in observations if row.capacity == largest and row.phase == phase]
    assert held_out
    fit = fit_rss(train, phase)
    assert all(fit.reserve(row.capacity, row.samples) >= row.peak_gib for row in held_out)


def test_real_matched_growth_is_used_and_capacity_is_not_trimmed():
    observations = OBSERVATIONS
    exponents = sample_exponents(observations)
    assert exponents[7] > 2.4
    record = dict(classic_samples=100, classic_log_Z_uncert=.02, state_capacity=1024)
    capacity, samples = project_tree(record, .01, 2.)
    assert samples == pytest.approx(440.)
    assert capacity == 1024


def test_oversized_memory_is_not_capped_to_machine_budget():
    fit = fit_rss(OBSERVATIONS, 'analysis')
    assert fit.reserve(100_000_000, 80_000_000) > 512
