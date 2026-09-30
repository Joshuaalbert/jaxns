"""Correctness guards for the paper's phantom-prefix experiment."""

import numpy as np
import pytest

from benchmarks.paper_phantom_conditioning.posterior import classic_mode_mass
from benchmarks.paper_phantom_conditioning.summarise_directions import (
    _validate_posterior_schema,
)


def test_mode_mass_uses_classic_expected_posterior_once():
    """Phantom-prefix choice cannot alter a completed tree's posterior."""
    log_dp = np.log(np.asarray([0.1, 0.2, 0.3, 0.4]))  # [N]
    first_mode = np.asarray([True, False, True, False])  # [N]

    mode_mass = classic_mode_mass(log_dp, first_mode)

    assert mode_mass == pytest.approx(0.4)


def test_mode_mass_rejects_non_sample_aligned_membership():
    """A diagnostic cannot silently classify a different sample axis."""
    log_dp = np.log(np.asarray([0.25, 0.75]))  # [N]
    first_mode = np.asarray([True])  # [M]

    with pytest.raises(ValueError, match="must match"):
        classic_mode_mass(log_dp, first_mode)


def test_prefix_records_reject_posterior_diagnostics():
    """Historical prefix-specific spike masses cannot enter new tables."""
    records = [{
        "evidence": {
            "0": {"log_Z_mean": 1.0},
            "1": {"log_Z_mean": 1.0, "mode_mass_mean": 0.4},
        },
    }]

    with pytest.raises(ValueError, match="once per completed classic tree"):
        _validate_posterior_schema(records)


def test_classic_runner_constructs_sampler_on_develop(tmp_path, monkeypatch):
    """Default runs must not require the experimental phantom-seeding API."""
    from benchmarks.paper_phantom_conditioning import run

    class SamplerConstructed(Exception):
        """Stop after constructing the real sampler, before any sampling."""

    def stop_before_sampling():
        raise SamplerConstructed

    monkeypatch.setattr(run, "_environment", stop_before_sampling)
    monkeypatch.setattr("sys.argv", [
        "run.py",
        "--case", "basic_mvn",
        "--direction", "isotropic",
        "--seeds", "0",
        "--phase", "core",
        "--output", str(tmp_path / "results.jsonl"),
        "--state-root", str(tmp_path / "states"),
    ])
    with pytest.raises(SamplerConstructed):
        run.main()
