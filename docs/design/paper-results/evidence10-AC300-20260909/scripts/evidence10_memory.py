"""Empirical memory forecasts for the frozen SS10 uncertainty ladder."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

STAGES = ('0.05', '0.02', '0.01', '0.005')


@dataclass(frozen=True, slots=True)
class Observation:
    stage: str
    seed: int
    phase: str
    capacity: int
    samples: int
    sigma: float
    peak_gib: float


@dataclass(frozen=True, slots=True)
class RssFit:
    intercept_gib: float
    gib_per_million_capacity: float
    gib_per_million_samples: float
    residual_allowance_gib: float
    observations: int
    headroom: float = 1.2

    def reserve(self, capacity: int, samples: float) -> int:
        """Return uncapped GiB; an oversized task must not be silently admitted."""
        predicted = (
            self.intercept_gib + self.gib_per_million_capacity * capacity / 1e6
            + self.gib_per_million_samples * samples / 1e6
            + self.residual_allowance_gib
        )
        return max(8, math.ceil(self.headroom * predicted))


def collect_observations(root: Path, through: str) -> list[Observation]:
    """Read completed phases up to the preceding stage, without loading states."""
    observations = []
    for stage in STAGES[:STAGES.index(through) + 1]:
        for path in sorted((root / stage / 'ss10').glob('seed-*/CORE.json')):
            core = json.loads(path.read_text())
            for phase in ('core', 'analysis'):
                phase_path = path.with_name(phase.upper() + '.json')
                if phase_path.exists():
                    row = json.loads(phase_path.read_text())
                    observations.append(Observation(
                        stage, core['seed'], phase, core['state_capacity'],
                        core['classic_samples'], core['classic_log_Z_uncert'],
                        row['peak_rss_bytes'] / 1024**3,
                    ))
    return observations


def fit_rss(observations: list[Observation], phase: str) -> RssFit:
    """Fit nonnegative fixed overhead, capacity storage, and sample storage."""
    rows = [row for row in observations if row.phase == phase]
    if len(rows) < 3 or len({row.capacity for row in rows}) < 2:
        raise ValueError(f'Insufficient completed {phase} data to fit memory')
    x = np.asarray([[1., row.capacity / 1e6, row.samples / 1e6] for row in rows])
    y = np.asarray([row.peak_gib for row in rows])  # [num_observations]
    coefficients, _ = nnls(x, y)
    allowance = max(0., float(np.max(y - x @ coefficients)))
    return RssFit(*map(float, coefficients), allowance, len(rows))


def sample_exponents(observations: list[Observation]) -> dict[int, float]:
    """Use each seed's most recent completed pair; retain a quadratic floor."""
    by_seed = {}
    for row in observations:
        if row.phase == 'core':
            by_seed.setdefault(row.seed, []).append(row)
    exponents = {}
    for seed, rows in by_seed.items():
        rows.sort(key=lambda row: STAGES.index(row.stage))
        if len(rows) < 2:
            continue
        old, new = rows[-2:]
        if new.sigma >= old.sigma or new.samples <= old.samples:
            raise ValueError(f'Invalid continuation records for seed {seed}')
        exponent = math.log(new.samples / old.samples) / math.log(old.sigma / new.sigma)
        exponents[seed] = max(2., exponent)
    return exponents


def project_tree(record: dict, target: float, exponent: float) -> tuple[int, float]:
    """Project sample growth, then round to the actual buffer growth schedule."""
    samples = record['classic_samples'] * (record['classic_log_Z_uncert'] / target)**exponent
    samples *= 1.1  # Additional count headroom before capacity rounding.
    capacity = record['state_capacity']
    while capacity < samples:
        capacity *= 2
    return capacity, samples


def calibrate(root: Path, previous: str) -> tuple[dict[str, RssFit], dict[int, float], dict]:
    observations = collect_observations(root, previous)
    fits = {phase: fit_rss(observations, phase) for phase in ('core', 'analysis')}
    exponents = sample_exponents(observations)
    snapshot = {
        'through_stage': previous,
        'fits': {phase: asdict(fit) for phase, fit in fits.items()},
        'sample_exponent_by_seed': exponents,
        'count_headroom': 1.1,
        'observations': [asdict(row) for row in observations],
    }
    return fits, exponents, snapshot
