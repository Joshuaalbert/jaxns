"""Explicit experiment arms and independent variable-work reference accounting."""

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import numpy as np
from jax import numpy as jnp

from benchmarks.issue_312.duration_schedule import (
    DurationContourReflectionSampler,
    ReflectionParameters,
)
from benchmarks.issue_312.reflection_sampler import ReflectionSampler
from benchmarks.issue_312.tuning_schedule import FrozenContourReflectionSampler
from jaxns.mixed_precision import mp_policy

BASELINE = '295893708805ff668178fc443cbda63faf2ac75d'
OUTPUT_ROOT = Path('/largedata/albert/jaxns-reflection-duration-20261027')
SCHEDULE_SHA256 = '21242cb5554c0b7c23c4750d1c8aa579e3d9856ff670a0f9c18133b1771a28d6'
ARMS = ('fixed', 'short', 'duration')


def make_arm(
    arm: str, schedule: dict,
) -> tuple[ReflectionSampler, Callable[[jax.Array], ReflectionParameters]]:
    """Return the actual sampler and its exact frozen per-contour work parameters."""
    kernel = dict(num_cycles=9, slices_per_cycle=1, num_bounces=10, step_size=0.003)
    if arm == 'fixed':
        sampler = ReflectionSampler(**kernel)

        def parameters(contour):
            return ReflectionParameters(
                jnp.asarray(10, mp_policy.count_dtype), jnp.asarray(0.003, jnp.float64),
            )

        return sampler, parameters
    frozen = dict(gaps=tuple(schedule['gaps']), steps=tuple(schedule['steps']),
                  peak=schedule['peak'])
    if arm == 'short':
        sampler = FrozenContourReflectionSampler(**kernel, **frozen)

        def parameters(contour):
            return ReflectionParameters(
                jnp.asarray(10, mp_policy.count_dtype), sampler.step_for_contour(contour),
            )

        return sampler, parameters
    if arm == 'duration':
        sampler = DurationContourReflectionSampler(**kernel, **frozen)
        return sampler, sampler.parameters_for_contour
    raise ValueError(f'Unknown reflection arm: {arm}')


def numpy_parameters(arm: str, schedule: dict, contours: np.ndarray) -> tuple:
    """Independent NumPy reconstruction of B/h, including the exact early policy."""
    gap = schedule['peak'] - contours
    if arm == 'fixed':
        return np.full(gap.shape, 10, np.int64), np.full(gap.shape, 0.003)
    desired = np.exp(np.interp(
        np.log(np.maximum(gap, schedule['gaps'][0])),
        np.log(schedule['gaps']), np.log(schedule['steps']),
    ))
    early = gap > schedule['gaps'][-1]
    desired = np.where(early, 0.003, desired)
    if arm == 'short':
        return np.full(gap.shape, 10, np.int64), desired
    if arm == 'duration':
        count = np.where(early, 10, np.ceil(0.03 / desired)).astype(np.int64)
        return count, np.where(early, 0.003, 0.03 / count)
    raise ValueError(f'Unknown reflection arm: {arm}')


def load_duration_freeze() -> dict:
    """Bind a worker to the reviewed source, protocol and inherited schedule."""
    from benchmarks.issue_312.tuning_common import SOURCE_ROOT

    freeze_path = OUTPUT_ROOT / 'SOURCE_FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    current_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=SOURCE_ROOT, text=True,
    ).strip()
    assert freeze['source_commit'] == current_commit
    assert freeze['schedule_sha256'] == SCHEDULE_SHA256
    assert hashlib.sha256((OUTPUT_ROOT / 'SCHEDULE.json').read_bytes()).hexdigest() == (
        SCHEDULE_SHA256
    )
    for relative, expected in freeze['source_sha256'].items():
        assert hashlib.sha256((SOURCE_ROOT / relative).read_bytes()).hexdigest() == expected, (
            relative
        )
    protocol = SOURCE_ROOT / 'benchmarks/issue_312/REFLECTION_DURATION_PROTOCOL.md'
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    return {**freeze, 'source_freeze_sha256': hashlib.sha256(freeze_path.read_bytes()).hexdigest()}


def work_parameters(
    parameters: Callable[[jax.Array], ReflectionParameters],
    contours: jax.Array,
    seed_indices: jax.Array,
) -> ReflectionParameters:
    """Count constrained paths only; initial and later adaptive roots are prior draws."""
    counts, steps = jax.vmap(parameters)(contours)  # [width], [width]
    is_child = seed_indices >= 0
    return ReflectionParameters(jnp.where(is_child, counts, 0), jnp.where(is_child, steps, 0.))
