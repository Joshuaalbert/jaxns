"""Explicit source-release and independent frozen-policy boundaries for this experiment."""

import hashlib
import json
from pathlib import Path
import subprocess

from jax import numpy as jnp
import numpy as np

from benchmarks.issue_312.online_frozen import FrozenLikelihoodPolicy
from benchmarks.issue_312.tuning_common import SOURCE_ROOT

BASELINE = '1d84f157496f5b7a20659fdc5da9ca41fd8cec09'
OUTPUT_ROOT = Path('/largedata/albert/jaxns-reflection-online-20261029')
ARMS = ('fixed', 'online', 'frozen')


def load_online_freeze(phase: str, seed: int, arm: str, policy_path: Path | None) -> dict:
    """Fail closed: neither training nor evaluation can run without a reviewed release."""
    freeze_path = OUTPUT_ROOT / 'SOURCE_FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE_ROOT,
                                     text=True).strip()
    if freeze['source_commit'] != commit or freeze['baseline_commit'] != BASELINE:
        raise ValueError('Online source freeze does not bind this worktree commit/baseline.')
    for relative, expected in freeze['source_sha256'].items():
        if hashlib.sha256((SOURCE_ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen online source differs: {relative}')
    protocol = SOURCE_ROOT / 'benchmarks/issue_312/REFLECTION_ONLINE_PROTOCOL.md'
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != freeze['protocol_sha256']:
        raise ValueError('Online protocol differs from source freeze.')
    freeze_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    if phase == 'smoke':
        if seed not in (3122920, 3122921):
            raise ValueError('Shared-runner smoke seeds are fixed outside training/evaluation.')
    elif phase in ('training', 'evaluation'):
        release = json.loads((OUTPUT_ROOT / f'{phase.upper()}_RELEASE.json').read_text())
        if (not release['approved'] or release['source_commit'] != commit
                or release['protocol_sha256'] != freeze['protocol_sha256']
                or release['source_freeze_sha256'] != freeze_hash):
            raise ValueError(f'{phase} lacks a compatible coordinator release.')
        if phase == 'training' and (seed not in range(3122910, 3122914) or arm != 'online'):
            raise ValueError('Training is the four independent online runs per allocation.')
        if phase == 'evaluation':
            if seed not in range(60, 90):
                raise ValueError('Evaluation uses all fresh matched seeds 60..89.')
            validation_path = OUTPUT_ROOT / 'validation/VALIDATION.json'
            if (not json.loads(validation_path.read_text())['passed']
                    or hashlib.sha256(validation_path.read_bytes()).hexdigest()
                    != release['validation_sha256']):
                raise ValueError('Evaluation lacks the exact passed held-out validation.')
            if arm == 'frozen':
                if policy_path is None:
                    raise ValueError('Frozen evaluation requires its independent policy path.')
                actual = hashlib.sha256(policy_path.read_bytes()).hexdigest()
                if actual not in release['frozen_policy_sha256'].values():
                    raise ValueError('Frozen policy is not one of the released training artifacts.')
    else:
        raise ValueError(f'Unknown experiment phase: {phase}')
    return {**freeze, 'source_freeze_sha256': freeze_hash}


def load_frozen_policy(path: Path, allocation: str) -> FrozenLikelihoodPolicy:
    """Read the exact trained lambda schema; evaluation-tree fields are absent."""
    payload = json.loads(path.read_text())
    if (payload['schema'] != 1 or payload['coordinate'] != 'parent_log_likelihood'
            or payload['allocation'] != allocation or payload['tail_policy'] != 'constant'
            or payload['training_seeds'] != list(range(3122910, 3122914))):
        raise ValueError('Frozen policy does not match the prespecified training contract.')
    contours = np.asarray(payload['contours'], np.float64)
    logs = np.asarray(payload['log_parameters'], np.float64)
    if (not 2 <= len(contours) <= 268 or logs.shape != (len(contours), 2)
            or not np.isfinite(contours).all() or not np.isfinite(logs).all()
            or np.any(np.diff(contours) <= 0)):
        raise ValueError('Frozen contour knots must be strictly ordered and finite.')
    padded_contours = np.full(268, np.inf)
    padded_contours[:len(contours)] = contours
    padded_logs = np.tile(logs[-1], (268, 1))
    padded_logs[:len(contours)] = logs
    return FrozenLikelihoodPolicy(jnp.asarray(padded_contours), jnp.asarray(padded_logs),
                                  jnp.asarray(len(contours), jnp.int64))
