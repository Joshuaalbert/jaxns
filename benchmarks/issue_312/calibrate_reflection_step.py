"""Execute the prespecified independent single-path calibration, then freeze it."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp

    from benchmarks.issue_312.reflection_sampler import reflection_trajectory
    from benchmarks.issue_312.tuning_common import (
        SOURCE_ROOT, atomic_json, atomic_npz, exact_draws, make_model,
    )
    from jaxns.pytree import pytree_ravel

    model = make_model()
    unit, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(3122799)))
    assert unit.dtype == jnp.float64 and unit.shape == (18,)

    def likelihood(point):
        return model.log_likelihood(unravel(point))

    peak = float(likelihood(jnp.full(18, .5, dtype=jnp.float64)))

    @jax.jit
    @jax.vmap
    def propose(key, point, contour, step):
        velocity_key, jitter_key = jax.random.split(key)
        velocity = jax.random.normal(velocity_key, (18,), dtype=jnp.float64)
        actual_step = step * jax.random.uniform(
            jitter_key, dtype=jnp.float64, minval=.8, maxval=1.2,
        )
        proposed, _ = reflection_trajectory(
            point, velocity, actual_step, 10, contour, jax.value_and_grad(likelihood),
        )
        value = likelihood(proposed)
        return proposed, jnp.isfinite(value) & (value > contour), actual_step

    options.output.mkdir(parents=True, exist_ok=True)
    if (options.output / 'SCHEDULE.json').exists():
        raise FileExistsError('Calibration is already frozen.')
    protocol = SOURCE_ROOT / 'benchmarks/issue_312/REFLECTION_TUNING_PROTOCOL.md'
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    started = time.monotonic()
    nodes = []
    for k in range(40):
        gap = 49 * np.exp(-2 * k / 15)
        log_step = np.log(.003)
        inputs, outputs, accepted, jitters, keys_saved = [], [], [], [], []
        log_steps, batches = [], []
        proposed_count = 0
        for t in range(64):
            initial, attempts = exact_draws(np.random.default_rng([3122701, k, t]), 256, gap)
            proposed_count += attempts
            key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(3122702), k), t)
            keys = jax.random.split(key, 256)
            step = np.exp(log_step)
            result = jax.block_until_ready(propose(
                keys, jnp.asarray(initial), jnp.full(256, peak - gap), jnp.full(256, step),
            ))
            points, accept, actual_step = map(np.asarray, result)
            probability = float(accept.mean())
            log_steps.append(log_step)
            batches.append({'t': t, 'step': float(step), 'acceptance': probability,
                            'accepted': int(accept.sum()), 'total': len(accept)})
            inputs.append(initial)
            outputs.append(points)
            accepted.append(accept)
            jitters.append(actual_step)
            keys_saved.append(np.asarray(keys))
            eta = .8 / (1 + t / 8)**.6
            log_step = np.clip(log_step + eta * (probability - .7), np.log(1e-6), np.log(.03))
        step = float(np.exp(np.mean(log_steps[48:64])))
        archive = options.output / f'node-{k:02d}.npz'
        atomic_npz(
            archive, initial=np.concatenate(inputs), proposed=np.concatenate(outputs),
            accepted=np.concatenate(accepted), realised_step=np.concatenate(jitters),
            keys=np.concatenate(keys_saved), log_steps=np.asarray(log_steps),
        )
        row = {
            'node': k, 'gap': float(gap), 'contour': peak - float(gap), 'step': step,
            'batches': batches, 'independent_paths': 16384, 'exact_draw_proposals': proposed_count,
            'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'cumulative_seconds': time.monotonic() - started,
        }
        nodes.append(row)
        atomic_json(options.output / 'PROGRESS.json', {
            'nodes_complete': k + 1, 'nodes_required': 40, 'last_node': row,
            'protocol_sha256': protocol_hash,
        })
        print(json.dumps({name: row[name] for name in (
            'node', 'gap', 'step', 'cumulative_seconds',
        )}), flush=True)
    ordered = sorted(nodes, key=lambda row: row['gap'])
    atomic_json(options.output / 'SCHEDULE.json', {
        'protocol_sha256': protocol_hash, 'peak': peak, 'nodes': nodes,
        'gaps': [row['gap'] for row in ordered], 'steps': [row['step'] for row in ordered],
        'paths': 655360, 'gradient_calls': 6553600, 'endpoint_value_calls': 655360,
        'initial_values_evaluated': 0, 'model_calls': 7208960,
        'seconds_including_compilation': time.monotonic() - started,
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'jax': jax.__version__,
        'status': 'frozen_before_heldout_validation_and_evaluation',
    })


if __name__ == '__main__':
    main()
