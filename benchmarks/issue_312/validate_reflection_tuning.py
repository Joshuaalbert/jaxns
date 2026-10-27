"""Held-out independent stationarity and single-path acceptance after schedule freeze."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp
    from scipy.stats import beta, ks_2samp, kstest

    from benchmarks.issue_312.reflection_sampler import ReflectionSampler, reflection_trajectory
    from benchmarks.issue_312.tuning_common import (
        atomic_json, atomic_npz, exact_draws, make_model, residuals,
    )
    from benchmarks.issue_312.tuning_schedule import FrozenContourReflectionSampler
    from jaxns.pytree import pytree_ravel

    model = make_model()
    _, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(3122799)))

    def likelihood(point):
        return model.log_likelihood(unravel(point))

    peak = float(likelihood(jnp.full(18, .5, dtype=jnp.float64)))
    schedule = json.loads(options.schedule.read_text())
    assert peak == schedule['peak']
    schedule_hash = hashlib.sha256(options.schedule.read_bytes()).hexdigest()
    samplers = [
        ReflectionSampler(num_cycles=9, slices_per_cycle=1, num_bounces=10,
                          step_size=.003, num_phantom_samples=8),
        FrozenContourReflectionSampler(
            num_cycles=9, slices_per_cycle=1, num_bounces=10,
            step_size=.003, num_phantom_samples=8, peak=peak,
            gaps=tuple(schedule['gaps']), steps=tuple(schedule['steps']),
        ),
    ]
    options.output.mkdir(parents=True, exist_ok=True)
    n = 8192
    all_records = []
    started = time.monotonic()
    for case, gap in enumerate((5., 15., 40., 100., 1000., np.inf)):
        initial, proposals = exact_draws(np.random.default_rng([3122703, case, 0]), n, gap)
        reference, _ = exact_draws(np.random.default_rng([3122703, case, 1]), n, gap)
        key = jax.random.fold_in(jax.random.PRNGKey(3122704), case)
        path_key, chain_key = jax.random.split(key)
        path_keys = jax.random.split(path_key, n)
        chain_keys = jax.random.split(chain_key, n)
        contour = jnp.asarray(peak - gap)
        initial_values = np.asarray(jax.jit(jax.vmap(likelihood))(initial))
        assert np.all(initial_values > float(contour))
        for arm, sampler in zip(('fixed', 'tuned'), samplers, strict=True):
            h = .003 if arm == 'fixed' else float(sampler.step_for_contour(contour))

            @jax.jit
            @jax.vmap
            def one_path(path_rng, point):
                velocity_key, time_key = jax.random.split(path_rng)
                velocity = jax.random.normal(velocity_key, (18,), dtype=jnp.float64)
                step = h * jax.random.uniform(time_key, dtype=jnp.float64, minval=.8, maxval=1.2)
                proposed, _ = reflection_trajectory(
                    point, velocity, step, 10, contour, jax.value_and_grad(likelihood),
                )
                value = likelihood(proposed)
                accept = jnp.isfinite(value) & (value > contour)
                return jnp.where(accept, proposed, point), accept, step

            @jax.jit
            @jax.vmap
            def chain(chain_rng, point, value):
                return sampler.run_chain(chain_rng, point, value, contour, likelihood)

            paths, accepts, steps, results = [], [], [], []
            for start in range(0, n, 256):
                part = slice(start, start+256)
                path, accept, step = map(np.asarray, jax.block_until_ready(
                    one_path(path_keys[part], jnp.asarray(initial[part])),
                ))
                paths.append(path)
                accepts.append(accept)
                steps.append(step)
                result, _ = jax.block_until_ready(chain(
                    chain_keys[part], jnp.asarray(initial[part]), jnp.asarray(initial_values[part]),
                ))
                results.append(jax.device_get(result))
            final = jax.tree.map(lambda *values: np.concatenate(values), *results)
            points = np.asarray(final.point)
            path_points = np.concatenate(paths)
            accept = np.concatenate(accepts)
            assert np.all((points >= 0) & (points <= 1))
            assert np.all(final.log_likelihood > float(contour))
            np.testing.assert_array_equal(final.gradients, 90)
            statistics = {}
            for law, locations in (('single_path', path_points), ('full_chain', points)):
                if gap < 50:
                    gaussian = residuals(locations)
                    radius = np.sqrt(2*gap)
                    pits = np.column_stack([
                        locations[:, [1, 16, 17]], beta.cdf((gaussian/radius+1)/2, 8, 8),
                        (np.linalg.norm(gaussian, axis=1)/radius)**15,
                    ])
                    values = [float(kstest(column, 'uniform').statistic) for column in pits.T]
                    limit = 5/np.sqrt(n)
                elif np.isposinf(gap):
                    values = [float(kstest(column, 'uniform').statistic)
                              for column in locations.T]
                    limit = 5/np.sqrt(n)
                else:
                    left = np.column_stack([
                        locations, residuals(locations),
                        np.linalg.norm(residuals(locations), axis=1),
                    ])
                    right = np.column_stack([
                        reference, residuals(reference),
                        np.linalg.norm(residuals(reference), axis=1),
                    ])
                    values = [float(ks_2samp(a, b).statistic)
                              for a, b in zip(left.T, right.T, strict=True)]
                    limit = 5*np.sqrt(2/n)
                statistics[law] = {'ks': values, 'max_ks': max(values), 'limit': float(limit),
                                   'passed': bool(max(values) < limit)}
            p = accept.mean()
            z = 1.959963984540054
            centre = (p+z*z/(2*n))/(1+z*z/n)
            half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            archive = options.output / f'case-{case}-{arm}.npz'
            atomic_npz(
                archive, initial=initial, independent_reference=reference, path_points=path_points,
                accepted=accept, realised_steps=np.concatenate(steps), final_points=points,
                final_log_likelihood=final.log_likelihood, chain_accepted=final.accepted,
                evaluations=final.evaluations, path_keys=np.asarray(path_keys),
                chain_keys=np.asarray(chain_keys),
            )
            record = {
                'case': case, 'gap': float(gap) if np.isfinite(gap) else 'infinity',
                'arm': arm, 'chains': n, 'h': h, 'accepted_paths': int(accept.sum()),
                'total_paths': n, 'single_path_acceptance': float(p),
                'single_path_acceptance_wilson95': [float(centre-half), float(centre+half)],
                'full_chain_path_acceptance': float(np.sum(final.accepted)/(9*n)),
                'stationarity': statistics,
                'path_rms': np.sqrt(np.mean((path_points-initial)**2, axis=0)).tolist(),
                'chain_rms': np.sqrt(np.mean((points-initial)**2, axis=0)).tolist(),
                'chain_correlations': [float(np.corrcoef(initial[:, i], points[:, i])[0, 1])
                                       for i in (1, 16, 17, 5)],
                'reference_proposals': proposals, 'schedule_sha256': schedule_hash,
                'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'seconds_cumulative': time.monotonic()-started,
            }
            all_records.append(record)
            atomic_json(options.output / 'PROGRESS.json', {'records': all_records})
            print(json.dumps(record), flush=True)
            if not all(item['passed'] for item in statistics.values()):
                raise RuntimeError(f'Stationarity gate failed: {case}, {arm}')
    atomic_json(options.output / 'VALIDATION.json', {
        'passed': True, 'records': all_records, 'schedule_sha256': schedule_hash,
        'seconds': time.monotonic()-started, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
    })


if __name__ == '__main__':
    main()
