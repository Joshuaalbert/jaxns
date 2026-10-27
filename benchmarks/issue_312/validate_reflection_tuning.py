"""Held-out independent stationarity and single-path acceptance after schedule freeze."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main(*, duration_comparison: bool = False):
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

    from benchmarks.issue_312.duration_common import load_duration_freeze, make_arm
    from benchmarks.issue_312.reflection_sampler import reflection_trajectory
    from benchmarks.issue_312.tuning_common import (
        atomic_json, atomic_npz, exact_draws, make_model, residuals,
    )
    from jaxns.pytree import pytree_ravel

    freeze = load_duration_freeze() if duration_comparison else {}
    model = make_model()
    _, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(3122799)))

    def likelihood(point):
        return model.log_likelihood(unravel(point))

    peak = float(likelihood(jnp.full(18, .5, dtype=jnp.float64)))
    schedule = json.loads(options.schedule.read_text())
    assert peak == schedule['peak']
    schedule_hash = hashlib.sha256(options.schedule.read_bytes()).hexdigest()
    arms = ('fixed', 'short', 'duration') if duration_comparison else ('fixed', 'tuned')
    specifications = [make_arm('short' if arm == 'tuned' else arm, schedule) for arm in arms]
    gaps = (.1, 5., 15., 40., 100., 1000., np.inf) if duration_comparison else (
        5., 15., 40., 100., 1000., np.inf,
    )
    numpy_seed, jax_seed = (3122711, 3122712) if duration_comparison else (3122703, 3122704)
    options.output.mkdir(parents=True, exist_ok=True)
    n = 8192
    all_records = []
    started = time.monotonic()
    for case, gap in enumerate(gaps):
        initial, proposals = exact_draws(np.random.default_rng([numpy_seed, case, 0]), n, gap)
        reference, _ = exact_draws(np.random.default_rng([numpy_seed, case, 1]), n, gap)
        key = jax.random.fold_in(jax.random.PRNGKey(jax_seed), case)
        path_key, chain_key = jax.random.split(key)
        path_keys = jax.random.split(path_key, n)
        chain_keys = jax.random.split(chain_key, n)
        contour = jnp.asarray(peak - gap)
        initial_values = np.asarray(jax.jit(jax.vmap(likelihood))(initial))
        assert np.all(initial_values > float(contour))
        early_control = None
        for arm, (sampler, parameters) in zip(arms, specifications, strict=True):
            bounce_count, base_h = parameters(contour)
            bounces, h = int(bounce_count), float(base_h)

            @jax.jit
            @jax.vmap
            def one_path(path_rng, point):
                velocity_key, time_key = jax.random.split(path_rng)
                velocity = jax.random.normal(velocity_key, (18,), dtype=jnp.float64)
                step = h * jax.random.uniform(time_key, dtype=jnp.float64, minval=.8, maxval=1.2)
                proposed, _ = reflection_trajectory(
                    point, velocity, step, bounce_count, contour, jax.value_and_grad(likelihood),
                )
                value = likelihood(proposed)
                accept = jnp.isfinite(value) & (value > contour)
                return jnp.where(accept, proposed, point), accept, step, proposed, value

            @jax.jit
            @jax.vmap
            def chain(chain_rng, point, value):
                return sampler.run_chain(chain_rng, point, value, contour, likelihood)

            paths, accepts, steps, proposals_out, proposed_values, results = [], [], [], [], [], []
            cycle_points, cycle_values = [], []
            for start in range(0, n, 256):
                part = slice(start, start+256)
                path, accept, step, proposed, proposed_value = map(
                    np.asarray, jax.block_until_ready(
                        one_path(path_keys[part], jnp.asarray(initial[part])),
                    ),
                )
                paths.append(path)
                accepts.append(accept)
                steps.append(step)
                proposals_out.append(proposed)
                proposed_values.append(proposed_value)
                result, (cycle_point, cycle_value) = jax.block_until_ready(chain(
                    chain_keys[part], jnp.asarray(initial[part]), jnp.asarray(initial_values[part]),
                ))
                results.append(jax.device_get(result))
                cycle_points.append(np.asarray(cycle_point))
                cycle_values.append(np.asarray(cycle_value))
            final = jax.tree.map(lambda *values: np.concatenate(values), *results)
            points = np.asarray(final.point)
            path_points = np.concatenate(paths)
            accept = np.concatenate(accepts)
            assert np.all((points >= 0) & (points <= 1))
            assert np.all(final.log_likelihood > float(contour))
            np.testing.assert_array_equal(final.gradients, 9 * bounces)
            assert np.all(final.evaluations >= final.gradients + 18)
            np.testing.assert_array_equal(path_points[~accept], initial[~accept])
            assert np.all(np.asarray(jax.jit(jax.vmap(likelihood))(path_points)) > float(contour))
            # Reconstruct all independent jitters without consuming continuation RNG.

            @jax.jit
            @jax.vmap
            def chain_steps(key):
                def jitter(cycle_key):
                    time_key = jax.random.split(cycle_key, 3)[2]
                    return base_h * jax.random.uniform(
                        time_key, dtype=jnp.float64, minval=.8, maxval=1.2,
                    )
                return jax.vmap(jitter)(jax.random.split(key, 9))

            actual_chain_steps = np.asarray(chain_steps(chain_keys))
            if gap > 49:
                raw_control = (path_points, accept, np.concatenate(steps), *jax.tree.leaves(final),
                               np.concatenate(cycle_points), np.concatenate(cycle_values))
                if early_control is None:
                    early_control = raw_control
                else:
                    for actual, expected in zip(raw_control, early_control, strict=True):
                        np.testing.assert_array_equal(actual, expected)
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
                proposed_points=np.concatenate(proposals_out),
                proposed_log_likelihood=np.concatenate(proposed_values),
                bounce_counts=np.full(n, bounces, np.int64),
                nominal_steps=np.full(n, h),
                realised_durations=np.concatenate(steps) * bounces,
                chain_realised_steps=actual_chain_steps,
                chain_realised_durations=actual_chain_steps * bounces,
                gradients=final.gradients,
                path_gradients=np.full(n, bounces, np.int64),
                value_only_evaluations=final.evaluations - final.gradients,
                cycle_points=np.concatenate(cycle_points),
                cycle_log_likelihood=np.concatenate(cycle_values),
            )
            record = {
                'case': case, 'gap': float(gap) if np.isfinite(gap) else 'infinity',
                'arm': arm, 'chains': n, 'h': h, 'num_bounces': bounces,
                'nominal_duration': h * bounces, 'accepted_paths': int(accept.sum()),
                'path_gradients_total': n * bounces,
                'chain_gradients_total': int(final.gradients.sum()),
                'chain_value_only_total': int((final.evaluations - final.gradients).sum()),
                'early_exact_parity': bool(gap > 49),
                'total_paths': n, 'single_path_acceptance': float(p),
                'single_path_acceptance_wilson95': [float(centre-half), float(centre+half)],
                'full_chain_path_acceptance': float(np.sum(final.accepted)/(9*n)),
                'stationarity': statistics,
                'path_rms': np.sqrt(np.mean((path_points-initial)**2, axis=0)).tolist(),
                'chain_rms': np.sqrt(np.mean((points-initial)**2, axis=0)).tolist(),
                'path_esjd_per_gradient': (
                    np.sum((path_points-initial)**2, axis=0) / (n * bounces)
                ).tolist(),
                'chain_esjd_per_gradient': (
                    np.sum((points-initial)**2, axis=0) / final.gradients.sum()
                ).tolist(),
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
        'source_freeze_sha256': freeze.get('source_freeze_sha256'),
        'source_commit': freeze.get('source_commit'),
        'protocol_sha256': freeze.get('protocol_sha256'),
        'seconds': time.monotonic()-started, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
    })


if __name__ == '__main__':
    main()
