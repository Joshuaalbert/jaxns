"""Focused CPU-only online-controller smoke, with chronological and exact chain replay.

This is not an evaluation entrypoint. Its fixed seeds/budgets exercise the
observer integration and learning mechanics before any source release.
"""

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import platform
import resource
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--allocation', choices=('uniform', 'evidence_improving'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp

    from benchmarks.issue_312.allocation_observer import run_with_observer
    from benchmarks.issue_312.online_controller import OnlineController, observe_completed
    from benchmarks.issue_312.online_policy import OnlinePolicy, choose_parameters
    from benchmarks.issue_312.online_session import OnlineSession
    from benchmarks.issue_312.reflection_sampler import ReflectionSampler
    from benchmarks.issue_312.tuning_common import (
        SOURCE_ROOT, atomic_json, atomic_npz, make_model, numpy_log_likelihood,
    )
    from jaxns.algorithm.depth import _run_depth
    from jaxns.core import NestedSampler
    from jaxns.pytree import pytree_ravel

    output = options.output
    output.mkdir(parents=True, exist_ok=False)
    model = make_model()
    import jaxns
    if not Path(jaxns.__file__).resolve().is_relative_to(SOURCE_ROOT):
        raise RuntimeError('Imported jaxns must be from the isolated experiment worktree.')
    allocation_index = ('uniform', 'evidence_improving').index(options.allocation)
    seed = 3122920 + allocation_index
    source = sorted((SOURCE_ROOT / 'src/jaxns').rglob('*.py'))
    source += sorted((SOURCE_ROOT / 'benchmarks/issue_312').glob('online_*.py'))
    source += [SOURCE_ROOT / 'benchmarks/issue_312' / name for name in (
        'allocation_observer.py', 'reflection_sampler.py', 'smoke_reflection_online.py',
        'tuning_common.py', 'REFLECTION_ONLINE_PROTOCOL.md',
    )]
    manifest = dict(baseline='1d84f157496f5b7a20659fdc5da9ca41fd8cec09',
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    phase='smoke-not-evaluation', allocation=options.allocation, seed=seed,
                    hostname=platform.node(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                    source_sha256={str(p.relative_to(SOURCE_ROOT)):
                                   hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
                    imported_jaxns=str(Path(jaxns.__file__).resolve()),
                    backend=jax.default_backend(), budgets=[30000, 60000, 300000])
    atomic_json(output / 'MANIFEST.json', manifest)
    sampler = ReflectionSampler(num_cycles=9, slices_per_cycle=1, num_bounces=10, step_size=.003)
    runner = NestedSampler(model=model, root_allocation_degree=150, replacement_width=150,
                           delta_K=150, unlimited_samples=True, sampler=sampler,
                           allocation_target=options.allocation, collect_phantom_samples=True)
    session = OnlineSession(150)
    initial = runner.initialise(jax.random.PRNGKey(seed))
    budget_index = 0
    snapshots = []
    started = time.monotonic()

    def observe(before, current, work):
        nonlocal budget_index
        record = session.observe(before, current, work)
        receipt = session.receipts[-1]
        evaluations = int(current.total_num_likelihood_evaluations)
        status = dict(evaluations=evaluations, samples=int(current.num_samples),
                      batches=int(current.depth_loop_iter),
                      version=int(session.controller.policy.version),
                      learned_strata=int(session.controller.learned.sum()),
                      total_exposure=float(session.controller.total_exposure.sum()),
                      deepest_compression=session.controller.deepest_compression,
                      capacity=current.samples.log_likelihoods.shape[0],
                      depth_compile_cache_size=_run_depth._cache_size(),
                      seconds=time.monotonic() - started)
        with (output / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(status) + '\n')
        with (output / 'controller-history.jsonl').open('a') as stream:
            stream.write(json.dumps(receipt) + '\n')
        if receipt['changes']:
            print(json.dumps(status), flush=True)
        while (budget_index < len(manifest['budgets'])
               and evaluations >= manifest['budgets'][budget_index]):
            budget = manifest['budgets'][budget_index]
            row = {**status, 'budget': budget,
                   'previous_evaluations': int(before.total_num_likelihood_evaluations),
                   'last_batch_evaluations': int(record['work'].sum())}
            atomic_json(output / f'CORE-{budget}.json', row)
            snapshots.append(row)
            budget_index += 1
        return budget_index == len(manifest['budgets'])

    state = run_with_observer(runner, initial, observe, lambda current: False,
                              prepare_sampler=session.prepare)
    sampling_seconds = time.monotonic() - started
    assert state.samples.phantom_samples.U_samples is None
    raw = session.arrays()
    atomic_npz(output / 'controller-ledger.npz', **raw)
    policy_arrays = {
        name: np.stack([np.asarray(getattr_policy) for getattr_policy in values])
        for name, values in (
            ('log_parameters', [policy.log_parameters for policy in session.policy_history]),
            ('support', [policy.support for policy in session.policy_history]),
            ('version', [policy.version for policy in session.policy_history]),
        )
    }
    atomic_npz(output / 'policy-history.npz', **policy_arrays)
    checkpoint = session.checkpoint(
        state, budgets=tuple(manifest['budgets']), next_budget=budget_index,
        snapshots=tuple(snapshots), sampling_elapsed_seconds=sampling_seconds,
        source_sha256=manifest['source_sha256'],
    )
    temporary = output / 'checkpoint.tmp.pkl'
    with temporary.open('wb') as stream:
        pickle.dump(checkpoint, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / 'checkpoint.pkl')
    digest = hashlib.sha256((output / 'checkpoint.pkl').read_bytes()).hexdigest()
    (output / 'checkpoint.pkl.sha256').write_text(digest + '  checkpoint.pkl\n')

    # No replay enters adaptation. Reconstruct feedback chronologically first,
    # then independently re-execute each recorded chain from its actual setting.
    audit_started = time.monotonic()
    audited = OnlineController.initialise(150)
    offset = 0
    for receipt in session.receipts:
        size = receipt['stop'] - receipt['start']
        rows = slice(offset, offset + size)
        audited, checked = observe_completed(
            audited, start=receipt['start'], seed_indices=raw['seed_indices'][rows],
            compression=raw['compression'][rows], incoming_k=raw['incoming_k'][rows],
            choices=raw['choice'][rows], squared_jumps=raw['squared_jumps'][rows],
            work=raw['work'][rows], policy_version=receipt['dispatch_version'],
        )
        assert checked == receipt
        offset += size
    np.testing.assert_array_equal(audited.policy.log_parameters,
                                  session.controller.policy.log_parameters)
    np.testing.assert_array_equal(audited.total_exposure, session.controller.total_exposure)
    points = np.asarray(jax.tree.leaves(state.samples.U_samples)[0])[:int(state.num_samples)]
    values = np.asarray(state.samples.log_likelihoods)[:int(state.num_samples)]
    _, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(3122999)))

    def likelihood(point):
        return model.log_likelihood(unravel(point))

    @jax.jit
    @jax.vmap
    def replay(key, point, value, contour, count, step):
        return sampler.run_chain(key, point, value, contour, likelihood,
                                 contour_num_bounces=count, contour_step_size=step)

    @jax.jit
    @jax.vmap
    def replay_choice(log_parameters, support, version, key, compression):
        return choose_parameters(OnlinePolicy(log_parameters, support, version), key, compression)

    replay_calls, replay_gradients = 0, 0
    children = np.flatnonzero(raw['seed_indices'] >= 0)
    for start in range(0, len(children), 150):
        selected = children[start:start + 150]
        padded = np.pad(selected, (0, 150 - len(selected)), mode='edge')
        seeds = raw['seed_indices'][padded]
        result, (_, phantom) = jax.device_get(replay(
            jnp.asarray(raw['chain_keys'][padded]), jnp.asarray(points[seeds]),
            jnp.asarray(values[seeds]), jnp.asarray(raw['log_L_constraints'][padded]),
            jnp.asarray(raw['num_bounces'][padded]), jnp.asarray(raw['step_size'][padded]),
        ))
        size = len(selected)
        indices = raw['sample_indices'][selected]
        np.testing.assert_array_equal(result.point[:size], points[indices])
        np.testing.assert_array_equal(result.log_likelihood[:size], values[indices])
        np.testing.assert_array_equal(result.evaluations[:size], raw['work'][selected])
        np.testing.assert_array_equal(result.gradients[:size],
                                      raw['gradient_evaluations'][selected])
        np.testing.assert_array_equal(phantom[:size, :8],
                                      np.asarray(state.samples.phantom_samples.log_L)[indices])
        versions = raw['controller_versions'][padded]
        chosen = jax.device_get(replay_choice(
            jnp.asarray(policy_arrays['log_parameters'][versions]),
            jnp.asarray(policy_arrays['support'][versions]),
            jnp.asarray(policy_arrays['version'][versions]),
            jnp.asarray(raw['chain_keys'][padded]), jnp.asarray(raw['compression'][padded]),
        ))
        np.testing.assert_array_equal(chosen.num_bounces[:size], raw['num_bounces'][selected])
        np.testing.assert_allclose(chosen.step_size[:size], raw['step_size'][selected], rtol=2e-14)
        np.testing.assert_array_equal(chosen.choice[:size], raw['choice'][selected])
        replay_calls += int(result.evaluations[:size].sum())
        replay_gradients += int(result.gradients[:size].sum())
    discrepancy = float(np.max(np.abs(numpy_log_likelihood(points) - values)))
    assert discrepancy < 5e-5
    complete = dict(
        passed=True, replay_exact=True, chronological_replay_exact=True,
        learning_exercised=any(row['real_parameter_change'] for row in session.receipts),
        real_parameter_change_batches=sum(row['real_parameter_change'] for row in session.receipts),
        controller_updates=int(session.controller.policy.version),
        learned_strata=int(session.controller.learned.sum()),
        cumulative_exposure=session.controller.total_exposure.tolist(),
        deepest_compression=session.controller.deepest_compression,
        max_likelihood_discrepancy=discrepancy, snapshots=snapshots,
        sampling_seconds=sampling_seconds, audit_seconds=time.monotonic() - audit_started,
        controller_seconds=session.controller_seconds, prepare_seconds=session.prepare_seconds,
        feedback_seconds=session.feedback_seconds, replay_calls=replay_calls,
        replay_gradients=replay_gradients, checkpoint_sha256=digest,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    atomic_json(output / 'COMPLETE.json', complete)
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
