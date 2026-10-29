"""Freeze equal-run actual-likelihood policies after the independent training cohort.

No evidence estimate or posterior diagnostic is used to select parameters.
All four final classic trees contribute equally, including unchanged regions.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import numpy as np

    from benchmarks.issue_312.online_audit import classic_volume_profile
    from benchmarks.issue_312.online_common import load_online_freeze
    from benchmarks.issue_312.online_frozen import TrainingProfile, freeze_training_profiles
    from benchmarks.issue_312.tuning_common import atomic_json

    freeze = load_online_freeze('training', 3122910, 'online', None)
    options.output.mkdir(parents=True, exist_ok=False)
    for allocation in ('uniform', 'evidence_improving'):
        profiles, inputs = [], []
        for seed in range(3122910, 3122914):
            folder = options.training_root / 'online' / allocation / f'seed-{seed}'
            manifest = json.loads((folder / 'MANIFEST.json').read_text())
            complete = json.loads((folder / 'COMPLETE.json').read_text())
            audit = json.loads((folder / 'CONTROLLER-AUDIT.json').read_text())
            if (manifest['phase'] != 'training' or manifest['seed'] != seed
                    or manifest['allocation'] != allocation or manifest['arm'] != 'online'
                    or manifest['source_freeze_sha256'] != freeze['source_freeze_sha256']
                    or not complete['core_complete'] or not audit['passed']):
                raise ValueError(f'Incomplete or incompatible independent training run: {folder}')
            core = json.loads((folder / 'CORE-4000000.json').read_text())
            raw_path, policies_path = folder / 'calls-4000000.npz', folder / 'policies-4000000.npz'
            if (hashlib.sha256(raw_path.read_bytes()).hexdigest() != core['archive_sha256']
                    or hashlib.sha256(policies_path.read_bytes()).hexdigest()
                    != core['controller_policies_sha256']):
                raise ValueError(f'Training raw/policy checksum mismatch: {folder}')
            with np.load(raw_path) as raw:
                contours, compression, _, _ = classic_volume_profile(
                    raw['log_likelihood'], raw['parent_indices'],
                )
            with np.load(policies_path) as policies:
                logs = policies['log_parameters'][-1]
            profiles.append(TrainingProfile(contours, compression, logs))
            inputs.append(dict(
                seed=seed, path=str(folder), evaluations=core['evaluations'],
                gradient_evaluations=core['gradient_evaluations'],
                seconds=complete['seconds'],
                controller_real_updates=core['controller_real_updates'],
                hashes={name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                        for name in ('MANIFEST.json', 'COMPLETE.json', 'CONTROLLER-AUDIT.json',
                                     'calls-4000000.npz', 'policies-4000000.npz')},
            ))
        policy = freeze_training_profiles(tuple(profiles))
        number = int(policy.num_knots)
        atomic_json(options.output / f'{allocation}.json', dict(
            schema=1, coordinate='parent_log_likelihood', allocation=allocation,
            tail_policy='constant', training_seeds=list(range(3122910, 3122914)),
            contours=np.asarray(policy.contours)[:number].tolist(),
            log_parameters=np.asarray(policy.log_parameters)[:number].tolist(),
            aggregation='equal-run log-parameter mean, all four final classic-volume maps',
            training_runs=inputs, training_evaluations=sum(row['evaluations'] for row in inputs),
            source_commit=freeze['source_commit'], protocol_sha256=freeze['protocol_sha256'],
            source_freeze_sha256=freeze['source_freeze_sha256'],
        ))


if __name__ == '__main__':
    main()
