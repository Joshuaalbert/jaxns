"""Repair six derived B/time descriptors from immutable raw arrays, without replay."""

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import numpy as np

    from benchmarks.issue_312.analyse_reflection_tuning import reflection_work_statistics

    root = options.root
    preserved = root / 'pre-movement-diagnostic-correction'
    receipt_path = root / 'DIAGNOSTIC_CORRECTION.json'
    if receipt_path.exists():
        raise FileExistsError('Correction receipt already exists; do not overwrite the audit.')
    status = json.loads((root / 'COHORT_STATUS.json').read_text())
    assert status['analyses_complete'] == 180 and not status['workers']
    approved_fields = {
        'bounce_count_min', 'bounce_count_max', 'mean_bounce_count',
        'mean_realised_duration', 'duration_min', 'duration_max',
    }
    peak = json.loads((root / 'SCHEDULE.json').read_text())['peak']
    plans, input_hashes = [], {}
    started = time.monotonic()
    for arm in ('fixed', 'short', 'duration'):
        for allocation in ('uniform', 'evidence_improving'):
            for seed in range(30, 60):
                folder = root / 'runs' / arm / allocation / f'seed-{seed:02d}'
                diagnostics = folder / 'chain-diagnostics.npz'
                input_hashes[str(diagnostics.relative_to(root))] = hashlib.sha256(
                    diagnostics.read_bytes()
                ).hexdigest()
                with np.load(diagnostics) as raw:
                    steps = raw['realised_steps']  # [final N, 9]
                    durations = raw['realised_durations']
                corrected_rows = []
                for budget in (2000000, 4000000):
                    path = folder / f'ANALYSIS-{budget}.json'
                    before = path.read_bytes()
                    assert before == (preserved / path.relative_to(root)).read_bytes()
                    original = json.loads(before)
                    updated = copy.deepcopy(original)
                    raw_path = folder / f'calls-{budget}.npz'
                    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                    assert digest == original['archive_sha256']
                    input_hashes[str(raw_path.relative_to(root))] = digest
                    with np.load(raw_path) as raw:
                        bounces = raw['bounce_counts']
                        gradients = raw['gradient_evaluations_per_sample']
                        constraints = raw['log_L_constraints']
                        roots = raw['seed_indices'] < 0
                    count = len(bounces)
                    np.testing.assert_array_equal(gradients, 9 * bounces)
                    np.testing.assert_array_equal(
                        durations[:count], steps[:count] * bounces[:, None],
                    )
                    gaps = peak - constraints
                    for old_bin, new_bin in zip(original['movement'], updated['movement']):
                        high = np.inf if old_bin['gap_high'] is None else old_bin['gap_high']
                        select = (~roots) & (gaps > old_bin['gap_low']) & (gaps <= high)
                        assert int(select.sum()) == old_bin['pairs']
                        if select.any():
                            new_bin.update(reflection_work_statistics(
                                bounces[select], gradients[select], steps[:count][select],
                            ))
                        # Every other movement metric and every other record field is immutable.
                        for field in new_bin.keys() | old_bin.keys():
                            if field not in approved_fields:
                                assert new_bin.get(field) == old_bin.get(field), field
                    for field in original:
                        if field != 'movement':
                            assert updated[field] == original[field], field
                    plans.append((path, before, updated))
                    corrected_rows.append(updated)
                path = folder / 'ANALYSIS-COMPLETE.json'
                before = path.read_bytes()
                assert before == (preserved / path.relative_to(root)).read_bytes()
                updated = json.loads(before)
                assert updated['records'] == [
                    json.loads((preserved / (folder / f'ANALYSIS-{b}.json').relative_to(root))
                               .read_bytes()) for b in (2000000, 4000000)
                ]
                updated['records'] = corrected_rows
                plans.append((path, before, updated))
    assert len(plans) == 540
    changed = []
    # All inputs and invariants are checked before any derived record is published.
    for path, before, updated in plans:
        after = (json.dumps(updated, indent=2, allow_nan=False) + '\n').encode()
        temporary = path.with_suffix(path.suffix + '.correction-tmp')
        temporary.write_bytes(after)
        temporary.replace(path)
        digest = hashlib.sha256(after).hexdigest()
        path.with_suffix(path.suffix + '.sha256').write_text(digest + '  ' + path.name + '\n')
        changed.append({
            'path': str(path.relative_to(root)),
            'before_sha256': hashlib.sha256(before).hexdigest(),
            'after_sha256': digest,
        })
    receipt = {
        'passed': True,
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'seconds': time.monotonic() - started,
        'runs': 180, 'snapshots': 360, 'corrected_files': changed,
        'corrected_fields': sorted(approved_fields),
        'all_other_fields_unchanged': True,
        'sampling_reruns': 0, 'evidence_reruns': 0, 'chain_replays': 0,
        'input_sha256': input_hashes,
        'original_derived_outputs': str(preserved),
        'executed_sampling_source': '724d5e9a94d2db417573cd8cc473224e82a60a13',
        'source_freeze_sha256': hashlib.sha256(
            (root / 'SOURCE_FREEZE.json').read_bytes(),
        ).hexdigest(),
        'release_sha256': hashlib.sha256(
            (root / 'EVALUATION_RELEASE.json').read_bytes(),
        ).hexdigest(),
        'correction_source_sha256': {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ('correct_reflection_duration_diagnostics.py',
                         'analyse_reflection_tuning.py')
        },
    }
    temporary = receipt_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    temporary.replace(receipt_path)
    receipt_path.with_suffix('.json.sha256').write_text(
        hashlib.sha256(receipt_path.read_bytes()).hexdigest() + '  ' + receipt_path.name + '\n'
    )
    print('Corrected 540 derived JSON files; all raw arrays/evidence and other fields unchanged.')


if __name__ == '__main__':
    main()
