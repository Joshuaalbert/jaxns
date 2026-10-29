"""Offline reconstruction of pre-dispatch information and chronological learning.

This audit never supplies feedback. It checks the saved width-sized ledger
against classic prefixes, then replays the host updates and frozen choices.
"""

import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np
from jax import numpy as jnp

from benchmarks.issue_312.online_common import load_frozen_policy
from benchmarks.issue_312.online_controller import OnlineController, observe_completed
from benchmarks.issue_312.online_policy import ChosenParameters, choose_parameters
from benchmarks.issue_312.tuning_common import atomic_json


def classic_volume_profile(
    values: np.ndarray, parent_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pure NumPy classic prefix reference, including genuine plateau blocks."""
    if values.ndim != 1 or parent_indices.shape != values.shape or not len(values):
        raise ValueError('Expected nonempty classic values and their parent indices.')
    child = parent_indices >= 0
    if np.any(parent_indices[child] >= np.flatnonzero(child)):
        raise ValueError('A classic prefix contains a parent from its future.')
    roots = int((~child).sum())
    contours, inverse, sizes = np.unique(values, return_inverse=True, return_counts=True)
    degree = np.bincount(parent_indices[child], minlength=len(values))
    outgoing = np.bincount(inverse, weights=degree).astype(np.int64)
    incoming = roots + np.concatenate([[0], np.cumsum(outgoing - sizes)[:-1]])
    if np.any(incoming < sizes):
        raise ValueError('Classic prefix violates lineage capacity.')
    greater = incoming - sizes + 1.
    equal = np.where(sizes > 1, sizes + .5, 0.)
    lower = np.where(sizes > 1, .5, 1.)
    compression = -np.cumsum(np.log(greater / (greater + equal + lower)))
    return contours, compression, incoming, roots


def audit_online_history(
    folder: Path, manifest: dict, raw: dict[str, np.ndarray], policy_path: Path | None,
) -> dict:
    """Bind all choices/updates to data available before their actual dispatch."""
    started = time.monotonic()
    final_budget = manifest['budgets'][-1]
    ledgers = {}
    for budget in manifest['budgets']:
        core = json.loads((folder / f'CORE-{budget}.json').read_text())
        for stem, field in (('controller', 'controller_ledger_sha256'),
                            ('policies', 'controller_policies_sha256')):
            path = folder / f'{stem}-{budget}.npz'
            if hashlib.sha256(path.read_bytes()).hexdigest() != core[field]:
                raise ValueError(f'Online audit input checksum failed: {path}')
        with np.load(folder / f'controller-{budget}.npz') as saved:
            ledgers[budget] = dict(saved)
    ledger = ledgers[final_budget]
    with np.load(folder / f'policies-{final_budget}.npz') as saved:
        policies = dict(saved)
    receipts = [json.loads(line) for line in (folder / 'controller-history.jsonl').read_text()
                .splitlines()]
    controller = OnlineController.initialise(manifest['initial_roots'])
    count = len(raw['samples'])
    indices = ledger['sample_indices']
    np.testing.assert_array_equal(indices, np.arange(manifest['initial_roots'], count))
    for name, other in (
        ('seed_indices', 'seed_indices'), ('parent_indices', 'parent_indices'),
        ('chain_keys', 'chain_keys'), ('log_L_constraints', 'log_L_constraints'),
        ('batch_indices', 'batch_indices'), ('work', 'evaluations_per_sample'),
        ('num_bounces', 'bounce_counts'), ('step_size', 'nominal_steps'),
        ('gradient_evaluations', 'gradient_evaluations_per_sample'),
        ('value_only_evaluations', 'value_only_evaluations_per_sample'),
    ):
        np.testing.assert_array_equal(ledger[name], raw[other][indices])
    for budget, prefix in ledgers.items():
        for name, values in prefix.items():
            np.testing.assert_array_equal(values, ledger[name][:len(values)])
    np.testing.assert_array_equal(policies['version'], np.arange(len(policies['version'])))
    frozen = (load_frozen_policy(policy_path, manifest['allocation'])
              if manifest['arm'] == 'frozen' else None)
    explore = manifest['arm'] != 'fixed'

    @jax.jit
    def choices(policy, keys, compression, contours):
        if frozen is not None:
            return jax.vmap(lambda key, level: frozen.choose(key, level, explore=explore))(
                keys, contours,
            )
        return jax.vmap(lambda key, c: choose_parameters(policy, key, c, explore=explore))(
            keys, compression,
        )

    summaries = {}
    offset = 0
    max_compression_difference = 0.
    for receipt in receipts:
        start, stop = receipt['start'], receipt['stop']
        if start != controller.next_sample or stop <= start or stop > count:
            raise ValueError('Controller ledger is not one chronological complete prefix.')
        size = stop - start
        rows = slice(offset, offset + size)
        child = ledger['seed_indices'][rows] >= 0
        contours, c, incoming, roots = classic_volume_profile(
            raw['log_likelihood'][:start], raw['parent_indices'][:start],
        )
        parent_contours = ledger['log_L_constraints'][rows]
        location = np.searchsorted(contours, parent_contours, side='right') - 1
        expected_c = np.where(location >= 0, c[np.maximum(location, 0)], 0.)
        expected_k = np.where(location >= 0, incoming[np.maximum(location, 0)], roots)
        max_compression_difference = max(max_compression_difference, float(
            np.max(np.abs(expected_c - ledger['compression'][rows])),
        ))
        np.testing.assert_allclose(ledger['compression'][rows], expected_c,
                                   rtol=2e-12, atol=2e-12)
        np.testing.assert_array_equal(ledger['incoming_k'][rows], expected_k)
        np.testing.assert_array_equal(ledger['controller_versions'][rows],
                                      int(controller.policy.version))
        strata = np.clip(np.floor(ledger['compression'][rows] / 2).astype(np.int64), 0, 63)
        np.testing.assert_array_equal(ledger['controller_generations'][rows],
                                      controller.generations[strata])
        version = int(controller.policy.version)
        np.testing.assert_array_equal(policies['log_parameters'][version],
                                      controller.policy.log_parameters)
        np.testing.assert_array_equal(policies['support'][version], controller.policy.support)
        padded = np.pad(np.arange(offset, offset + size), (0, 150 - size), mode='edge')
        selected = jax.device_get(choices(
            controller.policy, jnp.asarray(ledger['chain_keys'][padded]),
            jnp.asarray(ledger['compression'][padded]),
            jnp.asarray(ledger['log_L_constraints'][padded]),
        ))
        for name, expected in zip(ChosenParameters._fields, selected, strict=True):
            expected = np.where(child, expected[:size], -1 if name == 'choice' else 0)
            if expected.dtype.kind == 'f':
                np.testing.assert_allclose(ledger[name][rows], expected, rtol=2e-14, atol=0)
            else:
                np.testing.assert_array_equal(ledger[name][rows], expected)
        seeds = ledger['seed_indices'][rows]
        jump = np.sum((raw['unit_samples'][start:stop]
                       - raw['unit_samples'][np.maximum(seeds, 0)])**2, axis=1)
        np.testing.assert_allclose(ledger['squared_jumps'][rows], np.where(child, jump, 0.),
                                   rtol=2e-14, atol=2e-14)
        controller, actual_receipt = observe_completed(
            controller, start=start, seed_indices=seeds,
            compression=ledger['compression'][rows], incoming_k=ledger['incoming_k'][rows],
            choices=ledger['choice'][rows], squared_jumps=ledger['squared_jumps'][rows],
            work=ledger['work'][rows], policy_version=receipt['dispatch_version'],
            adapt=manifest['arm'] == 'online',
        )
        if actual_receipt != receipt:
            raise ValueError(f'Controller update differs at committed rows {start}:{stop}.')
        offset += size
        for budget, prefix in ledgers.items():
            if len(prefix['sample_indices']) != offset:
                continue
            with np.load(folder / f'policies-{budget}.npz') as saved:
                np.testing.assert_array_equal(saved['log_parameters'][-1],
                                              controller.policy.log_parameters)
                np.testing.assert_array_equal(saved['version'][-1], controller.policy.version)
            retained = prefix['seed_indices'] >= 0
            summaries[str(budget)] = dict(
                controller_version=int(controller.policy.version),
                real_parameter_change_batches=sum(
                    r['real_parameter_change'] for r in receipts if r['stop'] <= stop
                ),
                cumulative_information_exposure=controller.total_exposure.tolist(),
                pending_information_exposure=controller.exposure.tolist(),
                deepest_observed_compression=controller.deepest_compression,
                learned_strata=np.flatnonzero(controller.learned).tolist(),
                choice_counts=np.bincount(prefix['choice'][retained], minlength=5).tolist(),
                compression_movement=compression_movement(raw, prefix),
                flag_counts={name: int(prefix[name][retained].sum()) for name in (
                    'bound_clipped', 'duration_capped', 'extrapolated',
                    'extrapolation_clamped', 'compression_clipped',
                )},
            )
    if offset != len(indices) or len(summaries) != len(manifest['budgets']):
        raise ValueError('Missing complete controller snapshot or trailing ledger entries.')
    np.testing.assert_array_equal(policies['log_parameters'][-1],
                                  controller.policy.log_parameters)
    result = dict(passed=True, classic_prefix_reconstruction_passed=True,
                  chronological_replay_exact=True, parameter_replay_passed=True,
                  batches=len(receipts), samples=count,
                  max_compression_difference=max_compression_difference,
                  seconds=time.monotonic() - started, snapshots=summaries,
                  history_sha256=hashlib.sha256(
                      (folder / 'controller-history.jsonl').read_bytes()).hexdigest())
    atomic_json(folder / 'CONTROLLER-AUDIT.json', result)
    return result


def compression_movement(
    raw: dict[str, np.ndarray], ledger: dict[str, np.ndarray],
) -> list[dict]:
    """Whole-chain movement in the actual controller strata, all 18 coordinates."""
    child = ledger['seed_indices'] >= 0
    strata = np.clip(np.floor(ledger['compression'] / 2).astype(np.int64), 0, 63)
    records = []
    for stratum in np.flatnonzero(np.bincount(strata[child], minlength=64)):
        select = child & (strata == stratum)
        start = raw['unit_samples'][ledger['seed_indices'][select]]
        end = raw['unit_samples'][ledger['sample_indices'][select]]
        jumps = np.sum((end - start)**2, axis=0)
        square_jumps = np.sum((end**2 - start**2)**2, axis=0)
        total = int(ledger['work'][select].sum())
        gradient = int(ledger['gradient_evaluations'][select].sum())
        correlations, squared_correlations = [], []
        for i in range(18):
            nonconstant = len(start) > 1 and start[:, i].std() > 0 and end[:, i].std() > 0
            correlations.append(float(np.corrcoef(start[:, i], end[:, i])[0, 1])
                                if nonconstant else None)
            squared_correlations.append(float(np.corrcoef(start[:, i]**2, end[:, i]**2)[0, 1])
                                        if nonconstant else None)
        records.append(dict(
            stratum=int(stratum), pairs=int(select.sum()),
            compression_min=float(ledger['compression'][select].min()),
            compression_max=float(ledger['compression'][select].max()),
            parent_contour_min=float(ledger['log_L_constraints'][select].min()),
            parent_contour_max=float(ledger['log_L_constraints'][select].max()),
            squared_displacement_sums=jumps.tolist(),
            squared_coordinate_jump_sums=square_jumps.tolist(),
            total_calls=total, gradient_calls=gradient, value_only_calls=total-gradient,
            esjd_per_total_call=(jumps/total).tolist(), esjd_per_gradient=(jumps/gradient).tolist(),
            correlations=correlations, squared_coordinate_correlations=squared_correlations,
            whole_chain_objective=float(jumps.sum()/total),
        ))
    return records
