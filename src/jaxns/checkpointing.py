import dataclasses
import hashlib
import importlib.metadata
import json
import pickle
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import h5py
import jax
import numpy as np
from jax import tree_util

from jaxns.mixed_precision import mp_policy
from jaxns.samples import PhantomSamples, Samples
from jaxns.state import State

SCHEMA_VERSION = 1
ORDERING_STRATEGY = 'log_likelihood_then_sample_id'


@dataclasses.dataclass(frozen=True)
class LoadedCheckpoint:
    """Loaded checkpoint payload and runtime metadata."""

    key: Any
    state: State
    checkpoint_index: int
    checkpoint_every: int
    completed: bool


class CheckpointValidationError(ValueError):
    """Raised when an archive is incompatible with the current sampler."""

    pass


def initialise_archive(
    archive_path: str | Path,
    *,
    nested_sampler: Any,
    state: State,
    current_key: Any,
    checkpoint_every: int,
    completed: bool,
) -> None:
    """
    Create a fresh checkpoint archive from the current sampler state.

    Args:
        archive_path: Location of the HDF5 archive to create or overwrite.
        nested_sampler: NestedSampler instance whose configuration will be recorded.
        state: Initial committed state to persist into the archive.
        current_key: PRNG key to resume from after the committed state.
        checkpoint_every: Number of outer iterations grouped into each checkpoint.
        completed: Whether the run is already complete at archive creation time.
    """
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(archive_path, 'w') as handle:
        handle.create_group('meta')
        handle.create_group('config')
        handle.create_group('runtime')
        handle.create_group('journal')

        _write_meta(handle, checkpoint_every=checkpoint_every)
        _write_config(handle, nested_sampler)
        _create_journal_layout(handle, state)

        num_samples = int(np.asarray(jax.device_get(state.num_samples)))
        initial_samples = state.samples.slice(0, num_samples)
        initial_parent_sample_ids = np.full((num_samples,), -1, dtype=np.asarray(jax.device_get(state.root_out_degree)).dtype)

        _append_journal_rows(handle, samples=initial_samples, parent_sample_ids=initial_parent_sample_ids)
        _write_runtime(
            handle,
            current_key=current_key,
            state=state,
            checkpoint_index=0,
            completed=completed,
            checkpoint_every=checkpoint_every,
        )


def append_checkpoint(
    archive_path: str | Path,
    *,
    samples: Samples | None,
    parent_sample_ids: np.ndarray | None,
    current_key: Any,
    state: State,
    checkpoint_index: int,
    checkpoint_every: int,
    completed: bool,
) -> None:
    """
    Append a committed chunk delta and runtime metadata to an existing archive.

    Args:
        archive_path: Location of the HDF5 archive to update.
        samples: Flattened sample rows created since the last committed checkpoint.
        parent_sample_ids: Persistent parent identifiers for ``samples``.
        current_key: PRNG key to resume from after this committed checkpoint.
        state: Newly committed state boundary.
        checkpoint_index: Monotonic checkpoint counter for this commit.
        checkpoint_every: Number of outer iterations grouped into each checkpoint.
        completed: Whether this commit marks the final completed state.
    """
    with h5py.File(archive_path, 'a') as handle:
        if samples is not None and parent_sample_ids is not None and len(parent_sample_ids) > 0:
            _append_journal_rows(handle, samples=samples, parent_sample_ids=parent_sample_ids)
        _write_runtime(
            handle,
            current_key=current_key,
            state=state,
            checkpoint_index=checkpoint_index,
            completed=completed,
            checkpoint_every=checkpoint_every,
        )


def load_checkpoint(archive_path: str | Path, *, nested_sampler: Any) -> LoadedCheckpoint:
    """
    Load and validate a committed checkpoint archive.

    Args:
        archive_path: Location of the HDF5 archive to read.
        nested_sampler: Current NestedSampler instance used for compatibility validation.

    Returns:
        The loaded runtime key, reconstructed state, and checkpoint metadata.

    Raises:
        FileNotFoundError: If ``archive_path`` does not exist.
        CheckpointValidationError: If the archive is incompatible with ``nested_sampler``.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f'Checkpoint archive not found: {archive_path}')

    with h5py.File(archive_path, 'r') as handle:
        _validate_archive(handle, nested_sampler)

        runtime = handle['runtime']
        committed_num_samples = int(runtime['committed_num_samples'][()])
        checkpoint_index = int(runtime['committed_checkpoint_index'][()])
        checkpoint_every = int(handle['meta'].attrs['checkpoint_every'])
        completed = bool(runtime['completed'][()])
        current_key = _deserialise_key(np.asarray(runtime['current_key'][...], dtype=np.uint32))
        termination_reason = np.asarray(runtime['termination_reason'][()])
        log_L_supremum = np.asarray(runtime['log_L_supremum'][()])
        U_supremum = _read_pytree(handle['runtime'], 'U_supremum')

        journal = handle['journal']
        parent_sample_ids = np.asarray(journal['parent_sample_id'][:committed_num_samples])
        root_out_degree = np.asarray(np.count_nonzero(parent_sample_ids == -1), dtype=np.asarray(jax.device_get(mp_policy.count_dtype.type(0))).dtype)
        out_degree = np.zeros((committed_num_samples,), dtype=np.asarray(jax.device_get(mp_policy.count_dtype.type(0))).dtype)
        child_parent_ids = parent_sample_ids[parent_sample_ids >= 0]
        if child_parent_ids.size > 0:
            np.add.at(out_degree, child_parent_ids, 1)

        phantom_group = journal['phantom_samples']
        num_phantom = int(phantom_group.attrs['num_phantom'])
        if num_phantom == 0:
            phantom_valid_mask = np.zeros((committed_num_samples, 0), dtype=np.bool_)
            phantom_log_L = np.zeros((committed_num_samples, 0), dtype=journal['log_likelihoods'].dtype)
        else:
            phantom_valid_mask = np.asarray(phantom_group['valid_mask'][:committed_num_samples])
            phantom_log_L = np.asarray(phantom_group['log_L'][:committed_num_samples])

        compact_samples = Samples(
            log_L_constraints=np.asarray(journal['log_L_constraints'][:committed_num_samples]),
            log_likelihoods=np.asarray(journal['log_likelihoods'][:committed_num_samples]),
            sample_ids=np.arange(committed_num_samples, dtype=np.asarray(out_degree).dtype),
            U_samples=_read_pytree_rows(journal, 'U_samples', committed_num_samples),
            out_degree=out_degree,
            num_likelihood_evaluations=np.asarray(journal['num_likelihood_evaluations'][:committed_num_samples]),
            phantom_samples=PhantomSamples(
                U_samples=_read_pytree_rows(phantom_group, 'U_samples', committed_num_samples),
                valid_mask=phantom_valid_mask,
                log_L=phantom_log_L,
            ),
        ).sort().resize(int(nested_sampler.max_samples))

        state = State.from_checkpoint(
            root_out_degree=np.asarray(root_out_degree, dtype=np.asarray(compact_samples.out_degree).dtype),
            samples=compact_samples,
            num_samples=np.asarray(committed_num_samples, dtype=np.asarray(compact_samples.out_degree).dtype),
            log_L_supremum=log_L_supremum,
            U_supremum=U_supremum,
            termination_reason=termination_reason,
            model=nested_sampler.model,
            args=nested_sampler.args,
            params=nested_sampler.params,
        )

    return LoadedCheckpoint(
        key=current_key,
        state=state,
        checkpoint_index=checkpoint_index,
        checkpoint_every=checkpoint_every,
        completed=completed,
    )


def _write_meta(handle: h5py.File, *, checkpoint_every: int) -> None:
    """Persist schema and lifecycle metadata for a checkpoint archive."""

    meta = handle['meta']
    meta.attrs['schema_version'] = SCHEMA_VERSION
    meta.attrs['ordering_strategy'] = ORDERING_STRATEGY
    meta.attrs['jaxns_version'] = _package_version('jaxns')
    meta.attrs['jax_version'] = jax.__version__
    meta.attrs['created_utc'] = _utc_now()
    meta.attrs['updated_utc'] = meta.attrs['created_utc']
    meta.attrs['checkpoint_every'] = checkpoint_every
    meta.attrs['checkpoint_count'] = 0


def _write_config(handle: h5py.File, nested_sampler: Any) -> None:
    """Persist sampler, model, and termination configuration for resume validation."""

    config = handle['config']
    _write_payload(
        config,
        'nested_sampler',
        {
            'class_name': type(nested_sampler).__name__,
            'target_num_live_points': _jsonify(nested_sampler.target_num_live_points),
            'max_samples': _jsonify(nested_sampler.max_samples),
            'shell_size': _jsonify(nested_sampler.shell_size),
            'batch_size': _jsonify(nested_sampler.batch_size),
            'store_phantom_samples': _jsonify(nested_sampler.store_phantom_samples),
            'collect_phantom_samples': _jsonify(nested_sampler.collect_phantom_samples),
        },
    )
    _write_payload(
        config,
        'termination_condition',
        _dataclass_payload(nested_sampler.termination_condition),
    )
    _write_payload(
        config,
        'sampler',
        {
            'class_name': type(nested_sampler.sampler).__name__,
            'module': type(nested_sampler.sampler).__module__,
            'fields': _dataclass_payload(nested_sampler.sampler, exclude={'model'}),
        },
    )
    _write_payload(
        config,
        'model_signature',
        {
            'U_ndims': int(nested_sampler.model.U_ndims(args=nested_sampler.args, params=nested_sampler.params)),
            'prior_model_module': getattr(nested_sampler.model.prior_model, '__module__', None),
            'prior_model_qualname': getattr(nested_sampler.model.prior_model, '__qualname__', None),
            'args_fingerprint': _fingerprint_pytree(nested_sampler.args),
            'params_fingerprint': _fingerprint_pytree(nested_sampler.params),
        },
    )


def _create_journal_layout(handle: h5py.File, state: State) -> None:
    """Create appendable journal datasets sized from the committed sample structure."""

    journal = handle['journal']
    compact_samples = state.samples.slice(0, int(np.asarray(jax.device_get(state.num_samples))))
    phantom_group = journal.create_group('phantom_samples')
    num_phantom = int(np.asarray(jax.device_get(compact_samples.phantom_samples.valid_mask.shape[1])))

    _create_appendable_dataset(journal, 'parent_sample_id', np.asarray([-1], dtype=np.int64))
    _create_appendable_dataset(journal, 'log_L_constraints', compact_samples.log_L_constraints)
    _create_appendable_dataset(journal, 'log_likelihoods', compact_samples.log_likelihoods)
    _create_appendable_dataset(journal, 'num_likelihood_evaluations', compact_samples.num_likelihood_evaluations)
    _create_appendable_pytree(journal, 'U_samples', compact_samples.U_samples)
    phantom_group.attrs['num_phantom'] = num_phantom
    if num_phantom > 0:
        _create_appendable_dataset(phantom_group, 'valid_mask', compact_samples.phantom_samples.valid_mask)
        _create_appendable_dataset(phantom_group, 'log_L', compact_samples.phantom_samples.log_L)
    _create_appendable_pytree(phantom_group, 'U_samples', compact_samples.phantom_samples.U_samples)


def _append_journal_rows(handle: h5py.File, *, samples: Samples, parent_sample_ids: np.ndarray) -> None:
    """Append a batch of committed sample rows into the journal datasets."""

    journal = handle['journal']
    parent_sample_ids = np.asarray(parent_sample_ids)
    if parent_sample_ids.size == 0:
        return
    _append_dataset_rows(journal['parent_sample_id'], parent_sample_ids)
    _append_dataset_rows(journal['log_L_constraints'], _to_numpy(samples.log_L_constraints))
    _append_dataset_rows(journal['log_likelihoods'], _to_numpy(samples.log_likelihoods))
    _append_dataset_rows(journal['num_likelihood_evaluations'], _to_numpy(samples.num_likelihood_evaluations))
    _append_pytree_rows(journal, 'U_samples', samples.U_samples)

    phantom_group = journal['phantom_samples']
    if int(phantom_group.attrs['num_phantom']) > 0:
        _append_dataset_rows(phantom_group['valid_mask'], _to_numpy(samples.phantom_samples.valid_mask))
        _append_dataset_rows(phantom_group['log_L'], _to_numpy(samples.phantom_samples.log_L))
    _append_pytree_rows(phantom_group, 'U_samples', samples.phantom_samples.U_samples)


def _write_runtime(
    handle: h5py.File,
    *,
    current_key: Any,
    state: State,
    checkpoint_index: int,
    checkpoint_every: int,
    completed: bool,
) -> None:
    """Persist the checkpoint boundary state needed to continue the run."""

    runtime = handle['runtime']
    _write_or_update_dataset(runtime, 'current_key', _serialise_key(current_key))
    _write_or_update_dataset(runtime, 'committed_num_samples', _to_numpy(state.num_samples))
    _write_or_update_dataset(runtime, 'committed_checkpoint_index', np.asarray(checkpoint_index, dtype=np.int64))
    _write_or_update_dataset(runtime, 'termination_reason', _to_numpy(state.termination_reason))
    _write_or_update_dataset(runtime, 'completed', np.asarray(completed, dtype=np.bool_))
    _write_or_update_dataset(runtime, 'log_L_supremum', _to_numpy(state.log_L_supremum))
    _write_pytree(runtime, 'U_supremum', state.U_supremum)

    handle['meta'].attrs['updated_utc'] = _utc_now()
    handle['meta'].attrs['checkpoint_every'] = checkpoint_every
    handle['meta'].attrs['checkpoint_count'] = checkpoint_index + 1
    handle.flush()


def _validate_archive(handle: h5py.File, nested_sampler: Any) -> None:
    """Validate archive schema and sampler compatibility before resume."""

    schema_version = int(handle['meta'].attrs['schema_version'])
    if schema_version != SCHEMA_VERSION:
        raise CheckpointValidationError(
            f'Unsupported checkpoint schema version {schema_version}; expected {SCHEMA_VERSION}.'
        )

    expected_nested_sampler = _read_payload(handle['config'], 'nested_sampler')
    actual_nested_sampler = {
        'class_name': type(nested_sampler).__name__,
        'target_num_live_points': _jsonify(nested_sampler.target_num_live_points),
        'max_samples': _jsonify(nested_sampler.max_samples),
        'shell_size': _jsonify(nested_sampler.shell_size),
        'batch_size': _jsonify(nested_sampler.batch_size),
        'store_phantom_samples': _jsonify(nested_sampler.store_phantom_samples),
        'collect_phantom_samples': _jsonify(nested_sampler.collect_phantom_samples),
    }
    _validate_payload('nested_sampler', expected_nested_sampler, actual_nested_sampler)

    expected_termination = _read_payload(handle['config'], 'termination_condition')
    actual_termination = _dataclass_payload(nested_sampler.termination_condition)
    _validate_payload('termination_condition', expected_termination, actual_termination)

    expected_sampler = _read_payload(handle['config'], 'sampler')
    actual_sampler = {
        'class_name': type(nested_sampler.sampler).__name__,
        'module': type(nested_sampler.sampler).__module__,
        'fields': _dataclass_payload(nested_sampler.sampler, exclude={'model'}),
    }
    _validate_payload('sampler', expected_sampler, actual_sampler)

    expected_model_signature = _read_payload(handle['config'], 'model_signature')
    actual_model_signature = {
        'U_ndims': int(nested_sampler.model.U_ndims(args=nested_sampler.args, params=nested_sampler.params)),
        'prior_model_module': getattr(nested_sampler.model.prior_model, '__module__', None),
        'prior_model_qualname': getattr(nested_sampler.model.prior_model, '__qualname__', None),
        'args_fingerprint': _fingerprint_pytree(nested_sampler.args),
        'params_fingerprint': _fingerprint_pytree(nested_sampler.params),
    }
    _validate_payload('model_signature', expected_model_signature, actual_model_signature)

    ordering_strategy = handle['meta'].attrs['ordering_strategy']
    if ordering_strategy != ORDERING_STRATEGY:
        raise CheckpointValidationError(
            f'Checkpoint ordering strategy {ordering_strategy!r} is incompatible with {ORDERING_STRATEGY!r}.'
        )


def _validate_payload(name: str, expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """Raise a compatibility error when a stored payload differs from the live one."""

    if expected != actual:
        raise CheckpointValidationError(
            f'Checkpoint {name} mismatch. Expected {expected}, got {actual}.'
        )


def _package_version(package_name: str) -> str:
    """Return an installed package version, falling back to ``unknown``."""

    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return 'unknown'


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""

    return datetime.now(UTC).isoformat()


def _write_payload(parent: h5py.Group, name: str, payload: dict[str, Any]) -> None:
    """Store a small JSON payload under a named subgroup."""

    group = parent.create_group(name)
    group.attrs['payload_json'] = json.dumps(payload, sort_keys=True)


def _read_payload(parent: h5py.Group, name: str) -> dict[str, Any]:
    """Read a JSON payload written by ``_write_payload``."""

    return json.loads(parent[name].attrs['payload_json'])


def _dataclass_payload(value: Any, *, exclude: set[str] | None = None) -> dict[str, Any]:
    """Convert dataclass fields into a JSON-compatible comparison payload."""

    exclude = exclude or set()
    if not dataclasses.is_dataclass(value):
        return {}
    return {
        field.name: _jsonify(getattr(value, field.name))
        for field in dataclasses.fields(value)
        if field.name not in exclude
    }


def _jsonify(value: Any) -> Any:
    """Convert scalars and arrays into JSON-compatible values for metadata storage."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    device_value = jax.device_get(value)
    if isinstance(device_value, (bool, int, float, str)):
        return device_value
    try:
        array_value = np.asarray(device_value)
    except Exception:
        return repr(device_value)
    if array_value.shape == ():
        return array_value.item()
    return array_value.tolist()


def _fingerprint_pytree(tree: Any) -> dict[str, Any]:
    """Summarise a pytree with a structure-aware content fingerprint."""

    leaves, treedef = tree_util.tree_flatten(tree)
    digest = hashlib.sha256()
    digest.update(pickle.dumps(treedef, protocol=4))
    leaf_digests = []
    for leaf in leaves:
        leaf_digest = hashlib.sha256(_leaf_bytes(leaf)).hexdigest()
        leaf_digests.append(leaf_digest)
        digest.update(leaf_digest.encode())
    return {
        'fingerprint': digest.hexdigest(),
        'num_leaves': len(leaves),
        'treedef_repr': repr(treedef),
    }


def _leaf_bytes(leaf: Any) -> bytes:
    """Serialise a pytree leaf into bytes for hashing."""

    leaf = jax.device_get(leaf)
    if leaf is None:
        return b'None'
    try:
        array_value = np.asarray(leaf)
        return b'|'.join([
            str(array_value.dtype).encode(),
            str(array_value.shape).encode(),
            array_value.tobytes(),
        ])
    except Exception:
        return pickle.dumps(leaf, protocol=4)


def _create_appendable_dataset(parent: h5py.Group, name: str, example: Any) -> h5py.Dataset:
    """Create an empty resizable dataset matching an example row shape."""

    array_value = _to_numpy(example)
    shape = (0,) + array_value.shape[1:]
    maxshape = (None,) + array_value.shape[1:]
    return parent.create_dataset(name, shape=shape, maxshape=maxshape, dtype=array_value.dtype, chunks=True)


def _append_dataset_rows(dataset: h5py.Dataset, values: Any) -> None:
    """Append one or more rows to a resizable dataset."""

    array_value = _to_numpy(values)
    if array_value.shape[0] == 0:
        return
    start = dataset.shape[0]
    stop = start + array_value.shape[0]
    dataset.resize((stop,) + dataset.shape[1:])
    dataset[start:stop] = array_value


def _write_or_update_dataset(parent: h5py.Group, name: str, value: Any) -> None:
    """Create or replace a scalar or fixed-shape dataset in place."""

    array_value = _to_numpy(value)
    if name in parent:
        dataset = parent[name]
        if dataset.shape != array_value.shape or dataset.dtype != array_value.dtype:
            del parent[name]
            parent.create_dataset(name, data=array_value)
        else:
            dataset[...] = array_value
    else:
        parent.create_dataset(name, data=array_value)


def _write_pytree(parent: h5py.Group, name: str, pytree: Any) -> None:
    """Write a full pytree payload into a named subgroup."""

    if name in parent:
        del parent[name]
    group = parent.create_group(name)
    group.attrs['is_none'] = pytree is None
    if pytree is None:
        return
    leaves, treedef = tree_util.tree_flatten(pytree)
    group.attrs['num_leaves'] = len(leaves)
    group.attrs['treedef_pickle'] = np.void(pickle.dumps(treedef, protocol=4))
    for leaf_index, leaf in enumerate(leaves):
        group.create_dataset(f'leaf_{leaf_index:04d}', data=_to_numpy(leaf))


def _read_pytree(parent: h5py.Group, name: str) -> Any:
    """Reconstruct a full pytree payload from a named subgroup."""

    group = parent[name]
    if bool(group.attrs['is_none']):
        return None
    treedef = pickle.loads(bytes(group.attrs['treedef_pickle']))
    num_leaves = int(group.attrs['num_leaves'])
    leaves = [np.asarray(group[f'leaf_{leaf_index:04d}'][...]) for leaf_index in range(num_leaves)]
    return tree_util.tree_unflatten(treedef, leaves)


def _create_appendable_pytree(parent: h5py.Group, name: str, pytree: Any) -> None:
    """Create appendable datasets for each leaf of a pytree payload."""

    group = parent.create_group(name)
    group.attrs['is_none'] = pytree is None
    if pytree is None:
        return
    leaves, treedef = tree_util.tree_flatten(pytree)
    group.attrs['num_leaves'] = len(leaves)
    group.attrs['treedef_pickle'] = np.void(pickle.dumps(treedef, protocol=4))
    for leaf_index, leaf in enumerate(leaves):
        _create_appendable_dataset(group, f'leaf_{leaf_index:04d}', leaf)


def _append_pytree_rows(parent: h5py.Group, name: str, pytree: Any) -> None:
    """Append row batches to each leaf dataset in an appendable pytree."""

    group = parent[name]
    if bool(group.attrs['is_none']):
        return
    leaves, _ = tree_util.tree_flatten(pytree)
    for leaf_index, leaf in enumerate(leaves):
        _append_dataset_rows(group[f'leaf_{leaf_index:04d}'], leaf)


def _read_pytree_rows(parent: h5py.Group, name: str, length: int) -> Any:
    """Read the committed prefix of an appendable pytree payload."""

    group = parent[name]
    if bool(group.attrs['is_none']):
        return None
    treedef = pickle.loads(bytes(group.attrs['treedef_pickle']))
    num_leaves = int(group.attrs['num_leaves'])
    leaves = [
        np.asarray(group[f'leaf_{leaf_index:04d}'][:length])
        for leaf_index in range(num_leaves)
    ]
    return tree_util.tree_unflatten(treedef, leaves)


def _serialise_key(key: Any) -> np.ndarray:
    """Convert a PRNG key into an HDF5-storable uint32 array."""

    try:
        return np.asarray(jax.random.key_data(key), dtype=np.uint32)
    except Exception:
        return np.asarray(jax.device_get(key), dtype=np.uint32)


def _deserialise_key(key_data: np.ndarray) -> Any:
    """Reconstruct a PRNG key from stored uint32 key data."""

    try:
        return jax.random.wrap_key_data(np.asarray(key_data, dtype=np.uint32))
    except Exception:
        return np.asarray(key_data, dtype=np.uint32)


def _to_numpy(value: Any) -> np.ndarray:
    """Move a value to host memory and coerce it into a NumPy array."""

    return np.asarray(jax.device_get(value))