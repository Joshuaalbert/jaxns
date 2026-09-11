"""Cache-independent hashes of every accepted scientific sample field."""

import dataclasses
import hashlib

import jax
import numpy as np

from jaxns.state import State


def scientific_fingerprints(state: State) -> dict[str, dict]:
    """Hash original row order, dtypes and coordinates, excluding cache storage."""
    samples = dataclasses.replace(
        state.samples, phantom_samples=dataclasses.replace(
            state.samples.phantom_samples, seed_log_L_sorted=None,
        ),
    )
    fingerprints = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(samples)[0]:
        value = np.ascontiguousarray(np.asarray(leaf)[:int(state.num_samples)])
        fingerprints[jax.tree_util.keystr(path)] = {
            "shape": list(value.shape), "dtype": str(value.dtype),
            # Hash the contiguous buffer without another full sample-array copy.
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    return fingerprints
