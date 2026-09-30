"""Locate the first retained spike-dominated points in append order."""

import argparse
import json
from pathlib import Path
import pickle

import jax
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cell", type=Path, required=True)
cell = parser.parse_args().cell
core = json.loads((cell / "CORE.json").read_text())
counts = np.array([row["classic_samples"] for row in core["goal_progress"]])
means = np.zeros((2, 10))
means[:, :2] = np.array([6.0, 2.5])[:, None]
variances = np.array([0.08, 0.8])


def spike_dominates(x):
    """Compare both normalised component densities at each physical point."""
    spike_distance = np.sum((x - means[0]) ** 2, axis=-1) / variances[0]
    slab_distance = np.sum((x - means[1]) ** 2, axis=-1) / variances[1]
    return 5.0 * np.log(variances[1] / variances[0]) > 0.5 * (
        spike_distance - slab_distance
    )


posterior = np.load(cell / "classic_posterior.npz")
indices = np.flatnonzero(spike_dominates(posterior["x"]))
first = int(indices[0]) if len(indices) else None
analysis = json.loads((cell / "ANALYSIS.json").read_text())
weights = np.exp(posterior["log_dp"])
weights /= weights.sum()
np.testing.assert_allclose(
    weights[indices].sum(), analysis["classic_classified_spike_mass"], rtol=1e-12
)
record = {
    "definition": "spike density exceeds slab density",
    "classic_spike_points": len(indices),
    "first_classic_append_row": first,
    "first_classic_goal": int(np.searchsorted(counts, first + 1)) + 1
    if first is not None
    else None,
    "first_retained_phantom_goal": None,
    "new_classic_spike_points_per_goal": np.bincount(
        np.searchsorted(counts, indices + 1),
        minlength=len(counts),
    ).tolist(),
}
with (cell / "state.pkl").open("rb") as stream:
    state = pickle.load(stream)
phantoms = state.samples.phantom_samples
u = jax.tree.leaves(phantoms.U_samples)[0]  # [N,P,D]
for start in range(0, int(state.num_samples), 512):
    stop = min(start + 512, int(state.num_samples))
    x = -4.0 + 12.0 * np.asarray(u[start:stop], dtype=float)  # [tile,P,D]
    valid = np.asarray(phantoms.valid_mask[start:stop])  # [tile,P]
    identities = np.argwhere(valid & spike_dominates(x))
    if len(identities):
        row, slot = identities[0]
        row = int(start + row)
        record.update(
            first_retained_phantom_append_row=row,
            first_retained_phantom_slot=int(slot),
            first_retained_phantom_goal=int(np.searchsorted(counts, row + 1)) + 1,
        )
        break
(cell / "DISCOVERY.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
