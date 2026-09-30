"""Independently compare saved discovery samples for seed0 on all problems."""

import json
import os
from pathlib import Path
import pickle

import jax
import numpy as np
from jaxns.state import State


def discovery_arrays(state: State, count: int) -> tuple[np.ndarray, ...]:
    """Extract append-order rows from each scoped coordinate array."""
    phantoms = state.samples.phantom_samples
    payloads = (
        state.samples.U_samples,
        state.samples.log_likelihoods,
        state.samples.log_L_constraints,
        phantoms.U_samples,
        phantoms.log_L,
        phantoms.valid_mask,
    )
    return tuple(
        np.array(leaf[:count])
        for payload in payloads
        for leaf in jax.tree.leaves(payload)
    )


out = Path("/largedata/albert/jaxns-AC240-20260908")
assert all(
    (out / "AC240" / case / "seed-00" / "CORE.json").exists()
    for case in (
        "basic_mvn",
        "weak_curved_mvn8",
        "spike_slab",
        "curved_spike_slab8",
    )
)
assert len(os.sched_getaffinity(0)) == 1
rows = []
for case in (
    "basic_mvn",
    "weak_curved_mvn8",
    "spike_slab",
    "curved_spike_slab8",
):
    aref = (
        Path("/largedata/albert/jaxns-a181-A-20260907")
        if case in ("basic_mvn", "weak_curved_mvn8")
        else Path("/largedata/albert/jaxns-ss8-css8-20260908/A240")
    )
    ac = out / "AC240" / case / "seed-00"
    a = aref / case / "seed-00"
    count = json.loads((a / "CORE.json").read_text())["goal_progress"][0][
        "samples"
    ]
    # Later allocation can change out-degrees; append-order sample coordinates,
    # likelihoods and phantom payloads from discovery must remain identical.
    with (a / "state.pkl").open("rb") as stream:
        astate = pickle.load(stream)
    expected = discovery_arrays(astate, count)
    del astate
    with (ac / "state.pkl").open("rb") as stream:
        state = pickle.load(stream)
    actual = discovery_arrays(state, count)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, np.asarray(right))
    rows.append(
        dict(
            case=case,
            seed=0,
            first_goal_samples=count,
            identical_classic_and_phantom_payloads=True,
        )
    )
    del state, actual, expected
(out / "validation/DISCOVERY_PREFIX.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
print(json.dumps(rows, indent=2))
