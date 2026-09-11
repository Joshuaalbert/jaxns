"""Compare fresh optimized 0.05 runs with every scientific field of archived A."""

import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.phantom_seeding.classic_metrics import classic_metrics
from benchmarks.phantom_seeding.state_fingerprints import scientific_fingerprints

results = Path(__file__).parent / "results"
archive = Path("/largedata/albert/jaxns-evidence10-AC300-20260909/0.05")
checks = []
for case in ("g10", "cg10", "ss10"):
    source = archive / case / "seed-00/state.pkl"
    with source.open("rb") as stream:
        state = pickle.load(stream)
    actual = json.loads((results / f"full-{case}-combined/RESULT.json").read_text())
    expected = scientific_fingerprints(state)
    assert actual["fingerprints"] == expected, case
    state_values = {
        "classics": state.num_samples,
        "calls": state.total_num_likelihood_evaluations,
        "goals": state.goal_loop_iter,
        "allocation_goals": state.allocation_loop_iter,
        "depth_iterations": state.depth_loop_iter,
        "root_out_degree": state.root_out_degree,
    }
    for label, value in state_values.items():
        assert actual[label] == int(value), (case, label)
    state_keys = {
        "random_key": state.random_key,
        "goal_key": state.goal_key,
    }
    for key, value in state_keys.items():
        assert actual["keys"][key] == np.asarray(value).tolist(), (case, key)
    metrics = classic_metrics(state)
    metric_values = {
        "log_Z": metrics.log_Z,
        "uncertainty": metrics.log_Z_uncert,
        "kish_ess": metrics.kish_ess,
    }
    for label, value in metric_values.items():
        assert actual[label] == float(value), (case, label)
    check = {
        "case": case,
        "archive": str(source),
        "fields": expected,
        "classics": int(state.num_samples),
        "calls": int(state.total_num_likelihood_evaluations),
        "exact": True,
    }
    checks.append(check)
    print(case, check["classics"], check["calls"],
          "all scientific fields, keys and metrics EXACT", flush=True)
    del state
(results / "archive-parity.json").write_text(json.dumps(checks, indent=2) + "\n")
