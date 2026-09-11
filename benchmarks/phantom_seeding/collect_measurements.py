"""Validate saved comparisons and collect small, reviewable evidence records."""

import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).parent
results = root / "results"
records = {}
manifests = {}
parity = []
paths = [
    *sorted((results / "final").glob("*/*/*.json")),
    *sorted((results / "queries").glob("*.json")),
    *sorted((results / "scaling").glob("*.json")),
    *sorted(results.glob("run-*/RESULT.json")),
    *sorted(results.glob("full-*/RESULT.json")),
    *sorted(results.glob("resume-*.json")),
]
for path in paths:
    record = json.loads(path.read_text())
    for field in ("source_sha256", "benchmark_sha256"):
        if field in record:
            manifest = record.pop(field)
            digest = hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest()
            manifests[digest] = manifest
            record[field + "_manifest"] = digest
    label = str(path.relative_to(results))
    records[label] = record
    points_path = path.with_suffix(".npz")
    if not points_path.exists() or record["variant"] == "reference":
        continue
    if "final" in path.parts:
        reference_path = path.with_name("mixed-reference.npz")
    else:
        reference_path = path.with_name(path.stem.rsplit("-", 1)[0] + "-reference.npz")
    actual = np.load(points_path)
    expected = np.load(reference_path)
    assert actual.files == expected.files
    for field in actual.files:
        left = np.ascontiguousarray(actual[field])
        right = np.ascontiguousarray(expected[field])
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes(), (label, field)
    parity.append(
        {
            "record": label,
            "reference": str(reference_path.relative_to(results)),
            "identities": int(actual["identities"].size),
            "all_fields_bitwise_equal": True,
            "output_sha256": hashlib.sha256(points_path.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        }
    )
for case in ("g10", "ss10"):
    expected = records[f"resume-{case}-reference.json"]
    actual = records[f"resume-{case}-combined.json"]
    for field in (
        "fingerprints",
        "classics",
        "capacity",
        "calls",
        "new_calls",
        "goals",
        "random_key",
        "goal_key",
    ):
        assert actual[field] == expected[field], (case, field)
for case in ("g10", "cg10", "ss10"):
    expected = records[f"run-{case}-reference-3/RESULT.json"]
    actual = records[f"run-{case}-combined-3/RESULT.json"]
    for field in (
        "fingerprints",
        "keys",
        "classics",
        "calls",
        "goals",
        "allocation_goals",
        "depth_iterations",
        "root_out_degree",
        "log_Z",
        "uncertainty",
        "kish_ess",
    ):
        assert actual[field] == expected[field], (case, field)
archive_parity = json.loads((results / "archive-parity.json").read_text())
for row in archive_parity:
    assert row["exact"]
    assert (
        row["fields"]
        == records[f"full-{row['case']}-combined/RESULT.json"]["fingerprints"]
    )
output = {
    "reference_commit": "aa79a0dc395c0d64b85cbb88d0e3e732decd7361",
    "records": records,
    "manifests": manifests,
    "selector_parity": parity,
    "archive_parity": archive_parity,
}
(root / "measurements.json").write_text(json.dumps(output, indent=2) + "\n")
print(
    len(records),
    "measurements;",
    len(parity),
    "bitwise selector comparisons;",
    sum(row["identities"] for row in parity),
    "identities; all whole-run comparisons exact",
)
