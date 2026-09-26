"""Independently verify every State archived by the paper benchmark."""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.paper_phantom_conditioning.cases import PAPER_CASES
from jaxns.checkpoint import MANIFEST_NAME, CheckpointManager


def _sha256(path: Path) -> str:
    """Hash one artifact without holding the full State pickle in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(paths: list[Path]) -> list[dict]:
    """Load records and reject duplicate scientific runs."""
    records = []
    seen = set()
    for path in paths:
        for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
        ):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Incomplete JSON in {path} at line {line_number}."
                ) from error
            key = (record["case"], record["direction"], record["seed"])
            if key in seen:
                raise ValueError(f"Duplicate benchmark record {key}.")
            seen.add(key)
            records.append(record)
    return records


def _verify_record(record: dict, state_root: Path) -> int:
    """Verify recorded hashes, then deserialize through the public manager."""
    checkpoint = record["state_checkpoint"]
    if Path(checkpoint["archive_root"]).resolve() != state_root:
        raise ValueError(
            f"Unexpected archive root for {record['case']}/"
            f"{record['direction']}/seed-{record['seed']:02d}."
        )
    checkpoint_dir = state_root / checkpoint["directory"]
    manifest_path = checkpoint_dir / MANIFEST_NAME
    if _sha256(manifest_path) != checkpoint["manifest_sha256"]:
        raise ValueError(f"Manifest hash mismatch for {checkpoint_dir}.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_path = checkpoint_dir / checkpoint["state_file"]
    if manifest["state_file"] != checkpoint["state_file"]:
        raise ValueError(f"State filename mismatch for {checkpoint_dir}.")
    if state_path.stat().st_size != checkpoint["state_bytes"]:
        raise ValueError(f"State size mismatch for {state_path}.")
    if _sha256(state_path) != checkpoint["state_sha256"]:
        raise ValueError(f"State hash mismatch for {state_path}.")
    if manifest["checksum"] != checkpoint["state_sha256"]:
        raise ValueError(f"Manifest checksum mismatch for {state_path}.")

    # Loading is intentionally repeated in a new process after production.
    # This proves that the pickle is independently usable rather than merely
    # readable by the process that created and immediately reloaded it.
    with CheckpointManager(checkpoint_dir) as manager:
        state = manager.load()
    if state is None:
        raise ValueError(f"Checkpoint unexpectedly empty at {checkpoint_dir}.")
    del state
    gc.collect()
    return checkpoint["state_bytes"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--state-root", type=Path, required=True)
    args = parser.parse_args()

    # Importing the cases above establishes the global model functions needed
    # to unpickle States. It also makes a moved or incomplete source snapshot
    # fail here instead of during later scientific analysis.
    if not PAPER_CASES:
        raise ValueError("The paper benchmark case registry is empty.")

    state_root = args.state_root.resolve()
    records = _load_records(args.paths)
    total_bytes = 0
    for index, record in enumerate(records, start=1):
        total_bytes += _verify_record(record, state_root)
        print(
            f"verified {index}/{len(records)}: {record['direction']} "
            f"{record['case']} seed {record['seed']}"
        )
    print(f"verified {len(records)} States ({total_bytes:,} bytes)")


if __name__ == "__main__":
    main()
