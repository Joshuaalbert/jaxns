"""Append-only JSONL helpers for resumable benchmark groups."""

import json
from pathlib import Path


def completed_seeds(
        output_path: Path | None,
        *,
        implementation: str,
        source_id: str,
        case: str,
        phantoms: bool,
) -> set[int]:
    """Return completed seeds after validating an existing output group."""
    if output_path is None:
        return set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        output_path.touch()
        return set()

    seeds: set[int] = set()
    for line_number, line in enumerate(
            output_path.read_text(encoding="utf-8").splitlines(),
            start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        identity = (
            record.get("implementation"),
            record.get("source_id"),
            record.get("case"),
            record.get("phantoms"),
        )
        expected = (implementation, source_id, case, phantoms)
        if identity != expected:
            raise ValueError(
                f"{output_path}:{line_number} belongs to {identity}, "
                f"expected {expected}."
            )
        seed = int(record["seed"])
        if seed in seeds:
            raise ValueError(
                f"{output_path}:{line_number} duplicates seed {seed}."
            )
        seeds.add(seed)
    return seeds


def append_record(output_path: Path | None, line: str) -> None:
    """Append one complete record without rewriting earlier successful runs."""
    if output_path is None:
        return
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(line + "\n")
