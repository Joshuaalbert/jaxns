"""Pre-release check that every invariant has recorded unit-test coverage."""

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COVERAGE_RECORD = REPO_ROOT / "cicd" / "coverage_record.json"
INVARIANTS = REPO_ROOT / "docs" / "design" / "INVARIANTS.md"
INVARIANT_PREFIX = "- Invariant: "


def _invariants_by_section() -> dict[str, list[str]]:
    """Parse complete invariant text under its human-review section."""
    sections: dict[str, list[str]] = {}
    section = "Unsectioned"
    current: str | None = None
    for line in INVARIANTS.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if current is not None:
                sections.setdefault(section, []).append(current)
                current = None
            section = line.removeprefix("## ")
            continue
        if line.startswith(INVARIANT_PREFIX):
            if current is not None:
                sections.setdefault(section, []).append(current)
            current = line.removeprefix(INVARIANT_PREFIX)
        elif current is not None and line.startswith("  "):
            current += " " + line.strip()
        elif current is not None:
            sections.setdefault(section, []).append(current)
            current = None
    if current is not None:
        sections.setdefault(section, []).append(current)
    return sections


def _load_coverage_record() -> dict[str, list[str]]:
    raw_record = json.loads(COVERAGE_RECORD.read_text(encoding="utf-8"))
    if not isinstance(raw_record, dict):
        raise SystemExit("coverage_record.json must be a JSON object.")

    record: dict[str, list[str]] = {}
    for invariant, tests in raw_record.items():
        if not isinstance(invariant, str):
            raise SystemExit("Coverage-record keys must be strings.")
        if not isinstance(tests, list) or not tests:
            raise SystemExit(
                f"Coverage record for {invariant!r} must be a non-empty list."
            )
        if not all(isinstance(test, str) for test in tests):
            raise SystemExit(
                f"Coverage record for {invariant!r} must contain test names."
            )
        record[invariant] = tests
    return record


def main() -> int:
    invariants_by_section = _invariants_by_section()
    invariant_counts = Counter(
        invariant
        for invariants in invariants_by_section.values()
        for invariant in invariants
    )
    duplicate_invariants = sorted(
        invariant
        for invariant, count in invariant_counts.items()
        if count != 1
    )
    if duplicate_invariants:
        print(
            "Invariant text must be unique:\n"
            + "\n".join(duplicate_invariants)
        )
        return 1

    record = _load_coverage_record()
    bad_keys = sorted(
        invariant
        for invariant in record
        if invariant_counts[invariant] != 1
    )
    if bad_keys:
        print(
            "Coverage-record keys must match exactly one invariant:\n"
            + "\n".join(bad_keys)
        )
        return 1

    missing = set(invariant_counts) - set(record)
    if missing:
        print(
            "Every invariant must have recorded unit-test coverage before "
            "merging develop to main. Missing coverage:"
        )
        # Preserve design sections so reviewers can assign complete scientific
        # components instead of working through an alphabetized mixed list.
        for section, invariants in invariants_by_section.items():
            section_missing = [
                invariant for invariant in invariants if invariant in missing
            ]
            if not section_missing:
                continue
            print(f"\n{section} ({len(section_missing)}):")
            print("\n".join(f"- {item}" for item in section_missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
