"""Reviewer checks for invariant coverage-record integrity."""

import ast
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COVERAGE_RECORD = REPO_ROOT / "cicd" / "coverage_record.json"
NON_INVARIANT_TEST_COVERAGE = (
    REPO_ROOT / "cicd" / "non_invariant_test_coverage.json"
)
INVARIANTS = REPO_ROOT / "docs" / "design" / "INVARIANTS.md"
UNIT_TEST_ROOT = REPO_ROOT / "tests"
INVARIANT_PREFIX = "- Invariant: "


def _load_coverage_record() -> dict[str, list[str]]:
    raw_record = json.loads(COVERAGE_RECORD.read_text(encoding="utf-8"))
    assert isinstance(raw_record, dict), (
        "coverage_record.json must be a JSON object."
    )

    record: dict[str, list[str]] = {}
    for invariant, tests in raw_record.items():
        assert isinstance(invariant, str) and invariant, (
            "Coverage-record keys must be non-empty strings."
        )
        assert isinstance(tests, list) and tests, (
            f"Coverage record for {invariant!r} must be a non-empty list."
        )
        assert all(isinstance(test, str) for test in tests), (
            f"Coverage record for {invariant!r} must contain only test names."
        )
        assert len(tests) == len(set(tests)), (
            f"Coverage record for {invariant!r} contains duplicate tests."
        )
        record[invariant] = tests
    return record


def _invariant_text_counts() -> Counter[str]:
    texts: list[str] = []
    current: str | None = None
    for line in INVARIANTS.read_text(encoding="utf-8").splitlines():
        if line.startswith(INVARIANT_PREFIX):
            if current is not None:
                texts.append(current)
            current = line.removeprefix(INVARIANT_PREFIX)
        elif current is not None and line.startswith("  "):
            current += " " + line.strip()
        elif current is not None:
            texts.append(current)
            current = None
    if current is not None:
        texts.append(current)
    return Counter(texts)


def _is_test_function(node: ast.AST) -> bool:
    return (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def _is_pytest_skip(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "skip"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "mark"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "pytest"
    )


def _module_is_skipped(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            continue
        if _is_pytest_skip(node.value):
            return True
        if (
            isinstance(node.value, (ast.List, ast.Tuple))
            and any(_is_pytest_skip(item) for item in node.value.elts)
        ):
            return True
    return False


def _unit_test_functions() -> dict[str, Path]:
    tests: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(UNIT_TEST_ROOT.rglob("test*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _module_is_skipped(tree):
            continue
        for node in tree.body:
            if _is_test_function(node):
                name = node.name
                if name in tests:
                    duplicates.setdefault(name, [tests[name]]).append(path)
                tests[name] = path
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                for item in node.body:
                    if not _is_test_function(item):
                        continue
                    name = f"{node.name}.{item.name}"
                    if name in tests:
                        duplicates.setdefault(name, [tests[name]]).append(path)
                    tests[name] = path

    assert not duplicates, (
        "Unit test names must be unique for coverage_record.json:\n"
        + "\n".join(
            f"{name}: "
            + ", ".join(
                str(path.relative_to(REPO_ROOT)) for path in paths
            )
            for name, paths in sorted(duplicates.items())
        )
    )
    return tests


def _load_non_invariant_tests(
        unit_tests: dict[str, Path],
) -> dict[str, Path]:
    raw = json.loads(
        NON_INVARIANT_TEST_COVERAGE.read_text(encoding="utf-8")
    )
    assert isinstance(raw, dict), (
        "non_invariant_test_coverage.json must be a JSON object."
    )
    assert raw.get("schema_version") == 1, (
        "non_invariant_test_coverage.json schema_version must be 1."
    )
    entries = raw.get("tests")
    assert isinstance(entries, dict), (
        "non_invariant_test_coverage.json must contain a tests object."
    )

    classified: dict[str, Path] = {}
    for test_name, metadata in entries.items():
        assert isinstance(test_name, str), (
            "Non-invariant test names must be strings."
        )
        assert isinstance(metadata, dict), (
            f"Non-invariant metadata for {test_name!r} must be an object."
        )
        path = metadata.get("path")
        reason = metadata.get("reason")
        note = metadata.get("note")
        assert isinstance(path, str) and path.startswith("tests/"), (
            f"Non-invariant test {test_name!r} must declare its tests/ path."
        )
        assert isinstance(reason, str) and reason, (
            f"Non-invariant test {test_name!r} must declare a reason."
        )
        assert isinstance(note, str) and note, (
            f"Non-invariant test {test_name!r} must include a note."
        )
        assert test_name in unit_tests, (
            f"Non-invariant test {test_name!r} does not exist under tests/."
        )
        expected_path = unit_tests[test_name].relative_to(REPO_ROOT).as_posix()
        assert path == expected_path, (
            f"Non-invariant test {test_name!r} path mismatch: expected "
            f"{expected_path!r}, got {path!r}."
        )
        classified[test_name] = unit_tests[test_name]
    return classified


def test_coverage_record_entries_match_invariants_and_unit_tests() -> None:
    """Every test has one valid ownership classification."""
    record = _load_coverage_record()
    invariant_counts = _invariant_text_counts()
    unit_tests = _unit_test_functions()
    non_invariant_tests = _load_non_invariant_tests(unit_tests)

    duplicate_invariants = sorted(
        invariant
        for invariant, count in invariant_counts.items()
        if count != 1
    )
    assert not duplicate_invariants, (
        "Invariant text must be unique:\n" + "\n".join(duplicate_invariants)
    )

    bad_invariants = sorted(
        invariant
        for invariant in record
        if invariant_counts[invariant] != 1
    )
    assert not bad_invariants, (
        "Coverage-record keys must exactly match one invariant:\n"
        + "\n".join(bad_invariants)
    )

    referenced_tests = {
        test_name for test_names in record.values() for test_name in test_names
    }
    missing_tests = sorted(referenced_tests - set(unit_tests))
    assert not missing_tests, (
        "Coverage-record tests must exist under tests/:\n"
        + "\n".join(missing_tests)
    )

    overlap = sorted(referenced_tests & set(non_invariant_tests))
    assert not overlap, (
        "Tests cannot be both invariant and non-invariant coverage:\n"
        + "\n".join(overlap)
    )

    unclassified = sorted(
        set(unit_tests) - referenced_tests - set(non_invariant_tests)
    )
    assert not unclassified, (
        "Every unit test must have exactly one coverage classification:\n"
        + "\n".join(
            f"{name} ({unit_tests[name].relative_to(REPO_ROOT)})"
            for name in unclassified
        )
    )
