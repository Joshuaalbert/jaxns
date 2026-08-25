"""Mechanically reviewable JAXNS repository-structure requirements."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "jaxns"

REQUIRED_PATHS = (
    "docs/design/INVARIANTS.md",
    "docs/design/REQUIREMENTS.md",
    "cicd/coverage_record.json",
    "cicd/non_invariant_test_coverage.json",
    "cicd/pre_release_autochecks/check_all_invariants_covered.py",
    "cicd/demos/README.md",
    "cicd/benchmarks/README.md",
    "cicd/system_tests/README.md",
    "benchmarks/v2_v3/README.md",
    "benchmarks/v2_v3/environment-v2.yml",
    "benchmarks/v2_v3/environment-v3.yml",
    "benchmarks/v2_v3/prepare_environments.sh",
    "benchmarks/v2_v3/run_release_matrix.sh",
)

ARRAY_ANNOTATION_MARKERS = (
    "Array",
    "FloatArray",
    "IntArray",
    "BoolArray",
    "PRNGKey",
    "UType",
    "XType",
    "LogSpace",
    "TreeField",
)


def _decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return ""


def _is_dataclass(node: ast.ClassDef) -> bool:
    return any(_decorator_name(item) == "dataclass" for item in node.decorator_list)


def _dataclass_options(node: ast.ClassDef) -> dict[str, ast.expr]:
    for decorator in node.decorator_list:
        if _decorator_name(decorator) != "dataclass":
            continue
        if not isinstance(decorator, ast.Call):
            return {}
        return {
            keyword.arg: keyword.value
            for keyword in decorator.keywords
            if keyword.arg is not None
        }
    return {}


def _is_literal_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _annotation_text(path: Path, node: ast.AnnAssign) -> str:
    source = path.read_text(encoding="utf-8")
    return ast.get_source_segment(source, node.annotation) or ""


def _has_shape_comment(lines: list[str], node: ast.AnnAssign) -> bool:
    # Include every physical line occupied by a multi-line field declaration.
    end_lineno = node.end_lineno or node.lineno
    declaration = " ".join(lines[node.lineno - 1:end_lineno])
    return "# [" in declaration or "# ..." in declaration


def test_required_cicd_and_design_paths_exist() -> None:
    """The lifecycle structure must remain complete and discoverable."""
    missing = [path for path in REQUIRED_PATHS if not (REPO_ROOT / path).is_file()]
    assert not missing, "Missing required CI/CD paths:\n" + "\n".join(missing)


def test_array_dataclass_fields_have_shape_comments() -> None:
    """Array-bearing dataclass schemas remain readable without tracing code."""
    missing: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines = path.read_text(encoding="utf-8").splitlines()
        for class_node in (
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and _is_dataclass(node)
        ):
            for field in (
                node for node in class_node.body
                if isinstance(node, ast.AnnAssign)
            ):
                annotation = _annotation_text(path, field)
                if not any(
                    marker in annotation for marker in ARRAY_ANNOTATION_MARKERS
                ):
                    continue
                if _has_shape_comment(lines, field):
                    continue
                field_name = (
                    field.target.id
                    if isinstance(field.target, ast.Name)
                    else "<unknown>"
                )
                missing.append(
                    f"{path.relative_to(REPO_ROOT)}:{field.lineno} "
                    f"{class_node.name}.{field_name} ({annotation})"
                )

    assert not missing, (
        "Array-valued dataclass fields require an adjacent shape comment:\n"
        + "\n".join(missing)
    )


def test_production_dataclasses_are_frozen_and_slotted() -> None:
    """Scientific containers cannot be mutated or gain accidental fields."""
    invalid: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for class_node in (
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and _is_dataclass(node)
        ):
            options = _dataclass_options(class_node)
            if (
                _is_literal_true(options.get("frozen"))
                and _is_literal_true(options.get("slots"))
            ):
                continue
            invalid.append(
                f"{path.relative_to(REPO_ROOT)}:{class_node.lineno} "
                f"{class_node.name}"
            )

    assert not invalid, (
        "Production dataclasses must set frozen=True and slots=True:\n"
        + "\n".join(invalid)
    )
