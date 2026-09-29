"""Print a v3 prior-model snippet from a v2 generator without editing its file.

Usage:
    python migrate_v2_model.py model.py --prior prior_model_v2 \
        --likelihood log_likelihood_v2 > migrated_model.py

Keep the original likelihood, helper functions, and distribution imports.
The v2 generator must yield Prior objects and return positional likelihood
arguments. Comments and formatting inside the prior function are preserved.
"""

import argparse
import ast
from pathlib import Path


class _FindPrior(ast.NodeVisitor):
    def __init__(self, name: str):
        self.name = name
        self.matches: list[ast.FunctionDef] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Do not descend into other functions: a closure cannot be lifted out
        # safely without its enclosing variables.
        if node.name == self.name:
            self.matches.append(node)


class _ConvertBody(ast.NodeVisitor):
    def __init__(self, source: bytes, offsets: list[int], likelihood: str):
        self.source = source
        self.offsets = offsets
        self.likelihood = likelihood
        self.edits: list[tuple[int, int, bytes]] = []
        self.yields = 0
        self.returns = 0

    def span(self, node: ast.AST) -> tuple[int, int]:
        # AST columns count UTF-8 bytes, not characters. Using bytes preserves
        # comments and non-ASCII text without unparse rewriting the whole file.
        return (
            self.offsets[node.lineno - 1] + node.col_offset,
            self.offsets[node.end_lineno - 1] + node.end_col_offset,
        )

    def visit_Yield(self, node: ast.Yield) -> None:
        if node.value is None:
            raise ValueError("A bare yield is not a yielded Prior.")
        if any(
            type(child) in (ast.Yield, ast.YieldFrom)
            for child in ast.walk(node.value)
        ):
            raise ValueError(
                "Nested yield expressions require manual migration."
            )
        value_start, value_end = self.span(node.value)
        start, end = self.span(node)
        value = self.source[value_start:value_end]
        self.edits.append((start, end, b"(" + value + b").realise()"))
        self.yields += 1

    def visit_Return(self, node: ast.Return) -> None:
        # Validate the source syntax before emitting a positional call. A
        # non-tuple return may require a different likelihood calling rule.
        if type(node.value) is not ast.Tuple:
            raise ValueError(
                "Return the positional likelihood arguments as a tuple."
            )
        if any(
            type(child) in (ast.Yield, ast.YieldFrom)
            for child in ast.walk(node.value)
        ):
            raise ValueError(
                "Yielded return expressions require manual migration."
            )
        # Returned tuples remain intact, including multiline expressions and
        # comments. Expanding them preserves the v2 positional calling rule.
        start, end = self.span(node.value)
        arguments = self.source[start:end]
        replacement = self.likelihood.encode() + b"(*(" + arguments + b"))"
        self.edits.append((start, end, replacement))
        self.returns += 1

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        raise ValueError(
            "yield from requires manual migration of the delegated model."
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        raise ValueError("Nested functions require manual migration.")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        raise ValueError("Async functions require manual migration.")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        raise ValueError("Nested callables require manual migration.")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        raise ValueError("Nested classes require manual migration.")


def convert_prior_model(
    source: str,
    prior_name: str,
    likelihood_name: str,
    new_name: str = "prior_model_v3",
) -> str:
    """Convert one named generator to a reviewable v3 source snippet.

    This does not execute the input, edit imports, or migrate sampler settings.
    The caller supplies an ordinary positional likelihood function name.
    """
    for name in (prior_name, likelihood_name, new_name):
        if not name.isidentifier():
            raise ValueError(f"Expected a simple function name, got {name!r}.")
    finder = _FindPrior(prior_name)
    finder.visit(ast.parse(source))
    if len(finder.matches) != 1:
        raise ValueError(
            f"Expected exactly one function named {prior_name!r}."
        )
    function = finder.matches[0]
    if function.decorator_list or function.returns is not None:
        raise ValueError(
            "Remove or migrate decorators and return annotations manually."
        )
    if function.col_offset != 0:
        raise ValueError("The prior function must be defined at module level.")

    encoded = source.encode("utf-8")
    offsets = [0]
    for line in encoded.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    converter = _ConvertBody(encoded, offsets, likelihood_name)
    for statement in function.body:
        converter.visit(statement)
    if converter.yields == 0 or converter.returns == 0:
        raise ValueError(
            "Expected a generator with yielded priors and an explicit return."
        )
    start, _ = converter.span(function)
    # The AST ends before a trailing comment on the last statement. Include
    # that whole source line so an explanation beside the return is retained.
    end = offsets[function.end_lineno]
    name_start = start + len(b"def ")
    converter.edits.append(
        (name_start, name_start + len(prior_name.encode()), new_name.encode())
    )
    converted = encoded[start:end]
    for edit_start, edit_end, replacement in sorted(
        converter.edits, reverse=True
    ):
        converted = (
            converted[: edit_start - start]
            + replacement
            + converted[edit_end - start:]
        )
    snippet = (
        "from jaxns.model import Model\n"
        "from jaxns.priors import Prior\n\n"
        + converted.decode("utf-8").rstrip("\n")
        + f"\n\nmodel_v3 = Model(prior_model={new_name})\n"
    )
    ast.parse(snippet)
    return snippet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Original Python model file (read only)."
    )
    parser.add_argument(
        "--prior", default="prior_model", help="Generator function name."
    )
    parser.add_argument(
        "--likelihood",
        default="log_likelihood",
        help="Likelihood function name.",
    )
    parser.add_argument(
        "--new-name", default="prior_model_v3", help="New function name."
    )
    arguments = parser.parse_args()
    try:
        snippet = convert_prior_model(
            arguments.source.read_text(encoding="utf-8"),
            arguments.prior,
            arguments.likelihood,
            arguments.new_name,
        )
    except (ValueError, SyntaxError) as error:
        parser.error(str(error))
    print(snippet, end="")


if __name__ == "__main__":
    main()
