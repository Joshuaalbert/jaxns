# Agent Notes (jaxns)

- The Agent should have read LEARNINGS.md.
- Optional orientation: if the intended system behavior, scientific model, or
  collaboration style is still unclear, read [COMMON_CONTEXT.md](COMMON_CONTEXT.md)
  before acting. It is a mental-model aid, not a substitute for the current
  requirements, invariants, design docs, paper, or code.

## Repo Layout

- Packages: `packages/jaxns/`
- Packaging: `setuptools` via `pyproject.toml` (`build-backend = setuptools.build_meta`)
- Python: `>=3.10`
- Layout: `src/` (code lives in `src/jaxns/...`)
- Deps: `[project].dependencies` and `[project.optional-dependencies]` in `pyproject.toml`
- Tests: `pytest.ini` sets `python_files=test*` (name tests `test_*.py`)

### Run Things (all packages)

There is a conda env called `jaxns_py` for local development. To run tests and linters across all packages, use:

```bash
conda run -n jaxns_py ...
```

For terminal calls to python likewise, use:

```bash
conda run -n jaxns_py python -c "..."
```

Note: we use conda to manage the python version in our local development environment. The env is called `jaxns_py`.


## The primary goal of agent

The primary goal of the agent is to implement code with a clear intention. The common pitfall is that agents don't
understand the human intent and then human reviewers have difficulty asserting the intent.

## How agents should ensure code intention

- Linear non-refactored code: we prefer linear code that can be read top to bottom.
- Comments in the code: for every non-obvious segment of code, add a comment explaining the intent. This is especially
  important for things that have later consequences if altered. These prevent future humans/agents from making changes
  that break the intent of the code.
- Put full design doc in context: make sure to read the design doc and understand it entirely,
  following links and reading supplementary code, so that the entire relevant context is loaded.


## Code Style Guidelines

No formatter/type-checker is configured; aim for clean PEP-8 that passes `flake8` defaults.

- Imports: stdlib, third-party, local; one per line; Always absolute package imports; no `*`.
- Formatting: 4 spaces; no tabs; wrap long lines (~88-100); trailing commas in multi-line; f-strings.
- Types: use 3.10 typing (`List[int]`, `Dict[str, Any]`, `X | Y`); type public APIs and non-trivial internals; avoid
  `Any`.
- Naming: modules `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`; consts `UPPER_SNAKE_CASE`; private
  `_name`.
- Structure: production code in `src/jaxns/...`; tests in `cicd/tests/...`; Don't put anything inside `__init__.py` files.
- Errors: fail fast with validation; raise specific exceptions; no bare `except:`; add context with `raise ... from e`.
- Logging: prefer package logger over ad-hoc config; log actionable context; avoid spam in tight loops.
- Logging setup: package provides `jaxns.logging.jaxns_logger` that calls `dictConfig(...)` and exposes a
  package logger (e.g. `jaxns_logger`). Import and use those loggers directly when needed.
- Performance: keep code clear first; only optimize with evidence; keep a pure-Python reference path for any accelerated
  code. This will be mandated by design.
- How to be DRY: Refactor duplicate code into a separate function ONLY if the common code is non-trivial (> 5 lines) AND
  the abstraction is clear and doesn't require too much "mental plumbing" to understand. Avoid premature DRYing.
- It is better to have linearly readable code with some duplication than to have less code that is harder to understand
  due to abstraction. Use your judgment and err on the side of readability.
- Adhere to existing code styles. Infer from code.
- Tests: deterministic unit tests; isolate time/network; use `pytest` fixtures/parametrize; keep fixtures close to
  tests. Use NamedTuples to structure test fixtures, so that we have a clear schema for test data and can easily add
  fields without breaking existing tests.
- Avoid overusing fixtures for simple test data; sometimes it's clearer to just
  define the data in the test function.
- Dependencies: runtime deps and test/lint extras in `pyproject.toml`.
- Prefer Google style docstrings for APIs; include Args, Returns, Raises sections as appropriate; keep them up to
  date, and concise.
- Linter: `ruff`

Additional conventions (helpful when adding new code):

- Public modules/APIs: add docstrings for non-obvious behavior and edge cases.
- Data containers: prefer `@dataclass(frozen=True, slots=True)` when appropriate, and inherit `PureDataclassPytree` to
  register as pytrees and make picklable and serialisable.
- Numerics: be explicit about units/scales (seconds vs ms, basis points vs pct); encode in names. And leave a comment if
  there are any non-obvious conversions.
- Exceptions: avoid swallowing errors; if retrying, bound retries and include backoff/jitter. Failing fast is important
  in testing.
- Security: do not commit secrets; `.env` is gitignored.

## Adding New Code

- Ensure package imports works from `jaxns....` parent module.
- Prefer small modules with clear responsibilities; avoid mega-files.
- If adding CLI/entrypoints, document how to run them and add a minimal smoke test.

## Agent Workflow Expectations

- Assume other agents may be working in parallel. For work intended for a PR,
  create a dedicated worktree and PR branch from the correct base; do not edit
  the primary checkout unless the user explicitly asks you to update that
  checkout directly.
- Worktree isolation prevents file conflicts, not design conflicts. Before
  changing code, identify the feature's ownership and existing semantic
  boundaries. Do not force a new feature into a nearby pattern merely because
  the pattern exists; first verify that its intent, lifecycle, and consumers
  actually match.
- Work may span many files when it remains inside an understood boundary and
  the human and agent share a strong, concrete intention and the human
  understands and owns the scientific or product decision being made. If the
  work would change a shared contract, cross into another feature's ownership,
  or create a new boundary, stop before implementing that part and explain the
  crossing to the user. Recommend coordination with the relevant contributors
  rather than designing a shared boundary in isolation.
- Help the user see boundary crossings early. A surprising need to reach into
  another subsystem is evidence that the task or human intent may not yet be
  understood; trace the ownership and data flow, then clarify instead of
  allowing the feature to diffuse through unrelated code.
- Every PR must track its work in a GitHub issue. If the user requests a PR but
  does not provide an issue, create one and link the PR to it. Ephemeral
  investigation and exploratory work do not require an issue unless the user
  asks for one; if that work becomes a PR, create or identify its issue before
  opening the PR.
- After a PR is merged, remove its worktree. If its remote branch was deleted,
  delete the local PR branch too. Keep the primary checkout on the intended
  base branch and do not accumulate stale worktrees or branches.
- Keep changes scoped to one package unless intentionally cross-cutting.
- Add/adjust tests with behavior changes; run `pytest` + `flake8` + `ruff` for the touched package.
- Don’t add repo-wide tooling/config unless asked; propose it if it would materially help.

## Repo Hygiene

- Avoid committing local envs/credentials (`.env`, `.venv/`, `*.egg-info/` are gitignored).
- Don’t add editor-specific settings unless asked (`.idea/` exists locally but isn’t a repo standard).
- Keep generated build artifacts out of PRs (`dist/`, `build/`).
