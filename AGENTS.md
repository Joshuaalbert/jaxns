# Agent Notes (jaxns)

- The Agent should have read LEARNINGS.md.

## Repo Layout

- Packages: `packages/jaxns/`
- Packaging: `setuptools` via `pyproject.toml` (`build-backend = setuptools.build_meta`)
- Python: `>=3.10`
- Layout: `src/` (code lives in `src/jaxns/...`)
- Deps: `requirements.txt`, `requirements-tests.txt`, and `requirements-examples.txt` (if needed)
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

## Code Style Guidelines

No formatter/type-checker is configured; aim for clean PEP-8 that passes `flake8` defaults.

- Imports: stdlib, third-party, local; one per line; Always absolute package imports; no `*`.
- Formatting: 4 spaces; no tabs; wrap long lines (~88-100); trailing commas in multi-line; f-strings.
- Types: use 3.10 typing (`List[int]`, `Dict[str, Any]`, `X | Y`); type public APIs and non-trivial internals; avoid
  `Any`.
- Naming: modules `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`; consts `UPPER_SNAKE_CASE`; private
  `_name`.
- Structure: production code in `src/jaxns/...`; tests in `tests/...`; Don't put anything inside `__init__.py` files.
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
- Benchmarks and reporting scripts should read like a human execution log: setup, run, collect results, write report.
  Do not hide one-off setup/reporting steps behind helpers. Only factor a helper when the shared code is non-trivial
  (> 5 lines), used more than once, and the helper name genuinely reduces mental tracing.
- For internal numerical code, track array shapes at the point arrays are created or transformed. Add short comments like
  `# [num_live, D]` or `# [num_components, D, D]` near non-obvious arrays so later readers do not need to mentally trace
  shape flow across functions.
- Adhere to existing code styles. Infer from code.
- Tests: deterministic unit tests; isolate time/network; use `pytest` fixtures/parametrize; keep fixtures close to
  tests. Use NamedTuples to structure test fixtures, so that we have a clear schema for test data and can easily add
  fields without breaking existing tests.
- Avoid overusing fixtures for simple test data; sometimes it's clearer to just
  define the data in the test function.
- Dependencies: runtime deps in `requirements.txt`; test/lint deps in `requirements-tests.txt`.
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

- Keep changes scoped to one package unless intentionally cross-cutting.
- Add/adjust tests with behavior changes; run `pytest` + `flake8` + `ruff` for the touched package.
- Don’t add repo-wide tooling/config unless asked; propose it if it would materially help.

## Repo Hygiene

- Avoid committing local envs/credentials (`.env`, `.venv/`, `*.egg-info/` are gitignored).
- Don’t add editor-specific settings unless asked (`.idea/` exists locally but isn’t a repo standard).
- Keep generated build artifacts out of PRs (`dist/`, `build/`).
