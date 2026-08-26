# JAXNS CI/CD

This directory separates checks by lifecycle and intent.

- `reviewer_autochecks/` runs on every pull request. It checks review policy,
  repository structure, shape documentation, and the integrity of the
  invariant coverage ledgers.
- `pre_release_autochecks/` runs for `develop` to `main`. It may fail during
  feature development and requires every design invariant to have recorded
  test coverage before release.
- `tests/` contains the complete deterministic unit and regression suite.
- `system_tests/` contains composed public-API tests with matching design
  specifications under `docs/design/system_tests/`.
- `demos/` contains deterministic executable examples of supported workflows.
- `benchmarks/` catalogues maintained performance and accuracy measurements.
- `testing/` contains support code shared by demos and system tests.

All maintained tests and their support code live under `cicd/`; production
package discovery remains independently rooted at `src/`.

`coverage_record.json` maps exact text from
`docs/design/INVARIANTS.md` to tests that exercise it.
`non_invariant_test_coverage.json` classifies implementation, architecture,
regression, and utility tests which are valuable but are not evidence for a
scientific invariant. Every active unit test must appear in exactly one of
those ledgers.

The normal lifecycle is:

1. A feature pull request into `develop` runs unit tests and reviewer checks.
2. A push to `develop` runs the executable demos.
3. A `develop` to `main` pull request additionally runs system tests and the
   complete-invariant-coverage release gate.
4. Maintained benchmarks are run deliberately on controlled hardware and
   attached to release or performance-review evidence; noisy shared-runner
   timing is not a pull-request gate.
