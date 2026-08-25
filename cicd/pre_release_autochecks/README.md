# Pre-Release Autochecks

These checks run for `develop` to `main` pull requests. They are whole-design
release gates and are expected to expose unfinished work during development.

`check_all_invariants_covered.py` requires every invariant in
`docs/design/INVARIANTS.md` to have at least one test recorded in
`cicd/coverage_record.json`.
