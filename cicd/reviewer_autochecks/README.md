# Reviewer Autochecks

Reviewer autochecks enforce mechanically reviewable requirements which are
not themselves scientific invariants. They are intentionally separate from
unit and system tests.

`test_coverage_record.py` verifies that invariant keys exactly match the
design document, referenced test functions exist, test names are unique, and
every active unit test is classified exactly once as invariant or
non-invariant coverage. It deliberately permits uncovered invariants during a
feature pull request; the pre-release gate does not.

`test_repository_structure.py` checks the design/CI files and enforces the
array-shape comments required on pytree dataclass fields in production source.
