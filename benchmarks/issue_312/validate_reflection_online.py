"""Held-out independent ensembles for fixed and both frozen learned policies."""

from benchmarks.issue_312.validate_reflection_tuning import main

if __name__ == '__main__':
    main(online_comparison=True)
