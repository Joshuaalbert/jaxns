"""Run one source-gated online-reflection case using the shared scientific harness."""

from benchmarks.issue_312.run_reflection_tuning import main

if __name__ == '__main__':
    main(online_comparison=True)
