"""Run the prespecified three-arm duration comparison with the shared observer harness."""

from benchmarks.issue_312.run_reflection_tuning import main

if __name__ == "__main__":
    main(duration_comparison=True)
