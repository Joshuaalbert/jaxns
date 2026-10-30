"""Aggregate the complete released online-reflection CPU experiment."""

from benchmarks.issue_312.summarise_reflection_tuning import main


if __name__ == "__main__":
    main(online_comparison=True)
