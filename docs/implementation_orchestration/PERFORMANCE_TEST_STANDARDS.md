# Performance Test Standards

These are Python-baseline guardrails, not final production latency promises.

## What Performance Tests Must Prove

- Hot-path bounds are enforced.

## Required Performance Checks

- Ensure there is no regression, keeping a list of times over release versions to track trends and catch regressions.

## Benchmark Rule

Where the design now defines baseline numeric defaults, implement a tuning benchmark so those values can later be
revisited with evidence.
