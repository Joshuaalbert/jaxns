# V3 Performance Split Benchmark Report

Date: 2026-05-11

## Pure-Core Results

| row | mode | setup s | compile s | run s | result s | MC s | likelihood evals | accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| standard_basic_mvn_uniform | pure_core | 0.408 | 0.000 | 18.372 | 0.717 | 1.684 | 460364 | True |
| standard_basic_mvn_evidence_improving | pure_core | 0.001 | 0.000 | 27.908 | 0.083 | 0.789 | 319413 | True |
| standard_basic_mvn_posterior_improving | pure_core | 0.001 | 0.000 | 29.362 | 0.056 | 0.793 | 289770 | True |

## Distributed Results

| row | mode | setup s | compile s | run s | result s | MC s | likelihood evals | accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |

## Diagnostics

- JAX cache/static-shape diagnostics are reported for pure_core rows.
- worker topology diagnostics are reported for distributed rows.
