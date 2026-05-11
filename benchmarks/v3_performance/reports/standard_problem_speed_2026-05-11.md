# V3 Standard Problem Speed Report

Date: 2026-05-11

## Results

| allocation | workers | total s | run s | MC s | evals | log Z | log Z uncert | max active |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | cpu:*:2 | 268.663 | 266.318 | 1.684 | 301010 | -36.4762 | 0.779277 | 2 |
| evidence_improving | cpu:*:2 | 268.037 | 265.602 | 1.668 | 301010 | -36.4762 | 0.779277 | 2 |
| posterior_improving | cpu:*:2 | 269.165 | 266.525 | 1.865 | 301010 | -36.4762 | 0.779277 | 2 |

## Notes

- Timings are wall-clock seconds measured by this script.
- Worker sampler latency is reported as a diagnostic and is not part of the wall-clock timing fractions.
- Each allocation target owns and tears down its own local load balancer.
