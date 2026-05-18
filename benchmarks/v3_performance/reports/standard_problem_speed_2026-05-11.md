# V3 Standard Problem Speed Report

Date: 2026-05-11

## Results

| allocation | workers | total s | run s | MC s | evals | log Z | log Z uncert | max active |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | cpu:*:1 | 323.092 | 316.748 | 1.666 | 301010 | -36.4762 | 0.779277 | 1 |
| uniform | cpu:*:2 | 259.151 | 251.143 | 1.925 | 301010 | -36.4762 | 0.779277 | 2 |
| uniform | cpu:*:4 | 281.687 | 271.322 | 1.677 | 301010 | -36.4762 | 0.779277 | 4 |
| uniform | cpu:*:8 | 288.454 | 272.513 | 1.924 | 301010 | -36.4762 | 0.779277 | 6 |

## Notes

- Timings are wall-clock seconds measured by this script.
- Worker sampler latency is reported as a diagnostic and is not part of the wall-clock timing fractions.
- Each allocation target owns and tears down its own local load balancer.
