# Difficult Problems Benchmark Report

Date: 2026-05-11

## Configuration

- target_num_live_points: 30
- max_samples: 1200
- shell_size: 15
- num_slices: 24
- phantom_burn_in: 4
- allocation_target: uniform
- worker_specs: cpu:*:2

## Results

| model | runtime s | samples | likelihood evals | log Z | log Z uncert | workers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eggbox | 170.564 | 1200 | 128924 | 183.88 | 1.16116 | 2 |
| rastrigin | 213.318 | 1200 | 206271 | -59.3102 | 1.05559 | 2 |
| rosenbrock | 182.274 | 1200 | 145518 | -72.9489 | 1.31345 | 2 |
| spikeslab | 182.165 | 1200 | 124688 | -34.6361 | 0.692985 | 2 |
