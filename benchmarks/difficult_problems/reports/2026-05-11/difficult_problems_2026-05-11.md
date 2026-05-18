# Difficult Problems Benchmark Report

Date: 2026-05-11

## Configuration

- dimension: 10
- target_num_live_points: 1000
- live_points_per_dimension: 100.0
- max_samples: 2000
- shell_size: 500
- num_slices: 24
- phantom_burn_in: 4
- allocation_target: uniform
- worker_specs: cpu:*:2

## Results

| model | runtime s | samples | likelihood evals | log Z | log Z uncert | workers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eggbox | 145.118 | 2000 | 70088 | 75.9758 | 0.883578 | 2 |
| rastrigin | 144.980 | 2000 | 70643 | -87.3215 | 0.783838 | 2 |
| rosenbrock | 138.274 | 2000 | 57263 | -3013.24 | 0.932387 | 2 |
| spikeslab | 137.856 | 2000 | 53793 | -34.779 | 0.612907 | 2 |
