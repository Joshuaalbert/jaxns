# A+C300: 30-seed evidence and SS10 recovery

| Problem | Stop | Classic RMSE | Ph90 RMSE | Mean reported SD, classic / Ph90 | Calls (M) | Goals |
|---|---:|---:|---:|---:|---:|---:|
| G10 | 0.05 | 0.0473 ± 0.0054 | 0.0239 ± 0.0026 | 0.0494 / 0.0173 | 63.134 | 28.60 |
| CG10 | 0.05 | 0.0328 ± 0.0048 | 0.0286 ± 0.0038 | 0.0493 / 0.0179 | 64.570 | 29.00 |
| SS10 | 0.05 | 0.6256 ± 0.0435 | 0.6250 ± 0.0437 | 0.0491 / 0.0139 | 28.742 | 19.93 |
| SS10 | 0.02 | 0.5268 ± 0.0385 | 0.5226 ± 0.0389 | 0.0199 / 0.0059 | 193.296 | 135.10 |

| SS10 stop | Mean spike mass | Spike-mass RMSE | Range | Recovered |
|---:|---:|---:|---:|---|
| 0.05 | 0.21483 | 0.39752 ± 0.02193 | 0.00055–0.85290 | False |
| 0.02 | 0.27593 | 0.32273 ± 0.02045 | 0.02624–0.81902 | False |

The true spike fraction is 0.5000077444. RMSE uncertainties are whole-seed bootstrap SEs.
Each prefix uses the same classic posterior. Calls and goals for resumed stages are cumulative.
The 95% interval for Ph90 RMSE minus classic RMSE is paired across trees:

- G10 at 0.05: [-0.03439, -0.01190].
- CG10 at 0.05: [-0.01625, +0.00776].
- SS10 at 0.05: [-0.02143, +0.02161].
- SS10 at 0.02: [-0.01168, +0.00234].
