# Issue 244 end-to-end standard problems

Each row contains 30 independent seeds. Timings exclude seed zero's first compilation. The z-score is `(estimate - truth) / reported uncertainty`; calibrated repetitions should have mean near zero and standard deviation near one. When a paired timing file is supplied, its alternating same-process measurements replace the corresponding cross-process timing row.

## Accuracy and calibration

| problem | phantoms | n | develop bias | candidate bias | develop RMSE | candidate RMSE | develop mean z | candidate mean z | develop SD z | candidate SD z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| basic | False | 30 | -0.00661 | -0.00661 | 0.03950 | 0.03950 | -0.097 | -0.097 | 0.925 | 0.925 |
| basic | True | 30 | -0.00661 | -0.00661 | 0.03950 | 0.03950 | -0.097 | -0.097 | 0.925 | 0.925 |
| basic2 | False | 30 | -0.00625 | -0.00625 | 0.06906 | 0.06906 | +0.027 | +0.027 | 0.956 | 0.956 |
| basic2 | True | 30 | -0.00625 | -0.00625 | 0.06906 | 0.06906 | +0.027 | +0.027 | 0.956 | 0.956 |
| basic3 | False | 30 | -0.03142 | -0.03142 | 0.17759 | 0.17759 | -0.142 | -0.142 | 0.904 | 0.904 |
| basic3 | True | 30 | -0.03142 | -0.03142 | 0.17759 | 0.17759 | -0.142 | -0.142 | 0.904 | 0.904 |
| basic_mvn | False | 30 | -0.03823 | -0.03823 | 0.27912 | 0.27912 | -0.145 | -0.145 | 1.123 | 1.123 |
| basic_mvn | True | 30 | -0.03823 | -0.03823 | 0.27912 | 0.27912 | -0.145 | -0.145 | 1.123 | 1.123 |
| plateau | False | 30 | -0.03224 | -0.03224 | 0.03224 | 0.03224 | -1.031 | -1.031 | 0.000 | 0.000 |
| plateau | True | 30 | -0.03224 | -0.03224 | 0.03224 | 0.03224 | -1.031 | -1.031 | 0.000 | 0.000 |
| spike_slab | False | 30 | +0.02448 | +0.02448 | 0.27669 | 0.27669 | +0.113 | +0.113 | 1.217 | 1.217 |
| spike_slab | True | 30 | +0.02448 | +0.02448 | 0.27669 | 0.27669 | +0.113 | +0.113 | 1.217 | 1.217 |
| spike_slab10 | False | 30 | -0.05489 | -0.05489 | 0.22834 | 0.22834 | -0.245 | -0.245 | 1.029 | 1.029 |
| spike_slab10 | True | 30 | -0.05489 | -0.05489 | 0.22834 | 0.22834 | -0.245 | -0.245 | 1.029 | 1.029 |
| weak_curved_mvn8 | False | 30 | -0.01760 | -0.01760 | 0.14331 | 0.14331 | -0.109 | -0.109 | 0.955 | 0.955 |
| weak_curved_mvn8 | True | 30 | -0.01760 | -0.01760 | 0.14331 | 0.14331 | -0.109 | -0.109 | 0.955 | 0.955 |
| weak_curved_spike_slab10 | False | 30 | +0.00011 | +0.00011 | 0.18061 | 0.18061 | +0.009 | +0.009 | 1.102 | 1.102 |
| weak_curved_spike_slab10 | True | 30 | +0.00011 | +0.00011 | 0.18061 | 0.18061 | +0.009 | +0.009 | 1.102 | 1.102 |
| weak_curved_spike_slab8 | False | 30 | +0.00477 | +0.00477 | 0.16887 | 0.16887 | +0.038 | +0.038 | 1.066 | 1.066 |
| weak_curved_spike_slab8 | True | 30 | +0.00477 | +0.00477 | 0.16887 | 0.16887 | +0.038 | +0.038 | 1.066 | 1.066 |

## Performance and deterministic accounting

`Exact` counts seeds whose evidence estimate, uncertainty, ESS, classic/phantom sample counts, and depth-loop count match `develop` bitwise. The separate mechanism benchmark measures physical likelihood work exactly.

| problem | phantoms | n | exact | develop ESS | candidate ESS | develop core s | candidate core s | wall ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| basic | False | 30 | 30/30 | 39.6 | 39.6 | 0.0150 | 0.0136 | 0.911 |
| basic | True | 30 | 30/30 | 39.6 | 39.6 | 0.0161 | 0.0138 | 0.858 |
| basic2 | False | 30 | 30/30 | 40.8 | 40.8 | 0.0175 | 0.0137 | 0.783 |
| basic2 | True | 30 | 30/30 | 40.8 | 40.8 | 0.0201 | 0.0136 | 0.678 |
| basic3 | False | 30 | 30/30 | 142.0 | 142.0 | 0.0557 | 0.0486 | 0.872 |
| basic3 | True | 30 | 30/30 | 142.0 | 142.0 | 0.0482 | 0.0479 | 0.993 |
| basic_mvn | False | 30 | 30/30 | 740.7 | 740.7 | 3.3261 | 3.5657 | 1.072 |
| basic_mvn | True | 30 | 30/30 | 740.7 | 740.7 | 3.3005 | 3.7650 | 1.141 |
| plateau | False | 30 | 30/30 | 1.0 | 1.0 | 0.0020 | 0.0020 | 0.985 |
| plateau | True | 30 | 30/30 | 1.0 | 1.0 | 0.0020 | 0.0020 | 1.019 |
| spike_slab | False | 30 | 30/30 | 852.2 | 852.2 | 1.9508 | 1.5390 | 0.789 |
| spike_slab | True | 30 | 30/30 | 852.2 | 852.2 | 1.9603 | 1.6009 | 0.817 |
| spike_slab10 | False | 30 | 30/30 | 1130.1 | 1130.1 | 4.3035 | 3.3094 | 0.769 |
| spike_slab10 | True | 30 | 30/30 | 1130.1 | 1130.1 | 4.4401 | 3.4166 | 0.769 |
| weak_curved_mvn8 | False | 30 | 30/30 | 784.7 | 784.7 | 1.1905 | 0.9434 | 0.792 |
| weak_curved_mvn8 | True | 30 | 30/30 | 784.7 | 784.7 | 1.1645 | 1.0010 | 0.860 |
| weak_curved_spike_slab10 | False | 30 | 30/30 | 1125.3 | 1125.3 | 3.1974 | 2.0558 | 0.643 |
| weak_curved_spike_slab10 | True | 30 | 30/30 | 1125.3 | 1125.3 | 3.1615 | 1.4736 | 0.466 |
| weak_curved_spike_slab8 | False | 30 | 30/30 | 794.8 | 794.8 | 1.4548 | 1.0688 | 0.735 |
| weak_curved_spike_slab8 | True | 30 | 30/30 | 794.8 | 794.8 | 1.4971 | 1.1025 | 0.736 |
