# Current comparison including A+C240

[COMPARISON.md](COMPARISON.md) contains all five policies on all four current problems, with paired bootstrap contrasts and measured runtime/memory. [accuracy_cost.pdf](accuracy_cost.pdf) plots Ph72 evidence RMSE against likelihood work. summary.json retains all seed errors and exact source paths; resources.json retains per-seed times, RSS, classic counts and saved-state sizes.

A+C240 is a new 120-run cohort from frozen commit b6889bce3079ffa232e4da6b606e1af0ed0c3f00, on paper/AC240. All production src matches A. It starts with A's first uniform goal, then resumes the complete state/key under C's evidence-improving target. All79 phantom U points remain seed candidates; evidence conditioning uses the same leading 72 phantoms as every policy.

The existing R/A/C/Cprime summaries and raw input files were verified unchanged. Their definitions are the current SS8 covariance multiplier 0.43/rho0.031 and curved CSS8 with beta0.4 centered independently on each component. All paper tables remain R240; this comparison does not change the manuscript.

All new raw states and analyses are retained in /largedata/albert/jaxns-AC240-20260908/AC240/{basic_mvn,weak_curved_mvn8,spike_slab,curved_spike_slab8}/seed-00..29. AUDIT.json verifies 120 completed cells, 240 successful phases, exactly matched A discovery metrics, 2048×10 evidence draws, classic posterior normalization and the maximum 60 concurrent single-core workers. validation/ preserves the frozen manifests, dispatcher events, live affinity checks, protocol test log and exact analysis scripts.

Statistical intervals use 10,000 paired seed resamples with seed 20260908, without multiple-comparison correction. Timing includes JAX compilation and differing cohort host loads; likelihood counts are the more controlled work comparison. Peak RSS is per-process resident memory, not a universal production memory requirement.
