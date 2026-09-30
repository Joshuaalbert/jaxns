# Findings from A+C240

A+C combines C's likelihood savings with substantially better mixture accuracy. Against C, the Ph72 evidence RMSE differences are -0.0259 for SS8 (paired95% interval[-0.0430,-0.0089]) and -0.0250 for CSS8 ([-0.0403,-0.0096]). Mode RMSE also improves on both mixtures. Mean likelihood costs are within2% of C.

The accuracy differences between A+C and A remain unresolved at30 seeds. A+C uses about70% fewer likelihood evaluations than A on SS8/CSS8 and about50% fewer on G8/CG8. On G8/CG8, its evidence-accuracy differences from C are also unresolved.

Runtime and memory remain distinct costs. A+C's median core times are976s and1002s on G8/CG8, versus201s and203s for C. Median peak RSS is about4.3GiB versus2.9GiB. A+C's saved Gaussian states are about483--488MiB, compared with88--89MiB for C. On the smaller mixture trees, median core times are60--62s for A+C and57--59s for C. Core times include compilation and differing host loads; the controlled likelihood-work comparison is stronger than causal attribution of timing differences.

These are exploratory paired-bootstrap intervals without correction for multiple comparisons. An unresolved accuracy difference does not establish equivalence. See COMPARISON.md for all five policies and full contrasts, and resources.json for the per-seed resource measurements.

All120 new runs passed the stopping criterion without hitting the fixed screens. All120 first-goal sample counts and uncertainties match A. The seed0 saved classic and phantom discovery payloads match A exactly on all four problems. The two initial failures of the optional prefix-audit script were schema-access errors in the checker; their original logs are preserved beside the corrected successful audit. No experiment source or result was changed by those corrections.
