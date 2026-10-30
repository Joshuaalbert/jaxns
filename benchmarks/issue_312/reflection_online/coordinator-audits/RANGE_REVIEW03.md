# Numerical range decision before training/evaluation

The independent audit now supports a narrower FIRST-EXPERIMENT domain:
desired h in [0.001, 0.03], requested nominal T in [0.003, 0.1]. Keep the
declared integer rounding and effective h=T/B, then existing [.8,1.2] jitter.
Within these bounds B<=100, the old B<=256 cap is inactive, and T>=.003.
Ceil rounding means the effective h can approach .00075 near T=.003,
desired h=.001; the new probe explicitly tests just above that ratio boundary.
Do not claim that desired h is a lower bound on the effective/jittered h.

Evidence for this decision (all BEFORE training/evaluation):
- The original analytic-vector-field independent probe fails actual round-trip
  tolerances at B256,T.00256,h1e-5, including accepted paths: position2.6e-4,
  velocity1.46. Preserve this finding and its exact source/raw diagnostics.
- A wider production-model self-reversal grid also has failures in some small-h
  and long-duration settings. Its NumPy-vs-TFP path comparison is NOT a clean
  equivalent-vector-field gate: TFP computes standardized differences with
  float32 moderate parameters, whereas the manual reference uses centered
  residual arithmetic. Tiny vector-field differences are amplified in the
  sensitive regime. Preserve it as a diagnostic, do not conflate every NumPy
  discrepancy with a failed production-model self-reversal.
- Corrected independent verification separates (a) self-reversal using the
  unchanged normalized production toy and (b) NumPy/JAX phase comparison using
  the SAME explicit unnormalized analytic vector field. At the proposed range,
  4480 fresh paths,35 parameter settings,140 setting/gap cells ALL pass original
  position2e-9/velocity2e-8 thresholds, no relaxation. Maxima: production
  reversal position3.11e-15/velocity8.34e-13; independent phase position1.08e-14/
  velocity8.12e-12. This is numerical evidence on the probed cases, not a theorem
  that every floating-point trajectory is reversible.

Exact successful script,receipt,raw:
/tmp/jaxns-online-stable-range-probe-20261029.{py,json,npz}
Script SHA256:158b63ec4fbf53e57ea810a972a31170cacad7b0b2cdfcb4bf4f43f2caffacd7
Other preserved diagnostic stages:
/tmp/jaxns-online-independent-trajectory-probe-20261029.{py,json}
/tmp/jaxns-online-phase-failure-analysis-20261029.{py,json,npz}
/tmp/jaxns-online-range-probe-20261029.{py,json,npz}

Implement only the bounds change, document why the proposed range was narrowed,
and update affected tests/protocol before final source freeze. Preserve prior
source b8dffd9 and smoke receipts. Repeat relevant bound/frozen/source gates and
final-source smoke/replay checks; do not select any setting from evidence RMSE.
No core integrator change or tolerance change is requested. Continue runner /
reduction integration. Notify coordinator when final source, smokes and tests
are ready for training release; large evaluation still needs frozen-training
policies and corresponding held-out validation.
