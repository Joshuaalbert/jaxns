Migrating from JAXNS v2 to v3
=============================

JAXNS v3 replaces the v2 live-point implementation with the paper-faithful
race-tree core, explicit classic and phantom-conditioned final inference,
immutable resumable state, and a Python goal loop around compiled depth
epochs. Scientific users should switch for the maintained correctness model,
clearer evidence API, and the measured high-dimensional speed and mode-weight
improvements below. The v2 branch remains available for narrow backports under
the :doc:`version-support` policy.

Release comparison
------------------

The initial release matrix contains 30 independent complete runs for every
problem and configuration (``n=30``), not 30 live points or shrinkage draws.
Both lines used float64 on one Intel Core i7-8750H CPU with Python 3.12.9,
JAX/JAXLIB 0.10.0, root degree ``30D``, ``num_slices=5D``, and
``dlogZ=log1p(1e-3)``. Each final estimate used 1,000 MC shrinkage draws. The
table reports MC log-evidence RMSE and warmed core runtime; the full report
also lists bias, uncertainty, coverage, likelihood evaluations, ESS, result
and MC time, peak RSS, phantom counts, gate activation, and Kish cluster
counts.

.. list-table:: Per-problem v2/v3 release evidence (n=30 per cell)
   :header-rows: 1
   :widths: 27 13 13 16 13 13

   * - Problem
     - v2 classic RMSE
     - v3 classic RMSE
     - v3 phantom RMSE
     - v2 core s
     - v3 core s
   * - basic
     - 0.0465
     - 0.0399
     - 0.0398
     - 0.006
     - 0.013
   * - basic2
     - 0.0778
     - 0.0698
     - 0.0708
     - 0.006
     - 0.015
   * - basic3
     - 0.3049
     - 0.1773
     - 0.1691
     - 0.048
     - 0.050
   * - plateau
     - 0.0335
     - 0.0322
     - 0.0323
     - 0.000
     - 0.002
   * - basic MVN (8D)
     - 0.2381
     - 0.2780
     - 0.1559
     - 3.229
     - 1.710
   * - Spike–Slab (8D)
     - 0.3053
     - 0.2783
     - 0.2756
     - 1.816
     - 1.178
   * - Spike–Slab (10D)
     - 0.2669
     - 0.2299
     - 0.2398
     - 4.234
     - 2.510
   * - weak curved MVN (8D)
     - 0.1604
     - 0.1433
     - 0.1329
     - 0.817
     - 0.830
   * - weak curved Spike–Slab (8D)
     - 0.2067
     - 0.1675
     - 0.1497
     - 1.500
     - 0.996
   * - weak curved Spike–Slab (10D)
     - 0.2033
     - 0.1817
     - 0.1138
     - 2.976
     - 2.276

The substantive high-dimensional cases are generally faster in the v3 core.
The clearest scientific gains are basic3 and the curved or separated-mixture
problems. For weak-curved 10D Spike–Slab, phantom-conditioned RMSE is 0.1138
versus v2's 0.2033 classic result, with 93.3% two-sigma coverage. V3 also
reduces the 10D Spike–Slab classic RMSE from 0.2669 to 0.2299. The maintained
runner now records analytic posterior means and mixture mode weights directly,
so future releases cannot rely on evidence alone to hide a missed or wrongly
weighted mode.

Phantom conditioning
--------------------

In v3, phantoms are correlated auxiliary observations used only in an explicit
final shrinkage inference. They never become classic race arrivals and merely
collecting them does not change the classic run. Use ``conditioning="classic"``
for the race-tree-only posterior or ``conditioning="phantom"`` when the run
retained the required phantom metadata. The benchmark reports both modes and
the effective participating-cluster gate; raw phantom state counts must not be
interpreted as independent sample counts.

Known trade-offs
----------------

* Phantom-conditioned post-processing remains the largest cost. In the 8D
  basic-MVN case, 1,000 MC draws took a median 5.054 s and process peak RSS was
  2,726.5 MiB, versus 0.316 s without phantoms. Bounded-memory follow-up is
  tracked in issue 249.
* Basic-MVN phantom RMSE improves over v3 classic (0.1559 versus 0.2780), but is
  slightly worse than v2 phantom RMSE 0.1416 and has 86.7% two-sigma coverage.
* Separated-mode allocation is improved, not solved. V3 8D Spike–Slab coverage
  is 80% in both modes, and the 10D phantom-conditioned row has 70% coverage.
  Phantom conditioning cannot repair classic samples whose mode proportions
  were already wrong.
* V2 retains lower warmed core overhead on tiny one-dimensional cases, while
  v3's wider replacement batches win on the substantive high-dimensional
  cases. Compilation, result construction, and MC time must be considered
  separately for the user's likelihood cost.

API and default changes
-----------------------

V3 requires Python 3.10 or newer. Models use ``jaxctx`` priors and the v3
``jaxns.model.Model`` rather than the v2 generator framework. A
``NestedSampler`` returns immutable ``State`` objects through ``run`` and
``resume_until_goal``; convert a completed state with ``state.to_result()``.
Final MC evidence calls require an explicit classic or phantom conditioning
choice. The default root degree is ``30D``, slice transitions are ``5D``, the
replacement target is ``10D``, and the evidence goal is
``log1p(1e-3)``. State storage has a finite scientific limit by default, with
explicit opt-in transparent growth.

Artifacts and reproducibility
-----------------------------

The `full per-problem report
<https://github.com/Joshuaalbert/jaxns/blob/main/benchmarks/issue_247/REPORT.md>`_,
`raw v2 data
<https://github.com/Joshuaalbert/jaxns/tree/main/benchmarks/issue_247/results/final_v2>`_,
`raw v3 data
<https://github.com/Joshuaalbert/jaxns/tree/main/benchmarks/issue_247/results/final_comparison>`_,
and `representative corner and shrinkage plots
<https://github.com/Joshuaalbert/jaxns/tree/main/benchmarks/issue_247/diagnostics>`_
are versioned. The maintained runner, pinned environments, schema validation,
and exact reproduction commands live in ``benchmarks/v2_v3``. The initial raw
v3 corpus records its source as ``working-tree``; it is retained as the reviewed
v3.0 pre-release baseline, while final and subsequent gates must use full
commit-addressed records.
