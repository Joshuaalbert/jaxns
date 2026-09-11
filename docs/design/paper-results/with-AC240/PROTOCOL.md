# A+C240 protocol

Branch lineage: measured A240 -> A+C240. Production src is byte-for-byte A; only the experiment allocation policy changes. All four current problems (G8, CG8, SS8 scale0.43/rho0.031, component-centered curved CSS8 beta0.4), seeds0--29, root degree240, width80,80 isotropic no-step-out slice transitions, CPU float64.

All79 phantom U states remain eligible seeds throughout discovery and continuation, with no bounded phantom reservoir. The first complete goal iteration uses uniform delta_K=1, exactly as A. Each run checks sample count and expected evidence uncertainty against the corresponding saved A first goal. The complete state and random key are then resumed using C's evidence-improving direct gap ceil(240*unit_peak_utility). Stop at classic expected sigma(log Z)<0.05; depth log1p(.001). Fixed screens300M likelihood evaluations or300 goals.

Evidence analysis uses the same leading prefixes0,8,...72,2048 paired Monte Carlo draws, and C_min20. Posterior mode weights remain classic-only and independent of phantom prefix. The comparison includes existing R/A/C/Cprime measurements without rerunning or changing them. The paper tables remain R240.

Output: /largedata/albert/jaxns-AC240-20260908/AC240/{case}/seed-00..29. At most60 concurrent workers, each pinned to one distinct CPU. Frozen source manifests and per-cell raw State pickles are retained.
