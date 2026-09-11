"""Combine A's unchanged seed implementation with C's allocation harness."""
from pathlib import Path

root=Path('/largedata/albert/git/jaxns-paper-AC240')
croot=Path('/largedata/albert/git/jaxns-paper-ss8-css8-C240')
s=(croot/'benchmarks/paper_reproduction/run.py').read_text()
s=s.replace('"""Test evidence allocation after one shared uniform discovery iteration."""', '"""Combine all-phantom seeds with evidence allocation after A\'s first goal."""')
s=s.replace('max_phantom_samples=72', 'max_phantom_samples=79')
s=s.replace('assert core.sampler.num_phantom() == 72', 'assert core.sampler.num_phantom() == 79')
s=s.replace('"phantom_seeding": False', '"phantom_seeding": True')
s=s.replace('"retained_phantoms": 72,', '"retained_phantoms": 79,\n            "evidence_prefix_max": 72,\n            "intervention": "A+C",\n            "bootstrap_reference": str(reference),\n            "bootstrap_phantom_seeding": True,')
s=s.replace('log_L_phantom=results.log_L_phantom,', 'log_L_phantom=results.log_L_phantom[:, :72],')
s=s.replace('reference = Path("/largedata/albert/jaxns-ss8-css8-20260908/R240")', '''# A+C and A share discovery with all phantom seeds. C and R instead
    # share discovery with classic seeds. The unchanged Gaussian cases use
    # the earlier A cohort; revised mixtures use the current A cohort.
    reference = (
        Path("/largedata/albert/jaxns-a181-A-20260907")
        if args.case in ("basic_mvn", "weak_curved_mvn8") else
        Path("/largedata/albert/jaxns-ss8-css8-20260908/A240")
    )''')
s=s.replace('"C did not meet the goal within its limits."','"A+C did not meet the goal within its limits."')
(root/'benchmarks/paper_reproduction/run.py').write_text(s)
(root/'benchmarks/paper_reproduction/AC_PROTOCOL.md').write_text('''# A+C240 protocol

Branch lineage: measured A240 -> A+C240. Production src is byte-for-byte A; only the experiment allocation policy changes. All four current problems (G8, CG8, SS8 scale0.43/rho0.031, component-centered curved CSS8 beta0.4), seeds0--29, root degree240, width80,80 isotropic no-step-out slice transitions, CPU float64.

All79 phantom U states remain eligible seeds throughout discovery and continuation, with no bounded phantom reservoir. The first complete goal iteration uses uniform delta_K=1, exactly as A. Each run checks sample count and expected evidence uncertainty against the corresponding saved A first goal. The complete state and random key are then resumed using C's evidence-improving direct gap ceil(240*unit_peak_utility). Stop at classic expected sigma(log Z)<0.05; depth log1p(.001). Fixed screens300M likelihood evaluations or300 goals.

Evidence analysis uses the same leading prefixes0,8,...72,2048 paired Monte Carlo draws, and C_min20. Posterior mode weights remain classic-only and independent of phantom prefix. The comparison includes existing R/A/C/Cprime measurements without rerunning or changing them. The paper tables remain R240.

Output: /largedata/albert/jaxns-AC240-20260908/AC240/{case}/seed-00..29. At most60 concurrent workers, each pinned to one distinct CPU. Frozen source manifests and per-cell raw State pickles are retained.
''')
readme=root/'benchmarks/paper_reproduction/README.md'
readme.write_text('# A+C240: phantom seeds and evidence allocation\n\nSee [the combined protocol](AC_PROTOCOL.md). The inherited R/A/C/Cprime comparison is retained; the new A+C results will be appended after the cohort completes. All paper tables remain R240.\n')
print('A+C harness installed; production src unchanged.')
