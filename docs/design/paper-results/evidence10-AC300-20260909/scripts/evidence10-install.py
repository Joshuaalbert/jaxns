"""Install the latest audited complete cohorts into the paper checkout."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/largedata/albert/jaxns-evidence10-AC300-20260909')
MAIN = Path('/largedata/albert/git/jaxns')
REPORT = ROOT / 'report'
DEST = MAIN / 'docs/design/paper-results/evidence10-AC300-20260909'
SAMPLING = Path('/largedata/albert/git/jaxns-paper-evidence10-AC300')
ANALYSIS = Path('/largedata/albert/git/jaxns-paper-evidence10-analysis')
DEST.mkdir(parents=True, exist_ok=True)
summary = json.loads((REPORT / 'SUMMARY.json').read_text())
audit = json.loads((REPORT / 'AUDIT.json').read_text())
assert all(case in summary['0.05'] for case in ('g10', 'cg10', 'ss10'))
records = 0
for stage, cases in summary.items():
    for case in cases:
        for seed in range(30):
            rel = Path(stage) / case / f'seed-{seed:02d}'
            target = DEST / 'records' / rel
            target.mkdir(parents=True, exist_ok=True)
            for name in (
                'MANIFEST.json', 'CORE.json', 'ANALYSIS.json', 'evidence_draws.npz',
                'progress.jsonl',
            ):
                source = ROOT / rel / name
                expected = audit['record_sha256'][str(rel / name)]
                assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
                shutil.copy2(source, target / name)
            records += 1
assert records == audit['cells']
for name in (
    'SUMMARY.json', 'AUDIT.json', 'COMPARISON.md', 'PROTOCOL.md', 'REFERENCES.json',
    'ALL_TABLES.tex',
):
    shutil.copy2(REPORT / name, DEST / name)
for source in REPORT.glob('MEMORY_*'):
    if source.is_file():
        shutil.copy2(source, DEST / source.name)
for source in REPORT.glob('*_table.tex'):
    shutil.copy2(source, DEST / source.name)
for name in ('SOURCES.json', 'ANALYSIS_SOURCES.json', 'RETRIES.json', 'DECISIONS.json'):
    shutil.copy2(ROOT / name, DEST / name)
for name in ('FINISHED.json', 'ARCHIVE_FINISHED.json', 'REPORT_FINISHED.json'):
    if (ROOT / name).exists():
        shutil.copy2(ROOT / name, DEST / name)
shutil.copytree(REPORT / 'scripts', DEST / 'scripts', dirs_exist_ok=True)
# Freeze both source amendments and their validation with this report.
for source, name in ((SAMPLING, 'sampling'), (ANALYSIS, 'analysis')):
    target = DEST / 'source' / name
    target.mkdir(parents=True, exist_ok=True)
    names = ('evidence10_cases.py', 'run_evidence10.py') if name == 'sampling' else (
        'prefix_sweep.py', 'run_evidence10.py',
    )
    for filename in names:
        shutil.copy2(source / 'benchmarks/paper_reproduction' / filename, target / filename)
    test = 'test_evidence10.py' if name == 'sampling' else 'test_evidence10_plateau.py'
    shutil.copy2(source / 'cicd/tests' / test, target / test)
    original_report = 'evidence10-AC300-20260909' if name == 'sampling' else (
        'evidence10-plateau-20260909'
    )
    shutil.copytree(
        source / 'benchmarks/paper_reproduction/reports' / original_report,
        target / 'validation', dirs_exist_ok=True,
    )
    patch = subprocess.check_output(['git', 'show', '--format=fuller', 'HEAD'], cwd=source)
    (target / 'commit.patch').write_bytes(patch)
for name in ('evidence_problems', 'evidence_accuracy', 'ss10_recovery_ladder'):
    shutil.copy2(REPORT / 'figures' / f'{name}.pdf', MAIN / f'docs/design/images/{name}.pdf')
shutil.copytree(REPORT / 'figures', DEST / 'figures', dirs_exist_ok=True)
shutil.copy2(REPORT / 'paper.tex', MAIN / 'docs/design/paper.tex')
complete = (ROOT / 'REPORT_FINISHED.json').exists()
(DEST / 'SNAPSHOT.json').write_text(json.dumps(dict(
    installed_utc=datetime.now(timezone.utc).isoformat(), cells=records,
    cohorts={stage: list(cases) for stage, cases in summary.items()},
    full_ladder_complete=complete, raw_root=str(ROOT),
    raw_archive='/ceph_dorrie/albert/jaxns-evidence10-AC300-20260909',
), indent=2) + '\n')
print(f'Installed {records} audited cells; full ladder complete: {complete}')
