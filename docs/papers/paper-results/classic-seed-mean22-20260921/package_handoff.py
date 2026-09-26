"""Package audited small results for the workstation; leave full states in place."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

root = Path('/largedata/albert/jaxns-classic-seed-mean22-20260921')
assert json.loads((root / 'FINAL_AUDIT.json').read_text())['complete']
assert json.loads((root / 'REPORT_READY.json').read_text())['complete']
target = root / 'handoff'
target.mkdir(exist_ok=False)
names = [
    'PROTOCOL.json', 'PROTOCOL.sha256.json', 'MANIFEST.json', 'MANIFEST.sha256.json',
    'SOURCE.bundle', 'SOURCE.bundle.sha256.json', 'SOURCE.diff', 'MODEL.diff', 'SOURCE_DIFFS.sha256.json', 'SOURCE.tar.gz.sha256.json',
    'PREVIOUS_MEAN_PROVENANCE.json', 'MEAN_COMPARISON.json', 'MEAN_COMPARISON.csv',
    'MEAN_COMPARISON.md', 'MEAN_COMPARISON.sha256.json',
    'jaxns-mean22-records-20260921.tar.gz.sha256.json',
    'REFERENCES.json', 'REPRODUCTION.md', 'SUMMARY.json', 'CONTRASTS.json',
    'ALL_PREFIX_METRICS.csv', 'PER_SEED_DIAGNOSTICS.csv', 'TABLES.tex', 'COMPARISON.md',
    'PREFIX_COMPARISON.pdf', 'PREFIX_COMPARISON.png', 'REPORT_ARTIFACTS.sha256.json',
    'AUDIT.json', 'FINAL_AUDIT.json', 'REPORT_READY.json', 'MEMORY-core.json',
    'MEMORY-analysis.json', 'FINISHED-core.json', 'FINISHED-analysis.json', 'FINISHED.json',
    'LAUNCH.json', 'launch_mean22.py', 'TASK.txt', 'package_records.py',
    'freeze.log', 'supervisor.log', 'dispatch.log', 'report.log',
    'host-lscpu.txt', 'host-meminfo.txt', 'package_handoff.py',
]
for name in names:
    shutil.copy2(root / name, target / name)
for directory in ['validation']:
    shutil.copytree(root / directory, target / directory)
for case in ('g10', 'cg10'):
    for seed in range(30):
        cell = root / '0.05' / case / f'seed-{seed:02d}'
        dest = target / cell.relative_to(root)
        dest.mkdir(parents=True)
        for path in cell.iterdir():
            if path.suffix in ('.json', '.jsonl', '.log'):
                shutil.copy2(path, dest / path.name)
records = {}
for path in sorted(target.rglob('*')):
    if path.is_file():
        records[str(path.relative_to(target))] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
(target / 'COMPACT_SHA256.json').write_text(json.dumps(records, indent=2) + '\n')
archive = root / 'jaxns-mean22-results-20260921.tar.gz'
with tarfile.open(archive, 'w:gz') as stream:
    stream.add(target, arcname='jaxns-mean22-results-20260921')
archive.with_name(archive.name + '.sha256.json').write_text(json.dumps({
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size,
}, indent=2) + '\n')
print(json.dumps({'compact_archive': str(archive), 'files': len(records),
                      'bytes': archive.stat().st_size}))
