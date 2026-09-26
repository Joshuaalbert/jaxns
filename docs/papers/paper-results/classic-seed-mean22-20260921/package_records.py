"""Export the 60 completed cells for independent workstation verification."""

import hashlib
import json
import tarfile
from pathlib import Path

root = Path('/largedata/albert/jaxns-classic-seed-mean22-20260921')
assert json.loads((root / 'FINAL_AUDIT.json').read_text())['complete']
assert json.loads((root / 'REPORT_READY.json').read_text())['complete']
archive = root / 'jaxns-mean22-records-20260921.tar.gz'
assert not archive.exists()
checksums = {}
with tarfile.open(archive, 'w:gz') as stream:
    for case in ('g10', 'cg10'):
        for seed in range(30):
            cell = root / '0.05' / case / f'seed-{seed:02d}'
            analysis = json.loads((cell / 'ANALYSIS.json').read_text())
            assert analysis['complete']
            for name in ('CORE.json', 'ANALYSIS.json', 'progress.jsonl',
                         'MANIFEST.json', 'evidence_draws.npz'):
                path = cell / name
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if name == 'evidence_draws.npz':
                    assert digest == analysis['evidence_draws_sha256']
                relative = str(path.relative_to(root))
                checksums[relative] = {'sha256': digest, 'bytes': path.stat().st_size}
                stream.add(path, arcname=relative)
assert len(checksums) == 300
record = {'archive': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'bytes': archive.stat().st_size, 'cells': 60, 'files': checksums}
(root / (archive.name + '.sha256.json')).write_text(json.dumps(record, indent=2) + '\n')
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    assert len(members) == 300 and all(member.isfile() for member in members)
    for member in members:
        assert hashlib.sha256(stream.extractfile(member).read()).hexdigest() == checksums[member.name]['sha256']
print(json.dumps({key: value for key, value in record.items() if key != 'files'}))
