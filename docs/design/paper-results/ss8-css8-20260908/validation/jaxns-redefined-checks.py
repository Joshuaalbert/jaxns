import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

root = Path('/largedata/albert/jaxns-ss8-css8-20260908')
(root / 'validation').mkdir(parents=True, exist_ok=True)
files = ['benchmarks/paper_reproduction/' + name for name in ['cases.py', 'references.py', 'run.py', 'test_protocol.py']]

def check(item):
    index, variant = item
    worktree = Path(f'/largedata/albert/git/jaxns-paper-ss8-css8-{variant}')
    env = os.environ.copy()
    env.update(PYTHONPATH=f'{worktree}/src:{worktree}', JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    records = []
    commands = [
        ['ruff', 'check', *files],
        ['flake8', *files],
        ['taskset', '-c', str(60+index), sys.executable, '-m', 'pytest', '-q', 'benchmarks/paper_reproduction/test_protocol.py'],
    ]
    for number, command in enumerate(commands):
        log = root / 'validation' / f'{variant}-{number}.log'
        with log.open('w') as handle:
            result = subprocess.run(command, cwd=worktree, env=env, stdout=handle, stderr=subprocess.STDOUT)
        records.append(dict(command=command, exit_code=result.returncode, log=str(log)))
    return variant, records

with ThreadPoolExecutor(max_workers=4) as pool:
    results = dict(pool.map(check, enumerate(['R240', 'A240', 'C240', 'Cprime240'])))
(root / 'validation/checks.json').write_text(json.dumps(results, indent=2) + '\n')
for variant, records in results.items():
    print(variant, [(r['command'][0], r['exit_code']) for r in records])
    for record in records:
        if record['exit_code']:
            print(Path(record['log']).read_text())
assert all(r['exit_code'] == 0 for records in results.values() for r in records)
