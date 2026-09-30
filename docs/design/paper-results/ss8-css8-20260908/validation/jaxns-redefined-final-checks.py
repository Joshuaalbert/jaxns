"""Check post-run launcher/document edits and curate an integrity audit."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

base=Path('/largedata/albert/git')
out=Path('/largedata/albert/jaxns-ss8-css8-20260908')
variants=('R240','A240','C240','Cprime240')

def check(variant):
    root=base/f'jaxns-paper-ss8-css8-{variant}'
    learn=root/'LEARNINGS.md'
    s=learn.read_text().replace('multiplier is0.43','multiplier is 0.43').replace('correlation0.031','correlation 0.031').replace('beta0.4','beta 0.4')
    learn.write_text(s)
    checks=[]
    for command in (['ruff','check','benchmarks/paper_reproduction/launch.py'],['flake8','benchmarks/paper_reproduction/launch.py']):
        r=subprocess.run(['conda','run','--no-capture-output','-n','jaxns_py',*command],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        checks.append(dict(command=command,exit_code=r.returncode,output=r.stdout))
    assert all(c['exit_code']==0 for c in checks), checks
    return variant,checks
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    checks=dict(pool.map(check,variants))
# Verify every installed report against the recorded raw-byte hash.
report=out/'report'
hashes=json.loads((report/'ARTIFACT_SHA256.json').read_text())
destinations=[base/f'jaxns-paper-ss8-css8-{v}'/'benchmarks/paper_reproduction/reports/ss8-css8-20260908' for v in variants]
destinations.append(base/'jaxns/docs/design/paper-results/ss8-css8-20260908')
for dest in destinations:
    for path,digest in hashes.items():
        assert hashlib.sha256((dest/path).read_bytes()).hexdigest()==digest
    paper=base/'jaxns/docs/design/paper.tex' if 'docs/design' in str(dest) else dest.parents[3]/'docs/papers/paper.tex'
    assert paper.read_bytes()==(report/'paper.tex').read_bytes()
record=dict(launcher_checks=checks,identical_reports=len(destinations),verified_files_per_report=len(hashes),new_cells=240,all_five_installed_manuscripts_identical=True)
(out/'FINAL_VALIDATION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
