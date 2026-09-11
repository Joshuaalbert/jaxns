"""Preserve superseded mixture data outside the active result roots."""
import json
from pathlib import Path

base=Path('/largedata/albert')
new=base/'jaxns-ss8-css8-20260908'
archive=base/'jaxns-superseded-SS8-CSS8-pre-20260908'
archive.mkdir(exist_ok=True)
old=json.loads((base/'jaxns-C-family-comparison-20260907/240/summary.json').read_text())
(new/'report/G8_CG8_INPUTS.json').write_text(json.dumps({v:{c:r for c,r in rows.items() if c in ('basic_mvn','weak_curved_mvn8')} for v,rows in old.items()},indent=2)+'\n')
roots=['jaxns-a181-A-20260907','jaxns-a181-A-20260907-scalar-queries','jaxns-a181-A-perfcheck-20260907','jaxns-a181-B-20260907','jaxns-a181-Bprime-20260907','jaxns-a18124b-retest-20260907','jaxns-ablation-A960-20260907','jaxns-ablation-B240-20260907','jaxns-ablation-R960-20260907','jaxns-allocation-C240-20260907','jaxns-evidence-C240-20260907','jaxns-evidence-C960-20260907','jaxns-first-goal-allocation-20260907','jaxns-mixed-Cprime240-20260907','jaxns-mixed-Cprime960-20260907','jaxns-paper-pre289-reproduction-20260906']
plan=[]
for name in roots:
    for case in ('spike_slab','correlated_spike_slab8'):
        source=base/name/case
        if source.is_dir():
            plan.append((source,archive/name/case))
for name in ('jaxns-ss8-spike-C240-20260908','jaxns-ss8-resolved-C240-20260908','jaxns-ss8-correlation-C240-20260908'):
    source=base/name
    if source.is_dir(): plan.append((source,archive/name))
records=[]
for source,target in plan:
    assert not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True)
    inode=source.stat().st_ino
    source.rename(target)
    assert target.stat().st_ino == inode and not source.exists()
    records.append(dict(old=str(source),archive=str(target),directory_inode=inode))
for name in roots:
    p=base/name
    if p.exists():
        (p/'SS8_CSS8_SUPERSEDED.md').write_text('# Superseded mixture measurements\n\nOld SS8 and correlated CSS8 cells have moved to '+str(archive/name)+'. G8 and CG8 data remain here. Current SS8 and curved CSS8 measurements are in '+str(new)+'. Mixed summaries in this historical root must not supply current SS8/CSS8 values.\n')
(archive/'README.md').write_text('# Superseded SS8 and CSS8 results\n\nThese data use retired problem definitions, including the exploratory scale/correlation bisections. They are retained for provenance only and must not be merged with current results.\n\nCurrent results: '+str(new)+'\n')
(new/'report/RETIRED_PATHS.json').write_text(json.dumps(records,indent=2)+'\n')
print('Preserved and retired',len(records),'old mixture directories; G8/CG8 cells unchanged.')
