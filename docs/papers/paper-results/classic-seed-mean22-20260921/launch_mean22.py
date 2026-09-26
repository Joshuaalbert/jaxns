import datetime
import json
import os
import pathlib
import subprocess
import sys

root=pathlib.Path('/largedata/albert/git/jaxns-classic-seed-mean22-20260921')
out=pathlib.Path('/largedata/albert/jaxns-classic-seed-mean22-20260921')
out.mkdir(exist_ok=False)
env=dict(os.environ,PYTHONPATH=f'{root}/src:{root}',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
freeze=['taskset','-c','63',sys.executable,'-m','benchmarks.classic_seed_evidence.freeze']
with (out/'freeze.log').open('x') as log:
    subprocess.run(freeze,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
manifest=json.loads((out/'MANIFEST.json').read_text())
status=pathlib.Path('/tmp/jaxns-mean22-rerun-status.json')
s=json.loads(status.read_text());s.update(source_commit=manifest['source_commit'],protocol_sha256=manifest['protocol_sha256'],status='frozen_pre_run_validation',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
status.write_text(json.dumps(s,indent=2)+'\n')
cmd=['taskset','-c','62',sys.executable,'-m','benchmarks.classic_seed_evidence.supervise']
with (out/'supervisor.log').open('x') as log:
    p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch={'command': cmd,'freeze_command': freeze,'pid': p.pid,'environment': {k:env[k] for k in manifest['environment'] if k in env},'source_commit': manifest['source_commit'],'protocol_sha256': manifest['protocol_sha256'],'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),'launch_location': 'dorrie screen 58428.jaxns'}
(out/'LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(launch,indent=2))
