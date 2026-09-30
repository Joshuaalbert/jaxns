"""Run the frozen SS8/CSS8 redefinition suite on sixty distinct CPU cores."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path('/largedata/albert/jaxns-ss8-css8-20260908')
VARIANTS = ('R240', 'A240', 'C240', 'Cprime240')
CASES = ('spike_slab', 'curved_spike_slab8')
ROOTS = {v: Path('/largedata/albert/git') / f'jaxns-paper-ss8-css8-{v}' for v in VARIANTS}
CPUS = sorted(os.sched_getaffinity(0))[:60]
assert len(CPUS) == 60
sources = {}
for variant, root in ROOTS.items():
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root)
    sources[variant] = {
        'root': str(root),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'src_tree': subprocess.check_output(['git', 'rev-parse', 'HEAD:src'], cwd=root, text=True).strip(),
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'benchmarks/paper_reproduction').glob('*.py')},
    }
with (OUT / 'SOURCES.json').open('x') as stream:
    json.dump(sources, stream, indent=2)
pending = [(v, case, seed, phase) for v in VARIANTS for phase in ('core', 'analysis') for seed in range(30) for case in CASES]
active = {}
failures = []
started = time.time()

def event(kind, **record):
    print(json.dumps(dict(event=kind, time=time.time(), **record)), flush=True)

while pending or active:
    for cpu, (process, log, job) in list(active.items()):
        code = process.poll()
        if code is None:
            continue
        log.close()
        variant, case, seed, phase = job
        cell = OUT / variant / case / f'seed-{seed:02d}'
        success = code == 0 and (cell / f'{phase.upper()}.json').exists()
        event('finish', cpu=cpu, pid=process.pid, job=job, exit_code=code, success=success)
        del active[cpu]
        if not success:
            failures.append(dict(job=job, exit_code=code))
    if failures:
        # Drain existing workers but do not dispatch dependent or additional work.
        pending.clear()
    occupied = {tuple(row[2][:3]) for row in active.values()}
    for cpu in CPUS:
        if cpu in active or failures:
            continue
        ready = None
        for index, job in enumerate(pending):
            variant, case, seed, phase = job
            if tuple(job[:3]) in occupied:
                continue
            if phase == 'analysis' and not (OUT / variant / case / f'seed-{seed:02d}' / 'CORE.json').exists():
                continue
            if phase == 'core' and variant in ('C240', 'Cprime240') and not (OUT / 'R240' / case / f'seed-{seed:02d}' / 'CORE.json').exists():
                continue
            ready = index
            break
        if ready is None:
            continue
        job = pending.pop(ready)
        variant, case, seed, phase = job
        root = ROOTS[variant]
        cell = OUT / variant / case / f'seed-{seed:02d}'
        cell.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'JAX_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')})
        env.update(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true', PYTHONUNBUFFERED='1', MPLBACKEND='Agg', PYTHONPATH=f'{root / "src"}:{root}', XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
        log = (cell / f'{phase}.log').open('x')
        process = subprocess.Popen(['taskset', '-c', str(cpu), sys.executable, '-m', 'benchmarks.paper_reproduction.run', '--case', case, '--seed', str(seed), '--phase', phase, '--output', str(cell)], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        active[cpu] = (process, log, job)
        occupied.add(tuple(job[:3]))
        event('start', cpu=cpu, pid=process.pid, job=job)
    (OUT / 'STATUS.json').write_text(json.dumps(dict(active=len(active), pending=len(pending), failures=failures, elapsed_seconds=time.time()-started, active_jobs=[row[2] for row in active.values()]), indent=2))
    if not active and pending:
        raise RuntimeError('Pending work has no satisfied dependency')
    if active:
        time.sleep(2)
if failures:
    (OUT / 'FAILED.json').write_text(json.dumps(failures, indent=2))
    raise RuntimeError(f'Worker failures: {failures}')
assert all((OUT / v / c / f'seed-{s:02d}' / name).exists() for v in VARIANTS for c in CASES for s in range(30) for name in ('CORE.json', 'ANALYSIS.json'))
(OUT / 'FINISHED.json').write_text(json.dumps(dict(complete=True, cells=240, max_parallel=60, cpus=CPUS, elapsed_seconds=time.time()-started), indent=2))
