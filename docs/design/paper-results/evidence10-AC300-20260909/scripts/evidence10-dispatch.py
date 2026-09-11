"""Run frozen 10D paper cohorts and the matched SS10 uncertainty ladder."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/largedata/albert/git/jaxns-paper-evidence10-AC300')
OUT = Path('/largedata/albert/jaxns-evidence10-AC300-20260909')
CPUS = sorted(os.sched_getaffinity(0))[:60]
RESERVATIONS = {'0.05': 8, '0.02': 24, '0.01': 64, '0.005': 192}
BUDGET_GIB = 480
TRUTH_MASS = 0.5000077444258325
assert len(CPUS) == 60
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
OUT.mkdir(parents=True, exist_ok=True)
source = {
    'root': str(ROOT),
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                      text=True).strip(),
    'src_tree': subprocess.check_output(['git', 'rev-parse', 'HEAD:src'], cwd=ROOT,
                                        text=True).strip(),
    'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / 'benchmarks/paper_reproduction').glob('*.py')},
}
with (OUT / 'SOURCES.json').open('x') as stream:
    json.dump(source, stream, indent=2)
started = time.time()
max_active = 0
finished_cells = 0
decisions = []


def event(kind, **record):
    print(json.dumps(dict(event=kind, time=time.time(), **record)), flush=True)


for stage in ('0.05', '0.02', '0.01', '0.005'):
    previous = {'0.02': '0.05', '0.01': '0.02', '0.005': '0.01'}.get(stage)
    cases = ('ss10', 'g10', 'cg10') if stage == '0.05' else ('ss10',)
    pending = [(case, seed, phase) for phase in ('core', 'analysis')
               for seed in range(30) for case in cases]
    active = {}
    failures = []
    last_audit = 0.
    while pending or active:
        for cpu, (process, log, job) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            case, seed, phase = job
            cell = OUT / stage / case / f'seed-{seed:02d}'
            success = code == 0 and (cell / f'{phase.upper()}.json').exists()
            event('finish', stage=stage, cpu=cpu, pid=process.pid, job=job,
                  exit_code=code, success=success)
            del active[cpu]
            if not success:
                failures.append(dict(stage=stage, job=job, exit_code=code))
        if failures:
            pending.clear()
        occupied = {tuple(row[2][:2]) for row in active.values()}
        for cpu in CPUS:
            if cpu in active or failures:
                continue
            if (len(active) + 1) * RESERVATIONS[stage] > BUDGET_GIB:
                break
            ready = None
            for index, job in enumerate(pending):
                case, seed, phase = job
                if tuple(job[:2]) in occupied:
                    continue
                cell = OUT / stage / case / f'seed-{seed:02d}'
                if phase == 'analysis' and not (cell / 'CORE.json').exists():
                    continue
                ready = index
                break
            if ready is None:
                continue
            if shutil.disk_usage(OUT).free < 50 * 1024**3:
                event('low_disk', free_bytes=shutil.disk_usage(OUT).free)
                break
            job = pending.pop(ready)
            case, seed, phase = job
            cell = OUT / stage / case / f'seed-{seed:02d}'
            cell.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({name: '1' for name in (
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'JAX_NUM_THREADS',
                'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')})
            env.update(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true',
                       PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
                       PYTHONPATH=f'{ROOT / "src"}:{ROOT}',
                       XLA_FLAGS='--xla_cpu_multi_thread_eigen=false '
                       'intra_op_parallelism_threads=1')
            cmd = ['taskset', '-c', str(cpu), sys.executable, '-m',
                   'benchmarks.paper_reproduction.run_evidence10',
                   '--case', case, '--seed', str(seed), '--phase', phase,
                   '--goal-log-z-uncert', stage, '--output', str(cell)]
            if previous and phase == 'core':
                cmd.extend(['--resume-from', str(OUT / previous / case /
                                                 f'seed-{seed:02d}')])
            log = (cell / f'{phase}.log').open('x')
            process = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            os.sched_setaffinity(process.pid, {cpu})
            active[cpu] = (process, log, job)
            occupied.add(tuple(job[:2]))
            event('start', stage=stage, cpu=cpu, pid=process.pid, job=job)
        max_active = max(max_active, len(active))
        if time.time() - last_audit > 30:
            affinity = []
            for cpu, (process, _, job) in active.items():
                try:
                    actual = sorted(os.sched_getaffinity(process.pid))
                except ProcessLookupError:
                    continue
                if actual != [cpu]:
                    raise RuntimeError(f'Affinity changed: {process.pid}: {actual}')
                affinity.append(dict(pid=process.pid, cpu=cpu, job=job))
            event('affinity_audit', stage=stage, workers=affinity)
            last_audit = time.time()
        (OUT / 'STATUS.json').write_text(json.dumps(dict(
            stage=stage, active=len(active), pending=len(pending), failures=failures,
            elapsed_seconds=time.time()-started, max_active=max_active,
            reserved_gib=len(active)*RESERVATIONS[stage],
            active_jobs=[row[2] for row in active.values()]), indent=2))
        if not active and pending:
            raise RuntimeError('Pending work cannot be dispatched; inspect disk/dependencies')
        if active:
            time.sleep(2)
    if failures:
        (OUT / 'FAILED.json').write_text(json.dumps(failures, indent=2))
        raise RuntimeError(f'Worker failures: {failures}')
    for case in cases:
        for seed in range(30):
            cell = OUT / stage / case / f'seed-{seed:02d}'
            assert all((cell / name).exists() for name in ('CORE.json', 'ANALYSIS.json'))
    finished_cells += 30 * len(cases)
    masses = [json.loads((OUT / stage / 'ss10' / f'seed-{seed:02d}' /
                          'ANALYSIS.json').read_text())['classic_spike_mass']
              for seed in range(30)]
    errors = [mass - TRUTH_MASS for mass in masses]
    rmse = (sum(error**2 for error in errors) / 30)**.5
    max_error = max(abs(error) for error in errors)
    recovered = rmse <= .05 and max_error <= .15
    decision = dict(stage=stage, masses=masses, mass_rmse=rmse,
                    max_mass_error=max_error, recovered=recovered,
                    continue_ladder=stage == '0.05' or (not recovered and stage != '0.005'))
    decisions.append(decision)
    (OUT / 'DECISIONS.json').write_text(json.dumps(decisions, indent=2))
    event('stage_complete', **decision)
    if not decision['continue_ladder']:
        break
(OUT / 'FINISHED.json').write_text(json.dumps(dict(
    complete=True, cells=finished_cells, stages=[d['stage'] for d in decisions],
    max_parallel=max_active, cpus=CPUS, elapsed_seconds=time.time()-started,
    ss10_recovered=decisions[-1]['recovered']), indent=2))
event('finished', cells=finished_cells)
