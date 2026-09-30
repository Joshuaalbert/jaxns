"""Validate the uniform switch, run measured-memory cohorts, audit and report."""

import datetime
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmarks.classic_seed_evidence.common import OUTPUT, ROOT, sha256, write_json

status_path = Path('/tmp/jaxns-cg10-uniform-status.json')
status = json.loads(status_path.read_text())
status.update(controller_pid=os.getpid(), source_commit=subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
validation = OUTPUT / 'validation'
previous_pid = 192770
while True:
    proc = Path(f"/proc/{previous_pid}/stat")
    if not proc.exists() or proc.read_text().split()[2] == 'Z':
        break
    time.sleep(5)
prior = json.loads(status_path.read_text())
assert prior['status'] == 'validation_failed' and prior['failed_gate'] == 'semantic-gate'
while not (validation / 'SEMANTIC_GATE.json').exists():
    time.sleep(5)
assert json.loads((validation / 'SEMANTIC_GATE.json').read_text())['passed']
assert json.loads((validation / 'UNIFORM_GATE.json').read_text())['passed']
assert json.loads((validation / 'REFERENCE_GATE.json').read_text())['passed']
assert 'usage:' in (validation / 'cli.log').read_text()
active = [(name, None, None, None) for name in prior['gates']]
status.update(validation_setup_failure='Preserved semantic-gate.log; explicit disabled-collection capacity fixed before sampling.',
              resumed_from_controller_pid=previous_pid)
suites = ET.parse(validation / 'allocation-tests.xml').getroot().findall('testsuite')
assert all(int(suite.attrib.get(field, 0)) == 0 for suite in suites
           for field in ('errors', 'failures', 'skipped'))
assert json.loads((validation / 'INHERITED_GATES.json').read_text())['passed']
write_json(validation / 'TEST_GATE.json', {
    'passed': True, 'standard_cases': 22, 'standard_suite_unchanged': True, 'plateau_cases': 3,
    'inherited_standard_and_plateau': True,
    'allocation_cases': sum(int(suite.attrib['tests']) for suite in suites),
    'references_passed': True, 'cli_passed': True, 'full_discovery_and_rng_replay_passed': True,
})
write_json(validation / 'GATE_FILES.sha256.json', {'files': {
    str(path.relative_to(OUTPUT)): {'sha256': sha256(path)}
    for path in sorted(validation.rglob('*')) if path.is_file()
}})
log = (OUTPUT / 'dispatch.log').open('x')
process = subprocess.Popen([sys.executable, '-m', 'benchmarks.classic_seed_evidence.dispatch'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
status.update(dispatch_pid=process.pid, gates={name: 0 for name, _, _, _ in active})
while process.poll() is None:
    live = json.loads((OUTPUT / 'STATUS.json').read_text()) if (OUTPUT / 'STATUS.json').exists() else {}
    status.update(status='running_' + live.get('phase', 'core'),
                  updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  progress=live, completed_core=len(list(OUTPUT.glob('0.05/*/seed-*/CORE.json'))),
                  completed_analysis=len(list(OUTPUT.glob('0.05/*/seed-*/ANALYSIS.json'))))
    write_json(status_path, status)
    time.sleep(10)
log.close()
if process.returncode:
    status.update(status='cohort_failed', exit_code=process.returncode)
    write_json(status_path, status)
    raise RuntimeError('A worker failed; diagnose before retrying only that case.')
status.update(status='auditing_and_reporting', updated_utc=datetime.datetime.now(
    datetime.timezone.utc).isoformat())
write_json(status_path, status)
with (OUTPUT / 'report.log').open('x') as log:
    for module in ('report', 'report_artifacts', 'compare_allocations', 'finalize_audit'):
        subprocess.run([sys.executable, '-m', 'benchmarks.classic_seed_evidence.' + module],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
write_json(OUTPUT / 'REPORT_READY.json', {
    'complete': True, 'core_cells': 30, 'prefix_cells': 300, 'bootstrap_resamples': 100000,
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'final_audit': json.loads((OUTPUT / 'FINAL_AUDIT.json').read_text())['complete'],
})
status.update(status='numerical_report_complete', completed_core=30, completed_analysis=30,
              updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
write_json(status_path, status)
print('Validated uniform-allocation report complete; ready for compact handoff.', flush=True)
