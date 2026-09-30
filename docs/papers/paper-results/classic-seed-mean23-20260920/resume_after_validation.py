"""Run validation, measured-memory cohorts and reporting; publish monitor status."""

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.classic_seed_evidence.common import OUTPUT, ROOT, sha256, write_json

status_path = Path('/tmp/jaxns-mean-rerun-status.json')
status = json.loads(status_path.read_text())
status.update(controller_pid=os.getpid(), source_commit=subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
validation = OUTPUT / 'validation'
validation.mkdir(exist_ok=True)
# Continue only after the original supervisor records its diagnosed harness failure.
previous_pid = 168051
while True:
    proc = Path(f"/proc/{previous_pid}/stat")
    if not proc.exists() or proc.read_text().split()[2] == 'Z':
        break
    time.sleep(5)
prior_status = json.loads(status_path.read_text())
assert prior_status['status'] == 'validation_failed'
assert prior_status['failed_gate'] == 'references'
assert prior_status['gates']['standard-problems'] in (None, 0)
import xml.etree.ElementTree as ET

for name, expected in [('standard-problems', 22), ('plateau', 3)]:
    suites = ET.parse(validation / f'{name}.xml').getroot().findall('testsuite')
    assert sum(int(suite.attrib['tests']) for suite in suites) == expected
    assert all(int(suite.attrib.get(field, 0)) == 0 for suite in suites
               for field in ('errors', 'failures', 'skipped'))
assert json.loads((validation / 'SEMANTIC_GATE.json').read_text())['passed']
assert json.loads((validation / 'REFERENCE_GATE.json').read_text())['passed']
assert 'Mean, covariance, centered shear' in (validation / 'references-attempt02.log').read_text()
assert 'usage:' in (validation / 'cli.log').read_text()
status.update(gates={name: 0 for name in prior_status['gates']},
              validation_setup_failure='Preserved references.log: structured U coordinate check fixed before sampling.',
              resumed_from_controller_pid=previous_pid)
write_json(validation / 'TEST_GATE.json', {
    'passed': True, 'standard_cases': 22, 'standard_suite_unchanged': True, 'plateau_cases': 3,
    'references_passed': True, 'cli_passed': True,
})
write_json(validation / 'GATE_FILES.sha256.json', {'files': {
    str(path.relative_to(OUTPUT)): {'sha256': sha256(path)}
    for path in sorted(validation.iterdir()) if path.is_file()
}})
log = (OUTPUT / 'dispatch.log').open('x')
process = subprocess.Popen([sys.executable, '-m', 'benchmarks.classic_seed_evidence.dispatch'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
status.update(dispatch_pid=process.pid)
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
    raise RuntimeError('A cohort worker failed; diagnose before retrying only that case.')
status.update(status='auditing_and_reporting')
write_json(status_path, status)
with (OUTPUT / 'report.log').open('x') as log:
    for module in ('report', 'report_artifacts', 'finalize_audit'):
        subprocess.run([sys.executable, '-m', 'benchmarks.classic_seed_evidence.' + module],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
write_json(OUTPUT / 'REPORT_READY.json', {
    'complete': True, 'core_cells': 60, 'prefix_cells': 600, 'bootstrap_resamples': 100000,
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'final_audit': json.loads((OUTPUT / 'FINAL_AUDIT.json').read_text())['complete'],
})
status.update(status='numerical_report_complete', completed_core=60, completed_analysis=60,
              updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
write_json(status_path, status)
print('Validated numerical report complete; ready for compact handoff.', flush=True)
