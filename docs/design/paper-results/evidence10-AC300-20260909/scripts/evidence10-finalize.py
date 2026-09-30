"""Install the final audited ladder if the baseline paper has not been edited."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/largedata/albert/jaxns-evidence10-AC300-20260909')
MAIN = Path('/largedata/albert/git/jaxns')
SCRIPTS = ROOT / 'report/scripts'
DEST = MAIN / 'docs/design/paper-results/evidence10-AC300-20260909'
assert len(os.sched_getaffinity(0)) == 1
watched = [
    MAIN / 'docs/design/paper.tex',
    MAIN / 'docs/design/paper-results/README.md',
    DEST / 'README.md',
    *[MAIN / f'docs/design/images/{name}.pdf' for name in (
        'evidence_problems', 'evidence_accuracy', 'ss10_recovery_ladder',
    )],
]
expected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in watched}
(ROOT / 'FINALIZE_GUARD.json').write_text(json.dumps(expected, indent=2) + '\n')
while not (ROOT / 'REPORT_FINISHED.json').exists():
    if (ROOT / 'FAILED.json').exists():
        raise RuntimeError('Sampling failed; preserve the published baseline and failure record')
    time.sleep(30)
changed = [
    name for name, digest in expected.items()
    if not Path(name).exists() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
]
if changed:
    (ROOT / 'PUBLISH_READY.json').write_text(json.dumps(dict(
        complete_report=str(ROOT / 'report'), edited_files=changed,
        reason='Preserve edits made while the uncertainty ladder was running',
    ), indent=2) + '\n')
    print('Final report ready; paper edits preserved:', changed, flush=True)
    sys.exit(0)
subprocess.run([sys.executable, str(SCRIPTS / 'evidence10-install.py')], check=True)
subprocess.run([sys.executable, str(SCRIPTS / 'evidence10-validate-paper.py')], check=True)
summary = json.loads((DEST / 'SUMMARY.json').read_text())
stages = list(summary)
last = stages[-1]
posterior = summary[last]['ss10']['posterior']
status = 'met' if posterior['recovered'] else 'did not meet'
readme = DEST / 'README.md'
text = readme.read_text()
start = text.index('**Baseline complete:')
end = text.index('\nAt the baseline,', start)
text = text[:start] + (
    f'**Complete: {90 + 30 * (len(stages) - 1)} trees and paired prefix analyses.** '
    f'The completed matched SS10 stops are {", ".join(stages)}. At {last}, '
    f'the classic spike-mass RMSE is {posterior["rmse"]:.5f}; the cohort {status} '
    'the stated recovery criterion. Recovery means mass RMSE ≤0.05 and maximum '
    'absolute mass error ≤0.15. Phantom benefit is assessed separately in the '
    '[comparison](COMPARISON.md), including the paired bootstrap interval.\n'
) + text[end:]
readme.write_text(text)
index = MAIN / 'docs/design/paper-results/README.md'
text = index.read_text().replace(
    'Matched SS10 continuations to 0.02, then 0.01 and 0.005 if needed, are in progress.',
    'The matched SS10 uncertainty ladder is complete at stops ' + ', '.join(stages) + '.',
)
index.write_text(text)
record = dict(
    complete=True, paper=str(MAIN / 'docs/design/paper.tex'),
    report=str(DEST), stages=stages, recovered=posterior['recovered'],
)
(ROOT / 'PUBLISHED.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
