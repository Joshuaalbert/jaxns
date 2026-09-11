"""Check local manuscript structure and report pre-existing build gaps."""

from collections import Counter
import json
from pathlib import Path
import re
import shutil

paper = Path('/largedata/albert/git/jaxns/docs/design/paper.tex')
text = re.sub(r'(?<!\\)%[^\n]*', '', paper.read_text())
labels = re.findall(r'\\label\{([^}]+)\}', text)
refs = re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}', text)
assert not [key for key, count in Counter(labels).items() if count > 1]
assert not set(refs) - set(labels), set(refs) - set(labels)
stack = []
for match in re.finditer(r'\\(begin|end)\{([^}]+)\}', text):
    if match[1] == 'begin':
        stack.append(match[2])
    else:
        assert stack.pop() == match[2], match[0]
assert not stack
brace = 0
for match in re.finditer(r'(?<!\\)[{}]', text):
    brace += 1 if match[0] == '{' else -1
    assert brace >= 0
assert brace == 0
assert not any(word in text for word in ('G8', 'CG8', 'SS8', 'CSS8', 'R240'))
figs = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text)
missing = [filename for filename in figs if not (paper.parent / filename).exists()]
assert missing == ['images/censor.drawio.pdf'], missing
out = dict(
    labels=len(labels), references=len(refs), figures=len(figs),
    missing_preexisting_figures=missing, braces_balanced=True,
    environments_balanced=True, obsolete_active_cases=False,
    tex_compiler=shutil.which('pdflatex'),
)
report = paper.parent / 'paper-results/evidence10-AC300-20260909'
(report / 'PAPER_VALIDATION.json').write_text(json.dumps(out, indent=2) + '\n')
print(out)
