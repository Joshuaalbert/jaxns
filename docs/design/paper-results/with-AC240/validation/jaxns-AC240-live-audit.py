"""Check actual live worker affinities against the dispatcher CPU assignments."""
import json
import os
from pathlib import Path
import time

out=Path('/largedata/albert/jaxns-AC240-20260908')
active={}
for line in (out/'dispatch.log').read_text().splitlines():
    if not line.startswith('{'):
        continue
    row=json.loads(line)
    if row['event']=='start':
        active[row['pid']]=row
    elif row['event']=='finish':
        active.pop(row['pid'])
records=[]
for pid,row in active.items():
    try:
        affinity=sorted(os.sched_getaffinity(pid))
        assert affinity==[row['cpu']],(pid,affinity,row)
        text=Path(f'/proc/{pid}/status').read_text()
        rss=next((line.split()[1] for line in text.splitlines() if line.startswith('VmRSS:')),None)
        records.append(dict(pid=pid,job=row['job'],affinity=affinity,rss_kib=None if rss is None else int(rss)))
    except ProcessLookupError:
        continue
    except FileNotFoundError:
        continue
assert len(records)<=60
assert len({r['affinity'][0] for r in records})==len(records)
record=dict(timestamp=time.time(),workers=records)
with (out/'validation/live-affinity.jsonl').open('a') as stream:
    stream.write(json.dumps(record)+'\n')
print('Live workers checked:',len(records),'distinct single-core affinities.')
