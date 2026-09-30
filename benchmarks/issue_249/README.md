# Bounded final-MC benchmark

`run_bounded_mc.py` uses the issue-247 scientific configuration and measures
the final phantom-conditioned 1,000-draw calculation after the nested-sampling
state and result are ready. The first timing includes lowering and compilation;
later timings reuse the executable. RSS immediately before final MC and the
process peak are reported separately.

Run one batch-size point with this worktree first on `PYTHONPATH`:

```bash
conda run -n jaxns_py env \
  PYTHONPATH=/tmp/jaxns-issue-249/src \
  MPLCONFIGDIR=/tmp/matplotlib-issue249 \
  python benchmarks/issue_249/run_bounded_mc.py \
  --case basic_mvn --batch-size 128
```

The maintained comparison covers `basic_mvn`, `spike_slab`, and
`spike_slab10`, with 1,000 draws and the same random key for every batch size.
Add `--inspect-program` to report compiler argument, output, alias, and
temporary-memory estimates together with small structural HLO counts. The
compiler report is secondary evidence: measured wall time and process peak RSS
remain the acceptance measurements.

The accepted measurements and interpretation are recorded in
[`REPORT.md`](REPORT.md).
