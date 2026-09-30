# Demos

These deterministic, network-independent programs demonstrate supported
JAXNS workflows. Each demo is complete, uses public scientific objects, and
contains assertions so API or behavior drift fails visibly in CI.

Current demos:

- `local_nested_sampling.py`: define a model, run locally with retained
  phantoms, inspect results, and draw both classic and phantom-conditioned
  evidence samples.
- `readme_quick_start.py`: run the exact two-parameter regression shown in the
  top-level README and verify its text and graphical result surfaces with the
  headless Matplotlib backend.

Regenerate the maintained README summary and plot assets with:

```bash
conda run -n jaxns_py python \
  cicd/demos/readme_quick_start.py --write-assets
```
