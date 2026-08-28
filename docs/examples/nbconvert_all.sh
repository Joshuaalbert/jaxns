#!/bin/bash

# Exit on error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
echo "Script dir $SCRIPT_DIR"
cd "$SCRIPT_DIR"

# Documentation examples must render headlessly in CI and batch sessions.
# Override any inherited interactive backend before a notebook kernel starts.
export MPLBACKEND="Agg"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/jaxns-matplotlib}"
export IPYTHONDIR="${IPYTHONDIR:-/tmp/jaxns-ipython}"
mkdir -p "$MPLCONFIGDIR" "$IPYTHONDIR"
unset DISPLAY
unset WAYLAND_DISPLAY

python -m pip install "${SCRIPT_DIR}/../..[examples]"

# Execute each notebook that changed since the last successful run.
for file in *.ipynb; do
  timestamp_file=".${file}_timestamp"
  current_timestamp=$(stat -c %Y "$file")
  last_timestamp=0
  if [ -f "$timestamp_file" ]; then
    last_timestamp=$(cat "$timestamp_file")
  fi
  if [ "$current_timestamp" -gt "$last_timestamp" ]; then
    echo "Converting $file"
    env MPLBACKEND="Agg" jupyter nbconvert \
      --execute \
      --inplace \
      --ExecutePreprocessor.timeout=3600 \
      --ExecutePreprocessor.extra_arguments=--matplotlib=inline \
      "$file"
    post_run_timestamp=$(stat -c %Y "$file")
    # Save the captured timestamp, not the post-run timestamp
    echo "$post_run_timestamp" >"$timestamp_file"
  fi
done
