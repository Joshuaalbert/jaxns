#!/bin/bash

# Exit on error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
echo "Script dir $SCRIPT_DIR"

pip install -r "$SCRIPT_DIR"/../requirements.txt
pip install -r "$SCRIPT_DIR"/../requirements-examples.txt
pip install "$SCRIPT_DIR"/..

# Convert each notebook if no change since last run (stored variable)
for file in *.ipynb; do
  timestamp_file=".${file}_timestamp"
  current_timestamp=$(stat -c %Y "$file")
  last_timestamp=0
  if [ -f "$timestamp_file" ]; then
    last_timestamp=$(cat "$timestamp_file")
  fi
  if [ "$current_timestamp" -gt "$last_timestamp" ]; then
    jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=3600 "$file"
    # --to notebook  --stdout > /dev/null
    echo "$current_timestamp" >"$timestamp_file"
  fi
done
