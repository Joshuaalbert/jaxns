#!/usr/bin/env bash
set -euo pipefail

implementation="${1:?usage: run_matrix.sh IMPLEMENTATION OUTPUT_DIR}"
output_dir="${2:?usage: run_matrix.sh IMPLEMENTATION OUTPUT_DIR}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "${output_dir}"

cases=(
  basic
  basic2
  basic3
  plateau
  basic_mvn
  spike_slab
  spike_slab10
  weak_curved_mvn8
  weak_curved_spike_slab8
  weak_curved_spike_slab10
)

for case_name in "${cases[@]}"; do
  for phantom_flag in off on; do
    extra_args=()
    if [[ "${phantom_flag}" == "on" ]]; then
      extra_args+=(--phantoms)
    fi
    output_file="${output_dir}/${implementation}_${case_name}_${phantom_flag}.jsonl"
    if [[ "${implementation}" == "v3" ]]; then
      conda run --no-capture-output -n jaxns_py python \
        "${repo_root}/benchmarks/issue_247/run_v3_standard.py" \
        --case "${case_name}" --seeds 11,23,37 --mc-draws 1000 \
        --output "${output_file}" "${extra_args[@]}"
    elif [[ "${implementation}" == "v2-pypi" ]]; then
      PYTHONPATH="/tmp/jaxns_v2_269:${repo_root}" \
      MPLCONFIGDIR="/tmp/matplotlib-issue247" \
      conda run --no-capture-output -n jaxns_py python \
        "${repo_root}/benchmarks/issue_247/run_v2_standard.py" \
        --implementation-label v2-pypi --source-id jaxns==2.6.9 \
        --case "${case_name}" --seeds 11,23,37 --mc-draws 1000 \
        --output "${output_file}" "${extra_args[@]}"
    elif [[ "${implementation}" == "main" ]]; then
      PYTHONPATH="/tmp/jaxns_main_2f356d6/src:${repo_root}" \
      MPLCONFIGDIR="/tmp/matplotlib-issue247" \
      conda run --no-capture-output -n jaxns_py python \
        "${repo_root}/benchmarks/issue_247/run_v2_standard.py" \
        --implementation-label main --source-id 2f356d6 \
        --case "${case_name}" --seeds 11,23,37 --mc-draws 1000 \
        --output "${output_file}" "${extra_args[@]}"
    else
      echo "Unknown implementation: ${implementation}" >&2
      exit 2
    fi
  done
done
