#!/usr/bin/env bash
set -euo pipefail

implementation="${1:?usage: run_release_matrix.sh IMPLEMENTATION WORKSPACE OUTPUT_DIR [SEEDS]}"
workspace="${2:?usage: run_release_matrix.sh IMPLEMENTATION WORKSPACE OUTPUT_DIR [SEEDS]}"
output_dir="${3:?usage: run_release_matrix.sh IMPLEMENTATION WORKSPACE OUTPUT_DIR [SEEDS]}"
seeds="${4:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}"

if [[ "${implementation}" != "v2" && "${implementation}" != "v3" ]]; then
  echo "IMPLEMENTATION must be v2 or v3." >&2
  exit 2
fi

source_root="${workspace}/source-${implementation}"
env_root="${workspace}/env-${implementation}"
if [[ ! -e "${source_root}/.git" ]] || \
   [[ ! -f "${env_root}/conda-meta/history" ]]; then
  echo "Run prepare_environments.sh before the release matrix." >&2
  exit 2
fi
source_id="$(git -C "${source_root}" rev-parse HEAD)"
runner_root="${workspace}/source-v3"
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
  for conditioning in classic phantom; do
    extra_args=()
    if [[ "${conditioning}" == "phantom" ]]; then
      extra_args+=(--phantoms)
    fi
    output_file="${output_dir}/${implementation}_${case_name}_${conditioning}.jsonl"
    if [[ "${implementation}" == "v2" ]]; then
      PYTHONPATH="${source_root}/src" \
      MPLCONFIGDIR="${workspace}/matplotlib-v2" \
      conda run --no-capture-output --prefix "${env_root}" python \
        "${runner_root}/benchmarks/issue_247/run_v2_standard.py" \
        --implementation-label v2 --source-id "${source_id}" \
        --case "${case_name}" --seeds "${seeds}" --mc-draws 1000 \
        --output "${output_file}" "${extra_args[@]}"
    else
      MPLCONFIGDIR="${workspace}/matplotlib-v3" \
      conda run --no-capture-output --prefix "${env_root}" python \
        "${runner_root}/benchmarks/issue_247/run_current_standard.py" \
        --source-id "${source_id}" --case "${case_name}" \
        --seeds "${seeds}" --mc-draws 1000 --measure-depth-program \
        --output "${output_file}" "${extra_args[@]}"
    fi
  done
done
