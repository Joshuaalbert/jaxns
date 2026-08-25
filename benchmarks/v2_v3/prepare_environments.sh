#!/usr/bin/env bash
set -euo pipefail

workspace="${1:?usage: prepare_environments.sh WORKSPACE [V2_REF] [V3_REF]}"
v2_ref="${2:-v2}"
v3_ref="${3:-HEAD}"
repo_root="$(git rev-parse --show-toplevel)"
benchmark_root="${repo_root}/benchmarks/v2_v3"

if [[ "${workspace}" != /* ]] || [[ "${workspace}" == "/" ]] || \
   [[ "${workspace}" == "${repo_root}" ]]; then
  echo "WORKSPACE must be an absolute, dedicated directory." >&2
  exit 2
fi

mkdir -p "${workspace}"
v2_source="${workspace}/source-v2"
v3_source="${workspace}/source-v3"
v2_env="${workspace}/env-v2"
v3_env="${workspace}/env-v3"

resolve_commit() {
  local source_ref="$1"
  if git rev-parse --verify --quiet "${source_ref}^{commit}"; then
    return
  fi
  # A fresh clone may not have the maintenance branch locally. Fetch only the
  # requested ref, then pin the detached worktree to the resolved FETCH_HEAD.
  git fetch origin "${source_ref}"
  git rev-parse --verify "FETCH_HEAD^{commit}"
}

v2_commit="$(resolve_commit "${v2_ref}")"
v3_commit="$(resolve_commit "${v3_ref}")"

prepare_source() {
  local source_path="$1"
  local source_commit="$2"
  local actual_commit
  if [[ -e "${source_path}/.git" ]]; then
    actual_commit="$(git -C "${source_path}" rev-parse HEAD)"
    if [[ "${actual_commit}" != "${source_commit}" ]]; then
      echo "${source_path} is ${actual_commit}, expected ${source_commit}." >&2
      exit 2
    fi
    return
  fi
  if [[ -e "${source_path}" ]]; then
    echo "${source_path} exists but is not a Git worktree." >&2
    exit 2
  fi
  git worktree add --detach "${source_path}" "${source_commit}"
}

prepare_environment() {
  local env_path="$1"
  local environment_file="$2"
  local source_path="$3"
  if [[ ! -f "${env_path}/conda-meta/history" ]]; then
    conda env create --prefix "${env_path}" --file "${environment_file}"
  fi
  # Install the exact detached source after dependencies so neither release
  # can accidentally resolve the package from PyPI or the caller's checkout.
  conda run --prefix "${env_path}" \
    python -m pip install --no-deps --force-reinstall "${source_path}"
}

prepare_source "${v2_source}" "${v2_commit}"
prepare_source "${v3_source}" "${v3_commit}"
prepare_environment "${v2_env}" "${benchmark_root}/environment-v2.yml" "${v2_source}"
prepare_environment "${v3_env}" "${benchmark_root}/environment-v3.yml" "${v3_source}"

conda run --prefix "${v2_env}" python \
  "${v3_source}/benchmarks/v2_v3/verify_environment.py" \
  --implementation v2 --source-id "${v2_commit}" \
  --output "${workspace}/environment-v2.json"
conda run --prefix "${v3_env}" python \
  "${v3_source}/benchmarks/v2_v3/verify_environment.py" \
  --implementation v3 --source-id "${v3_commit}" \
  --output "${workspace}/environment-v3.json"

echo "Prepared v2=${v2_commit} and v3=${v3_commit} in ${workspace}."
