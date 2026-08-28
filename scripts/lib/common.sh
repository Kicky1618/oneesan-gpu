#!/usr/bin/env bash
set -euo pipefail

ONEESAN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ONEESAN_BUILD_DIR="${ONEESAN_BUILD_DIR:-$ONEESAN_ROOT/build}"
ONEESAN_TMP_DIR="${ONEESAN_TMP_DIR:-$ONEESAN_BUILD_DIR/tmp}"

mkdir -p "$ONEESAN_BUILD_DIR" "$ONEESAN_TMP_DIR"

repo_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ONEESAN_ROOT" "$path"
  fi
}

build_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  elif [[ "$path" == */* ]]; then
    printf '%s/%s\n' "$ONEESAN_ROOT" "$path"
  else
    printf '%s/%s\n' "$ONEESAN_BUILD_DIR" "$path"
  fi
}

require_nvcc_version_at_least() {
  local nvcc_bin="${1:-nvcc}" min_major="${2:?missing minimum CUDA major}" min_minor="${3:-0}" reason="${4:-requested CUDA target}"
  command -v "$nvcc_bin" >/dev/null || { echo "$nvcc_bin not found" >&2; return 2; }
  local line version major minor
  line="$("$nvcc_bin" --version 2>&1 | grep -E 'Cuda compilation tools, release [0-9]+\.[0-9]+' | tail -n1 || true)"
  version="$(sed -nE 's/.*release ([0-9]+)\.([0-9]+).*/\1.\2/p' <<<"$line")"
  if [[ -z "$version" ]]; then
    echo "could not determine CUDA Toolkit version from $nvcc_bin --version" >&2
    return 2
  fi
  major="${version%%.*}"; minor="${version#*.}"
  if (( major < min_major || (major == min_major && minor < min_minor) )); then
    echo "$reason requires CUDA Toolkit >= ${min_major}.${min_minor}; found ${major}.${minor} via $nvcc_bin" >&2
    return 2
  fi
  printf 'cuda_toolkit=%s nvcc=%s requirement=%s\n' "$version" "$nvcc_bin" "${min_major}.${min_minor}" >&2
}
