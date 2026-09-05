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

require_uint() {
  local name="$1" value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be an unsigned decimal integer; got: $value" >&2
    return 2
  fi
}
