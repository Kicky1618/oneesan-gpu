#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PATCH_ONLY="${PIPE2_PATCH_ONLY:-0}"
[[ "$PATCH_ONLY" == 0 || "$PATCH_ONLY" == 1 ]] || { echo 'PIPE2_PATCH_ONLY must be 0/1' >&2; exit 2; }
[[ "${ORBITCTA_FLAT:-1}" == 1 ]] || { echo 'pipe2 requires ORBITCTA_FLAT=1' >&2; exit 2; }
[[ "${ORBITCTA_FLAT_DYNAMIC:-1}" == 1 ]] || { echo 'pipe2 requires ORBITCTA_FLAT_DYNAMIC=1' >&2; exit 2; }
[[ "${ORBITCTA_FLAT_CHUNK:-1}" == 1 ]] || { echo 'pipe2 requires ORBITCTA_FLAT_CHUNK=1' >&2; exit 2; }
[[ "${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP:-0}" == 0 ]] || {
  echo 'pipe2 replaces ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP; set it to 0' >&2; exit 2;
}

base="$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh"
if [[ "$PATCH_ONLY" == 1 ]]; then
  bash -n "$base"
  grep -Fq -- '-DP10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2="$ORBITCTA_FLAT_DYNAMIC_PIPE2"' "$base" || { echo 'native pipe2 nvcc macro missing' >&2; exit 3; }
  grep -Fq 'flat_dynamic_pipe2=$ORBITCTA_FLAT_DYNAMIC_PIPE2' "$base" || { echo 'native pipe2 build marker missing' >&2; exit 3; }
  echo 'b300_directgather_orbitcta_pipe2_native=OK gpu_work=0'
  exit 0
fi

exec env ORBITCTA_FLAT=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_CHUNK=1 \
  ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 ORBITCTA_FLAT_DYNAMIC_PIPE2=1 \
  bash "$base"
