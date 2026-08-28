#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O3 -std=c++17}"
W="${W:-28}"
K="${K:-13}"
NGPU="${NGPU:-8}"
SUPER_MAX_W="${SUPER_MAX_W:-11}"
MIN_B300_HEADROOM_GIB="${MIN_B300_HEADROOM_GIB:-40}"
MIN_EXACT_B300_HEADROOM_GIB="${MIN_EXACT_B300_HEADROOM_GIB:-20}"
MIN_COMPRESSED_B300_HEADROOM_GIB="${MIN_COMPRESSED_B300_HEADROOM_GIB:-20}"
MIN_LIST_COMPRESSION="${MIN_LIST_COMPRESSION:-2.4}"
NODE_HBM_TBPS="${NODE_HBM_TBPS:-64}"
NODE_NVLINK_TBPS="${NODE_NVLINK_TBPS:-14.4}"

SEG_SRC="$(repo_path src/cpp/probes/gridfp_p2p_owner_local_segment_probe.cpp)"
HASH_SRC="$(repo_path src/cpp/probes/gridfp_p2p_cycle_batch_hash_probe.cpp)"
HOST_LIST_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_host_persistent_list_probe.cpp)"
MEM_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_persistent_memory_probe.cpp)"
EXACT_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_persistent_exact_plan_probe.cpp)"
COMPRESS_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_cycle_owner_compression_probe.cpp)"
SUPER_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_stationary_supercomponent_probe.cpp)"
BW_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_persistent_bandwidth_probe.cpp)"
SEG_BIN="$(build_path gridfp_p2p_owner_local_segment_probe)"
HASH_BIN="$(build_path gridfp_p2p_cycle_batch_hash_probe)"
HOST_LIST_BIN="$(build_path gridfp_reduced_production_host_persistent_list_probe)"
MEM_BIN="$(build_path gridfp_reduced_production_persistent_memory_probe)"
EXACT_BIN="$(build_path gridfp_reduced_production_persistent_exact_plan_probe)"
COMPRESS_BIN="$(build_path gridfp_reduced_production_cycle_owner_compression_probe)"
SUPER_BIN="$(build_path gridfp_reduced_production_stationary_supercomponent_probe)"
BW_BIN="$(build_path gridfp_reduced_production_persistent_bandwidth_probe)"

# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$SEG_SRC" -o "$SEG_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$HASH_SRC" -o "$HASH_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$HOST_LIST_SRC" -o "$HOST_LIST_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$MEM_SRC" -o "$MEM_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$EXACT_SRC" -o "$EXACT_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$COMPRESS_SRC" -o "$COMPRESS_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$SUPER_SRC" -o "$SUPER_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$BW_SRC" -o "$BW_BIN"

echo "== owner-local maximal-segment proof =="
"$SEG_BIN"

echo "== cycle-closed batch hash proof =="
"$HASH_BIN"

echo "== host-built persistent-list exact coverage =="
"$HOST_LIST_BIN"

echo "== stationary component-invariant owner probe =="
"$SUPER_BIN" "$SUPER_MAX_W"

echo "== persistent-list W28 dual-direction memory bound =="
MEM_OUTPUT="$($MEM_BIN "$W" "$K" "$NGPU" 2)"
printf '%s\n' "$MEM_OUTPUT"

HEADROOM="$(printf '%s\n' "$MEM_OUTPUT" | awk '
  /gridfp-persistent-memory-bound/ {
    for (i=1; i<=NF; ++i) {
      if ($i ~ /^worst_B300_headroom_before_scratch_GiB=/) {
        split($i,a,"="); print a[2]; exit
      }
    }
  }
')"
if [[ -z "$HEADROOM" ]]; then
  echo "persistent memory probe did not report B300 headroom" >&2
  exit 4
fi
if ! awk -v got="$HEADROOM" -v want="$MIN_B300_HEADROOM_GIB" \
  'BEGIN { exit !(got + 0 >= want + 0) }'; then
  echo "persistent memory headroom too small: got=${HEADROOM}GiB want>=${MIN_B300_HEADROOM_GIB}GiB" >&2
  exit 5
fi

echo "== exact W28/K13 cycle-batched persistent plan =="
EXACT_OUTPUT="$($EXACT_BIN)"
printf '%s\n' "$EXACT_OUTPUT"
EXACT_HEADROOM="$(printf '%s\n' "$EXACT_OUTPUT" | awk '
  /ALL_OK gridfp_persistent_exact_plan=1/ {
    for (i=1; i<=NF; ++i) {
      if ($i ~ /^B300_headroom_GiB=/) {
        split($i,a,"="); print a[2]; exit
      }
    }
  }
')"
if [[ -z "$EXACT_HEADROOM" ]]; then
  echo "exact persistent planner did not report B300 headroom" >&2
  exit 6
fi
if ! awk -v got="$EXACT_HEADROOM" -v want="$MIN_EXACT_B300_HEADROOM_GIB" \
  'BEGIN { exit !(got + 0 >= want + 0) }'; then
  echo "exact persistent peak headroom too small: got=${EXACT_HEADROOM}GiB want>=${MIN_EXACT_B300_HEADROOM_GIB}GiB" >&2
  exit 7
fi

echo "== exact W28/K13 cycle-owner list compression =="
COMPRESS_OUTPUT="$($COMPRESS_BIN)"
printf '%s\n' "$COMPRESS_OUTPUT"
COMPRESSED_HEADROOM="$(printf '%s\n' "$COMPRESS_OUTPUT" | awk '
  /cycle-owner-compression-peak/ {
    for (i=1; i<=NF; ++i) {
      if ($i ~ /^B300_headroom_GiB=/) {
        split($i,a,"=");
        if (!seen || a[2] + 0 < min) min=a[2]+0;
        seen=1;
      }
    }
  }
  END { if(seen) printf "%.9f", min }
')"
LIST_COMPRESSION="$(printf '%s\n' "$COMPRESS_OUTPUT" | awk '
  /cycle-owner-compression-direction/ && /direction=forward/ {
    for (i=1; i<=NF; ++i) {
      if ($i ~ /^compression=/) { split($i,a,"="); print a[2]; exit }
    }
  }
')"
if [[ -z "$COMPRESSED_HEADROOM" || -z "$LIST_COMPRESSION" ]]; then
  echo "cycle-owner compression probe missing summary fields" >&2
  exit 8
fi
if ! awk -v got="$COMPRESSED_HEADROOM" -v want="$MIN_COMPRESSED_B300_HEADROOM_GIB" \
  'BEGIN { exit !(got + 0 >= want + 0) }'; then
  echo "cycle-owner compressed headroom too small: got=${COMPRESSED_HEADROOM}GiB want>=${MIN_COMPRESSED_B300_HEADROOM_GIB}GiB" >&2
  exit 9
fi
if ! awk -v got="$LIST_COMPRESSION" -v want="$MIN_LIST_COMPRESSION" \
  'BEGIN { exit !(got + 0 >= want + 0) }'; then
  echo "cycle-owner list compression too small: got=${LIST_COMPRESSION}x want>=${MIN_LIST_COMPRESSION}x" >&2
  exit 10
fi

echo "== optimistic HGX B300 redistribution bandwidth floor =="
"$BW_BIN" "$NODE_HBM_TBPS" "$NODE_NVLINK_TBPS"

echo "persistent-redistribution-research exact=OK W=$W K=$K ngpu=$NGPU startup_gpu_support_scan_candidate=0 B300_headroom_before_scratch_GiB=$HEADROOM exact_peak_headroom_GiB=$EXACT_HEADROOM compressed_peak_headroom_GiB=$COMPRESSED_HEADROOM list_compression=$LIST_COMPRESSION node_HBM_TBps=$NODE_HBM_TBPS node_NVLink_TBps=$NODE_NVLINK_TBPS"
