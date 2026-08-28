#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O3 -std=c++17}"
W="${W:-28}"
K="${K:-13}"
NGPU="${NGPU:-8}"
MIN_B300_HEADROOM_GIB="${MIN_B300_HEADROOM_GIB:-40}"

SEG_SRC="$(repo_path src/cpp/probes/gridfp_p2p_owner_local_segment_probe.cpp)"
HASH_SRC="$(repo_path src/cpp/probes/gridfp_p2p_cycle_batch_hash_probe.cpp)"
MEM_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_persistent_memory_probe.cpp)"
SEG_BIN="$(build_path gridfp_p2p_owner_local_segment_probe)"
HASH_BIN="$(build_path gridfp_p2p_cycle_batch_hash_probe)"
MEM_BIN="$(build_path gridfp_reduced_production_persistent_memory_probe)"

# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$SEG_SRC" -o "$SEG_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$HASH_SRC" -o "$HASH_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$MEM_SRC" -o "$MEM_BIN"

echo "== owner-local maximal-segment proof =="
"$SEG_BIN"

echo "== cycle-closed batch hash proof =="
"$HASH_BIN"

echo "== persistent-list W28 memory bound =="
MEM_OUTPUT="$($MEM_BIN "$W" "$K" "$NGPU")"
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

echo "persistent-redistribution-research exact=OK W=$W K=$K ngpu=$NGPU B300_headroom_before_scratch_GiB=$HEADROOM"
