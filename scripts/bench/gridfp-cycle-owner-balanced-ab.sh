#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
W="${W:-10}"
K="${K:-4}"
S="${S:-4}"
BATCHES="${BATCHES:-16}"
BLOCKS="${BLOCKS:-256}"
NGPU="${NGPU:-8}"

BASE_BUILD="$(repo_path scripts/build/gridfp-reduced-component-probe.sh)"
BAL_BUILD="$(repo_path scripts/build/gridfp-cycle-owner-balanced-probe.sh)"
BASE_BIN="$(build_path gridfp_reduced_component_p2p-cycle-owner-oldhash-ab)"
BAL_BIN="$(build_path gridfp_reduced_component_p2p-cycle-owner-balanced-ab)"

MODE=p2p-cycle-owner-descriptorless ARCH="$ARCH" PTXAS_VERBOSE=0 \
  OUT="$(basename "$BASE_BIN")" bash "$BASE_BUILD"
ARCH="$ARCH" PTXAS_VERBOSE=0 OUT="$(basename "$BAL_BIN")" bash "$BAL_BUILD"

BASE_OUTPUT="$($BASE_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
BAL_OUTPUT="$($BAL_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
printf '%s\n' "$BASE_OUTPUT"
printf '%s\n' "$BAL_OUTPUT"

grep -q 'ALL_OK gridfp_p2p_cycle_owner_descriptorless=1' <<<"$BASE_OUTPUT" || exit 4
grep -q 'ALL_OK gridfp_p2p_cycle_owner_balanced_descriptorless=1' <<<"$BAL_OUTPUT" || exit 5
grep -q 'leader_batch_hash=1' <<<"$BAL_OUTPUT" || exit 6
grep -q 'batch_hash_shift=12' <<<"$BAL_OUTPUT" || exit 7

extract() {
  local text="$1" tag="$2" direction="$3" key="$4"
  printf '%s\n' "$text" | awk -v tag="$tag" -v want="$direction" -v key="$key" '
    index($0, tag) {
      have=0; value="";
      for(i=1;i<=NF;++i){
        if($i=="direction=" want) have=1;
        if(index($i,key "=")==1){split($i,a,"=");value=a[2];}
      }
      if(have&&value!=""){print value;exit}
    }
  '
}

for direction in forward reverse; do
  BASE_MS="$(extract "$BASE_OUTPUT" gridfp-p2p-cycle-owner-descriptorless "$direction" runtime_ms)"
  BAL_MS="$(extract "$BAL_OUTPUT" gridfp-p2p-cycle-owner-balanced-descriptorless "$direction" runtime_ms)"
  BASE_PEER="$(extract "$BASE_OUTPUT" gridfp-p2p-cycle-owner-descriptorless "$direction" logical_peer_KiB)"
  BAL_PEER="$(extract "$BAL_OUTPUT" gridfp-p2p-cycle-owner-balanced-descriptorless "$direction" logical_peer_KiB)"
  BASE_ENTRIES="$(extract "$BASE_OUTPUT" gridfp-p2p-cycle-owner-descriptorless "$direction" total_list_entries)"
  BAL_ENTRIES="$(extract "$BAL_OUTPUT" gridfp-p2p-cycle-owner-balanced-descriptorless "$direction" total_list_entries)"
  if [[ -z "$BASE_MS" || -z "$BAL_MS" || -z "$BASE_PEER" || -z "$BAL_PEER" ]]; then
    echo "missing balanced cycle-owner metrics direction=$direction" >&2
    exit 8
  fi
  if [[ "$BASE_PEER" != "$BAL_PEER" ]]; then
    echo "logical peer traffic changed direction=$direction old=$BASE_PEER balanced=$BAL_PEER" >&2
    exit 9
  fi
  if [[ -n "$BASE_ENTRIES" && -n "$BAL_ENTRIES" && "$BASE_ENTRIES" != "$BAL_ENTRIES" ]]; then
    echo "list entry count changed direction=$direction old=$BASE_ENTRIES balanced=$BAL_ENTRIES" >&2
    exit 10
  fi
  SPEEDUP="$(awk -v a="$BASE_MS" -v b="$BAL_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  echo "cycle-owner-balanced-ab direction=$direction batches=$BATCHES oldhash_ms=$BASE_MS balanced_ms=$BAL_MS balanced_speedup=$SPEEDUP logical_peer_KiB=$BAL_PEER list_entries=${BAL_ENTRIES:-unknown}"
done

echo "cycle-owner-balanced-ab exact=OK batches=$BATCHES leader_batch_hash=1 batch_hash_shift=12"
