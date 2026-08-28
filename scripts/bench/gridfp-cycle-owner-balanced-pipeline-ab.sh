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

OLD_BUILD="$(repo_path scripts/build/gridfp-persistent-pipeline-probe.sh)"
BAL_BUILD="$(repo_path scripts/build/gridfp-cycle-owner-balanced-pipeline-probe.sh)"
OLD_BIN="$(build_path gridfp_reduced_component_p2p-cycle-owner-pipeline-oldhash-ab)"
BAL_BIN="$(build_path gridfp_reduced_component_p2p-cycle-owner-balanced-pipeline-ab)"

VARIANT=cycle-owner ARCH="$ARCH" PTXAS_VERBOSE=0 OUT="$(basename "$OLD_BIN")" \
  bash "$OLD_BUILD"
ARCH="$ARCH" PTXAS_VERBOSE=0 OUT="$(basename "$BAL_BIN")" bash "$BAL_BUILD"

OLD_OUTPUT="$($OLD_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
BAL_OUTPUT="$($BAL_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
printf '%s\n' "$OLD_OUTPUT"
printf '%s\n' "$BAL_OUTPUT"

grep -q 'ALL_OK gridfp_p2p_cycle_owner_pipeline=1' <<<"$OLD_OUTPUT" || exit 4
grep -q 'ALL_OK gridfp_p2p_cycle_owner_balanced_pipeline=1' <<<"$BAL_OUTPUT" || exit 5
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
  OLD_MS="$(extract "$OLD_OUTPUT" gridfp-p2p-cycle-owner-pipeline "$direction" runtime_ms)"
  BAL_MS="$(extract "$BAL_OUTPUT" gridfp-p2p-cycle-owner-balanced-pipeline "$direction" runtime_ms)"
  OLD_PEER="$(extract "$OLD_OUTPUT" gridfp-p2p-cycle-owner-pipeline "$direction" logical_peer_KiB)"
  BAL_PEER="$(extract "$BAL_OUTPUT" gridfp-p2p-cycle-owner-balanced-pipeline "$direction" logical_peer_KiB)"
  OLD_ENTRIES="$(extract "$OLD_OUTPUT" gridfp-p2p-cycle-owner-pipeline "$direction" total_list_entries)"
  BAL_ENTRIES="$(extract "$BAL_OUTPUT" gridfp-p2p-cycle-owner-balanced-pipeline "$direction" total_list_entries)"
  OLD_P0="$(extract "$OLD_OUTPUT" gridfp-p2p-cycle-owner-pipeline "$direction" scratch_plane0_max_KiB)"
  OLD_P1="$(extract "$OLD_OUTPUT" gridfp-p2p-cycle-owner-pipeline "$direction" scratch_plane1_max_KiB)"
  BAL_P0="$(extract "$BAL_OUTPUT" gridfp-p2p-cycle-owner-balanced-pipeline "$direction" scratch_plane0_max_KiB)"
  BAL_P1="$(extract "$BAL_OUTPUT" gridfp-p2p-cycle-owner-balanced-pipeline "$direction" scratch_plane1_max_KiB)"
  if [[ -z "$OLD_MS" || -z "$BAL_MS" || -z "$OLD_PEER" || -z "$BAL_PEER" ]]; then
    echo "missing balanced pipeline metrics direction=$direction" >&2
    exit 8
  fi
  [[ "$OLD_PEER" == "$BAL_PEER" ]] || { echo "logical peer traffic changed direction=$direction" >&2; exit 9; }
  if [[ -n "$OLD_ENTRIES" && -n "$BAL_ENTRIES" ]]; then
    [[ "$OLD_ENTRIES" == "$BAL_ENTRIES" ]] || { echo "list entries changed direction=$direction" >&2; exit 10; }
  fi
  SPEEDUP="$(awk -v a="$OLD_MS" -v b="$BAL_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  OLD_SCRATCH="$(awk -v a="${OLD_P0:-0}" -v b="${OLD_P1:-0}" 'BEGIN{printf "%.3f",a+b}')"
  BAL_SCRATCH="$(awk -v a="${BAL_P0:-0}" -v b="${BAL_P1:-0}" 'BEGIN{printf "%.3f",a+b}')"
  echo "cycle-owner-balanced-pipeline-ab direction=$direction batches=$BATCHES oldhash_ms=$OLD_MS balanced_ms=$BAL_MS balanced_speedup=$SPEEDUP old_scratch_planes_KiB=$OLD_SCRATCH balanced_scratch_planes_KiB=$BAL_SCRATCH logical_peer_KiB=$BAL_PEER list_entries=${BAL_ENTRIES:-unknown}"
done

echo "cycle-owner-balanced-pipeline-ab exact=OK batches=$BATCHES leader_batch_hash=1 batch_hash_shift=12 double_scratch=1"
