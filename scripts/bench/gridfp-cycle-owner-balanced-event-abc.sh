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
EVENT_BUILD="$(repo_path scripts/build/gridfp-cycle-owner-balanced-event-probe.sh)"
FANIN_SRC="$(repo_path src/cuda/gridfp/gridfp_reduced_production_cross_device_event_fanin_probe.cu)"
OLD_BIN="$(build_path gridfp_reduced_component_cycle-owner-oldhash-event-abc)"
BAL_BIN="$(build_path gridfp_reduced_component_cycle-owner-balanced-host-event-abc)"
EVENT_BIN="$(build_path gridfp_reduced_component_cycle-owner-balanced-event-abc)"
FANIN_BIN="$(build_path gridfp_reduced_production_cross_device_event_fanin_balanced_abc)"

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "$FANIN_SRC" -o "$FANIN_BIN"
VARIANT=cycle-owner ARCH="$ARCH" PTXAS_VERBOSE=0 OUT="$(basename "$OLD_BIN")" \
  bash "$OLD_BUILD"
ARCH="$ARCH" PTXAS_VERBOSE=0 OUT="$(basename "$BAL_BIN")" bash "$BAL_BUILD"
ARCH="$ARCH" PTXAS_VERBOSE=0 OUT="$(basename "$EVENT_BIN")" bash "$EVENT_BUILD"

FANIN_OUTPUT="$($FANIN_BIN "$NGPU")"
printf '%s\n' "$FANIN_OUTPUT"
grep -q 'ALL_OK gridfp_cross_device_event_fanin=1' <<<"$FANIN_OUTPUT" || exit 3

OLD_OUTPUT="$($OLD_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
BAL_OUTPUT="$($BAL_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
EVENT_OUTPUT="$($EVENT_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
printf '%s\n' "$OLD_OUTPUT"
printf '%s\n' "$BAL_OUTPUT"
printf '%s\n' "$EVENT_OUTPUT"

grep -q 'ALL_OK gridfp_p2p_cycle_owner_pipeline=1' <<<"$OLD_OUTPUT" || exit 4
grep -q 'ALL_OK gridfp_p2p_cycle_owner_balanced_pipeline=1' <<<"$BAL_OUTPUT" || exit 5
grep -q 'ALL_OK gridfp_p2p_cycle_owner_balanced_event_pipeline=1' <<<"$EVENT_OUTPUT" || exit 6
grep -q 'leader_batch_hash=1' <<<"$BAL_OUTPUT" || exit 7
grep -q 'leader_batch_hash=1' <<<"$EVENT_OUTPUT" || exit 8
grep -q 'event_schedule=staged' <<<"$EVENT_OUTPUT" || exit 9
grep -q 'host_batch_barriers=0' <<<"$EVENT_OUTPUT" || exit 10

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
  EVENT_MS="$(extract "$EVENT_OUTPUT" gridfp-p2p-cycle-owner-balanced-event-pipeline "$direction" runtime_ms)"
  OLD_PEER="$(extract "$OLD_OUTPUT" gridfp-p2p-cycle-owner-pipeline "$direction" logical_peer_KiB)"
  BAL_PEER="$(extract "$BAL_OUTPUT" gridfp-p2p-cycle-owner-balanced-pipeline "$direction" logical_peer_KiB)"
  EVENT_PEER="$(extract "$EVENT_OUTPUT" gridfp-p2p-cycle-owner-balanced-event-pipeline "$direction" logical_peer_KiB)"
  OLD_ENTRIES="$(extract "$OLD_OUTPUT" gridfp-p2p-cycle-owner-pipeline "$direction" total_list_entries)"
  BAL_ENTRIES="$(extract "$BAL_OUTPUT" gridfp-p2p-cycle-owner-balanced-pipeline "$direction" total_list_entries)"
  EVENT_ENTRIES="$(extract "$EVENT_OUTPUT" gridfp-p2p-cycle-owner-balanced-event-pipeline "$direction" total_list_entries)"
  if [[ -z "$OLD_MS" || -z "$BAL_MS" || -z "$EVENT_MS" ]]; then
    echo "missing balanced event A/B/C timing direction=$direction" >&2
    exit 11
  fi
  if [[ "$OLD_PEER" != "$BAL_PEER" || "$BAL_PEER" != "$EVENT_PEER" ]]; then
    echo "logical peer traffic differs direction=$direction" >&2
    exit 12
  fi
  if [[ -n "$OLD_ENTRIES" && -n "$BAL_ENTRIES" && -n "$EVENT_ENTRIES" ]]; then
    if [[ "$OLD_ENTRIES" != "$BAL_ENTRIES" || "$BAL_ENTRIES" != "$EVENT_ENTRIES" ]]; then
      echo "list entries differ direction=$direction" >&2
      exit 13
    fi
  fi
  HASH_SPEEDUP="$(awk -v a="$OLD_MS" -v b="$BAL_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  EVENT_SPEEDUP="$(awk -v a="$BAL_MS" -v b="$EVENT_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  TOTAL_SPEEDUP="$(awk -v a="$OLD_MS" -v b="$EVENT_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  echo "cycle-owner-balanced-event-abc direction=$direction batches=$BATCHES oldhash_host_ms=$OLD_MS balanced_host_ms=$BAL_MS balanced_event_ms=$EVENT_MS hash_speedup=$HASH_SPEEDUP event_speedup=$EVENT_SPEEDUP total_speedup=$TOTAL_SPEEDUP logical_peer_KiB=$EVENT_PEER list_entries=${EVENT_ENTRIES:-unknown}"
done

echo "cycle-owner-balanced-event-abc exact=OK batches=$BATCHES leader_batch_hash=1 batch_hash_shift=12 event_schedule=staged host_batch_barriers=0"
