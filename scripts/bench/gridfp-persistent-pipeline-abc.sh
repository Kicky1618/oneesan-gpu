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
PIPE_BUILD="$(repo_path scripts/build/gridfp-persistent-pipeline-probe.sh)"
BASE_BIN="$(build_path gridfp_reduced_component_p2p-host-persistent-descriptorless-b16)"
SEG_BIN="$(build_path gridfp_reduced_component_p2p-host-persistent-pipeline-b16)"
COMP_BIN="$(build_path gridfp_reduced_component_p2p-cycle-owner-pipeline-b16)"

MODE=p2p-host-persistent-descriptorless ARCH="$ARCH" OUT="$(basename "$BASE_BIN")" \
  bash "$BASE_BUILD"
VARIANT=segment ARCH="$ARCH" OUT="$(basename "$SEG_BIN")" bash "$PIPE_BUILD"
VARIANT=cycle-owner ARCH="$ARCH" OUT="$(basename "$COMP_BIN")" bash "$PIPE_BUILD"

BASE_OUTPUT="$($BASE_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
SEG_OUTPUT="$($SEG_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
COMP_OUTPUT="$($COMP_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
printf '%s\n' "$BASE_OUTPUT"
printf '%s\n' "$SEG_OUTPUT"
printf '%s\n' "$COMP_OUTPUT"

printf '%s\n' "$BASE_OUTPUT" | grep -q 'ALL_OK gridfp_p2p_host_persistent_descriptorless=1' || exit 4
printf '%s\n' "$SEG_OUTPUT" | grep -q 'ALL_OK gridfp_p2p_host_persistent_pipeline=1' || exit 5
printf '%s\n' "$COMP_OUTPUT" | grep -q 'ALL_OK gridfp_p2p_cycle_owner_pipeline=1' || exit 6

extract_ms() {
  local text="$1" tag="$2" direction="$3"
  printf '%s\n' "$text" | awk -v tag="$tag" -v want="$direction" '
    index($0, tag) {
      have=0; value="";
      for(i=1;i<=NF;++i){
        if($i=="direction=" want) have=1;
        if($i~/^runtime_ms=/){split($i,a,"=");value=a[2];}
      }
      if(have&&value!=""){print value;exit}
    }
  '
}

extract_entries() {
  local text="$1" tag="$2" direction="$3"
  printf '%s\n' "$text" | awk -v tag="$tag" -v want="$direction" '
    index($0, tag) {
      have=0; value="";
      for(i=1;i<=NF;++i){
        if($i=="direction=" want) have=1;
        if($i~/^total_list_entries=/){split($i,a,"=");value=a[2];}
        if($i~/^list_entries=/ && value==""){split($i,a,"=");value=a[2];}
      }
      if(have&&value!=""){print value;exit}
    }
  '
}

for direction in forward reverse; do
  BASE_MS="$(extract_ms "$BASE_OUTPUT" gridfp-p2p-host-persistent-descriptorless "$direction")"
  SEG_MS="$(extract_ms "$SEG_OUTPUT" gridfp-p2p-host-persistent-pipeline "$direction")"
  COMP_MS="$(extract_ms "$COMP_OUTPUT" gridfp-p2p-cycle-owner-pipeline "$direction")"
  BASE_ENTRIES="$(extract_entries "$BASE_OUTPUT" gridfp-p2p-host-persistent-descriptorless "$direction")"
  SEG_ENTRIES="$(extract_entries "$SEG_OUTPUT" gridfp-p2p-host-persistent-pipeline "$direction")"
  COMP_ENTRIES="$(extract_entries "$COMP_OUTPUT" gridfp-p2p-cycle-owner-pipeline "$direction")"
  if [[ -z "$BASE_MS" || -z "$SEG_MS" || -z "$COMP_MS" ]]; then
    echo "missing pipeline A/B/C timing direction=$direction" >&2
    exit 7
  fi
  SEG_SPEEDUP="$(awk -v a="$BASE_MS" -v b="$SEG_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  COMP_SPEEDUP="$(awk -v a="$BASE_MS" -v b="$COMP_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  COMP_VS_SEG="$(awk -v a="$SEG_MS" -v b="$COMP_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  echo "persistent-pipeline-abc direction=$direction batches=$BATCHES sequential_ms=$BASE_MS segment_pipeline_ms=$SEG_MS compressed_pipeline_ms=$COMP_MS segment_speedup=$SEG_SPEEDUP compressed_speedup=$COMP_SPEEDUP compressed_vs_segment=$COMP_VS_SEG sequential_entries=${BASE_ENTRIES:-unknown} segment_entries=${SEG_ENTRIES:-unknown} compressed_entries=${COMP_ENTRIES:-unknown}"
done

echo "persistent-pipeline-abc exact=OK batches=$BATCHES descriptor_bytes=0 scratch_allocator_atomics=0"
