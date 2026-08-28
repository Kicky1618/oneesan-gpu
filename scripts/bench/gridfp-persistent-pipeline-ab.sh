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
PIPE_BIN="$(build_path gridfp_reduced_component_p2p-host-persistent-pipeline)"

MODE=p2p-host-persistent-descriptorless ARCH="$ARCH" OUT="$(basename "$BASE_BIN")" \
  bash "$BASE_BUILD"
ARCH="$ARCH" OUT="$(basename "$PIPE_BIN")" bash "$PIPE_BUILD"

BASE_OUTPUT="$($BASE_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
PIPE_OUTPUT="$($PIPE_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
printf '%s\n' "$BASE_OUTPUT"
printf '%s\n' "$PIPE_OUTPUT"

if ! printf '%s\n' "$BASE_OUTPUT" | grep -q 'ALL_OK gridfp_p2p_host_persistent_descriptorless=1'; then
  echo "sequential descriptorless baseline missing ALL_OK" >&2
  exit 4
fi
if ! printf '%s\n' "$PIPE_OUTPUT" | grep -q 'ALL_OK gridfp_p2p_host_persistent_pipeline=1'; then
  echo "pipelined persistent probe missing ALL_OK" >&2
  exit 5
fi

for direction in forward reverse; do
  BASE_MS="$(printf '%s\n' "$BASE_OUTPUT" | awk -v want="$direction" '
    /gridfp-p2p-host-persistent-descriptorless/ {
      have=0; value="";
      for(i=1;i<=NF;++i){
        if($i=="direction=" want) have=1;
        if($i~/^runtime_ms=/){split($i,a,"=");value=a[2];}
      }
      if(have&&value!=""){print value;exit}
    }
  ')"
  PIPE_MS="$(printf '%s\n' "$PIPE_OUTPUT" | awk -v want="$direction" '
    /gridfp-p2p-host-persistent-pipeline/ {
      have=0; value="";
      for(i=1;i<=NF;++i){
        if($i=="direction=" want) have=1;
        if($i~/^runtime_ms=/){split($i,a,"=");value=a[2];}
      }
      if(have&&value!=""){print value;exit}
    }
  ')"
  if [[ -z "$BASE_MS" || -z "$PIPE_MS" ]]; then
    echo "missing pipeline A/B timing direction=$direction" >&2
    exit 6
  fi
  SPEEDUP="$(awk -v a="$BASE_MS" -v b="$PIPE_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  echo "persistent-pipeline-ab direction=$direction batches=$BATCHES sequential_ms=$BASE_MS pipeline_ms=$PIPE_MS speedup=$SPEEDUP"
done

PIPE_EXACT="$(printf '%s\n' "$PIPE_OUTPUT" | awk '
  /gridfp-p2p-host-persistent-pipeline/ && /scratch_planes=2/ &&
  /streams_per_gpu=2/ && /phase_b_next_a_overlap=1/ && /exact=OK/ { ++n }
  END { print n + 0 }
')"
if [[ "$PIPE_EXACT" != 2 ]]; then
  echo "pipeline probe did not prove both directions exact" >&2
  exit 7
fi

echo "persistent-pipeline-ab exact=OK batches=$BATCHES scratch_planes=2 streams_per_gpu=2"
