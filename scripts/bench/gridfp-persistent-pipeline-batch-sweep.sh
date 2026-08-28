#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
W="${W:-10}"
K="${K:-4}"
S="${S:-4}"
BLOCKS="${BLOCKS:-256}"
NGPU="${NGPU:-8}"
BATCH_LIST="${BATCH_LIST:-16 32}"

BASE_BUILD="$(repo_path scripts/build/gridfp-reduced-component-probe.sh)"
PIPE_BUILD="$(repo_path scripts/build/gridfp-persistent-pipeline-probe.sh)"
BASE_BIN="$(build_path gridfp_reduced_component_p2p-host-persistent-descriptorless-sweep)"
PIPE_BIN="$(build_path gridfp_reduced_component_p2p-host-persistent-pipeline-sweep)"

MODE=p2p-host-persistent-descriptorless ARCH="$ARCH" OUT="$(basename "$BASE_BIN")" \
  bash "$BASE_BUILD"
ARCH="$ARCH" OUT="$(basename "$PIPE_BIN")" bash "$PIPE_BUILD"

for batches in $BATCH_LIST; do
  case "$batches" in
    16|32) ;;
    *) echo "unsupported production pipeline batch count: $batches" >&2; exit 2 ;;
  esac

  BASE_OUTPUT="$($BASE_BIN "$W" "$K" "$S" "$batches" "$BLOCKS" "$NGPU")"
  PIPE_OUTPUT="$($PIPE_BIN "$W" "$K" "$S" "$batches" "$BLOCKS" "$NGPU")"

  if ! printf '%s\n' "$BASE_OUTPUT" | grep -q 'ALL_OK gridfp_p2p_host_persistent_descriptorless=1'; then
    echo "baseline missing ALL_OK batches=$batches" >&2
    exit 4
  fi
  if ! printf '%s\n' "$PIPE_OUTPUT" | grep -q 'ALL_OK gridfp_p2p_host_persistent_pipeline=1'; then
    echo "pipeline missing ALL_OK batches=$batches" >&2
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
      echo "missing sweep timing batches=$batches direction=$direction" >&2
      exit 6
    fi
    SPEEDUP="$(awk -v a="$BASE_MS" -v b="$PIPE_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
    echo "persistent-pipeline-sweep batches=$batches direction=$direction sequential_ms=$BASE_MS pipeline_ms=$PIPE_MS speedup=$SPEEDUP"
  done

done

echo "persistent-pipeline-sweep exact=OK batch_list=[$BATCH_LIST]"
