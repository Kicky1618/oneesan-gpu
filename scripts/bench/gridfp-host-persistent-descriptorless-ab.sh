#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
W="${W:-10}"
K="${K:-4}"
S="${S:-4}"
BATCHES="${BATCHES:-8}"
BLOCKS="${BLOCKS:-256}"
NGPU="${NGPU:-8}"

BUILD_SCRIPT="$(repo_path scripts/build/gridfp-reduced-component-probe.sh)"
DESC_BIN="$(build_path gridfp_reduced_component_p2p-host-persistent-list)"
NODESC_BIN="$(build_path gridfp_reduced_component_p2p-host-persistent-descriptorless)"

MODE=p2p-host-persistent-list ARCH="$ARCH" OUT="$(basename "$DESC_BIN")" \
  bash "$BUILD_SCRIPT"
MODE=p2p-host-persistent-descriptorless ARCH="$ARCH" OUT="$(basename "$NODESC_BIN")" \
  bash "$BUILD_SCRIPT"

DESC_OUTPUT="$($DESC_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
NODESC_OUTPUT="$($NODESC_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
printf '%s\n' "$DESC_OUTPUT"
printf '%s\n' "$NODESC_OUTPUT"

for direction in forward reverse; do
  DESC_MS="$(printf '%s\n' "$DESC_OUTPUT" | awk -v want="$direction" '
    /gridfp-p2p-host-persistent-list/ {
      have=0; value="";
      for(i=1;i<=NF;++i){
        if($i=="direction=" want) have=1;
        if($i~/^runtime_ms=/){split($i,a,"=");value=a[2];}
      }
      if(have&&value!=""){print value;exit}
    }
  ')"
  NODESC_MS="$(printf '%s\n' "$NODESC_OUTPUT" | awk -v want="$direction" '
    /gridfp-p2p-host-persistent-descriptorless/ {
      have=0; value="";
      for(i=1;i<=NF;++i){
        if($i=="direction=" want) have=1;
        if($i~/^runtime_ms=/){split($i,a,"=");value=a[2];}
      }
      if(have&&value!=""){print value;exit}
    }
  ')"
  if [[ -z "$DESC_MS" || -z "$NODESC_MS" ]]; then
    echo "missing descriptor A/B timing direction=$direction" >&2
    exit 4
  fi
  SPEEDUP="$(awk -v a="$DESC_MS" -v b="$NODESC_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  echo "persistent-descriptor-ab direction=$direction descriptor_ms=$DESC_MS descriptorless_ms=$NODESC_MS descriptorless_speedup=$SPEEDUP"
done

if ! printf '%s\n' "$NODESC_OUTPUT" | grep -q 'ALL_OK gridfp_p2p_host_persistent_descriptorless=1'; then
  echo "descriptorless persistent probe missing ALL_OK" >&2
  exit 5
fi

echo "persistent-descriptor-ab exact=OK descriptor_bytes=0 scratch_allocator_atomics=0"
