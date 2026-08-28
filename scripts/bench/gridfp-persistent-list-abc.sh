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
CYCLE_BIN="$(build_path gridfp_reduced_component_p2p-cycle-owner-descriptorless)"

for spec in \
  "p2p-host-persistent-list:$DESC_BIN" \
  "p2p-host-persistent-descriptorless:$NODESC_BIN" \
  "p2p-cycle-owner-descriptorless:$CYCLE_BIN"; do
  mode="${spec%%:*}"
  bin="${spec#*:}"
  MODE="$mode" ARCH="$ARCH" OUT="$(basename "$bin")" bash "$BUILD_SCRIPT"
done

DESC_OUTPUT="$($DESC_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
NODESC_OUTPUT="$($NODESC_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
CYCLE_OUTPUT="$($CYCLE_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
printf '%s\n' "$DESC_OUTPUT"
printf '%s\n' "$NODESC_OUTPUT"
printf '%s\n' "$CYCLE_OUTPUT"

for direction in forward reverse; do
  DESC_LINE="$(printf '%s\n' "$DESC_OUTPUT" | grep "gridfp-p2p-host-persistent-list .*direction=$direction " | head -1)"
  NODESC_LINE="$(printf '%s\n' "$NODESC_OUTPUT" | grep "gridfp-p2p-host-persistent-descriptorless .*direction=$direction " | head -1)"
  CYCLE_LINE="$(printf '%s\n' "$CYCLE_OUTPUT" | grep "gridfp-p2p-cycle-owner-descriptorless .*direction=$direction " | head -1)"
  if [[ -z "$DESC_LINE" || -z "$NODESC_LINE" || -z "$CYCLE_LINE" ]]; then
    echo "persistent A/B/C missing result direction=$direction" >&2
    exit 4
  fi

  field() {
    local line="$1" key="$2"
    printf '%s\n' "$line" | awk -v k="$key" '{
      for(i=1;i<=NF;++i){
        split($i,a,"=");
        if(a[1]==k){print a[2]; exit}
      }
    }'
  }

  DESC_MS="$(field "$DESC_LINE" runtime_ms)"
  NODESC_MS="$(field "$NODESC_LINE" runtime_ms)"
  CYCLE_MS="$(field "$CYCLE_LINE" runtime_ms)"
  DESC_LIST="$(field "$DESC_LINE" list_entries)"
  NODESC_LIST="$(field "$NODESC_LINE" list_entries)"
  CYCLE_LIST="$(field "$CYCLE_LINE" total_list_entries)"
  if [[ -z "$DESC_MS" || -z "$NODESC_MS" || -z "$CYCLE_MS" ||
        -z "$DESC_LIST" || -z "$NODESC_LIST" || -z "$CYCLE_LIST" ]]; then
    echo "persistent A/B/C missing fields direction=$direction" >&2
    exit 5
  fi

  NODESC_SPEEDUP="$(awk -v a="$DESC_MS" -v b="$NODESC_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  CYCLE_SPEEDUP="$(awk -v a="$DESC_MS" -v b="$CYCLE_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  LIST_COMPRESSION="$(awk -v a="$DESC_LIST" -v b="$CYCLE_LIST" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  echo "persistent-list-abc direction=$direction descriptor_ms=$DESC_MS descriptorless_ms=$NODESC_MS cycle_owner_ms=$CYCLE_MS descriptorless_speedup=$NODESC_SPEEDUP cycle_owner_speedup=$CYCLE_SPEEDUP descriptor_list_entries=$DESC_LIST descriptorless_list_entries=$NODESC_LIST cycle_owner_list_entries=$CYCLE_LIST cycle_owner_list_compression=$LIST_COMPRESSION"
done

for marker in \
  'ALL_OK gridfp_p2p_host_persistent_list=1' \
  'ALL_OK gridfp_p2p_host_persistent_descriptorless=1' \
  'ALL_OK gridfp_p2p_cycle_owner_descriptorless=1'; do
  case "$marker" in
    *cycle_owner*) haystack="$CYCLE_OUTPUT" ;;
    *descriptorless*) haystack="$NODESC_OUTPUT" ;;
    *) haystack="$DESC_OUTPUT" ;;
  esac
  if ! printf '%s\n' "$haystack" | grep -q "$marker"; then
    echo "persistent A/B/C missing marker: $marker" >&2
    exit 6
  fi
done

echo "persistent-list-abc exact=OK startup_gpu_support_scan_passes=0 descriptorless_candidate=1 cycle_owner_compression_candidate=1"
