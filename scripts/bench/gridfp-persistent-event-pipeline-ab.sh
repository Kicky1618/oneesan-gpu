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

PIPE_BUILD="$(repo_path scripts/build/gridfp-persistent-pipeline-probe.sh)"
EVENT_BUILD="$(repo_path scripts/build/gridfp-persistent-event-pipeline-probe.sh)"
HOST_BIN="$(build_path gridfp_reduced_component_p2p-host-persistent-pipeline-event-ab)"
EVENT_BIN="$(build_path gridfp_reduced_component_p2p-host-persistent-event-pipeline-ab)"
CAP_BIN="$(build_path gridfp_reduced_production_cross_device_event_probe)"
CAP_SRC="$(repo_path src/cuda/gridfp/gridfp_reduced_production_cross_device_event_probe.cu)"

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "$CAP_SRC" -o "$CAP_BIN"
VARIANT=segment ARCH="$ARCH" PTXAS_VERBOSE=0 OUT="$(basename "$HOST_BIN")" \
  bash "$PIPE_BUILD"
ARCH="$ARCH" PTXAS_VERBOSE=0 OUT="$(basename "$EVENT_BIN")" \
  bash "$EVENT_BUILD"

if (( NGPU < 2 )); then
  echo "NGPU must be >=2" >&2
  exit 2
fi

for ((g=1; g<NGPU; ++g)); do
  CAP_OUTPUT="$($CAP_BIN 0 "$g")"
  printf '%s\n' "$CAP_OUTPUT"
  if ! grep -q 'ALL_OK gridfp_cross_device_event=1' <<<"$CAP_OUTPUT"; then
    echo "cross-device event capability failed gpu0<->gpu$g" >&2
    exit 3
  fi
done

echo "persistent-event-star-capability ngpu=$NGPU exact=OK"

HOST_OUTPUT="$($HOST_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
EVENT_OUTPUT="$($EVENT_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
printf '%s\n' "$HOST_OUTPUT"
printf '%s\n' "$EVENT_OUTPUT"

grep -q 'ALL_OK gridfp_p2p_host_persistent_pipeline=1' <<<"$HOST_OUTPUT" || exit 4
grep -q 'ALL_OK gridfp_p2p_host_persistent_event_pipeline=1' <<<"$EVENT_OUTPUT" || exit 5
grep -q 'host_batch_barriers=0' <<<"$EVENT_OUTPUT" || exit 6
grep -q 'cross_device_events=1' <<<"$EVENT_OUTPUT" || exit 7

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

for direction in forward reverse; do
  HOST_MS="$(extract_ms "$HOST_OUTPUT" gridfp-p2p-host-persistent-pipeline "$direction")"
  EVENT_MS="$(extract_ms "$EVENT_OUTPUT" gridfp-p2p-host-persistent-event-pipeline "$direction")"
  if [[ -z "$HOST_MS" || -z "$EVENT_MS" ]]; then
    echo "missing event pipeline timing direction=$direction" >&2
    exit 8
  fi
  SPEEDUP="$(awk -v a="$HOST_MS" -v b="$EVENT_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  echo "persistent-event-pipeline-ab direction=$direction batches=$BATCHES host_barrier_ms=$HOST_MS event_fanin_ms=$EVENT_MS event_speedup=$SPEEDUP"
done

echo "persistent-event-pipeline-ab exact=OK batches=$BATCHES host_batch_barriers=0 event_barrier_fanin=1"
