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
HOST_BIN="$(build_path gridfp_reduced_component_p2p-pipeline-host-schedule-abc)"
STAGED_BIN="$(build_path gridfp_reduced_component_p2p-event-staged-schedule-abc)"
EAGER_BIN="$(build_path gridfp_reduced_component_p2p-event-eager-schedule-abc)"
FANIN_BIN="$(build_path gridfp_reduced_production_cross_device_event_fanin_schedule_abc)"
FANIN_SRC="$(repo_path src/cuda/gridfp/gridfp_reduced_production_cross_device_event_fanin_probe.cu)"

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "$FANIN_SRC" -o "$FANIN_BIN"
VARIANT=segment ARCH="$ARCH" PTXAS_VERBOSE=0 OUT="$(basename "$HOST_BIN")" \
  bash "$PIPE_BUILD"
SCHEDULE=staged ARCH="$ARCH" PTXAS_VERBOSE=0 OUT="$(basename "$STAGED_BIN")" \
  bash "$EVENT_BUILD"
SCHEDULE=eager ARCH="$ARCH" PTXAS_VERBOSE=0 OUT="$(basename "$EAGER_BIN")" \
  bash "$EVENT_BUILD"

FANIN_OUTPUT="$($FANIN_BIN "$NGPU")"
printf '%s\n' "$FANIN_OUTPUT"
grep -q 'ALL_OK gridfp_cross_device_event_fanin=1' <<<"$FANIN_OUTPUT" || exit 3

HOST_OUTPUT="$($HOST_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
STAGED_OUTPUT="$($STAGED_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
EAGER_OUTPUT="$($EAGER_BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
printf '%s\n' "$HOST_OUTPUT"
printf '%s\n' "$STAGED_OUTPUT"
printf '%s\n' "$EAGER_OUTPUT"

grep -q 'ALL_OK gridfp_p2p_host_persistent_pipeline=1' <<<"$HOST_OUTPUT" || exit 4
grep -q 'ALL_OK gridfp_p2p_host_persistent_event_pipeline=1' <<<"$STAGED_OUTPUT" || exit 5
grep -q 'ALL_OK gridfp_p2p_host_persistent_event_pipeline=1' <<<"$EAGER_OUTPUT" || exit 6
grep -q 'event_schedule=staged' <<<"$STAGED_OUTPUT" || exit 7
grep -q 'adjacent_A_overlap=0' <<<"$STAGED_OUTPUT" || exit 8
grep -q 'event_schedule=eager' <<<"$EAGER_OUTPUT" || exit 9
grep -q 'adjacent_A_overlap=1' <<<"$EAGER_OUTPUT" || exit 10

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
  STAGED_MS="$(extract_ms "$STAGED_OUTPUT" gridfp-p2p-host-persistent-event-pipeline "$direction")"
  EAGER_MS="$(extract_ms "$EAGER_OUTPUT" gridfp-p2p-host-persistent-event-pipeline "$direction")"
  if [[ -z "$HOST_MS" || -z "$STAGED_MS" || -z "$EAGER_MS" ]]; then
    echo "missing event schedule A/B/C timing direction=$direction" >&2
    exit 11
  fi
  STAGED_SPEEDUP="$(awk -v a="$HOST_MS" -v b="$STAGED_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  EAGER_SPEEDUP="$(awk -v a="$HOST_MS" -v b="$EAGER_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  EAGER_VS_STAGED="$(awk -v a="$STAGED_MS" -v b="$EAGER_MS" 'BEGIN{if(b>0)printf "%.6f",a/b;else print "0"}')"
  echo "persistent-event-schedule-abc direction=$direction batches=$BATCHES host_ms=$HOST_MS staged_ms=$STAGED_MS eager_ms=$EAGER_MS staged_speedup=$STAGED_SPEEDUP eager_speedup=$EAGER_SPEEDUP eager_vs_staged=$EAGER_VS_STAGED"
done

echo "persistent-event-schedule-abc exact=OK batches=$BATCHES event_fanin=1 schedules=host,staged,eager"
