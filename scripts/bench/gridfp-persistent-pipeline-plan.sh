#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O3 -std=c++17}"
NODE_HBM_TBPS="${NODE_HBM_TBPS:-64}"
NODE_NVLINK_TBPS="${NODE_NVLINK_TBPS:-14.4}"
MIN_B300_HEADROOM_GIB="${MIN_B300_HEADROOM_GIB:-18}"

PLAN_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_persistent_pipeline_plan_probe.cpp)"
ORDER_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_persistent_pipeline_order_probe.cpp)"
RACE_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_persistent_pipeline_race_probe.cpp)"
EVENT_RACE_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_persistent_event_race_probe.cpp)"
EAGER_LOAD_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_persistent_eager_load_probe.cpp)"
LEADER_HASH_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_leader_batch_hash_probe.cpp)"
B8_BOUND_SRC="$(repo_path src/cpp/probes/gridfp_reduced_production_persistent_b8_lower_bound_probe.cpp)"
PLAN_BIN="$(build_path gridfp_reduced_production_persistent_pipeline_plan_probe)"
ORDER_BIN="$(build_path gridfp_reduced_production_persistent_pipeline_order_probe)"
RACE_BIN="$(build_path gridfp_reduced_production_persistent_pipeline_race_probe)"
EVENT_RACE_BIN="$(build_path gridfp_reduced_production_persistent_event_race_probe)"
EAGER_LOAD_BIN="$(build_path gridfp_reduced_production_persistent_eager_load_probe)"
LEADER_HASH_BIN="$(build_path gridfp_reduced_production_leader_batch_hash_probe)"
B8_BOUND_BIN="$(build_path gridfp_reduced_production_persistent_b8_lower_bound_probe)"

# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$PLAN_SRC" -o "$PLAN_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$ORDER_SRC" -o "$ORDER_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$RACE_SRC" -o "$RACE_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$EVENT_RACE_SRC" -o "$EVENT_RACE_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$EAGER_LOAD_SRC" -o "$EAGER_LOAD_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$LEADER_HASH_SRC" -o "$LEADER_HASH_BIN"
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS "$B8_BOUND_SRC" -o "$B8_BOUND_BIN"

PLAN_OUTPUT="$($PLAN_BIN "$NODE_HBM_TBPS" "$NODE_NVLINK_TBPS")"
ORDER_OUTPUT="$($ORDER_BIN "$NODE_HBM_TBPS" "$NODE_NVLINK_TBPS")"
RACE_OUTPUT="$($RACE_BIN)"
EVENT_RACE_OUTPUT="$($EVENT_RACE_BIN)"
EAGER_LOAD_OUTPUT="$($EAGER_LOAD_BIN "$NODE_HBM_TBPS")"
LEADER_HASH_OUTPUT="$($LEADER_HASH_BIN)"
B8_BOUND_OUTPUT="$($B8_BOUND_BIN)"
printf '%s\n' "$PLAN_OUTPUT"
printf '%s\n' "$ORDER_OUTPUT"
printf '%s\n' "$RACE_OUTPUT"
printf '%s\n' "$EVENT_RACE_OUTPUT"
printf '%s\n' "$EAGER_LOAD_OUTPUT"
printf '%s\n' "$LEADER_HASH_OUTPUT"
printf '%s\n' "$B8_BOUND_OUTPUT"

if ! printf '%s\n' "$PLAN_OUTPUT" | grep -q 'ALL_OK production_persistent_pipeline_plan=1'; then
  echo "persistent pipeline plan missing ALL_OK" >&2
  exit 4
fi
if ! printf '%s\n' "$ORDER_OUTPUT" | grep -q 'ALL_OK production_persistent_pipeline_order=1 exact_dp=1'; then
  echo "persistent pipeline order DP missing ALL_OK" >&2
  exit 5
fi
if ! printf '%s\n' "$RACE_OUTPUT" | grep -q 'ALL_OK persistent_pipeline_race=1'; then
  echo "persistent pipeline race proof missing ALL_OK" >&2
  exit 8
fi
if ! printf '%s\n' "$EVENT_RACE_OUTPUT" | grep -q 'ALL_OK persistent_event_pipeline_race=1'; then
  echo "persistent event pipeline race proof missing ALL_OK" >&2
  exit 9
fi
if ! printf '%s\n' "$EAGER_LOAD_OUTPUT" | grep -q 'ALL_OK production_persistent_eager_load=1'; then
  echo "persistent eager load plan missing ALL_OK" >&2
  exit 10
fi
if ! printf '%s\n' "$LEADER_HASH_OUTPUT" | grep -q 'ALL_OK production_leader_batch_hash=1 exact_W28=1'; then
  echo "production leader batch hash audit missing ALL_OK" >&2
  exit 11
fi
if ! printf '%s\n' "$B8_BOUND_OUTPUT" | grep -q 'ALL_OK production_persistent_b8_lower_bound=1 B8_double_scratch_impossible=1'; then
  echo "B8 double-scratch lower-bound proof missing ALL_OK" >&2
  exit 14
fi

for direction in forward reverse; do
  HEADROOM="$(printf '%s\n' "$PLAN_OUTPUT" | awk -v want="$direction" '
    /persistent-pipeline-plan/ {
      have=0; value="";
      for(i=1;i<=NF;++i){
        if($i=="direction=" want) have=1;
        if($i~/^B300_headroom_GiB=/){split($i,a,"=");value=a[2];}
      }
      if(have&&value!=""){print value;exit}
    }
  ')"
  if [[ -z "$HEADROOM" ]]; then
    echo "pipeline plan missing headroom direction=$direction" >&2
    exit 6
  fi
  if ! awk -v got="$HEADROOM" -v want="$MIN_B300_HEADROOM_GIB" \
      'BEGIN { exit !(got + 0 >= want + 0) }'; then
    echo "pipeline B300 headroom too small direction=$direction got=$HEADROOM want>=$MIN_B300_HEADROOM_GIB" >&2
    exit 7
  fi
done

BAL_HEADROOM="$(printf '%s\n' "$LEADER_HASH_OUTPUT" | awk '
  /leader-batch-hash/ && /scheme=leader-shift12/ {
    for(i=1;i<=NF;++i) if($i~/^B300_headroom_GiB=/){split($i,a,"=");print a[2];exit}
  }
')"
if [[ -z "$BAL_HEADROOM" ]]; then
  echo "leader batch hash audit missing balanced B300 headroom" >&2
  exit 12
fi
if ! awk -v got="$BAL_HEADROOM" -v want="20" 'BEGIN { exit !(got + 0 >= want + 0) }'; then
  echo "balanced leader hash B300 headroom unexpectedly small got=$BAL_HEADROOM" >&2
  exit 13
fi

echo "persistent-pipeline-plan exact=OK batches=16 double_scratch=1 B8_double_scratch_impossible=1 minimal_capacity_batch_count=16 exact_order_dp=1 overlap_race_proof=1 event_fanin_race_proof=1 eager_load_smoothing=1 balanced_leader_hash=1 batch_hash_shift=12 node_HBM_TBps=$NODE_HBM_TBPS node_NVLink_TBps=$NODE_NVLINK_TBPS"
