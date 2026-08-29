#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
W=$((N + 1))
ARCH="${ARCH:-native}"
SRC="$(repo_path "${SRC:-src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu}")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_n${N}}")"
LOW_LUT_K="${LOW_LUT_K:-}"
HIGH_LUT_K="${HIGH_LUT_K:-}"
FAST_SHARD_ADDRESS8="${FAST_SHARD_ADDRESS8:-0}"
MAIN_MATE_CACHE="${MAIN_MATE_CACHE:-1}"
MAIN_PULL="${MAIN_PULL:-0}"
BLOCK_PULL="${BLOCK_PULL:-0}"
BLOCK_MATE_CACHE="${BLOCK_MATE_CACHE:-$BLOCK_PULL}"
MAIN_PULL_ILP2="${MAIN_PULL_ILP2:-0}"
HEIGHT_CACHE="${HEIGHT_CACHE:-0}"
RANK_DELTA_CACHE="${RANK_DELTA_CACHE:-0}"
RANK_STATE_PACKED="${RANK_STATE_PACKED:-0}"
RANK_STATE_ILP2="${RANK_STATE_ILP2:-0}"
RANK_STATE_ILP4="${RANK_STATE_ILP4:-0}"
BLOCK_CLOSURE_QUAD="${BLOCK_CLOSURE_QUAD:-0}"
HOT_DELTA_TABLE="${HOT_DELTA_TABLE:-0}"
CONCURRENT_GROUP_IO="${CONCURRENT_GROUP_IO:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"

for name in FAST_SHARD_ADDRESS8 MAIN_MATE_CACHE MAIN_PULL BLOCK_PULL BLOCK_MATE_CACHE MAIN_PULL_ILP2 HEIGHT_CACHE RANK_DELTA_CACHE RANK_STATE_PACKED RANK_STATE_ILP2 RANK_STATE_ILP4 BLOCK_CLOSURE_QUAD HOT_DELTA_TABLE CONCURRENT_GROUP_IO PTXAS_VERBOSE; do
  value="${!name}"
  if [[ "$value" != 0 && "$value" != 1 ]]; then echo "$name must be 0 or 1" >&2; exit 2; fi
done
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && (( MAXRREGCOUNT == 0 || (MAXRREGCOUNT >= 32 && MAXRREGCOUNT <= 255) )) || { echo "MAXRREGCOUNT must be 0 or 32..255" >&2; exit 2; }
if [[ "$MAIN_PULL" == 1 && "$MAIN_MATE_CACHE" != 1 ]]; then echo "MAIN_PULL=1 currently requires MAIN_MATE_CACHE=1" >&2; exit 2; fi
if [[ "$BLOCK_PULL" == 1 && "$MAIN_PULL" != 1 ]]; then echo "BLOCK_PULL=1 requires MAIN_PULL=1" >&2; exit 2; fi
if [[ "$BLOCK_MATE_CACHE" == 1 && "$BLOCK_PULL" != 1 ]]; then echo "BLOCK_MATE_CACHE=1 requires BLOCK_PULL=1" >&2; exit 2; fi
for name in MAIN_PULL_ILP2 HEIGHT_CACHE RANK_DELTA_CACHE; do
  if [[ "${!name}" == 1 && ( "$MAIN_PULL" != 1 || "$BLOCK_PULL" != 1 || "$MAIN_MATE_CACHE" != 1 || "$BLOCK_MATE_CACHE" != 1 ) ]]; then echo "$name=1 requires full-pull plus both MateID caches" >&2; exit 2; fi
done
if (( MAIN_PULL_ILP2 + HEIGHT_CACHE + RANK_DELTA_CACHE > 1 )); then echo "MAIN_PULL_ILP2, HEIGHT_CACHE and RANK_DELTA_CACHE are separate base experiments; use packed rank-state ILP2/4 to combine ILP with rank recurrence" >&2; exit 2; fi
if [[ "$RANK_STATE_PACKED" == 1 && "$RANK_DELTA_CACHE" != 1 ]]; then echo "RANK_STATE_PACKED=1 requires RANK_DELTA_CACHE=1" >&2; exit 2; fi
if [[ "$RANK_STATE_ILP2" == 1 && "$RANK_STATE_PACKED" != 1 ]]; then echo "RANK_STATE_ILP2=1 requires RANK_STATE_PACKED=1" >&2; exit 2; fi
if [[ "$RANK_STATE_ILP4" == 1 && "$RANK_STATE_PACKED" != 1 ]]; then echo "RANK_STATE_ILP4=1 requires RANK_STATE_PACKED=1" >&2; exit 2; fi
if (( RANK_STATE_ILP2 + RANK_STATE_ILP4 > 1 )); then echo "RANK_STATE_ILP2 and RANK_STATE_ILP4 are mutually exclusive" >&2; exit 2; fi
if [[ "$BLOCK_CLOSURE_QUAD" == 1 && "$RANK_STATE_ILP4" != 1 ]]; then echo "BLOCK_CLOSURE_QUAD=1 requires RANK_STATE_ILP4=1" >&2; exit 2; fi
if [[ "$HOT_DELTA_TABLE" == 1 && "$RANK_DELTA_CACHE" != 1 ]]; then echo "HOT_DELTA_TABLE=1 requires RANK_DELTA_CACHE=1" >&2; exit 2; fi

if [[ -z "$LOW_LUT_K" ]]; then if (( N >= 24 )); then LOW_LUT_K=13; else LOW_LUT_K=0; fi; fi
if [[ -z "$HIGH_LUT_K" ]]; then if (( N >= 24 )); then HIGH_LUT_K=13; else HIGH_LUT_K=0; fi; fi
if (( LOW_LUT_K > W || HIGH_LUT_K > W - 1 )); then echo "LUT K exceeds width: n=$N W=$W low=$LOW_LUT_K high=$HIGH_LUT_K" >&2; exit 2; fi

if [[ "$FAST_SHARD_ADDRESS8" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-shard-address8-proof.sh"; fi
if [[ "$MAIN_PULL" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-main-pull-operator-proof.sh"; fi
if [[ "$BLOCK_PULL" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-operator-proof.sh"
  bash "$ONEESAN_ROOT/scripts/bench/b300-group-rank-drop-insert-proof.sh"
  bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-rank-incremental-proof.sh"
fi
if [[ "$MAIN_PULL_ILP2" == 1 || "$RANK_STATE_ILP2" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-ilp2-partition-proof.sh"; fi
if [[ "$RANK_STATE_ILP4" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-ilp4-partition-proof.sh"; fi
if [[ "$HEIGHT_CACHE" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-height-recurrence-proof.sh"; fi
if [[ "$RANK_DELTA_CACHE" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/b300-rank-delta-recurrence-proof.sh"
  bash "$ONEESAN_ROOT/scripts/bench/b300-rank-delta-window-free-proof.sh"
fi
if [[ "$RANK_STATE_PACKED" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/b300-height-recurrence-proof.sh"
  bash "$ONEESAN_ROOT/scripts/bench/b300-rank-state-i56-bound-proof.sh"
fi

BUILD_SRC="$SRC"
if [[ "$MAIN_MATE_CACHE" == 1 ]]; then BUILD_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_main_mate_cache.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-mate-cache.py" "$SRC" "$BUILD_SRC";fi
if [[ "$MAIN_PULL" == 1 ]]; then PULL_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_main_pull.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull.py" "$BUILD_SRC" "$PULL_SRC";BUILD_SRC="$PULL_SRC";fi
if [[ "$BLOCK_PULL" == 1 ]]; then BLOCK_PULL_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_full_pull.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull.py" "$BUILD_SRC" "$BLOCK_PULL_SRC";BUILD_SRC="$BLOCK_PULL_SRC";fi
if [[ "$BLOCK_MATE_CACHE" == 1 ]]; then BLOCK_MATE_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_full_pull_block_mate.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-mate-cache.py" "$BUILD_SRC" "$BLOCK_MATE_SRC";BUILD_SRC="$BLOCK_MATE_SRC";fi
if [[ "$MAIN_PULL_ILP2" == 1 ]]; then ILP_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_main_ilp2.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull-ilp2.py" "$BUILD_SRC" "$ILP_SRC";BUILD_SRC="$ILP_SRC";fi
if [[ "$HEIGHT_CACHE" == 1 ]]; then HEIGHT_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_height_cache.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-height-cache.py" "$BUILD_SRC" "$HEIGHT_SRC";BUILD_SRC="$HEIGHT_SRC";fi
if [[ "$RANK_DELTA_CACHE" == 1 ]]; then
  RANK_DELTA_INPUT_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_rank_delta_input.cu";python3 "$ONEESAN_ROOT/scripts/build/normalize-b300-rank-delta-input.py" "$BUILD_SRC" "$RANK_DELTA_INPUT_SRC"
  RANK_DELTA_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_rank_delta_cache.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-rank-delta-cache.py" "$RANK_DELTA_INPUT_SRC" "$RANK_DELTA_SRC"
  RANK_DELTA_FREE_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_rank_delta_free_step.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-rank-delta-free-step.py" "$RANK_DELTA_SRC" "$RANK_DELTA_FREE_SRC"
  RANK_DELTA_REPORT_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_rank_delta_report.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-rank-delta-report.py" "$RANK_DELTA_FREE_SRC" "$RANK_DELTA_REPORT_SRC";BUILD_SRC="$RANK_DELTA_REPORT_SRC"
  if [[ "$RANK_STATE_PACKED" == 1 ]]; then RANK_STATE_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_rank_state_packed.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-rank-state-packed.py" "$BUILD_SRC" "$RANK_STATE_SRC";BUILD_SRC="$RANK_STATE_SRC";fi
  if [[ "$RANK_STATE_ILP2" == 1 ]]; then
    RANK_STATE_ILP2_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_rank_state_ilp2.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-rank-state-ilp2.py" "$BUILD_SRC" "$RANK_STATE_ILP2_SRC";BUILD_SRC="$RANK_STATE_ILP2_SRC"
    BLOCK_RANK_STATE_ILP2_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_rank_state_ilp2_block.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-rank-state-ilp2.py" "$BUILD_SRC" "$BLOCK_RANK_STATE_ILP2_SRC";BUILD_SRC="$BLOCK_RANK_STATE_ILP2_SRC"
  elif [[ "$RANK_STATE_ILP4" == 1 ]]; then
    RANK_STATE_ILP4_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_rank_state_ilp4.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-rank-state-ilp4.py" "$BUILD_SRC" "$RANK_STATE_ILP4_SRC";BUILD_SRC="$RANK_STATE_ILP4_SRC"
    BLOCK_RANK_STATE_ILP4_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_rank_state_ilp4_block.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-rank-state-ilp4.py" "$BUILD_SRC" "$BLOCK_RANK_STATE_ILP4_SRC";BUILD_SRC="$BLOCK_RANK_STATE_ILP4_SRC"
  fi
  if [[ "$HOT_DELTA_TABLE" == 1 ]]; then
    HOT_DELTA_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_hot_delta_table.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-hot-delta-table.py" "$BUILD_SRC" "$HOT_DELTA_SRC";BUILD_SRC="$HOT_DELTA_SRC"
  fi
fi
if [[ "$CONCURRENT_GROUP_IO" == 1 ]]; then CONCURRENT_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_concurrent_group_io.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-concurrent-group-io.py" "$BUILD_SRC" "$CONCURRENT_SRC";BUILD_SRC="$CONCURRENT_SRC";fi

ROW_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_rowlimit.cu";cp "$BUILD_SRC" "$ROW_SRC";python3 "$ONEESAN_ROOT/scripts/build/lower-b300-row-limit.py" "$ROW_SRC" "$ROW_SRC";BUILD_SRC="$ROW_SRC"
THREAD_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_runtime_threads.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-runtime-threads.py" "$BUILD_SRC" "$THREAD_SRC";BUILD_SRC="$THREAD_SRC"
PLAN_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_plan_target.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-plan-target.py" "$BUILD_SRC" "$PLAN_SRC";BUILD_SRC="$PLAN_SRC"

PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
REG_FLAGS=(); (( MAXRREGCOUNT > 0 )) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" -DB300_FAST_SHARD_ADDRESS8="$FAST_SHARD_ADDRESS8" -DB300_BLOCK_CLOSURE_QUAD="$BLOCK_CLOSURE_QUAD" "$BUILD_SRC" -o "$OUT"

echo "built $OUT"
echo "  source=$SRC"
echo "  build_source=$BUILD_SRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K fast_shard_address8=$FAST_SHARD_ADDRESS8 main_mate_cache=$MAIN_MATE_CACHE main_pull=$MAIN_PULL block_pull=$BLOCK_PULL block_mate_cache=$BLOCK_MATE_CACHE main_pull_ilp2=$MAIN_PULL_ILP2 height_cache=$HEIGHT_CACHE rank_delta_cache=$RANK_DELTA_CACHE rank_state_packed=$RANK_STATE_PACKED rank_state_ilp2=$RANK_STATE_ILP2 rank_state_ilp4=$RANK_STATE_ILP4 block_closure_quad=$BLOCK_CLOSURE_QUAD hot_delta_table=$HOT_DELTA_TABLE concurrent_group_io=$CONCURRENT_GROUP_IO maxrregcount=$MAXRREGCOUNT ptxas_verbose=$PTXAS_VERBOSE"
echo "  row_limit_env=B300_ROW_LIMIT default_rows=$W runtime_threads_env=GRIDFP_THREADS default_threads=256 planner_target_env=GRIDFP_PLAN_TARGET_MIB scratch_target_separate=1"
echo "  block_closure_scan=endpoint_setbits block_closure_candidate_rank=incremental_delta rank_same_calls_per_closure_candidate=0"
if [[ "$RANK_STATE_ILP2" == 1 ]]; then echo "  rank_state_main_destinations_per_thread=2 rank_state_block_destinations_per_thread=2 rank_state_index_first=1 rank_state_hbm_request_overlap=pair,block,endpoint,closure,two_destinations register_pressure_requires_ab=1";fi
if [[ "$RANK_STATE_ILP4" == 1 ]]; then echo "  rank_state_main_destinations_per_thread=4 rank_state_block_destinations_per_thread=4 rank_state_index_first=1 rank_state_hbm_request_overlap=pair,block,endpoint,four_destinations closure_slow_path=noinline closure_quad=$BLOCK_CLOSURE_QUAD register_pressure_requires_ab=1";fi
if [[ "$HOT_DELTA_TABLE" == 1 ]]; then echo "  hot_delta_bits=32 hot_delta_constant_bytes_added=17400 hot_delta_step_n_stored=0 hot_delta_int32_checked_per_group=1 rank_step_constant_loads=1 pair_rank_constant_loads=1";fi
if [[ "$CONCURRENT_GROUP_IO" == 1 ]]; then echo "  group_io_main_block_overlap=1 mate_materialize_overlap=1 rank_state_init_overlap=1 scatter_overlap=1 devicewide_group_io_sync=0";fi
