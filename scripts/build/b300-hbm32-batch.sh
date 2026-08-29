#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
W=$((N + 1))
ARCH="${ARCH:-native}"
SRC="${SRC:-}"
if [[ -z "$SRC" ]]; then
  if (( N >= 27 )); then SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_forced2window_opt_batch.cu"; else SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN_batch.cu"; fi
fi
SRC="$(repo_path "$SRC")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_batch_n${N}}")"
LOW_LUT_K="${LOW_LUT_K:-}"
HIGH_LUT_K="${HIGH_LUT_K:-}"
MAIN_MATE_CACHE="${MAIN_MATE_CACHE:-1}"
MAIN_PULL="${MAIN_PULL:-1}"
BLOCK_PULL="${BLOCK_PULL:-1}"
BLOCK_MATE_CACHE="${BLOCK_MATE_CACHE:-1}"
if (( N >= 27 )); then DEFAULT_SHARD8=1; DEFAULT_LOW_DROP_CACHE=1; DEFAULT_LOW_DROP_CHUNK=1; DEFAULT_LOW_BLOCK_CACHE=1; DEFAULT_MAIN_PULL_ILP=2; else DEFAULT_SHARD8=0; DEFAULT_LOW_DROP_CACHE=0; DEFAULT_LOW_DROP_CHUNK=0; DEFAULT_LOW_BLOCK_CACHE=0; DEFAULT_MAIN_PULL_ILP=1; fi
FAST_SHARD_ADDRESS8="${FAST_SHARD_ADDRESS8:-$DEFAULT_SHARD8}"
LOW_DROP_CACHE="${LOW_DROP_CACHE:-$DEFAULT_LOW_DROP_CACHE}"
LOW_DROP_CHUNK="${LOW_DROP_CHUNK:-$DEFAULT_LOW_DROP_CHUNK}"
LOW_BLOCK_CACHE="${LOW_BLOCK_CACHE:-$DEFAULT_LOW_BLOCK_CACHE}"
MAIN_PULL_ILP="${MAIN_PULL_ILP:-$DEFAULT_MAIN_PULL_ILP}"
# Experimental A/B axes. Keep OFF until actual B300 residue+wall validation.
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
LOW_MAIN_RECURRENCE="${LOW_MAIN_RECURRENCE:-0}"
HIGH_MAIN_RECURRENCE="${HIGH_MAIN_RECURRENCE:-0}"
RUNTIME_THREADS="${RUNTIME_THREADS:-1}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"

for name in MAIN_MATE_CACHE MAIN_PULL BLOCK_PULL BLOCK_MATE_CACHE FAST_SHARD_ADDRESS8 LOW_DROP_CACHE LOW_DROP_CHUNK LOW_BLOCK_CACHE HIGH_DROP_CHUNK LOW_MAIN_RECURRENCE HIGH_MAIN_RECURRENCE RUNTIME_THREADS PTXAS_VERBOSE; do
  value="${!name}"; [[ "$value" == 0 || "$value" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }
done
case "$MAIN_PULL_ILP" in 1|2|3|4) ;; *) echo "MAIN_PULL_ILP must be 1, 2, 3, or 4" >&2; exit 2;; esac
if [[ "$MAIN_PULL" == 1 && "$MAIN_MATE_CACHE" != 1 ]]; then echo "MAIN_PULL=1 requires MAIN_MATE_CACHE=1" >&2; exit 2; fi
if [[ "$BLOCK_PULL" == 1 && "$MAIN_PULL" != 1 ]]; then echo "BLOCK_PULL=1 requires MAIN_PULL=1" >&2; exit 2; fi
if [[ "$BLOCK_MATE_CACHE" == 1 && "$BLOCK_PULL" != 1 ]]; then echo "BLOCK_MATE_CACHE=1 requires BLOCK_PULL=1" >&2; exit 2; fi
if [[ "$LOW_DROP_CACHE" == 1 && ( "$MAIN_PULL" != 1 || "$MAIN_MATE_CACHE" != 1 ) ]]; then echo "LOW_DROP_CACHE=1 requires MAIN_PULL=1 MAIN_MATE_CACHE=1" >&2; exit 2; fi
if [[ "$LOW_DROP_CHUNK" == 1 && "$LOW_DROP_CACHE" != 1 ]]; then echo "LOW_DROP_CHUNK=1 requires LOW_DROP_CACHE=1" >&2; exit 2; fi
if [[ "$LOW_BLOCK_CACHE" == 1 && ( "$LOW_DROP_CHUNK" != 1 || "$BLOCK_MATE_CACHE" != 1 ) ]]; then echo "LOW_BLOCK_CACHE=1 requires LOW_DROP_CHUNK=1 BLOCK_MATE_CACHE=1" >&2; exit 2; fi
if [[ "$HIGH_DROP_CHUNK" == 1 && ( "$LOW_DROP_CACHE" != 1 || "$BLOCK_PULL" != 1 ) ]]; then echo "HIGH_DROP_CHUNK=1 requires LOW_DROP_CACHE=1 BLOCK_PULL=1" >&2; exit 2; fi
if [[ "$LOW_MAIN_RECURRENCE" == 1 && ( "$LOW_DROP_CACHE" != 1 || "$LOW_DROP_CHUNK" != 1 || "$MAIN_MATE_CACHE" != 1 || "$MAIN_PULL" != 1 ) ]]; then echo "LOW_MAIN_RECURRENCE=1 requires LOW_DROP_CACHE=LOW_DROP_CHUNK=MAIN_MATE_CACHE=MAIN_PULL=1" >&2; exit 2; fi
if [[ "$HIGH_MAIN_RECURRENCE" == 1 && ( "$MAIN_PULL_ILP" != 2 || "$MAIN_MATE_CACHE" != 1 || "$MAIN_PULL" != 1 ) ]]; then echo "HIGH_MAIN_RECURRENCE=1 currently requires MAIN_PULL_ILP=2 MAIN_MATE_CACHE=MAIN_PULL=1" >&2; exit 2; fi
if [[ "$HIGH_MAIN_RECURRENCE" == 1 && "$LOW_MAIN_RECURRENCE" == 1 ]]; then echo "HIGH_MAIN_RECURRENCE and LOW_MAIN_RECURRENCE are separate A/B transforms for now" >&2; exit 2; fi
if [[ "$MAIN_PULL_ILP" != 1 && "$MAIN_PULL" != 1 ]]; then echo "MAIN_PULL_ILP=$MAIN_PULL_ILP requires MAIN_PULL=1" >&2; exit 2; fi

if [[ -z "$LOW_LUT_K" ]]; then if (( N >= 27 )); then LOW_LUT_K=14; else LOW_LUT_K=0; fi; fi
if [[ -z "$HIGH_LUT_K" ]]; then if (( N >= 27 )); then HIGH_LUT_K=13; else HIGH_LUT_K=0; fi; fi
if [[ "$LOW_DROP_CACHE" == 1 && ( "$W" != 28 || "$LOW_LUT_K" != 14 || "$HIGH_LUT_K" != 13 ) ]]; then echo "LOW_DROP_CACHE=1 is specialized for W=28 LOW_LUT_K=14 HIGH_LUT_K=13" >&2; exit 2; fi

if [[ "$MAIN_PULL" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-main-pull-operator-proof.sh"; bash "$ONEESAN_ROOT/scripts/bench/b300-main-pull-direct-pair-rank-proof.sh"; fi
if [[ "$BLOCK_PULL" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-operator-proof.sh"; bash "$ONEESAN_ROOT/scripts/bench/b300-group-rank-drop-insert-proof.sh"; fi
if [[ "$LOW_DROP_CACHE" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-low-window-drop-cache-proof.sh"; fi
if [[ "$LOW_DROP_CHUNK" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-low-drop-chunk-proof.sh"; fi
if [[ "$LOW_BLOCK_CACHE" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-low-block-cache-proof.sh"; fi
if [[ "$HIGH_DROP_CHUNK" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-high-drop-chunk-proof.sh"; fi
if [[ "$LOW_MAIN_RECURRENCE" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-low-main-recurrence-proof.sh"; fi
if [[ "$HIGH_MAIN_RECURRENCE" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-high-main-recurrence-proof.sh"; fi
if [[ "$MAIN_PULL_ILP" == 2 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-ilp2-partition-proof.sh"; fi
if [[ "$MAIN_PULL_ILP" == 3 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-main-pull-ilp3-partition-proof.sh"; fi
if [[ "$MAIN_PULL_ILP" == 4 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-ilp4-partition-proof.sh"; fi
if [[ "$FAST_SHARD_ADDRESS8" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-shard-address8-proof.sh"; fi

BUILD_SRC="$SRC"
if [[ "$MAIN_MATE_CACHE" == 1 ]]; then BUILD_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_main_mate_cache.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-mate-cache.py" "$SRC" "$BUILD_SRC"; fi
if [[ "$MAIN_PULL" == 1 ]]; then PULL_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_main_pull.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull.py" "$BUILD_SRC" "$PULL_SRC"; BUILD_SRC="$PULL_SRC"; fi
if [[ "$BLOCK_PULL" == 1 ]]; then FULL_PULL_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_full_pull.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull.py" "$BUILD_SRC" "$FULL_PULL_SRC"; BUILD_SRC="$FULL_PULL_SRC"; fi
if [[ "$BLOCK_MATE_CACHE" == 1 ]]; then BLOCK_CACHE_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_full_pull_block_cache.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-mate-cache.py" "$BUILD_SRC" "$BLOCK_CACHE_SRC"; BUILD_SRC="$BLOCK_CACHE_SRC"; fi
if [[ "$LOW_DROP_CACHE" == 1 ]]; then LOW_DROP_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_low_drop_cache.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-window-drop-cache.py" "$BUILD_SRC" "$LOW_DROP_SRC"; BUILD_SRC="$LOW_DROP_SRC"; fi
if [[ "$LOW_DROP_CHUNK" == 1 ]]; then LOW_DROP_CHUNK_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_low_drop_chunk.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-drop-chunk.py" "$BUILD_SRC" "$LOW_DROP_CHUNK_SRC"; BUILD_SRC="$LOW_DROP_CHUNK_SRC"; fi
if [[ "$HIGH_DROP_CHUNK" == 1 ]]; then HIGH_DROP_CHUNK_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_high_drop_chunk.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-high-drop-chunk.py" "$BUILD_SRC" "$HIGH_DROP_CHUNK_SRC"; BUILD_SRC="$HIGH_DROP_CHUNK_SRC"; fi
if [[ "$LOW_BLOCK_CACHE" == 1 ]]; then LOW_BLOCK_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_low_block_cache.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-block-cache.py" "$BUILD_SRC" "$LOW_BLOCK_SRC"; BUILD_SRC="$LOW_BLOCK_SRC"; fi
if [[ "$MAIN_PULL_ILP" == 2 ]]; then ILP_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_main_pull_ilp2.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull-ilp2.py" "$BUILD_SRC" "$ILP_SRC"; BUILD_SRC="$ILP_SRC"; fi
if [[ "$MAIN_PULL_ILP" == 3 ]]; then ILP_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_main_pull_ilp3.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull-ilp3.py" "$BUILD_SRC" "$ILP_SRC"; BUILD_SRC="$ILP_SRC"; fi
if [[ "$MAIN_PULL_ILP" == 4 ]]; then ILP_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_main_pull_ilp4.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull-ilp4.py" "$BUILD_SRC" "$ILP_SRC"; BUILD_SRC="$ILP_SRC"; fi
if [[ "$HIGH_MAIN_RECURRENCE" == 1 ]]; then HIGH_REC_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_high_main_recurrence.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-high-main-recurrence.py" "$BUILD_SRC" "$HIGH_REC_SRC"; BUILD_SRC="$HIGH_REC_SRC"; fi
if [[ "$LOW_MAIN_RECURRENCE" == 1 ]]; then LOW_REC_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_low_main_recurrence.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-main-recurrence.py" "$BUILD_SRC" "$LOW_REC_SRC"; BUILD_SRC="$LOW_REC_SRC"; fi
if [[ "$FAST_SHARD_ADDRESS8" == 1 ]]; then SHARD_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_shard8.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-batch-shard-address8.py" "$BUILD_SRC" "$SHARD_SRC"; BUILD_SRC="$SHARD_SRC"; fi
ROW_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_rowlimit.cu"; python3 "$ONEESAN_ROOT/scripts/build/lower-b300-batch-row-limit.py" "$BUILD_SRC" "$ROW_SRC"; BUILD_SRC="$ROW_SRC"
if [[ "$RUNTIME_THREADS" == 1 ]]; then THREAD_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_runtime_threads.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-runtime-threads.py" "$BUILD_SRC" "$THREAD_SRC"; BUILD_SRC="$THREAD_SRC"; fi

PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" "$BUILD_SRC" -o "$OUT"
echo "built $OUT"
echo "  source=$SRC"
echo "  build_source=$BUILD_SRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K main_mate_cache=$MAIN_MATE_CACHE main_pull=$MAIN_PULL main_pull_ilp=$MAIN_PULL_ILP block_pull=$BLOCK_PULL block_mate_cache=$BLOCK_MATE_CACHE low_drop_cache=$LOW_DROP_CACHE low_drop_chunk=$LOW_DROP_CHUNK low_block_cache=$LOW_BLOCK_CACHE high_drop_chunk=$HIGH_DROP_CHUNK low_main_recurrence=$LOW_MAIN_RECURRENCE high_main_recurrence=$HIGH_MAIN_RECURRENCE fast_shard_address8=$FAST_SHARD_ADDRESS8 runtime_threads=$RUNTIME_THREADS"
echo "  batch_row_limit_env=B300_ROW_LIMIT default_rows=$W calibration_default=0"
if [[ "$MAIN_PULL_ILP" != 1 ]]; then echo "  main_pull_destinations_per_thread=$MAIN_PULL_ILP memory_request_phases=mate,self,pair,block register_pressure_requires_ab=1"; fi
if [[ "$HIGH_DROP_CHUNK" == 1 ]]; then echo "  high_drop_chunk_table_bytes_per_gpu=184320 max_table_loads=3 max_scalar_tail=3 proof_gate=1"; fi
if [[ "$LOW_MAIN_RECURRENCE" == 1 ]]; then echo "  low_main_state_bits=60 extra_state_bytes=0 mate_hbm_store_per_state_step=8 low_drop_table_loads_per_state_step=0 low_height_popcounts_per_state_step=0 proof_gate=1"; fi
if [[ "$HIGH_MAIN_RECURRENCE" == 1 ]]; then echo "  high_main_state_bits=64 extra_state_bytes=0 mate_hbm_store_per_state_step=8 high_drop_walk_or_table_loads_per_state_step=0 high_height_walk_per_state_step=0 proof_gate=1"; fi
