#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
MOD="${2:-4294967291}"
TARGET_MIB_WAS_SET="${TARGET_MIB+x}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-8}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
ROW7_TENSOR="${ROW7_TENSOR:-0}"
ROW8_TENSOR="${ROW8_TENSOR:-0}"
ROW8_STRUCTURAL="${ROW8_STRUCTURAL:-0}"
OWNERFUSED="${OWNERFUSED:-auto}"
BLOCKFUSED="${BLOCKFUSED:-0}"
GROUPBATCH="${GROUPBATCH:-0}"
GRIDFP_GROUPBATCH_DYNAMIC="${GRIDFP_GROUPBATCH_DYNAMIC:-1}"
GRIDFP_GROUPBATCH_AFFINITY="${GRIDFP_GROUPBATCH_AFFINITY:-1}"
GRIDFP_GROUPBATCH_STICKY="${GRIDFP_GROUPBATCH_STICKY:-1}"
GRIDFP_GROUPBATCH_RECLAIM="${GRIDFP_GROUPBATCH_RECLAIM:-1}"
GRIDFP_GROUPBATCH_GRAPH="${GRIDFP_GROUPBATCH_GRAPH:-1}"
GRIDFP_GROUPBATCH_P1_DEST="${GRIDFP_GROUPBATCH_P1_DEST:-1}"
GRIDFP_GROUPBATCH_WRAP32="${GRIDFP_GROUPBATCH_WRAP32:-0}"
GRIDFP_GROUPBATCH_MATCHLUT="${GRIDFP_GROUPBATCH_MATCHLUT:-0}"
GRIDFP_GROUPBATCH_BATCHES_PER_GPU="${GRIDFP_GROUPBATCH_BATCHES_PER_GPU:-6}"
if (( GROUPBATCH && N >= 27 )) && [[ -z "$TARGET_MIB_WAS_SET" ]]; then TARGET_MIB=10240; fi
require_uint ROW7_TENSOR "$ROW7_TENSOR" || exit 2
require_uint ROW8_TENSOR "$ROW8_TENSOR" || exit 2
require_uint ROW8_STRUCTURAL "$ROW8_STRUCTURAL" || exit 2
require_uint BLOCKFUSED "$BLOCKFUSED" || exit 2
require_uint GROUPBATCH "$GROUPBATCH" || exit 2
if (( ROW7_TENSOR != 0 && ROW7_TENSOR != 1 )); then echo "ROW7_TENSOR must be 0 or 1" >&2; exit 2; fi
if (( ROW8_TENSOR != 0 && ROW8_TENSOR != 1 )); then echo "ROW8_TENSOR must be 0 or 1" >&2; exit 2; fi
if (( ROW8_STRUCTURAL != 0 && ROW8_STRUCTURAL != 1 )); then echo "ROW8_STRUCTURAL must be 0 or 1" >&2; exit 2; fi
if (( BLOCKFUSED != 0 && BLOCKFUSED != 1 )); then echo "BLOCKFUSED must be 0 or 1" >&2; exit 2; fi
if (( GROUPBATCH != 0 && GROUPBATCH != 1 )); then echo "GROUPBATCH must be 0 or 1" >&2; exit 2; fi
if (( GROUPBATCH )); then
  for spec in "GRIDFP_GROUPBATCH_DYNAMIC:$GRIDFP_GROUPBATCH_DYNAMIC" "GRIDFP_GROUPBATCH_AFFINITY:$GRIDFP_GROUPBATCH_AFFINITY" "GRIDFP_GROUPBATCH_STICKY:$GRIDFP_GROUPBATCH_STICKY" "GRIDFP_GROUPBATCH_RECLAIM:$GRIDFP_GROUPBATCH_RECLAIM" "GRIDFP_GROUPBATCH_GRAPH:$GRIDFP_GROUPBATCH_GRAPH" "GRIDFP_GROUPBATCH_P1_DEST:$GRIDFP_GROUPBATCH_P1_DEST" "GRIDFP_GROUPBATCH_WRAP32:$GRIDFP_GROUPBATCH_WRAP32" "GRIDFP_GROUPBATCH_MATCHLUT:$GRIDFP_GROUPBATCH_MATCHLUT" "GRIDFP_GROUPBATCH_BATCHES_PER_GPU:$GRIDFP_GROUPBATCH_BATCHES_PER_GPU"; do require_uint "${spec%%:*}" "${spec#*:}" || exit 2; done
  if (( GRIDFP_GROUPBATCH_DYNAMIC > 1 || GRIDFP_GROUPBATCH_AFFINITY > 1 || GRIDFP_GROUPBATCH_STICKY > 1 || GRIDFP_GROUPBATCH_RECLAIM > 1 )); then echo "GRIDFP_GROUPBATCH_DYNAMIC/AFFINITY/STICKY/RECLAIM must be 0 or 1" >&2; exit 2; fi
  if (( GRIDFP_GROUPBATCH_GRAPH > 2 )); then echo "GRIDFP_GROUPBATCH_GRAPH must be 0, 1, or 2" >&2; exit 2; fi
  if (( GRIDFP_GROUPBATCH_P1_DEST > 1 )); then echo "GRIDFP_GROUPBATCH_P1_DEST must be 0 or 1" >&2; exit 2; fi
  if (( GRIDFP_GROUPBATCH_WRAP32 > 1 )); then echo "GRIDFP_GROUPBATCH_WRAP32 must be 0 or 1" >&2; exit 2; fi
  if (( GRIDFP_GROUPBATCH_MATCHLUT > 1 )); then echo "GRIDFP_GROUPBATCH_MATCHLUT must be 0 or 1" >&2; exit 2; fi
fi
if (( ROW8_STRUCTURAL && ! ROW8_TENSOR )); then echo "ROW8_STRUCTURAL requires ROW8_TENSOR=1" >&2; exit 2; fi
if (( ROW7_TENSOR && ROW8_TENSOR )); then echo "ROW7_TENSOR and ROW8_TENSOR are mutually exclusive" >&2; exit 2; fi
if [[ "$OWNERFUSED" == "auto" ]]; then
  OWNERFUSED=0
  if (( N >= 20 && N < 27 && ! BLOCKFUSED && ! GROUPBATCH && ! ROW7_TENSOR && ! ROW8_TENSOR )) && grep -Eq "(^|[^0-9])${MOD}u([^0-9]|$)" "$ONEESAN_ROOT/src/cuda/b300/row6_automaton_crt20_generated.hpp"; then OWNERFUSED=1; fi
fi
require_uint OWNERFUSED "$OWNERFUSED" || exit 2
if (( OWNERFUSED != 0 && OWNERFUSED != 1 )); then echo "OWNERFUSED must be 0, 1, or auto" >&2; exit 2; fi
if (( OWNERFUSED + BLOCKFUSED + GROUPBATCH > 1 )); then echo "OWNERFUSED, BLOCKFUSED and GROUPBATCH are mutually exclusive" >&2; exit 2; fi
if (( OWNERFUSED && (ROW7_TENSOR || ROW8_TENSOR) )); then echo "OWNERFUSED is mutually exclusive with tensor modes" >&2; exit 2; fi
if (( BLOCKFUSED && (ROW7_TENSOR || ROW8_TENSOR) )); then echo "BLOCKFUSED is mutually exclusive with tensor modes" >&2; exit 2; fi
if (( GROUPBATCH && (ROW7_TENSOR || ROW8_TENSOR) )); then echo "GROUPBATCH is mutually exclusive with tensor modes" >&2; exit 2; fi
CUSTOM_BIN=0
if [[ -n "${BIN:-}" ]]; then CUSTOM_BIN=1; fi
DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_batch_n${N}"
if (( OWNERFUSED )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_ownerfused_batch_n${N}"; fi
if (( BLOCKFUSED )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_blockfused_batch_n${N}"; fi
if (( GROUPBATCH )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_groupbatch_batch_n${N}"; fi
if (( ROW7_TENSOR )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_row7tensor_batch_n${N}"; fi
if (( ROW8_TENSOR )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_row8tensor_batch_n${N}"; fi
BIN="${BIN:-$DEFAULT_BIN}"

for spec in "N:$N" "MOD:$MOD" "TARGET_MIB:$TARGET_MIB" "MAX_WINDOW:$MAX_WINDOW" "NGPU:$NGPU" "GRIDFP_VRAM_RESERVE_MIB:$GRIDFP_VRAM_RESERVE_MIB"; do
  require_uint "${spec%%:*}" "${spec#*:}" || exit 2
done

if (( MOD < 2 || MOD > 4294967295 )); then
  echo "HBM32 requires 2 <= modulus <= 4294967295; got $MOD" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found" >&2
  exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < NGPU )); then
  echo "requested $NGPU GPUs, but only $visible are visible" >&2
  exit 2
fi

PROVENANCE="${BIN}.provenance.json"
needs_build=0
if [[ ! -x "$BIN" ]]; then
  needs_build=1
elif [[ ! -f "$PROVENANCE" ]] || ! python3 "$ONEESAN_ROOT/scripts/solve/build_provenance.py" verify     "$PROVENANCE" --binary "$BIN" --root "$ONEESAN_ROOT" --verify-sources     --expect-compile-arg="-DTARGET_W=$((N + 1))" >/dev/null 2>&1; then
  if (( CUSTOM_BIN )); then
    echo "custom BIN has missing/stale build provenance: $BIN" >&2
    exit 2
  fi
  needs_build=1
fi
if (( needs_build )); then
  echo "$BIN missing or stale; building batch binary for n=$N" >&2
  N="$N" OUT="$BIN" OWNERFUSED="$OWNERFUSED" BLOCKFUSED="$BLOCKFUSED" GROUPBATCH="$GROUPBATCH" ROW7_TENSOR="$ROW7_TENSOR" ROW8_TENSOR="$ROW8_TENSOR"     "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh"
fi

nvidia-smi -L
nvidia-smi topo -m || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

echo "N=$N MOD=$MOD GPUs=$NGPU requested_scratch=${TARGET_MIB}MiB reserve=${GRIDFP_VRAM_RESERVE_MIB}MiB max_window=$MAX_WINDOW"
echo "BIN=$BIN"
export GRIDFP_VRAM_RESERVE_MIB
if (( GROUPBATCH )); then
  export GRIDFP_GROUPBATCH_DYNAMIC GRIDFP_GROUPBATCH_AFFINITY GRIDFP_GROUPBATCH_STICKY GRIDFP_GROUPBATCH_RECLAIM GRIDFP_GROUPBATCH_GRAPH GRIDFP_GROUPBATCH_P1_DEST GRIDFP_GROUPBATCH_WRAP32 GRIDFP_GROUPBATCH_MATCHLUT GRIDFP_GROUPBATCH_BATCHES_PER_GPU
  echo "GROUPBATCH dynamic=$GRIDFP_GROUPBATCH_DYNAMIC affinity=$GRIDFP_GROUPBATCH_AFFINITY sticky=$GRIDFP_GROUPBATCH_STICKY reclaim=$GRIDFP_GROUPBATCH_RECLAIM graph=$GRIDFP_GROUPBATCH_GRAPH p1_dest=$GRIDFP_GROUPBATCH_P1_DEST wrap32=$GRIDFP_GROUPBATCH_WRAP32 matchlut=$GRIDFP_GROUPBATCH_MATCHLUT batches_per_gpu=$GRIDFP_GROUPBATCH_BATCHES_PER_GPU" >&2
fi
if (( ROW7_TENSOR )); then
  export GRIDFP_BOUNDED_PREFIX_K=7
  export GRIDFP_DIRECT_ROW7_TENSOR=1
fi
if (( ROW8_TENSOR )); then
  export GRIDFP_BOUNDED_PREFIX_K=8
  export GRIDFP_DIRECT_ROW8_TENSOR=1
  if (( ROW8_STRUCTURAL )); then export GRIDFP_ROW8_STRUCTURAL=1; fi
fi
exec "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD"
