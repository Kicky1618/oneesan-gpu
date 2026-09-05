#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
TARGET_MIB_WAS_SET="${TARGET_MIB+x}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-8}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
ROW6_VMM="${ROW6_VMM:-0}"
ROW6_PRERANK="${ROW6_PRERANK:-0}"
ROW6_OCCVMM="${ROW6_OCCVMM:-0}"
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
require_uint ROW6_VMM "$ROW6_VMM" || exit 2
require_uint ROW6_PRERANK "$ROW6_PRERANK" || exit 2
require_uint ROW6_OCCVMM "$ROW6_OCCVMM" || exit 2
require_uint ROW7_TENSOR "$ROW7_TENSOR" || exit 2
require_uint ROW8_TENSOR "$ROW8_TENSOR" || exit 2
require_uint ROW8_STRUCTURAL "$ROW8_STRUCTURAL" || exit 2
require_uint BLOCKFUSED "$BLOCKFUSED" || exit 2
require_uint GROUPBATCH "$GROUPBATCH" || exit 2
if (( ROW6_VMM != 0 && ROW6_VMM != 1 )); then echo "ROW6_VMM must be 0 or 1" >&2; exit 2; fi
if (( ROW6_PRERANK != 0 && ROW6_PRERANK != 1 )); then echo "ROW6_PRERANK must be 0 or 1" >&2; exit 2; fi
if (( ROW6_OCCVMM != 0 && ROW6_OCCVMM != 1 )); then echo "ROW6_OCCVMM must be 0 or 1" >&2; exit 2; fi
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
  if (( N >= 20 && N < 27 && ! BLOCKFUSED && ! GROUPBATCH && ! ROW7_TENSOR && ! ROW8_TENSOR && ! ROW6_VMM && ! ROW6_PRERANK && ! ROW6_OCCVMM )); then OWNERFUSED=1; fi
fi
require_uint OWNERFUSED "$OWNERFUSED" || exit 2
if (( OWNERFUSED != 0 && OWNERFUSED != 1 )); then echo "OWNERFUSED must be 0, 1, or auto" >&2; exit 2; fi
if (( OWNERFUSED + BLOCKFUSED + GROUPBATCH > 1 )); then echo "OWNERFUSED, BLOCKFUSED and GROUPBATCH are mutually exclusive" >&2; exit 2; fi
if (( OWNERFUSED && (ROW6_VMM || ROW6_PRERANK || ROW6_OCCVMM || ROW7_TENSOR || ROW8_TENSOR) )); then echo "OWNERFUSED is mutually exclusive with other specialized modes" >&2; exit 2; fi
if (( BLOCKFUSED && (ROW6_VMM || ROW6_PRERANK || ROW6_OCCVMM || ROW7_TENSOR || ROW8_TENSOR) )); then echo "BLOCKFUSED is mutually exclusive with other specialized modes" >&2; exit 2; fi
if (( GROUPBATCH && (ROW6_VMM || ROW6_PRERANK || ROW6_OCCVMM || ROW7_TENSOR || ROW8_TENSOR) )); then echo "GROUPBATCH is mutually exclusive with other specialized modes" >&2; exit 2; fi
if (( ROW6_VMM + ROW6_PRERANK + ROW6_OCCVMM > 1 )); then echo "ROW6_VMM, ROW6_PRERANK and ROW6_OCCVMM are mutually exclusive" >&2; exit 2; fi
if (( (ROW6_VMM || ROW6_PRERANK || ROW6_OCCVMM) && (ROW7_TENSOR || ROW8_TENSOR) )); then echo "ROW6 VMM/pre-rank/occvmm modes are mutually exclusive with ROW7_TENSOR/ROW8_TENSOR" >&2; exit 2; fi
if (( ROW7_TENSOR )); then
  echo "ROW7_TENSOR is benchmark-only: its exact compact table does not yet have a clean-clone reproducible certificate chain" >&2
  echo "use scripts/run/b300x8.sh with ROW7_TENSOR=1 for modular benchmarking" >&2
  exit 2
fi
if (( ROW8_TENSOR )); then
  if (( ! ROW8_STRUCTURAL )); then
    echo "ROW8_TENSOR exact mode requires ROW8_STRUCTURAL=1 and a width-specific Grid-FP/structural integer-equality certificate" >&2
    exit 2
  fi
  ROW8_CERT="${ROW8_GRIDFP_CERT:-$ONEESAN_ROOT/formal/certificates/row8_gridfp_structural_w$((N + 1)).json}"
  if [[ ! -f "$ROW8_CERT" ]]; then
    echo "ROW8 structural exact certificate missing for width $((N + 1)): $ROW8_CERT" >&2
    echo "generate it with GRIDFP_ROW8_CERT_COMPARE=1 on the target width before enabling exact mode" >&2
    exit 2
  fi
  python3 "$ONEESAN_ROOT/scripts/tools/row8_gridfp_structural_cert.py" --verify "$ROW8_CERT" >/dev/null || {
    echo "ROW8 structural exact certificate verification failed: $ROW8_CERT" >&2; exit 2; }
  python3 - "$ROW8_CERT" "$N" <<'PY' || exit 2
import json,sys
p,n=sys.argv[1],int(sys.argv[2]); d=json.load(open(p))
if d.get('n')!=n or d.get('width')!=n+1 or not d.get('integer_vector_equal'):
    raise SystemExit(f"ROW8 certificate target mismatch: cert n={d.get('n')} width={d.get('width')} requested n={n}")
PY
  echo "ROW8 structural exact certificate verified: $ROW8_CERT" >&2
fi
CUSTOM_BIN=0
if [[ -n "${BIN:-}" ]]; then CUSTOM_BIN=1; fi
DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_batch_n${N}"
if (( OWNERFUSED )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_ownerfused_batch_n${N}"; fi
if (( BLOCKFUSED )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_blockfused_batch_n${N}"; fi
if (( GROUPBATCH )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_groupbatch_batch_n${N}"; fi
if (( ROW6_VMM )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_vmm_batch_n${N}"; fi
if (( ROW6_PRERANK )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_vmm_prerank_batch_n${N}"; fi
if (( ROW6_OCCVMM )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_occvmm_batch_n${N}"; fi
if (( ROW8_TENSOR )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_row8tensor_batch_n${N}"; fi
BIN="${BIN:-$DEFAULT_BIN}"

for spec in "N:$N" "TARGET_MIB:$TARGET_MIB" "MAX_WINDOW:$MAX_WINDOW" "NGPU:$NGPU" "GRIDFP_VRAM_RESERVE_MIB:$GRIDFP_VRAM_RESERVE_MIB"; do
  require_uint "${spec%%:*}" "${spec#*:}" || exit 2
done
if (( N < 2 || N > 27 )); then
  echo "N must be in 2..27 for the B300 production solvers" >&2
  exit 2
fi
if (( (ROW6_VMM || ROW6_PRERANK || ROW6_OCCVMM) && N < 27 )); then
  echo "ROW6 VMM production modes currently require N>=27" >&2
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
elif [[ ! -f "$PROVENANCE" ]] || ! python3 "$ONEESAN_ROOT/scripts/solve/build_provenance.py" verify \
    "$PROVENANCE" --binary "$BIN" --root "$ONEESAN_ROOT" --verify-sources \
    --expect-compile-arg="-DTARGET_W=$((N + 1))" >/dev/null 2>&1; then
  if (( CUSTOM_BIN )); then
    echo "custom BIN has missing/stale build provenance: $BIN" >&2
    echo "provide a matching ${BIN}.provenance.json or invoke solve_b300_exact_batch.py directly to opt out" >&2
    exit 2
  fi
  echo "$BIN has missing/stale build provenance; rebuilding optimized batch binary for n=$N" >&2
  needs_build=1
fi
if (( needs_build )); then
  N="$N" OUT="$BIN" OWNERFUSED="$OWNERFUSED" BLOCKFUSED="$BLOCKFUSED" GROUPBATCH="$GROUPBATCH" ROW6_VMM="$ROW6_VMM" ROW6_PRERANK="$ROW6_PRERANK" ROW6_OCCVMM="$ROW6_OCCVMM" ROW8_TENSOR="$ROW8_TENSOR" "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh"
fi

export GRIDFP_VRAM_RESERVE_MIB
if (( GROUPBATCH )); then
  export GRIDFP_GROUPBATCH_DYNAMIC GRIDFP_GROUPBATCH_AFFINITY GRIDFP_GROUPBATCH_STICKY GRIDFP_GROUPBATCH_RECLAIM GRIDFP_GROUPBATCH_GRAPH GRIDFP_GROUPBATCH_P1_DEST GRIDFP_GROUPBATCH_WRAP32 GRIDFP_GROUPBATCH_MATCHLUT GRIDFP_GROUPBATCH_BATCHES_PER_GPU
  echo "GROUPBATCH dynamic=$GRIDFP_GROUPBATCH_DYNAMIC affinity=$GRIDFP_GROUPBATCH_AFFINITY sticky=$GRIDFP_GROUPBATCH_STICKY reclaim=$GRIDFP_GROUPBATCH_RECLAIM graph=$GRIDFP_GROUPBATCH_GRAPH p1_dest=$GRIDFP_GROUPBATCH_P1_DEST wrap32=$GRIDFP_GROUPBATCH_WRAP32 matchlut=$GRIDFP_GROUPBATCH_MATCHLUT batches_per_gpu=$GRIDFP_GROUPBATCH_BATCHES_PER_GPU" >&2
fi
if (( ROW6_OCCVMM )); then
  export GRIDFP_AUTH_VMM=1
  export GRIDFP_DIRECT_AUTH=3
fi
EXACT_ADMISSION_ARGS=()
if (( ROW8_TENSOR )); then
  export GRIDFP_BOUNDED_PREFIX_K=8
  export GRIDFP_DIRECT_ROW8_TENSOR=1
  if (( ROW8_STRUCTURAL )); then
    export GRIDFP_ROW8_STRUCTURAL=1
    EXACT_ADMISSION_ARGS+=(--admission-certificate "$ROW8_CERT")
  fi
fi
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" "$N" \
  --binary "$BIN" \
  --target-mib "$TARGET_MIB" \
  --max-window "$MAX_WINDOW" \
  --gpus "$NGPU" \
  "${EXACT_ADMISSION_ARGS[@]}" \
  "$@"
