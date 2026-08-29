#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
[[ "$N" == 27 ]] || { echo 'batch rank-state source preflight currently targets n=27' >&2; exit 2; }
SRC="${SRC:-$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_forced2window_opt_batch.cu}"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_batch_rankstate_source_preflight_n${N}}"
mkdir -p "$OUTDIR"
[[ -f "$SRC" ]] || { echo "missing source: $SRC" >&2; exit 2; }
command -v python3 >/dev/null || { echo 'python3 required' >&2; exit 2; }

step(){
  local script="$1" in="$2" out="$3" tag="$4"
  python3 "$ONEESAN_ROOT/scripts/build/$script" "$in" "$out" >"$OUTDIR/$tag.log"
  [[ -s "$out" ]] || { echo "empty transform output: $tag" >&2; exit 3; }
}

BASE0="$OUTDIR/00_main_mate.cu"
BASE1="$OUTDIR/01_main_pull.cu"
BASE2="$OUTDIR/02_block_pull.cu"
BASE3="$OUTDIR/03_block_mate.cu"
step gen-b300-main-mate-cache.py "$SRC" "$BASE0" main_mate
step gen-b300-main-pull.py "$BASE0" "$BASE1" main_pull
step gen-b300-block-pull.py "$BASE1" "$BASE2" block_pull
step gen-b300-block-mate-cache.py "$BASE2" "$BASE3" block_mate

rank_common(){
  local stem="$1" in="$2"
  local n0="$OUTDIR/${stem}_10_normalized.cu"
  local n1="$OUTDIR/${stem}_11_rank_delta.cu"
  local n2="$OUTDIR/${stem}_12_rank_free.cu"
  local n3="$OUTDIR/${stem}_13_rank_report.cu"
  local n4="$OUTDIR/${stem}_14_packed.cu"
  step normalize-b300-rank-delta-input.py "$in" "$n0" "${stem}_normalize"
  step gen-b300-rank-delta-cache.py "$n0" "$n1" "${stem}_rank_delta"
  step gen-b300-rank-delta-free-step.py "$n1" "$n2" "${stem}_rank_free"
  step gen-b300-rank-delta-report.py "$n2" "$n3" "${stem}_rank_report"
  step gen-b300-rank-state-packed.py "$n3" "$n4" "${stem}_packed"
  printf '%s\n' "$n4"
}

finish_batch(){
  local stem="$1" in="$2"
  local s0="$OUTDIR/${stem}_30_shard8.cu"
  local s1="$OUTDIR/${stem}_31_rowlimit.cu"
  local s2="$OUTDIR/${stem}_32_runtime_threads.cu"
  step gen-b300-batch-shard-address8.py "$in" "$s0" "${stem}_shard8"
  step lower-b300-batch-row-limit.py "$s0" "$s1" "${stem}_rowlimit"
  step gen-b300-runtime-threads.py "$s1" "$s2" "${stem}_runtime_threads"
  printf '%s\n' "$s2"
}

PACKED2="$(rank_common ilp2 "$BASE3")"
ILP2_MAIN="$OUTDIR/ilp2_20_main.cu"
ILP2_BLOCK="$OUTDIR/ilp2_21_block.cu"
step gen-b300-rank-state-ilp2.py "$PACKED2" "$ILP2_MAIN" ilp2_main
step gen-b300-block-rank-state-ilp2.py "$ILP2_MAIN" "$ILP2_BLOCK" ilp2_block
FINAL2="$(finish_batch ilp2 "$ILP2_BLOCK")"

PACKED4="$(rank_common ilp4 "$BASE3")"
ILP4_MAIN="$OUTDIR/ilp4_20_main.cu"
ILP4_BLOCK="$OUTDIR/ilp4_21_block.cu"
step gen-b300-rank-state-ilp4.py "$PACKED4" "$ILP4_MAIN" ilp4_main
step gen-b300-block-rank-state-ilp4.py "$ILP4_MAIN" "$ILP4_BLOCK" ilp4_block
FINAL4="$(finish_batch ilp4 "$ILP4_BLOCK")"

python3 - "$SRC" "$FINAL2" "$FINAL4" <<'PY'
from pathlib import Path
import sys
base=Path(sys.argv[1]).read_text()
variants={"ilp2":Path(sys.argv[2]).read_text(),"ilp4":Path(sys.argv[3]).read_text()}
# These are intentionally broad ABI invariants rather than fragile formatting
# anchors. The batch frontend must still parse argv, install D_MOD repeatedly and
# emit machine-readable residue/modulus records after all hot-path transforms.
abi=("argc","argv","D_MOD","cudaMemcpyToSymbol","residue=","modulus=")
for marker in abi:
    if marker not in base:
        raise SystemExit(f"base batch ABI marker missing: {marker}")
for name,s in variants.items():
    for marker in abi:
        if marker not in s:
            raise SystemExit(f"{name}: batch ABI marker lost: {marker}")
    required=(
        "using RankState = unsigned long long;",
        "dMainRankDelta","dBlockRankDelta",
        "B300_ROW_LIMIT","GRIDFP_THREADS",
    )
    for marker in required:
        if marker not in s:
            raise SystemExit(f"{name}: transformed artifact missing: {marker}")
if "b300_main_pull_rankstate_ilp2_kernel" not in variants["ilp2"]:
    raise SystemExit("ilp2: main packed rank-state kernel missing")
if "b300_block_pull_rankstate_ilp2_kernel" not in variants["ilp2"]:
    raise SystemExit("ilp2: block packed rank-state kernel missing")
if "b300_main_pull_rankstate_ilp4_kernel" not in variants["ilp4"]:
    raise SystemExit("ilp4: main packed rank-state kernel missing")
if "b300_block_pull_rankstate_ilp4_kernel" not in variants["ilp4"]:
    raise SystemExit("ilp4: block packed rank-state kernel missing")
print("batch_rankstate_source_preflight=OK variants=ilp2,ilp4 low_drop_cache=0 low_drop_chunk=0 low_block_cache=0 batch_abi_markers=preserved compile=NOT_RUN")
PY

echo "batch rank-state source preflight OK outdir=$OUTDIR (source transforms only; nvcc not run)" >&2
