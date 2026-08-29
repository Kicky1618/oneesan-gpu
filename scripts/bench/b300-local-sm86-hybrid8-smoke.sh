#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'local hybrid8 smoke targets n=27' >&2; exit 2; }
ARCH="${ARCH:-sm_86}"
NGPU="${NGPU:-1}"
ROWS="${ROWS:-1}"
THREADS="${THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-1024}"
MAX_WINDOW="${MAX_WINDOW:-14}"
MOD="${MOD:-4294967291}"
HYBRID_THRESHOLD="${HYBRID_THRESHOLD:-0}"
WORK="${WORK:-$ONEESAN_ROOT/work/b300_local_sm86_hybrid8_smoke}"
mkdir -p "$WORK" "$ONEESAN_BUILD_DIR"

[[ "$NGPU" =~ ^[1-8]$ ]] || { echo 'NGPU must be 1..8' >&2; exit 2; }
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32 && THREADS<=1024 && THREADS%32==0)) || { echo 'THREADS must be a warp multiple in 32..1024' >&2; exit 2; }
[[ "$TARGET_MIB" =~ ^[1-9][0-9]*$ ]] || { echo 'TARGET_MIB must be positive' >&2; exit 2; }
[[ "$MAX_WINDOW" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_WINDOW must be positive' >&2; exit 2; }
[[ "$MOD" =~ ^[1-9][0-9]*$ ]] || { echo 'MOD must be positive' >&2; exit 2; }
[[ "$HYBRID_THRESHOLD" =~ ^[0-9]+$ ]] || { echo 'HYBRID_THRESHOLD must be non-negative' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU visible GPU(s), found $visible" >&2; exit 2; }

BASE_BIN="$ONEESAN_BUILD_DIR/b300_local_sm86_ilp2_n27"
HYBRID_BIN="$ONEESAN_BUILD_DIR/b300_local_sm86_hybrid8_t${HYBRID_THRESHOLD}_n27"
BASE_ERR="$WORK/baseline.build.err"
HYBRID_ERR="$WORK/hybrid.build.err"

build_one(){
  local mode="$1" out="$2" err="$3" hybrid="$4" threshold="$5"
  echo "=== build mode=$mode arch=$ARCH hybrid=$hybrid threshold=$threshold ===" >&2
  N=27 ARCH="$ARCH" OUT="$out" \
    HIGH_DROP_CHUNK=0 RECURRENCE_ILP=2 \
    RECURRENCE_HYBRID_ILP8="$hybrid" RECURRENCE_HYBRID_ILP8_MIN_STATES="$threshold" \
    RECURRENCE_HYBRID_ILP8_NEXTSELF=0 RECURRENCE_HYBRID_ILP8_NEXTSELF_WIDTH=8 \
    RANDOM_CG=0 RANDOM_CG_L2_FETCH_BYTES=0 PREFETCH_L2=0 DUALMASK=0 CLOSURE_BATCH=0 \
    MAXRREGCOUNT=0 PTXAS_VERBOSE=1 BUILD_ERR="$err" \
    bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" \
    >"$WORK/${mode}.build.out" 2>"$WORK/${mode}.build.driver.err"
  [[ -x "$out" ]] || { echo "$mode binary missing: $out" >&2; exit 3; }
}

build_one baseline "$BASE_BIN" "$BASE_ERR" 0 0
build_one hybrid "$HYBRID_BIN" "$HYBRID_ERR" 1 "$HYBRID_THRESHOLD"

grep -Fq 'main_pull_kernel_ilp8_hybrid' "$WORK/hybrid.build.out" || \
  grep -Fq 'recurrence_hybrid_ilp8=1' "$WORK/hybrid.build.out" || {
    echo 'hybrid build summary missing hybrid8 marker' >&2; exit 3;
  }

field(){
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

run_one(){
  local mode="$1" bin="$2" so="$WORK/${mode}.run.out" se="$WORK/${mode}.run.err"
  echo "=== run mode=$mode rows=$ROWS threads=$THREADS ngpu=$NGPU target_mib=$TARGET_MIB ===" >&2
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" \
    "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$mode missing backend result line" >&2; tail -n 80 "$se" >&2 || true; exit 4; }
  local residue wall high fallback
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  high="$(field high_rec_groups "$line")"; fallback="$(field high_rec_fallback_groups "$line")"
  [[ -n "$residue" && -n "$wall" ]] || { echo "$mode missing residue/wall_s" >&2; exit 4; }
  printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$residue" "$wall" "${high:-NA}" "${fallback:-NA}"
}

printf 'mode\tresidue\twall_s\thigh_rec_groups\thigh_rec_fallback_groups\n' >"$WORK/results.tsv"
run_one baseline "$BASE_BIN" | tee -a "$WORK/results.tsv"
run_one hybrid "$HYBRID_BIN" | tee -a "$WORK/results.tsv"

read -r base_res base_wall < <(awk -F'\t' '$1=="baseline"{print $2,$3}' "$WORK/results.tsv")
read -r hyb_res hyb_wall < <(awk -F'\t' '$1=="hybrid"{print $2,$3}' "$WORK/results.tsv")
[[ -n "$base_res" && "$base_res" == "$hyb_res" ]] || {
  echo "FATAL local hybrid8 residue mismatch baseline=$base_res hybrid=$hyb_res" >&2
  exit 5
}
speedup="$(python3 - "$base_wall" "$hyb_wall" <<'PY'
import sys
b=float(sys.argv[1]); h=float(sys.argv[2])
print(f'{b/h:.6f}' if h>0 else 'nan')
PY
)"

echo "b300_local_sm86_hybrid8_smoke=OK residue=$base_res baseline_wall_s=$base_wall hybrid_wall_s=$hyb_wall speedup=${speedup}x rows=$ROWS threads=$THREADS ngpu=$NGPU threshold=$HYBRID_THRESHOLD arch=$ARCH"
echo "results=$WORK/results.tsv" >&2
