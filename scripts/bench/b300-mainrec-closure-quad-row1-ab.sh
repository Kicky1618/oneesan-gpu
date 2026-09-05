#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27; GPUS=8
MOD="${MOD:-4294967291}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"
THREADS="${THREADS:-256}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
DUALMASK="${DUALMASK:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_closure_quad_row${ROWS}_hd${HIGH_DROP_CHUNK}_dual${DUALMASK}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
ISO="${ISO:-$ONEESAN_BUILD_DIR/mainrec_closure_quad_$$}"
BASE_BIN="$ONEESAN_BUILD_DIR/b300_mainrec_closure_quad_base_hd${HIGH_DROP_CHUNK}_dual${DUALMASK}_n27"
CAND_BIN="$ONEESAN_BUILD_DIR/b300_mainrec_closure_quad_hd${HIGH_DROP_CHUNK}_dual${DUALMASK}_n27"
mkdir -p "$LOGDIR" "$ISO" "$ISO/tmp" "$(dirname "$PREFIX")"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=1024&&THREADS%32==0)) || { echo 'THREADS must be a warp multiple 32..1024' >&2; exit 2; }
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || { echo 'HIGH_DROP_CHUNK must be 0 or 1' >&2; exit 2; }
[[ "$DUALMASK" == 0 || "$DUALMASK" == 1 ]] || { echo 'DUALMASK must be 0 or 1' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= GPUS )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

BASE_BUILD_OUT="$LOGDIR/base.build.out"; BASE_BUILD_ERR="$LOGDIR/base.build.err"
echo "=== build MAIN_RECURRENCE baseline highdrop=$HIGH_DROP_CHUNK dualmask=$DUALMASK ===" >&2
N=27 ARCH="$ARCH" OUT="$BASE_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" DUALMASK="$DUALMASK" \
BUILD_ERR="$BASE_BUILD_ERR" CALIBRATED_BUILD_DIR="$ISO/basechain" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-calibrated.sh" >"$BASE_BUILD_OUT" 2>"$LOGDIR/base.build.driver.err"
[[ -x "$BASE_BIN" ]] || { echo 'baseline binary missing' >&2; exit 3; }
grep -Fq "calibrated_forced=1 main_recurrence=1 high_drop_chunk=$HIGH_DROP_CHUNK dualmask=$DUALMASK" "$BASE_BUILD_OUT"
if [[ "$DUALMASK" == 1 ]]; then
  BUILD_SRC="$(sed -nE 's/^  source_after_dualmask=(.*)$/\1/p' "$BASE_BUILD_OUT" | tail -n1)"
else
  BUILD_SRC="$(sed -nE 's/^  source_after_proof_gates=(.*)$/\1/p' "$BASE_BUILD_OUT" | tail -n1)"
fi
[[ -n "$BUILD_SRC" && -f "$BUILD_SRC" ]] || { echo "could not resolve calibrated baseline source from $BASE_BUILD_OUT" >&2; exit 3; }
grep -Fq 'high_rec_groups=' "$BUILD_SRC"
[[ "$DUALMASK" == 0 ]] || grep -Fq 'b300_block_endpoint_masks(d)' "$BUILD_SRC"

QUAD_SRC="$ISO/final_closure_quad.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-quad.py" "$BUILD_SRC" "$QUAD_SRC" >"$LOGDIR/quad.transform.out"
grep -Fq 'B300BlockClosureQuad' "$QUAD_SRC"
grep -Fq 'flush_scope=closure' "$LOGDIR/quad.transform.out"
grep -Fq 'high_rec_groups=' "$QUAD_SRC"
[[ "$DUALMASK" == 0 ]] || grep -Fq 'b300_block_endpoint_masks(d)' "$QUAD_SRC"

echo '=== compile closure-quad candidate ===' >&2
CAND_BUILD_ERR="$LOGDIR/cand.build.err"
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
  -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 "$QUAD_SRC" -o "$CAND_BIN" \
  >"$LOGDIR/cand.build.out" 2>"$CAND_BUILD_ERR"
[[ -x "$CAND_BIN" ]] || { echo 'closure-quad candidate binary missing' >&2; exit 3; }

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
RESOURCE="${PREFIX}.ptxas.tsv"
printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$BASE_BUILD_ERR" --label base --contains block_pull_kernel >>"$RESOURCE" || true
python3 "$PARSER" "$CAND_BUILD_ERR" --label quad --contains block_pull_kernel >>"$RESOURCE" || true

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
summarize(){ python3 - "$1" <<'PY'
import csv,sys
mem=[];busy=[];sm=[];power=[]
for r in csv.reader(open(sys.argv[1])):
    if len(r)<5:continue
    try:s=float(r[2]);m=float(r[3]);p=float(r[4])
    except ValueError:continue
    sm.append(s);mem.append(m);power.append(p)
    if s>=50:busy.append(m)
def avg(x):return sum(x)/len(x) if x else float('nan')
print(f'{avg(mem):.6f} {max(mem) if mem else float("nan"):.6f} {avg(busy):.6f} {avg(sm):.6f} {avg(power):.6f} {len(mem)}')
PY
}
run_one(){
  local mode="$1" bin="$2" out="$LOGDIR/$mode.out" err="$LOGDIR/$mode.err" tele="$LOGDIR/$mode.gpu.csv"
  : >"$tele";nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null & local mon=$!
  sleep 1;set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$out" 2>"$err";local rc=$?;set -e
  kill "$mon" 2>/dev/null||true;wait "$mon" 2>/dev/null||true
  ((rc==0))||{ echo "$mode failed rc=$rc" >&2;tail -n 160 "$err" >&2||true;return "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$out"|tail -n1||true)";[[ -n "$line" ]]||{ echo "$mode missing result" >&2;return 4; }
  grep -Fq " rows=$ROWS calibration=$((ROWS<28?1:0)) " <<<"$line" || { echo "$mode row metadata mismatch" >&2;return 5; }
  local stats="$(summarize "$tele")"
  printf '%s\t%s\t%s\n' "$(field residue "$line")" "$(field wall_s "$line")" "$stats"
}
read -r BR BW BM BMAX BBUSY BSM BP BSAMPLES <<<"$(run_one base "$BASE_BIN")"
read -r CR CW CM CMAX CBUSY CSM CP CSAMPLES <<<"$(run_one quad "$CAND_BIN")"
[[ "$BR" == "$CR" ]]||{ echo "FATAL closure-quad residue mismatch base=$BR cand=$CR" >&2;exit 6; }
python3 - "$BW" "$CW" "$BM" "$CM" "$BBUSY" "$CBUSY" "$BSM" "$CSM" <<'PY'
import sys
bw,cw,bm,cm,bb,cb,bs,cs=map(float,sys.argv[1:])
print('b300_mainrec_closure_quad_exact_intermediate_match=1')
print(f'b300_mainrec_closure_quad_wall_baseline_s={bw:.9f}')
print(f'b300_mainrec_closure_quad_wall_candidate_s={cw:.9f}')
print(f'b300_mainrec_closure_quad_wall_speedup={bw/cw:.9f}x')
print(f'b300_mainrec_closure_quad_memctl_all_baseline_avg_pct={bm:.3f}')
print(f'b300_mainrec_closure_quad_memctl_all_candidate_avg_pct={cm:.3f}')
print(f'b300_mainrec_closure_quad_memctl_busy_baseline_avg_pct={bb:.3f}')
print(f'b300_mainrec_closure_quad_memctl_busy_candidate_avg_pct={cb:.3f}')
print(f'b300_mainrec_closure_quad_sm_baseline_avg_pct={bs:.3f}')
print(f'b300_mainrec_closure_quad_sm_candidate_avg_pct={cs:.3f}')
PY
printf 'b300_mainrec_closure_quad_threads=%s\n' "$THREADS"
printf 'b300_mainrec_closure_quad_high_drop_chunk=%s\n' "$HIGH_DROP_CHUNK"
printf 'b300_mainrec_closure_quad_dualmask=%s\n' "$DUALMASK"
printf 'b300_mainrec_closure_quad_rows=%s\n' "$ROWS"
printf 'b300_mainrec_closure_quad_residue=%s\n' "$BR"
printf 'b300_mainrec_closure_quad_note=only closure candidate source loads are queued four-at-a-time; endpoint load and rank generation remain unchanged; adopt only on exact match plus wall win\n'
cat "$RESOURCE"
