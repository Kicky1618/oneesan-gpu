#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27; W=28; GPUS=8
MOD="${MOD:-4294967291}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"
THREADS="${THREADS:-256}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_dualmask_row${ROWS}_high${HIGH_DROP_CHUNK}_t${THREADS}}"
TMP="${TMP:-$ONEESAN_BUILD_DIR/b300_mainrec_dualmask_gen}"
BASE_BIN="$ONEESAN_BUILD_DIR/b300_mainrec_dualmask_base_n27_high${HIGH_DROP_CHUNK}"
CAND_BIN="$ONEESAN_BUILD_DIR/b300_mainrec_dualmask_n27_high${HIGH_DROP_CHUNK}"
mkdir -p "$(dirname "$PREFIX")" "$TMP" "$ONEESAN_BUILD_DIR"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=1024)) || { echo 'THREADS must be 32..1024' >&2; exit 2; }
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || { echo 'HIGH_DROP_CHUNK must be 0 or 1' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= GPUS )) || { echo "need $GPUS visible GPUs" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-dualmask-proof.sh"

echo "=== build baseline MAIN_RECURRENCE=1 high=$HIGH_DROP_CHUNK ===" >&2
N=27 ARCH="$ARCH" OUT="$BASE_BIN" MAIN_PULL_ILP=2 HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" \
  LOW_MAIN_RECURRENCE=0 HIGH_MAIN_RECURRENCE=0 MAIN_RECURRENCE=1 \
  MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 FAST_SHARD_ADDRESS8=1 \
  LOW_DROP_CACHE=1 LOW_DROP_CHUNK=1 LOW_BLOCK_CACHE=1 RUNTIME_THREADS=1 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >"${PREFIX}.base.build.out" 2>"${PREFIX}.base.build.err"
grep -Fq 'main_recurrence=1' "${PREFIX}.base.build.out"
grep -Fq 'high_recurrence_p_range=27..15 high_symbol_range=14..27' "${PREFIX}.base.build.out"
grep -Fq "high_drop_chunk=$HIGH_DROP_CHUNK" "${PREFIX}.base.build.out"

SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_forced2window_opt_batch.cu"
S1="$TMP/01_main_mate.cu"; S2="$TMP/02_main_pull.cu"; S3="$TMP/03_block_pull.cu"
S4="$TMP/04_block_mate.cu"; S5="$TMP/05_low_cache.cu"; S6="$TMP/06_low_chunk.cu"
S7="$TMP/07_high_chunk.cu"; S8="$TMP/08_low_block.cu"; S9="$TMP/09_dualmask.cu"
S10="$TMP/10_ilp2.cu"; S11="$TMP/11_mainrec_raw.cu"; S12="$TMP/12_mainrec_gate.cu"
S13="$TMP/13_shard8.cu"; S14="$TMP/14_rowlimit.cu"; S15="$TMP/15_threads.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-mate-cache.py" "$SRC" "$S1"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull.py" "$S1" "$S2"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull.py" "$S2" "$S3"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-mate-cache.py" "$S3" "$S4"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-window-drop-cache.py" "$S4" "$S5"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-drop-chunk.py" "$S5" "$S6"
BUILD_SRC="$S6"
if [[ "$HIGH_DROP_CHUNK" == 1 ]]; then
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-high-drop-chunk.py" "$BUILD_SRC" "$S7"; BUILD_SRC="$S7"
fi
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-block-cache.py" "$BUILD_SRC" "$S8"; BUILD_SRC="$S8"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull-dualmask.py" "$BUILD_SRC" "$S9"; BUILD_SRC="$S9"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull-ilp2.py" "$BUILD_SRC" "$S10"; BUILD_SRC="$S10"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence.py" "$BUILD_SRC" "$S11"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-high-recurrence-fixed-gate.py" "$S11" "$S12"; BUILD_SRC="$S12"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-batch-shard-address8.py" "$BUILD_SRC" "$S13"; BUILD_SRC="$S13"
python3 "$ONEESAN_ROOT/scripts/build/lower-b300-batch-row-limit.py" "$BUILD_SRC" "$S14"; BUILD_SRC="$S14"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-runtime-threads.py" "$BUILD_SRC" "$S15"; BUILD_SRC="$S15"
grep -Fq 'b300_block_endpoint_masks(d)' "$BUILD_SRC"
grep -Fq 'high_rec_groups=' "$BUILD_SRC"
grep -Fq 'b300_main_trit_get(x,p-14)' "$BUILD_SRC"
if grep -Fq 'const MateValue v=mget(d,q);' "$BUILD_SRC"; then echo 'stale block scan mget remains' >&2; exit 3; fi
if grep -Fq 'b300_main_trit_get(x,p-13)' "$BUILD_SRC"; then echo 'stale recurrence p13 artifact remains' >&2; exit 3; fi

echo "=== compile MAIN_RECURRENCE + dualmask candidate ===" >&2
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
  -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 "$BUILD_SRC" -o "$CAND_BIN" \
  >"${PREFIX}.cand.build.out" 2>"${PREFIX}.cand.build.err"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
run_one(){
  local mode="$1" bin="$2" out="${PREFIX}.${1}.out" err="${PREFIX}.${1}.err" tele="${PREFIX}.${1}.gpu.csv"
  : >"$tele"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null & local mon=$!
  sleep 1
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$out" 2>"$err"
  local rc=$?; set -e
  kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  ((rc==0)) || { echo "$mode failed rc=$rc" >&2; tail -n 180 "$err" >&2 || true; return "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$mode missing result" >&2; return 4; }
  grep -Fq " rows=$ROWS calibration=$((ROWS<28?1:0)) " <<<"$line" || { echo "$mode row metadata mismatch" >&2; return 5; }
  local hg="$(field high_rec_groups "$line")" hf="$(field high_rec_fallback_groups "$line")"
  [[ "$hg" =~ ^[0-9]+$ ]] && ((hg>0)) || { echo "$mode zero high recurrence coverage hg=$hg hf=$hf" >&2; return 6; }
  local stats
  stats="$(python3 - "$tele" <<'PY'
import csv,sys
mem=[];busy=[];sm=[];power=[]
for r in csv.reader(open(sys.argv[1])):
    if len(r)<5:continue
    try:s=float(r[2]);m=float(r[3]);p=float(r[4])
    except ValueError:continue
    sm.append(s);mem.append(m);power.append(p)
    if s>=50:busy.append(m)
def avg(v):return sum(v)/len(v) if v else float('nan')
print(f'{avg(mem):.6f} {max(mem) if mem else float("nan"):.6f} {avg(busy):.6f} {avg(sm):.6f} {avg(power):.6f}')
PY
)"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(field residue "$line")" "$(field wall_s "$line")" "$hg" "$hf" "$stats"
}
read -r BR BW BHG BHF BM BMAX BBUSY BSM BPWR <<<"$(run_one base "$BASE_BIN")"
read -r CR CW CHG CHF CM CMAX CBUSY CSM CPWR <<<"$(run_one cand "$CAND_BIN")"
[[ "$BR" == "$CR" ]] || { echo "FATAL mainrec+dualmask residue mismatch base=$BR cand=$CR" >&2; exit 7; }

python3 - "$BW" "$CW" "$BM" "$CM" "$BBUSY" "$CBUSY" "$BSM" "$CSM" "$BPWR" "$CPWR" "$BHG" "$CHG" "$BHF" "$CHF" <<'PY'
import sys
bw,cw,bm,cm,bb,cb,bs,cs,bp,cp=map(float,sys.argv[1:11]);bhg,chg,bhf,chf=map(int,sys.argv[11:15])
print('b300_mainrec_dualmask_exact_intermediate_match=1')
print(f'b300_mainrec_dualmask_wall_baseline_s={bw:.9f}')
print(f'b300_mainrec_dualmask_wall_candidate_s={cw:.9f}')
print(f'b300_mainrec_dualmask_wall_speedup={bw/cw:.9f}x')
print(f'b300_mainrec_dualmask_memctl_busy_baseline_avg_pct={bb:.3f}')
print(f'b300_mainrec_dualmask_memctl_busy_candidate_avg_pct={cb:.3f}')
print(f'b300_mainrec_dualmask_sm_baseline_avg_pct={bs:.3f}')
print(f'b300_mainrec_dualmask_sm_candidate_avg_pct={cs:.3f}')
print(f'b300_mainrec_dualmask_power_baseline_avg_w={bp:.3f}')
print(f'b300_mainrec_dualmask_power_candidate_avg_w={cp:.3f}')
print(f'b300_mainrec_dualmask_high_rec_groups_baseline={bhg}')
print(f'b300_mainrec_dualmask_high_rec_groups_candidate={chg}')
print(f'b300_mainrec_dualmask_high_rec_fallback_baseline={bhf}')
print(f'b300_mainrec_dualmask_high_rec_fallback_candidate={chf}')
PY
printf 'b300_mainrec_dualmask_rows=%s\n' "$ROWS"
printf 'b300_mainrec_dualmask_threads=%s\n' "$THREADS"
printf 'b300_mainrec_dualmask_high_drop_chunk=%s\n' "$HIGH_DROP_CHUNK"
printf 'b300_mainrec_dualmask_residue=%s\n' "$BR"
printf 'b300_mainrec_dualmask_note=production-like composition; adopt dualmask only on exact match plus wall-time win\n'
