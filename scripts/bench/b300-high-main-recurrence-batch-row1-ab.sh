#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27;MOD="${MOD:-4294967291}";GPUS=8
TARGET_MIB="${TARGET_MIB:-65536}";MAX_WINDOW="${MAX_WINDOW:-14}";ROWS="${ROWS:-1}"
THREADS="${THREADS:-256}";HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}";ARCH="${ARCH:-native}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_high_main_recurrence_row${ROWS}_highdrop${HIGH_DROP_CHUNK}}"
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || { echo "HIGH_DROP_CHUNK must be 0 or 1" >&2;exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2;exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= GPUS )) || { echo "need $GPUS GPUs" >&2;exit 2; }
mkdir -p "$(dirname "$PREFIX")" "$ONEESAN_BUILD_DIR"

build_one(){
  local mode="$1" rec="$2" bin="$3"
  echo "=== build $mode high_main_recurrence=$rec high_drop_chunk=$HIGH_DROP_CHUNK ===" >&2
  N="$N" ARCH="$ARCH" OUT="$bin" MAIN_PULL_ILP=2 HIGH_MAIN_RECURRENCE="$rec" LOW_MAIN_RECURRENCE=0 HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" \
    MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 FAST_SHARD_ADDRESS8=1 \
    LOW_DROP_CACHE=1 LOW_DROP_CHUNK=1 LOW_BLOCK_CACHE=1 RUNTIME_THREADS=1 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >"${PREFIX}.${mode}.build.out" 2>"${PREFIX}.${mode}.build.err"
  grep -Fq "main_pull_ilp=2" "${PREFIX}.${mode}.build.out"
  grep -Fq "high_main_recurrence=$rec" "${PREFIX}.${mode}.build.out"
  if [[ "$rec" == 1 ]]; then
    grep -Fq 'high_main_state_bits=64 extra_state_bytes=0 high_p_lo=14 high_symbol_range=13..27 high_trit_positions=15 high_min_fixed=7' "${PREFIX}.${mode}.build.out"
  fi
}
BASE_BIN="$ONEESAN_BUILD_DIR/b300_batch_n27_highmainrec0_high${HIGH_DROP_CHUNK}"
CAND_BIN="$ONEESAN_BUILD_DIR/b300_batch_n27_highmainrec1_high${HIGH_DROP_CHUNK}"
build_one base 0 "$BASE_BIN";build_one cand 1 "$CAND_BIN"

run_one(){
  local mode="$1" bin="$2";local out="${PREFIX}.${mode}.out" err="${PREFIX}.${mode}.err" tele="${PREFIX}.${mode}.gpu.csv"
  : >"$tele"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null & local mon=$!
  sleep 1;set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$GPUS" "$MOD" >"$out" 2>"$err";local rc=$?;set -e
  kill "$mon" 2>/dev/null||true;wait "$mon" 2>/dev/null||true
  ((rc==0))||{ echo "$mode failed rc=$rc" >&2;tail -n 120 "$err" >&2||true;return "$rc"; }
  local line residue wall
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$out"|tail -n1||true)";[[ -n "$line" ]]||return 4
  grep -Fq " rows=$ROWS calibration=$((ROWS<28?1:0)) " <<<"$line"||{ echo "$mode row metadata mismatch" >&2;return 5; }
  residue="$(sed -nE 's/.* residue=([^[:space:]]+).*/\1/p'<<<"$line")";wall="$(sed -nE 's/.* wall_s=([^[:space:]]+).*/\1/p'<<<"$line")"
  local st
  st="$(python3 - "$tele" <<'PY'
import csv,sys
m=[];b=[];s=[]
for r in csv.reader(open(sys.argv[1])):
    if len(r)<4:continue
    try:sm=float(r[2]);mem=float(r[3])
    except ValueError:continue
    s.append(sm);m.append(mem)
    if sm>=50:b.append(mem)
def a(x):return sum(x)/len(x) if x else float('nan')
print(f'{a(m):.6f} {max(m) if m else float("nan"):.6f} {a(b):.6f} {a(s):.6f}')
PY
)"
  printf '%s\t%s\t%s\n' "$residue" "$wall" "$st"
}
read -r BR BW BM BMAX BBUSY BSM <<<"$(run_one base "$BASE_BIN")"
read -r CR CW CM CMAX CBUSY CSM <<<"$(run_one cand "$CAND_BIN")"
[[ "$BR" == "$CR" ]]||{ echo "high-main recurrence residue mismatch base=$BR cand=$CR" >&2;exit 6; }
python3 - "$BW" "$CW" "$BM" "$CM" "$BBUSY" "$CBUSY" "$BSM" "$CSM" <<'PY'
import sys
bw,cw,bm,cm,bb,cb,bs,cs=map(float,sys.argv[1:])
print('b300_high_main_recurrence_exact_intermediate_match=1')
print(f'b300_high_main_recurrence_wall_baseline_s={bw:.9f}')
print(f'b300_high_main_recurrence_wall_candidate_s={cw:.9f}')
print(f'b300_high_main_recurrence_wall_speedup={bw/cw:.9f}x')
print(f'b300_high_main_recurrence_memctl_all_baseline_avg_pct={bm:.3f}')
print(f'b300_high_main_recurrence_memctl_all_candidate_avg_pct={cm:.3f}')
print(f'b300_high_main_recurrence_memctl_busy_baseline_avg_pct={bb:.3f}')
print(f'b300_high_main_recurrence_memctl_busy_candidate_avg_pct={cb:.3f}')
print(f'b300_high_main_recurrence_memctl_busy_delta_pp={cb-bb:.3f}')
print(f'b300_high_main_recurrence_sm_baseline_avg_pct={bs:.3f}')
print(f'b300_high_main_recurrence_sm_candidate_avg_pct={cs:.3f}')
PY
printf 'b300_high_main_recurrence_ilp=2\n'
printf 'b300_high_main_recurrence_threads=%s\n' "$THREADS"
printf 'b300_high_main_recurrence_high_drop_chunk=%s\n' "$HIGH_DROP_CHUNK"
printf 'b300_high_main_recurrence_rows=%s\n' "$ROWS"
printf 'b300_high_main_recurrence_residue=%s\n' "$BR"
printf 'b300_high_main_recurrence_layout=p27..14_symbols13..27_trits15_fixed_gate7\n'
printf 'b300_high_main_recurrence_note=8B/state/p Mate streaming store replaces high-prefix address work; adopt only on wall-time win\n'
