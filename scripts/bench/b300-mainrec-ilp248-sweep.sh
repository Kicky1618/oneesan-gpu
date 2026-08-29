#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27; ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_ilp248_hd${HIGH_DROP_CHUNK}_row${ROWS}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")"
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || exit 2
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || exit 2
[[ "$REPEATS" =~ ^[0-9]+$ ]] && ((REPEATS>=1)) || exit 2
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }
python3 "$ONEESAN_ROOT/scripts/bench/b300-main-recurrence-ilp-partition-proof.py"

BASE_BIN="$ONEESAN_BUILD_DIR/b300_mainrec_ilp2_hd${HIGH_DROP_CHUNK}_n27"; BASE_OUT="$LOGDIR/ilp2.build.out"; BASE_ERR="$LOGDIR/ilp2.build.err"
N=27 ARCH="$ARCH" OUT="$BASE_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" DUALMASK=0 BUILD_ERR="$BASE_ERR" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-calibrated.sh" >"$BASE_OUT" 2>"$LOGDIR/ilp2.build.driver.err"
SRC="$(sed -nE 's/^  source_after_proof_gates=(.*)$/\1/p' "$BASE_OUT" | tail -n1)"
[[ -x "$BASE_BIN" && -n "$SRC" && -f "$SRC" ]] || { echo 'failed to resolve recurrence ILP2 generated source' >&2; exit 3; }
grep -Fq 'main_recurrence=1' "$BASE_OUT"; grep -Fq 'high_rec_groups=' "$SRC"; grep -Fq 'main_pull_kernel_ilp2' "$SRC"

: >"$LOGDIR/binaries.tsv"
printf 'ilp2\t%s\n' "$BASE_BIN" >>"$LOGDIR/binaries.tsv"
for lanes in 4 8; do
  outsrc="$LOGDIR/ilp${lanes}.cu"; bin="$ONEESAN_BUILD_DIR/b300_mainrec_ilp${lanes}_hd${HIGH_DROP_CHUNK}_n27"; err="$LOGDIR/ilp${lanes}.build.err"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-ilp.py" "$SRC" "$outsrc" "$lanes" >"$LOGDIR/ilp${lanes}.transform.out"
  grep -Fq "unified_main_recurrence_ilp=$lanes" "$LOGDIR/ilp${lanes}.transform.out"
  grep -Fq "base+=Code(${lanes})*grid" "$outsrc"
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 "$outsrc" -o "$bin" >"$LOGDIR/ilp${lanes}.compile.out" 2>"$err"
  [[ -x "$bin" ]] || exit 3
  printf 'ilp%s\t%s\n' "$lanes" "$bin" >>"$LOGDIR/binaries.tsv"
done

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"; printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for mode in ilp2 ilp4 ilp8; do python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" --contains main_pull_kernel >>"$RESOURCE" || true; done
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'mode\tthreads\trepeat\tresidue\twall_s\thigh_rec_groups\thigh_rec_fallback_groups\n' >"$RESULT"
run_one(){
  local mode="$1" bin="$2" threads="$3" rep="$4" so="$LOGDIR/${mode}_t${threads}_r${rep}.out" se="$LOGDIR/${mode}_t${threads}_r${rep}.err"
  set +e; B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$threads" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se"; local rc=$?; set -e
  ((rc==0)) || { echo "$mode threads=$threads failed rc=$rc" >&2; tail -n 120 "$se" >&2 || true; return "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing result" >&2; return 4; }
  local hg="$(field high_rec_groups "$line")" hf="$(field high_rec_fallback_groups "$line")"; [[ "$hg" =~ ^[0-9]+$ ]] && ((hg>0)) || { echo "$mode has zero high recurrence coverage" >&2; return 5; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$threads" "$rep" "$(field residue "$line")" "$(field wall_s "$line")" "$hg" "$hf" >>"$RESULT"
}
for t in $THREADS_LIST; do [[ "$t" =~ ^[0-9]+$ ]] && ((t>=32&&t<=1024&&t%32==0)) || exit 2; for mode in ilp2 ilp4 ilp8; do bin="$(awk -F '\t' -v m="$mode" '$1==m{print $2}' "$LOGDIR/binaries.tsv")"; for ((r=1;r<=REPEATS;++r)); do echo "=== $mode threads=$t repeat=$r ===" >&2; run_one "$mode" "$bin" "$t" "$r"; done; done; done

python3 - "$RESULT" "$HIGH_DROP_CHUNK" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); hd=sys.argv[2]
if not rows:raise SystemExit('no ILP248 results')
res={r['residue'] for r in rows}
if len(res)!=1:raise SystemExit('FATAL mainrec ILP248 partial residue mismatch '+repr({(r['mode'],r['threads']):r['residue'] for r in rows}))
by={}
for r in rows:by.setdefault((r['mode'],int(r['threads'])),[]).append(float(r['wall_s']))
med=[(statistics.median(v),m,t) for (m,t),v in by.items()]
for w,m,t in sorted(med):print(f'{m} threads={t} median_wall_s={w:.9f}',file=sys.stderr)
best=min(med);base=min(x for x in med if x[1]=='ilp2')
print('b300_mainrec_ilp248_exact_intermediate_match=1')
print(f'b300_mainrec_ilp248_high_drop_chunk={hd}')
print(f'b300_mainrec_ilp248_residue={next(iter(res))}')
print(f'b300_mainrec_ilp248_base_best_threads={base[2]}')
print(f'b300_mainrec_ilp248_base_best_wall_s={base[0]:.9f}')
print(f'b300_mainrec_ilp248_best_mode={best[1]}')
print(f'b300_mainrec_ilp248_best_threads={best[2]}')
print(f'b300_mainrec_ilp248_best_wall_s={best[0]:.9f}')
print(f'b300_mainrec_ilp248_speedup_vs_ilp2={base[0]/best[0]:.9f}x')
PY
cat "$RESOURCE"
echo "b300-mainrec-ilp248-sweep OK result=$RESULT resources=$RESOURCE" >&2
