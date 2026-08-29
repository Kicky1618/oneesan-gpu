#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}";MOD="${MOD:-4294967291}";NGPU="${NGPU:-8}";TARGET_MIB="${TARGET_MIB:-65536}";PLAN_TARGET_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";ROWS="${ROWS:-1}";GRIDFP_THREADS="${GRIDFP_THREADS:-256}";RUNS="${RUNS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_exact_pull_ab}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";RESULT="${RESULT:-${PREFIX}.tsv}";mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
command -v nvidia-smi >/dev/null||exit 2;(( $(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)>=NGPU ))||exit 2
modes=(push pull blockmate);declare -A bin
for m in "${modes[@]}";do bin[$m]="$ONEESAN_BUILD_DIR/b300_exact_${m}_n${N}";done
N="$N" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=0 BLOCK_PULL=0 BLOCK_MATE_CACHE=0 OUT="${bin[push]}" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/push.build.out" 2>"$LOGDIR/push.build.err"
N="$N" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=0 OUT="${bin[pull]}" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/pull.build.out" 2>"$LOGDIR/pull.build.err"
N="$N" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 OUT="${bin[blockmate]}" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/blockmate.build.out" 2>"$LOGDIR/blockmate.build.err"
printf 'mode\trun\tresidue\twall_s\tactive_max_s\tactive_sum_s\tprepare_s\n' >"$RESULT"
run_one(){local m="$1" r="$2" out="$LOGDIR/${m}_${r}.out" err="$LOGDIR/${m}_${r}.err";B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$GRIDFP_THREADS" GRIDFP_PLAN_TARGET_MIB="$PLAN_TARGET_MIB" GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}" "${bin[$m]}" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" >"$out" 2>"$err";local line;line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$out"|tail -n1)";[[ -n "$line" ]]||{ tail -n100 "$err" >&2;exit 3;};f(){sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p"<<<"$line";};printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$m" "$r" "$(f residue)" "$(f wall_s)" "$(f active_max_s)" "$(f active_sum_s)" "$(f prepare_s)" >>"$RESULT";}
for((r=0;r<RUNS;++r));do for((j=0;j<3;++j));do m="${modes[$(((j+r)%3))]}";run_one "$m" "$r";done;done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));modes=('push','pull','blockmate');res={m:{x['residue'] for x in r if x['mode']==m} for m in modes}
if len(res['push'])!=1 or any(res[m]!=res['push'] for m in modes):raise SystemExit(f'residue mismatch {res}')
med=lambda m,k:statistics.median(float(x[k]) for x in r if x['mode']==m);b=med('push','wall_s')
for m in modes:print(f'{m}_wall_s={med(m,"wall_s"):.6f} speedup_vs_push={b/med(m,"wall_s"):.6f}x')
print(f'winner={min(modes,key=lambda m:med(m,"wall_s"))} residue={next(iter(res["push"]))}')
PY
