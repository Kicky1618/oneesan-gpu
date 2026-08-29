#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}";MOD="${MOD:-4294967291}";NGPU="${NGPU:-8}";TARGET_MIB="${TARGET_MIB:-65536}";MAX_WINDOW="${MAX_WINDOW:-14}";ROWS="${ROWS:-1}";THREADS_LIST="${THREADS_LIST:-64 128 256 512}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_exact_thread_sweep}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";RESULT="${RESULT:-${PREFIX}.tsv}";mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
BIN="$ONEESAN_BUILD_DIR/b300_exact_thread_sweep_n${N}"
N="$N" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 PTXAS_VERBOSE=1 OUT="$BIN" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"
printf 'threads\tresidue\twall_s\tactive_max_s\tactive_sum_s\tprepare_s\n' >"$RESULT"
for t in $THREADS_LIST;do
  out="$LOGDIR/t${t}.out";err="$LOGDIR/t${t}.err"
  GRIDFP_THREADS="$t" B300_ROW_LIMIT="$ROWS" GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}" "$BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" >"$out" 2>"$err"
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$out"|tail -n1)";[[ -n "$line" ]]||{ tail -n80 "$err" >&2;exit 3; }
  f(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p"<<<"$line"; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$(f residue)" "$(f wall_s)" "$(f active_max_s)" "$(f active_sum_s)" "$(f prepare_s)" >>"$RESULT"
done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));res={x['residue'] for x in r}
if len(res)!=1:raise SystemExit(f'residue mismatch {res}')
b=min(r,key=lambda x:float(x['wall_s']));print(f'best_threads={b["threads"]} best_wall_s={b["wall_s"]} residue={b["residue"]}')
PY
