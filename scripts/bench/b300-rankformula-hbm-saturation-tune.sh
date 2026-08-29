#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"; REPEATS="${REPEATS:-1}"
MODES="${MODES:-full window4 force7}"; CAPS="${CAPS:-0 96 128 160}"; LAYOUTS="${LAYOUTS:-128:16:8 256:16:8 512:16:8 256:8:16 256:32:4}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }; fi
command -v nvcc >/dev/null && command -v nvidia-smi >/dev/null || { echo "nvcc+nvidia-smi required" >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l) >= NGPU )) || exit 2
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_saturation_n${N}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; MODE_TSV="${PREFIX}_modes.tsv"; CAPS_TSV="${PREFIX}_caps.tsv"; LAYOUT_TSV="${PREFIX}_layout.tsv"; mkdir -p "$LOGDIR"
field(){ local k="$1" s="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$s"|tail -n1; }
run_one(){ local bin="$1" tag="$2" rep="$3" th="$4" gx="$5" gy="$6" so="$LOGDIR/${tag}_r${rep}.out" se="$LOGDIR/${tag}_r${rep}.err"; BUCKET_THREADS="$th" BUCKET_GRID_X="$gx" BUCKET_GRID_Y="$gy" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"; local l d res wall fh rh;l="$(grep '^residue=' "$so"|tail -n1)";d="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1)";res="$(field residue "$l")";[[ "$res" == "$EXPECT" ]]||exit 4;wall="$(field wall_s "$l")";fh="$(field forward_high_s "$d")";rh="$(field reverse_high_s "$d")";python3 - "$wall" "$fh" "$rh" <<'PY'
import sys
w,a,b=map(float,sys.argv[1:]);print(f'{w:.9f} {a+b:.9f}')
PY
}
flags(){ case "$1" in full) echo '0 0';; window4) echo '0 1';; force7) echo '1 0';; *) exit 2;; esac; }
build_one(){ local mode="$1" cap="$2" out="$3" bo="$4" be="$5";read -r force win < <(flags "$mode");N="$N" ARCH="$ARCH" OUT="$out" RANKFORMULA_DIRECTGATHER_FORCE7="$force" RANKFORMULA_MLP_WINDOW4="$win" MAXRREGCOUNT="$cap" PTXAS_VERBOSE=1 bash "$ONEESAN_ROOT/scripts/build/b300-rankformula-hbm-fast.sh" >"$bo" 2>"$be"; }

printf 'mode\trepeat\twall_s\thigh_s\n' >"$MODE_TSV";declare -A MODE_BIN
for m in $MODES;do b="$ONEESAN_BUILD_DIR/b300_hbm_mode_${m}_n${N}";MODE_BIN[$m]="$b";build_one "$m" 0 "$b" "$LOGDIR/mode_${m}.build.out" "$LOGDIR/mode_${m}.build.err";for((r=1;r<=REPEATS;++r));do read -r w h < <(run_one "$b" "mode_${m}" "$r" 256 16 8);printf '%s\t%s\t%s\t%s\n' "$m" "$r" "$w" "$h" >>"$MODE_TSV";done;done
BEST_MODE="$(python3 - "$MODE_TSV" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));ms=sorted({x['mode'] for x in r});med=lambda m,k:statistics.median(float(x[k]) for x in r if x['mode']==m)
for m in ms:print(f'mode={m} wall={med(m,"wall_s"):.6f} high={med(m,"high_s"):.6f}',file=sys.stderr)
print(min(ms,key=lambda m:(med(m,'high_s'),med(m,'wall_s'))))
PY
)";echo "best_mode=$BEST_MODE" >&2

printf 'cap\trepeat\twall_s\thigh_s\tmax_regs\tspill_store_bytes\tspill_load_bytes\n' >"$CAPS_TSV";declare -A CAP_BIN
for cap in $CAPS;do b="$ONEESAN_BUILD_DIR/b300_hbm_${BEST_MODE}_cap${cap}_n${N}";CAP_BIN[$cap]="$b";be="$LOGDIR/cap${cap}.build.err";build_one "$BEST_MODE" "$cap" "$b" "$LOGDIR/cap${cap}.build.out" "$be";read -r regs ss sl < <(python3 - "$be" <<'PY'
import re,sys
s=open(sys.argv[1],errors='ignore').read();r=[int(x) for x in re.findall(r'Used\s+(\d+)\s+registers',s)];p=[tuple(map(int,x)) for x in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',s)];print(max(r) if r else -1,sum(x for x,y in p),sum(y for x,y in p))
PY
);for((r=1;r<=REPEATS;++r));do read -r w h < <(run_one "$b" "cap${cap}" "$r" 256 16 8);printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cap" "$r" "$w" "$h" "$regs" "$ss" "$sl" >>"$CAPS_TSV";done;done
BEST_CAP="$(python3 - "$CAPS_TSV" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));cs=sorted({x['cap'] for x in r},key=int);med=lambda c,k:statistics.median(float(x[k]) for x in r if x['cap']==c)
for c in cs:print(f'cap={c} high={med(c,"high_s"):.6f} wall={med(c,"wall_s"):.6f}',file=sys.stderr)
print(min(cs,key=lambda c:(med(c,'high_s'),med(c,'wall_s'))))
PY
)";echo "best_cap=$BEST_CAP" >&2;BEST_BIN="${CAP_BIN[$BEST_CAP]}"

printf 'threads\tgx\tgy\trepeat\twall_s\thigh_s\n' >"$LAYOUT_TSV"
for s in $LAYOUTS;do IFS=: read -r th gx gy <<<"$s";for((r=1;r<=REPEATS;++r));do read -r w h < <(run_one "$BEST_BIN" "layout_${th}_${gx}_${gy}" "$r" "$th" "$gx" "$gy");printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$th" "$gx" "$gy" "$r" "$w" "$h" >>"$LAYOUT_TSV";done;done
python3 - "$LAYOUT_TSV" "$BEST_MODE" "$BEST_CAP" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));mode,cap=sys.argv[2:];ks=sorted({(x['threads'],x['gx'],x['gy']) for x in r});med=lambda k,f:statistics.median(float(x[f]) for x in r if (x['threads'],x['gx'],x['gy'])==k);b=min(ks,key=lambda k:(med(k,'high_s'),med(k,'wall_s')))
for k in ks:print(f'layout={":".join(k)} high={med(k,"high_s"):.6f} wall={med(k,"wall_s"):.6f}')
print(f'BEST mode={mode} cap={cap} threads={b[0]} gx={b[1]} gy={b[2]} high_s={med(b,"high_s"):.6f} wall_s={med(b,"wall_s"):.6f}')
PY
