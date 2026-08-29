#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; REPEATS="${REPEATS:-1}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"
THREADS="${BUCKET_THREADS:-256}"; GX="${BUCKET_GRID_X:-16}"; GY="${BUCKET_GRID_Y:-8}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo EXPECT required >&2; exit 2; }; fi
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_force7_n${N}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; TSV="${TSV:-${PREFIX}.tsv}"; mkdir -p "$LOGDIR"
field(){ local k="$1" s="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$s"|tail -n1; }
printf 'mode\trepeat\tresidue\twall_s\thigh_s\n' >"$TSV"
for force in 0 1; do
  mode=$([[ "$force" == 1 ]]&&echo force7||echo selective); bin="$ONEESAN_BUILD_DIR/b300_directgather_${mode}_n${N}"
  N="$N" ARCH="$ARCH" OUT="$bin" RANKFORMULA_DIRECTGATHER_FORCE7="$force" PTXAS_VERBOSE=1 bash "$ONEESAN_ROOT/scripts/build/b300-rankformula-hbm-fast.sh" >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
  for((r=1;r<=REPEATS;++r));do
    so="$LOGDIR/${mode}_r${r}.out";se="$LOGDIR/${mode}_r${r}.err"
    BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$GX" BUCKET_GRID_Y="$GY" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    line="$(grep '^residue=' "$so"|tail -n1)";detail="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1)";res="$(field residue "$line")";[[ "$res" == "$EXPECT" ]]||exit 4
    wall="$(field wall_s "$line")";fh="$(field forward_high_s "$detail")";rh="$(field reverse_high_s "$detail")";high="$(python3 -c 'import sys;print(float(sys.argv[1])+float(sys.argv[2]))' "$fh" "$rh")"
    printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$r" "$res" "$wall" "$high" >>"$TSV"
  done
done
python3 - "$TSV" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
for m in ('selective','force7'):
 g=[x for x in r if x['mode']==m];print(m,'wall',statistics.median(float(x['wall_s']) for x in g),'high',statistics.median(float(x['high_s']) for x in g))
a=[x for x in r if x['mode']=='selective'];b=[x for x in r if x['mode']=='force7'];ma=statistics.median(float(x['high_s']) for x in a);mb=statistics.median(float(x['high_s']) for x in b);print(f'force7_high_speedup={ma/mb:.6f}x')
PY
