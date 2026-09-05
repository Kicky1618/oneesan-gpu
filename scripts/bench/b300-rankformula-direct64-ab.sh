#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; REPEATS="${REPEATS:-2}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-16}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankformula_direct64_ab_n${N}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU GPUs" >&2; exit 2; }

BASE="$ONEESAN_BUILD_DIR/b300_rankformula_group61_ab_n${N}"
DIRECT="$ONEESAN_BUILD_DIR/b300_rankformula_direct64_ab_n${N}"
N="$N" ARCH="$ARCH" OUT="$BASE" RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 RANKFORMULA_NOMETA_COOPGROUP=1 RANKFORMULA_NOMETA_COOP_UNROLL=0 RANKFORMULA_NOMETA_GROUP56=0 RANKFORMULA_NOMETA_GROUP61=1 RANKFORMULA_ABSTRACT_SELECT8=1 RANKFORMULA_ABSTRACT_DEPTH4=1 RANKFORMULA_ABSTRACT_SRCPACK10=1 TRANSPOSE_MODE=pipeline bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh" >"$LOGDIR/base.build.out" 2>"$LOGDIR/base.build.err"
N="$N" ARCH="$ARCH" OUT="$DIRECT" bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-rankformula-direct64.sh" >"$LOGDIR/direct.build.out" 2>"$LOGDIR/direct.build.err"

echo -e 'mode\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tdirect64_mib' >"$RESULT"
field(){ local k="$1" s="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$s" | tail -n1; }
run_one(){ local mode="$1" bin="$2" r="$3" out="$LOGDIR/${mode}_${r}.out" err="$LOGDIR/${mode}_${r}.err"; BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$out" 2>"$err"; local line detail residue wall fh rh mib; line="$(grep '^residue=' "$out"|tail -n1)"; detail="$(grep 'snake_onepass_graph_batch modulus=' "$err"|tail -n1)"; residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; mib=0; if [[ "$mode" == direct64 ]];then mib="$(grep 'p10dc_low_rankformula_direct64 fixed_owner=' "$err"|sed -nE 's/.* mib=([^ ]+).*/\1/p'|sort -nr|head -n1)"; mib="${mib:-NA}";fi; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$r" "$residue" "$wall" "$fh" "$rh" "$mib" >>"$RESULT"; }
for ((r=1;r<=REPEATS;++r));do if ((r&1));then run_one base "$BASE" "$r";run_one direct64 "$DIRECT" "$r";else run_one direct64 "$DIRECT" "$r";run_one base "$BASE" "$r";fi;done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:];r=list(csv.DictReader(open(src),delimiter='\t'));modes=('base','direct64');res={m:{x['residue'] for x in r if x['mode']==m} for m in modes}
if len(res['base'])!=1 or res['base']!=res['direct64']:raise SystemExit(f'residue mismatch {res}')
med=lambda m,k:statistics.median(float(x[k]) for x in r if x['mode']==m)
rows=[]
for m in modes: rows.append((m,med(m,'wall_s'),med(m,'forward_high_s'),med(m,'reverse_high_s')))
with open(dst,'w') as f:
 f.write('mode\tmedian_wall_s\tmedian_forward_high_s\tmedian_reverse_high_s\n')
 for x in rows:f.write('\t'.join(map(str,x))+'\n')
b=dict(zip(('mode','wall','fh','rh'),rows[0]));d=dict(zip(('mode','wall','fh','rh'),rows[1]))
print(f'direct64_wall_speedup={b["wall"]/d["wall"]:.6f}x')
print(f'direct64_forward_high_speedup={b["fh"]/d["fh"]:.6f}x')
print(f'direct64_reverse_high_speedup={b["rh"]/d["rh"]:.6f}x')
print(f'residue={next(iter(res["base"]))}')
PY
