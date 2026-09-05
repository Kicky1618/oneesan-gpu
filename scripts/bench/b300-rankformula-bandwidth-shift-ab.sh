#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; REPEATS="${REPEATS:-2}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-16}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankformula_bandwidth_shift_n${N}}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";RESULT="${RESULT:-${PREFIX}.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
command -v nvcc >/dev/null||exit 2;command -v nvidia-smi >/dev/null||exit 2
(( $(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l) >= NGPU ))||{ echo "need $NGPU GPUs" >&2;exit 2; }

declare -A bin build
bin[compact]="$ONEESAN_BUILD_DIR/b300_bwshift_compact_n${N}";bin[direct64]="$ONEESAN_BUILD_DIR/b300_bwshift_direct64_n${N}";bin[directplan16]="$ONEESAN_BUILD_DIR/b300_bwshift_directplan16_n${N}"
build[compact]=b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh;build[direct64]=b300-bucket-snake-rankformula-direct64.sh;build[directplan16]=b300-bucket-snake-rankformula-directplan16.sh
N="$N" ARCH="$ARCH" OUT="${bin[compact]}" RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 RANKFORMULA_NOMETA_COOPGROUP=1 RANKFORMULA_NOMETA_COOP_UNROLL=0 RANKFORMULA_NOMETA_GROUP56=0 RANKFORMULA_NOMETA_GROUP61=1 RANKFORMULA_ABSTRACT_SELECT8=1 RANKFORMULA_ABSTRACT_DEPTH4=1 RANKFORMULA_ABSTRACT_SRCPACK10=1 TRANSPOSE_MODE=pipeline bash "$ONEESAN_ROOT/scripts/build/${build[compact]}" >"$LOGDIR/compact.build.out" 2>"$LOGDIR/compact.build.err"
for m in direct64 directplan16;do N="$N" ARCH="$ARCH" OUT="${bin[$m]}" bash "$ONEESAN_ROOT/scripts/build/${build[$m]}" >"$LOGDIR/$m.build.out" 2>"$LOGDIR/$m.build.err";done

printf 'mode\trun\tresidue\twall_s\tforward_high_s\treverse_high_s\tmeta_mib\n' >"$RESULT"
field(){ sed -nE "s/(^|.*[[:space:]])$1=([^[:space:]]+).*/\\2/p" <<<"$2"|tail -n1; }
run_one(){ local m="$1" r="$2" out="$LOGDIR/${m}_${r}.out" err="$LOGDIR/${m}_${r}.err";BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" "${bin[$m]}" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$out" 2>"$err";local line detail rr w fh rh meta=0;line="$(grep '^residue=' "$out"|tail -n1)";detail="$(grep 'snake_onepass_graph_batch modulus=' "$err"|tail -n1)";rr="$(field residue "$line")";w="$(field wall_s "$line")";fh="$(field forward_high_s "$detail")";rh="$(field reverse_high_s "$detail")";if [[ "$m" == direct64 ]];then meta="$(grep 'p10dc_low_rankformula_direct64 fixed_owner=' "$err"|sed -nE 's/.* mib=([^ ]+).*/\1/p'|sort -nr|head -n1)";elif [[ "$m" == directplan16 ]];then meta="$(grep 'p10dc_low_rankformula_directplan16 fixed_owner=' "$err"|sed -nE 's/.* mib=([^ ]+).*/\1/p'|sort -nr|head -n1)";fi;printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$m" "$r" "$rr" "$w" "$fh" "$rh" "${meta:-NA}" >>"$RESULT"; }
modes=(compact direct64 directplan16)
for((r=0;r<REPEATS;++r));do for((j=0;j<3;++j));do m="${modes[$(((j+r)%3))]}";run_one "$m" "$r";done;done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));modes=('compact','direct64','directplan16');res={m:{x['residue'] for x in r if x['mode']==m} for m in modes}
if len(res['compact'])!=1 or any(res[m]!=res['compact'] for m in modes):raise SystemExit(f'residue mismatch {res}')
med=lambda m,k:statistics.median(float(x[k]) for x in r if x['mode']==m)
base=med('compact','wall_s')
for m in modes:
 print(f'{m}_median_wall_s={med(m,"wall_s"):.6f} high_s={med(m,"forward_high_s")+med(m,"reverse_high_s"):.6f} speedup_vs_compact={base/med(m,"wall_s"):.6f}x')
print(f'winner={min(modes,key=lambda m:med(m,"wall_s"))} residue={next(iter(res["compact"]))}')
PY
