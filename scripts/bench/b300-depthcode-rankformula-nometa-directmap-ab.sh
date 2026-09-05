#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; REPEATS="${REPEATS:-3}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
GATHER_MLP="${GATHER_MLP:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-16}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
RUN_SELFTEST="${RUN_SELFTEST:-1}"; RUN_PTXAS="${RUN_PTXAS:-1}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }; fi
for x in PM_ACCUM TERNARY_KEY4 GATHER_MLP RUN_SELFTEST RUN_PTXAS; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) exit 2;; esac
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankformula_nometa_directmap_ab_n${N}_${TRANSPOSE_MODE}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

build_one(){
  local direct="$1" gather="$2" label="$3" bin="$4"
  N="$N" ARCH="$ARCH" OUT="$bin" \
    RANKFORMULA_NOMETA_BLOCK=16 \
    RANKFORMULA_NOMETA_WARPSHARE=1 \
    RANKFORMULA_NOMETA_COOPGROUP=1 \
    RANKFORMULA_NOMETA_COOP_UNROLL=0 \
    RANKFORMULA_NOMETA_GROUP56=0 \
    RANKFORMULA_NOMETA_GROUP61=1 \
    RANKFORMULA_NOMETA_DIRECTMAP="$direct" \
    RANKFORMULA_DIRECTGATHER="$gather" \
    RANKFORMULA_ABSTRACT_SELECT8=1 \
    RANKFORMULA_ABSTRACT_DEPTH4=1 \
    RANKFORMULA_ABSTRACT_SRCPACK10=1 \
    RANKFORMULA_GATHER_MLP="$GATHER_MLP" \
    DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
    RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
    PTXAS_VERBOSE="$RUN_PTXAS" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh" \
    >"$LOGDIR/$label.build.out" 2>"$LOGDIR/$label.build.err"
}

run_one(){
  local label="$1" bin="$2" rep="$3"
  local so="$LOGDIR/${label}_r${rep}.out" se="$LOGDIR/${label}_r${rep}.err"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail residue wall fh fl rl rh ts
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || exit 3
  residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$label residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  wall="$(field wall_s "$line")"
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; fl="$(field forward_low_s "$detail")"
  rl="$(field reverse_low_s "$detail")"; rh="$(field reverse_high_s "$detail")"; ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$rep" "$residue" "$wall" "${fh:-NA}" "${fl:-NA}" "${rl:-NA}" "${rh:-NA}" "${ts:-NA}" >>"$RESULT"
}

printf 'mode\trepeat\tresidue\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\n' >"$RESULT"
[[ "$RUN_PTXAS" == 1 ]] && printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

for spec in 'coop 0 0' 'directmap 1 0' 'directgather 1 1'; do
  read -r label direct gather <<<"$spec"
  bin="$ONEESAN_BUILD_DIR/ab_group61_${label}_n${N}"
  build_one "$direct" "$gather" "$label" "$bin"
  if [[ "$RUN_PTXAS" == 1 ]]; then python3 "$PARSER" "$LOGDIR/$label.build.err" --label "$label" >>"$RESOURCE"; fi
  for ((r=1;r<=REPEATS;++r)); do run_one "$label" "$bin" "$r"; done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
keys=('wall_s','forward_high_s','forward_low_s','reverse_low_s','reverse_high_s','transpose_s')
out=[]
for mode in ('coop','directmap','directgather'):
    g=[r for r in rows if r['mode']==mode]
    z={'mode':mode,'repeats':len(g)}
    for k in keys:
        xs=[float(r[k]) for r in g if r[k]!='NA']
        z[k]=statistics.median(xs) if xs else None
    z['high_total_s']=(z['forward_high_s']+z['reverse_high_s']) if z['forward_high_s'] is not None and z['reverse_high_s'] is not None else None
    out.append(z)
with open(dst,'w') as f:
    f.write('mode\trepeats\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\thigh_total_s\ttranspose_s\n')
    for z in out:
        f.write('\t'.join(str(z[k]) for k in ('mode','repeats','wall_s','forward_high_s','forward_low_s','reverse_low_s','reverse_high_s','high_total_s','transpose_s'))+'\n')
q={z['mode']:z for z in out}
for mode in ('directmap','directgather'):
    for k in ('wall_s','forward_high_s','reverse_high_s','high_total_s'):
        a,b=q['coop'][k],q[mode][k]
        if a and b:
            print(f'{mode}_{k}_speedup={a/b:.6f} coop={a:.6f} candidate={b:.6f}')
print('summary='+dst)
PY
cat "$SUMMARY"

echo "directmap: 1 rank descriptor load; directgather: 1 16-byte gather descriptor then Count loads. Inspect p10dc_low_rankformula_direct* lines for exact table sizes." >&2
