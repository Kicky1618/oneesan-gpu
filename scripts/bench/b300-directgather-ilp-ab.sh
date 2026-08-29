#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
REPEATS="${REPEATS:-1}"; THREADS="${THREADS:-256}"
HIGH_GX="${HIGH_GX:-128}"; HIGH_GY="${HIGH_GY:-1}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }; fi
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather_ilp_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

# mode ilp prefetch pair window4 maxrreg
CASES=(
  "base 1 0 0 0 0"
  "prefetch 1 1 0 0 0"
  "ilp2 2 0 0 0 0"
  "ilp2_prefetch 2 1 0 0 0"
  "ilp2_pair 2 0 1 0 0"
  "ilp2_pair_prefetch 2 1 1 0 0"
  "ilp4_pair 4 0 1 0 0"
  "ilp4_pair_prefetch 4 1 1 0 0"
  "ilp2_pair_r128 2 0 1 0 128"
  "ilp2_window4 2 0 0 1 0"
)

printf 'mode\tilp\tprefetch\tpair\twindow4\tmaxrreg\trepeat\tresidue\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

for spec in "${CASES[@]}"; do
  read -r mode ilp prefetch pair window4 rcap <<<"$spec"
  bin="$ONEESAN_BUILD_DIR/directgather_ilp_${mode}_n${N}"
  N="$N" ARCH="$ARCH" OUT="$bin" \
    RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 \
    RANKFORMULA_NOMETA_COOPGROUP=1 RANKFORMULA_NOMETA_COOP_UNROLL=0 \
    RANKFORMULA_NOMETA_GROUP56=0 RANKFORMULA_NOMETA_GROUP61=1 \
    RANKFORMULA_NOMETA_DIRECTMAP=1 RANKFORMULA_DIRECTGATHER=1 \
    RANKFORMULA_DIRECTGATHER_DEPTHMAJOR=1 RANKFORMULA_DIRECTGATHER_FORCE7=0 \
    RANKFORMULA_MLP_WINDOW4="$window4" RANKFORMULA_PREFETCH_NEXT="$prefetch" \
    RANKFORMULA_PAIR_MLP="$pair" WARPSTRIPED_COL_ILP="$ilp" \
    RANKFORMULA_ABSTRACT_SELECT8=1 RANKFORMULA_ABSTRACT_DEPTH4=1 \
    RANKFORMULA_ABSTRACT_SRCPACK10=1 RANKFORMULA_GATHER_MLP=1 \
    DEPTHCODE_DECODE_LOAD=ldg RANKSTREAM_LUT_LOAD=ldg PM_ACCUM="$PM_ACCUM" \
    MAXRREGCOUNT="$rcap" TERNARY_KEY4="$TERNARY_KEY4" TRANSPOSE_MODE="$TRANSPOSE_MODE" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh" \
    >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
  python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$RESOURCE" || true

  for ((rep=1; rep<=REPEATS; ++rep)); do
    so="$LOGDIR/${mode}_r${rep}.out"; se="$LOGDIR/${mode}_r${rep}.err"
    BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
    BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
    BUCKET_HIGH_GRID_X="$HIGH_GX" BUCKET_HIGH_GRID_Y="$HIGH_GY" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$mode" "$ilp" "$prefetch" "$pair" "$window4" "$rcap" "$rep" "$residue" \
      "$(field wall_s "$line")" "$(field forward_high_s "$detail")" "$(field forward_low_s "$detail")" \
      "$(field reverse_low_s "$detail")" "$(field reverse_high_s "$detail")" "$(field transpose_s "$detail")" >>"$RESULT"
  done
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
by={}
for r in rows: by.setdefault(r['mode'],[]).append(r)
out=[]
for mode,g in by.items():
    z={k:statistics.median(float(x[k]) for x in g) for k in ('wall_s','forward_high_s','forward_low_s','reverse_low_s','reverse_high_s','transpose_s')}
    z['high_total_s']=z['forward_high_s']+z['reverse_high_s']
    out.append((z['wall_s'],mode,z))
for _,mode,z in sorted(out):
    print(mode,f"wall={z['wall_s']:.6f}",f"high={z['high_total_s']:.6f}",f"fh={z['forward_high_s']:.6f}",f"rh={z['reverse_high_s']:.6f}")
q={m:z for _,m,z in out}
if 'base' in q:
    for _,m,z in sorted(out):
        print(f"speedup_vs_base {m} wall={q['base']['wall_s']/z['wall_s']:.6f} high={q['base']['high_total_s']/z['high_total_s']:.6f}")
print('BEST',sorted(out)[0][1],f"wall={sorted(out)[0][0]:.6f}")
PY

echo "result=$RESULT ptxas=$RESOURCE depthmajor=1 geometry=threads${THREADS}-high${HIGH_GX}x${HIGH_GY}-low${LOW_GX}x${LOW_GY}" >&2
