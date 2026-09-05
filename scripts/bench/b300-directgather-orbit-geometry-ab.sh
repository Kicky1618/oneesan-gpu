#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
REPEATS="${REPEATS:-1}"; PM_ACCUM="${PM_ACCUM:-0}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }; fi
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }

BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_directgather_orbitgeom_n${N}}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather_orbitgeom_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

N="$N" ARCH="$ARCH" OUT="$BIN" \
  RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 \
  RANKFORMULA_NOMETA_COOPGROUP=1 RANKFORMULA_NOMETA_COOP_UNROLL=0 \
  RANKFORMULA_NOMETA_GROUP56=0 RANKFORMULA_NOMETA_GROUP61=1 \
  RANKFORMULA_NOMETA_DIRECTMAP=1 RANKFORMULA_DIRECTGATHER=1 \
  RANKFORMULA_DIRECTGATHER_DEPTHMAJOR=1 RANKFORMULA_DIRECTGATHER_FORCE7=0 \
  RANKFORMULA_MLP_WINDOW4=0 RANKFORMULA_PREFETCH_NEXT=0 RANKFORMULA_PAIR_MLP=0 \
  WARPSTRIPED_COL_ILP=1 RANKFORMULA_ABSTRACT_SELECT8=1 \
  RANKFORMULA_ABSTRACT_DEPTH4=1 RANKFORMULA_ABSTRACT_SRCPACK10=1 \
  RANKFORMULA_GATHER_MLP=1 DEPTHCODE_DECODE_LOAD=ldg RANKSTREAM_LUT_LOAD=ldg \
  PM_ACCUM="$PM_ACCUM" TRANSPOSE_MODE="$TRANSPOSE_MODE" \
  bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh" \
  >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
# Keep gx*gy roughly constant while reducing duplicated orbit setup in X.
# Every X CTA rebuilds the same warp-private orbit contexts for its column slice.
CASES=(
  "256 32 4"
  "256 16 8"
  "256 8 16"
  "256 4 32"
  "256 2 64"
  "256 1 128"
  "128 16 8"
  "128 8 16"
  "128 4 32"
  "128 2 64"
  "128 1 128"
  "64 8 16"
  "64 4 32"
  "64 2 64"
  "64 1 128"
)
printf 'threads\thigh_gx\thigh_gy\tsetup_dup_x\trepeat\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"

for spec in "${CASES[@]}"; do
  read -r threads gx gy <<<"$spec"
  for ((rep=1;rep<=REPEATS;++rep)); do
    so="$LOGDIR/t${threads}_x${gx}_y${gy}_r${rep}.out"; se="$LOGDIR/t${threads}_x${gx}_y${gy}_r${rep}.err"
    BUCKET_THREADS="$threads" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
    BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
    BUCKET_HIGH_GRID_X="$gx" BUCKET_HIGH_GRID_Y="$gy" \
      "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || exit 3
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "residue mismatch t=$threads x=$gx y=$gy got=$residue" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$threads" "$gx" "$gy" "$gx" "$rep" "$(field wall_s "$line")" \
      "$(field forward_high_s "$detail")" "$(field reverse_high_s "$detail")" \
      "$(field forward_low_s "$detail")" "$(field reverse_low_s "$detail")" \
      "$(field transpose_s "$detail")" >>"$RESULT"
  done
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
g={}
for r in rows:
    k=(int(r['threads']),int(r['high_gx']),int(r['high_gy']))
    g.setdefault(k,[]).append(r)
out=[]
for k,rs in g.items():
    z={x:statistics.median(float(r[x]) for r in rs) for x in ('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s')}
    z['high']=z['forward_high_s']+z['reverse_high_s']
    out.append((z['wall_s'],k,z))
for _,k,z in sorted(out):
    print('orbit_geometry',*k,f"wall={z['wall_s']:.6f}",f"high={z['high']:.6f}",f"setup_dup_x={k[1]}")
print('BEST',sorted(out)[0][1],f"wall={sorted(out)[0][0]:.6f}")
PY

echo "result=$RESULT depthmajor=1 pm_accum=$PM_ACCUM" >&2
