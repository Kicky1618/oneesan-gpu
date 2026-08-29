#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
REPEATS="${REPEATS:-2}"; PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }; fi
command -v nvcc >/dev/null || exit 2
command -v nvidia-smi >/dev/null || exit 2

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather_geometry_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_directgather_geometry_n${N}}"

N="$N" ARCH="$ARCH" OUT="$BIN" \
  RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 \
  RANKFORMULA_NOMETA_COOPGROUP=1 RANKFORMULA_NOMETA_COOP_UNROLL=0 \
  RANKFORMULA_NOMETA_GROUP56=0 RANKFORMULA_NOMETA_GROUP61=1 \
  RANKFORMULA_NOMETA_DIRECTMAP=1 RANKFORMULA_DIRECTGATHER=1 \
  RANKFORMULA_DIRECTGATHER_DEPTHMAJOR=1 RANKFORMULA_DIRECTGATHER_FORCE7=0 \
  RANKFORMULA_MLP_WINDOW4=0 RANKFORMULA_PREFETCH_NEXT=0 RANKFORMULA_PAIR_MLP=0 \
  WARPSTRIPED_COL_ILP=1 \
  RANKFORMULA_ABSTRACT_SELECT8=1 RANKFORMULA_ABSTRACT_DEPTH4=1 \
  RANKFORMULA_ABSTRACT_SRCPACK10=1 RANKFORMULA_GATHER_MLP=1 \
  DEPTHCODE_DECODE_LOAD=ldg RANKSTREAM_LUT_LOAD=ldg \
  TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
  bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh" \
  >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
# HIGH uses X for 32-column stripes and Y for orbit groups.  With 256 threads a
# single Y CTA already exposes eight warps/orbits, so test increasingly X-heavy
# layouts. LOW keeps its conservative geometry and is not conflated with HIGH.
CASES=(
  "256 16 8"
  "256 32 4"
  "256 64 2"
  "256 128 1"
  "256 64 4"
  "256 128 2"
  "256 256 1"
  "128 128 1"
  "128 256 1"
  "64 256 1"
  "64 512 1"
)
printf 'threads\thigh_gx\thigh_gy\tlow_gx\tlow_gy\trepeat\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\n' >"$RESULT"

for spec in "${CASES[@]}"; do
  read -r threads hgx hgy <<<"$spec"
  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/t${threads}_hx${hgx}_hy${hgy}_r${r}.out"
    se="$LOGDIR/t${threads}_hx${hgx}_hy${hgy}_r${r}.err"
    BUCKET_THREADS="$threads" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
    BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
    BUCKET_HIGH_GRID_X="$hgx" BUCKET_HIGH_GRID_Y="$hgy" \
      "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || exit 3
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "residue mismatch geometry=$spec got=$residue" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$threads" "$hgx" "$hgy" "$LOW_GX" "$LOW_GY" "$r" \
      "$(field wall_s "$line")" "$(field forward_high_s "$detail")" \
      "$(field forward_low_s "$detail")" "$(field reverse_low_s "$detail")" \
      "$(field reverse_high_s "$detail")" "$(field transpose_s "$detail")" >>"$RESULT"
  done
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
groups={}
for r in rows:
    key=(int(r['threads']),int(r['high_gx']),int(r['high_gy']))
    groups.setdefault(key,[]).append(r)
out=[]
for key,g in groups.items():
    z={k:statistics.median(float(r[k]) for r in g) for k in ('wall_s','forward_high_s','forward_low_s','reverse_low_s','reverse_high_s','transpose_s')}
    z['high_total_s']=z['forward_high_s']+z['reverse_high_s']
    out.append((z['wall_s'],key,z))
for _,key,z in sorted(out):
    print('high_geometry',*key,
          f"wall={z['wall_s']:.6f}",
          f"high={z['high_total_s']:.6f}",
          f"fh={z['forward_high_s']:.6f}",
          f"rh={z['reverse_high_s']:.6f}",
          f"low={z['forward_low_s']+z['reverse_low_s']:.6f}",
          f"transpose={z['transpose_s']:.6f}")
print('BEST',sorted(out)[0][1],f"wall={sorted(out)[0][0]:.6f}")
PY

echo "result=$RESULT pm_accum=$PM_ACCUM depthmajor=1 low_geometry=${LOW_GX}x${LOW_GY}" >&2
