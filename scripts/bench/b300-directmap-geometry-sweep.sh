#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
REPEATS="${REPEATS:-2}"; PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }; fi
command -v nvcc >/dev/null || exit 2
command -v nvidia-smi >/dev/null || exit 2

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directmap_geometry_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_directmap_geometry_n${N}}"

N="$N" ARCH="$ARCH" OUT="$BIN" \
  RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 \
  RANKFORMULA_NOMETA_COOPGROUP=1 RANKFORMULA_NOMETA_COOP_UNROLL=0 \
  RANKFORMULA_NOMETA_GROUP56=0 RANKFORMULA_NOMETA_GROUP61=1 \
  RANKFORMULA_NOMETA_DIRECTMAP=1 RANKFORMULA_ABSTRACT_SELECT8=1 \
  RANKFORMULA_ABSTRACT_DEPTH4=1 RANKFORMULA_ABSTRACT_SRCPACK10=1 \
  RANKFORMULA_GATHER_MLP=1 DEPTHCODE_DECODE_LOAD=ldg RANKSTREAM_LUT_LOAD=ldg \
  TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
  bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh" \
  >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
# HIGH uses x for rank columns and y for orbit operations; LOW swaps the useful
# dimensions.  Sweep both axes rather than simply maximizing total CTA count.
CASES=(
  "256 16 8"
  "256 32 8"
  "256 64 8"
  "256 16 16"
  "128 32 16"
  "128 64 16"
  "128 64 32"
)
printf 'threads\tgx\tgy\trepeat\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\n' >"$RESULT"

for spec in "${CASES[@]}"; do
  read -r threads gx gy <<<"$spec"
  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/t${threads}_x${gx}_y${gy}_r${r}.out"
    se="$LOGDIR/t${threads}_x${gx}_y${gy}_r${r}.err"
    BUCKET_THREADS="$threads" BUCKET_GRID_X="$gx" BUCKET_GRID_Y="$gy" \
      "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || exit 3
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "residue mismatch geometry=$spec got=$residue" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$threads" "$gx" "$gy" "$r" \
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
    key=(int(r['threads']),int(r['gx']),int(r['gy']))
    groups.setdefault(key,[]).append(r)
out=[]
for key,g in groups.items():
    z={k:statistics.median(float(r[k]) for r in g) for k in ('wall_s','forward_high_s','forward_low_s','reverse_low_s','reverse_high_s','transpose_s')}
    z['high_total_s']=z['forward_high_s']+z['reverse_high_s']
    out.append((z['wall_s'],key,z))
for _,key,z in sorted(out):
    print('geometry',*key,
          f"wall={z['wall_s']:.6f}",
          f"high={z['high_total_s']:.6f}",
          f"fh={z['forward_high_s']:.6f}",
          f"rh={z['reverse_high_s']:.6f}",
          f"low={z['forward_low_s']+z['reverse_low_s']:.6f}",
          f"transpose={z['transpose_s']:.6f}")
print('BEST',sorted(out)[0][1],f"wall={sorted(out)[0][0]:.6f}")
PY

echo "result=$RESULT" >&2
