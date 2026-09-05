#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; REPEATS="${REPEATS:-2}"
THREADS="${THREADS:-256}"; ORBIT_GY="${ORBIT_GY:-128}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
PRECTX_FORWARD="${PRECTX_FORWARD:-0}"; PRECTX_REVERSE="${PRECTX_REVERSE:-0}"
PRECTX_COMPACT="${PRECTX_COMPACT:-0}"; SPARSE64="${SPARSE64:-0}"
PM_ACCUM="${PM_ACCUM:-0}"; COL_ILP="${COL_ILP:-1}"; WINDOW4="${WINDOW4:-0}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || {
    echo "EXPECT required unless N=21 MOD=4294967291" >&2; exit 2;
  }
fi
for x in PRECTX_FORWARD PRECTX_REVERSE PRECTX_COMPACT SPARSE64 PM_ACCUM WINDOW4; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
case "$COL_ILP" in 1|2|4) ;; *) echo "COL_ILP must be 1,2,4" >&2; exit 2;; esac
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbitcta_sort_ranks_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
printf 'sort_ranks\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"

for sort in 0 1; do
  bin="$ONEESAN_BUILD_DIR/orbitcta_sort${sort}_n${N}"
  N="$N" ARCH="$ARCH" OUT="$bin" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" \
    DIRECTGATHER_SORT_RANKS="$sort" PRECTX_FORWARD="$PRECTX_FORWARD" \
    PRECTX_REVERSE="$PRECTX_REVERSE" PRECTX_COMPACT="$PRECTX_COMPACT" \
    ORBITCTA_COL_ILP="$COL_ILP" PAIR_MLP=0 CPASYNC_PAIR=0 \
    RANKFORMULA_MLP_WINDOW4="$WINDOW4" PM_ACCUM="$PM_ACCUM" PTXAS_VERBOSE=0 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
    >"$LOGDIR/sort${sort}.build.out" 2>"$LOGDIR/sort${sort}.build.err"
  for ((rep=1; rep<=REPEATS; ++rep)); do
    so="$LOGDIR/sort${sort}_r${rep}.out"; se="$LOGDIR/sort${sort}_r${rep}.err"
    BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
    BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
    BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || exit 3
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || {
      echo "sort=$sort residue mismatch got=$residue expected=$EXPECT" >&2; exit 4;
    }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$sort" "$rep" "$residue" "$(field wall_s "$line")" \
      "$(field forward_high_s "$detail")" "$(field reverse_high_s "$detail")" \
      "$(field forward_low_s "$detail")" "$(field reverse_low_s "$detail")" \
      "$(field transpose_s "$detail")" >>"$RESULT"
  done
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
by={}
for r in rows: by.setdefault(r['sort_ranks'],[]).append(r)
z={}
for k,g in by.items():
    q={x:statistics.median(float(r[x]) for r in g)
       for x in ('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s')}
    q['high']=q['forward_high_s']+q['reverse_high_s']; z[k]=q
    print('sorted='+k,f"wall={q['wall_s']:.6f}",f"high={q['high']:.6f}",
          f"fh={q['forward_high_s']:.6f}",f"rh={q['reverse_high_s']:.6f}")
if '0' in z and '1' in z:
    print(f"sorted_speedup wall={z['0']['wall_s']/z['1']['wall_s']:.6f} "
          f"high={z['0']['high']/z['1']['high']:.6f} "
          f"fh={z['0']['forward_high_s']/z['1']['forward_high_s']:.6f} "
          f"rh={z['0']['reverse_high_s']/z['1']['reverse_high_s']:.6f}")
PY

echo "result=$RESULT prectx=${PRECTX_FORWARD}/${PRECTX_REVERSE} compact=$PRECTX_COMPACT sparse64=$SPARSE64 geometry=t${THREADS}-y${ORBIT_GY}" >&2
