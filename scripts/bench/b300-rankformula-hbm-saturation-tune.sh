#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; REPEATS="${REPEATS:-1}"
CAPS="${CAPS:-0 96 128 160}"
LAYOUTS="${LAYOUTS:-128:16:8 256:16:8 512:16:8 256:8:16 256:32:4}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || {
    echo "EXPECT required for N=$N MOD=$MOD" >&2; exit 2;
  }
fi
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || {
  echo "need at least $NGPU GPUs" >&2; exit 2;
}

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_saturation_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; CAPS_TSV="${CAPS_TSV:-${PREFIX}_caps.tsv}"
LAYOUT_TSV="${LAYOUT_TSV:-${PREFIX}_layout.tsv}"; mkdir -p "$LOGDIR"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
run_one(){
  local bin="$1" tag="$2" rep="$3" threads="$4" gx="$5" gy="$6"
  local so="$LOGDIR/${tag}_r${rep}.out" se="$LOGDIR/${tag}_r${rep}.err"
  BUCKET_THREADS="$threads" BUCKET_GRID_X="$gx" BUCKET_GRID_Y="$gy" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail residue wall fh rh
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$tag missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$tag residue=$residue expect=$EXPECT" >&2; exit 4; }
  wall="$(field wall_s "$line")"
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  python3 - "$wall" "${fh:-0}" "${rh:-0}" <<'PY'
import sys
w,fh,rh=map(float,sys.argv[1:])
print(f"{w:.9f} {fh+rh:.9f}")
PY
}

printf 'cap\trepeat\twall_s\thigh_s\tmax_regs\tspill_store_bytes\tspill_load_bytes\n' >"$CAPS_TSV"
declare -A BIN
for cap in $CAPS; do
  bin="$ONEESAN_BUILD_DIR/b300_hbm_sat_cap${cap}_n${N}"; BIN[$cap]="$bin"
  bo="$LOGDIR/cap${cap}.build.out"; be="$LOGDIR/cap${cap}.build.err"
  N="$N" ARCH="$ARCH" OUT="$bin" MAXRREGCOUNT="$cap" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-rankformula-hbm-fast.sh" >"$bo" 2>"$be"
  read -r regs ss sl < <(python3 - "$be" <<'PY'
import re,sys
s=open(sys.argv[1],errors='ignore').read()
regs=[int(x) for x in re.findall(r'Used\s+(\d+)\s+registers',s)]
sp=[(int(a),int(b)) for a,b in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',s)]
print(max(regs) if regs else -1, sum(a for a,b in sp), sum(b for a,b in sp))
PY
)
  for ((r=1;r<=REPEATS;++r)); do
    read -r wall high < <(run_one "$bin" "cap${cap}" "$r" 256 16 8)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cap" "$r" "$wall" "$high" "$regs" "$ss" "$sl" >>"$CAPS_TSV"
  done
done

BEST_CAP="$(python3 - "$CAPS_TSV" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
caps=sorted({x['cap'] for x in r},key=int)
def med(c,k):return statistics.median(float(x[k]) for x in r if x['cap']==c)
for c in caps:
 print(f"cap={c} median_wall={med(c,'wall_s'):.6f} median_high={med(c,'high_s'):.6f}",file=sys.stderr)
print(min(caps,key=lambda c:(med(c,'high_s'),med(c,'wall_s'))))
PY
)"
echo "b300_hbm_saturation best_cap=$BEST_CAP" >&2
BEST_BIN="${BIN[$BEST_CAP]}"

printf 'threads\tgx\tgy\trepeat\twall_s\thigh_s\n' >"$LAYOUT_TSV"
for spec in $LAYOUTS; do
  IFS=: read -r threads gx gy <<<"$spec"
  [[ "$threads" =~ ^(32|64|96|128|160|192|224|256|288|320|352|384|416|448|480|512|544|576|608|640|672|704|736|768|800|832|864|896|928|960|992|1024)$ ]] || { echo "bad threads=$threads" >&2; exit 2; }
  for ((r=1;r<=REPEATS;++r)); do
    read -r wall high < <(run_one "$BEST_BIN" "layout_t${threads}_x${gx}_y${gy}" "$r" "$threads" "$gx" "$gy")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$threads" "$gx" "$gy" "$r" "$wall" "$high" >>"$LAYOUT_TSV"
  done
done
python3 - "$LAYOUT_TSV" "$BEST_CAP" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); cap=sys.argv[2]
keys=sorted({(x['threads'],x['gx'],x['gy']) for x in r},key=lambda z:tuple(map(int,z)))
def med(k,f):return statistics.median(float(x[f]) for x in r if (x['threads'],x['gx'],x['gy'])==k)
best=min(keys,key=lambda k:(med(k,'high_s'),med(k,'wall_s')))
for k in keys: print(f"layout={k[0]}:{k[1]}:{k[2]} median_wall={med(k,'wall_s'):.6f} median_high={med(k,'high_s'):.6f}")
print(f"BEST cap={cap} threads={best[0]} gx={best[1]} gy={best[2]} wall_s={med(best,'wall_s'):.6f} high_s={med(best,'high_s'):.6f}")
PY

echo "b300-rankformula-hbm-saturation-tune OK caps=$CAPS_TSV layouts=$LAYOUT_TSV" >&2
