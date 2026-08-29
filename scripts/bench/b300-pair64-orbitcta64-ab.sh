#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; REPEATS="${REPEATS:-1}"
THREADS="${BUCKET_THREADS:-256}"; PAIR_GX="${PAIR_GX:-32}"; PAIR_GY="${PAIR_GY:-8}"
ORBIT_GY="${ORBIT_GY:-128}"; LOW_GX="${LOW_GX:-32}"; LOW_GY="${LOW_GY:-8}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.20}"
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

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_pair64_orbitcta64_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

sample_process(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' \
      >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

build_pair(){
  local label="$1" dg64="$2" bin="$3"
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP=2 DEPTHMAJOR=1 PAIR_MLP=1 \
    MLP_WINDOW4=1 DIRECTGATHER64="$dg64" CPASYNC_PAIR=0 PREFETCH_NEXT=0 \
    FORCE7=0 SORTED=0 PM_ACCUM=1 PTXAS_VERBOSE=1 TRANSPOSE_MODE=pipeline \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
    >"$LOGDIR/$label.build.out" 2>"$LOGDIR/$label.build.err"
}

build_orbit(){
  local label="$1" bin="$2"
  N="$N" ARCH="$ARCH" OUT="$bin" DIRECTGATHER64=1 \
    RANKFORMULA_MLP_WINDOW4=1 PM_ACCUM=1 RANKFORMULA_DIRECTGATHER_FORCE7=0 \
    PTXAS_VERBOSE=1 bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
    >"$LOGDIR/$label.build.out" 2>"$LOGDIR/$label.build.err"
}

printf 'backend\tmax_registers\tspill_store_bytes\tspill_load_bytes\n' >"$RESOURCE"
for spec in 'pair16 0' 'pair64 1'; do
  read -r label dg64 <<<"$spec"
  bin="$ONEESAN_BUILD_DIR/b300_${label}_ab_n${N}"
  build_pair "$label" "$dg64" "$bin"
  python3 - "$label" "$LOGDIR/$label.build.err" >>"$RESOURCE" <<'PY'
import re,sys
label,path=sys.argv[1:]
s=open(path,errors='ignore').read()
r=[int(x) for x in re.findall(r'Used\s+(\d+)\s+registers',s)]
sp=[(int(a),int(b)) for a,b in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',s)]
print(label, max(r) if r else -1, sum(a for a,b in sp), sum(b for a,b in sp), sep='\t')
PY
done
orbit_bin="$ONEESAN_BUILD_DIR/b300_orbitcta64_ab_n${N}"
build_orbit orbitcta64 "$orbit_bin"
python3 - orbitcta64 "$LOGDIR/orbitcta64.build.err" >>"$RESOURCE" <<'PY'
import re,sys
label,path=sys.argv[1:]
s=open(path,errors='ignore').read()
r=[int(x) for x in re.findall(r'Used\s+(\d+)\s+registers',s)]
sp=[(int(a),int(b)) for a,b in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',s)]
print(label, max(r) if r else -1, sum(a for a,b in sp), sum(b for a,b in sp), sep='\t')
PY

printf 'mode\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tforward_occupancy_pct\treverse_occupancy_pct\n' >"$RESULT"

run_one(){
  local mode="$1" bin="$2" rep="$3"
  local so="$LOGDIR/${mode}_r${rep}.out" se="$LOGDIR/${mode}_r${rep}.err" util="$LOGDIR/${mode}_r${rep}.util"
  if [[ "$mode" == orbitcta64 ]]; then
    BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" \
      BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  else
    BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$PAIR_GX" BUCKET_GRID_Y="$PAIR_GY" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  fi
  local pid=$!; sample_process "$pid" "$util" & local sampler=$!
  set +e; wait "$pid"; local rc=$?; set -e
  wait "$sampler" || true
  (( rc == 0 )) || { echo "$mode failed rc=$rc" >&2; return "$rc"; }

  local line detail occ residue wall fh rh high ug um mg mm fo ro
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; return 3; }
  residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue=$residue expected=$EXPECT" >&2; return 4; }
  wall="$(field wall_s "$line")"
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  high="$(python3 - "${fh:-0}" "${rh:-0}" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  read -r ug um mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
  if [[ "$mode" == orbitcta64 ]]; then
    occ="$(grep 'rankformula_orbitcta_occupancy ' "$se" | head -n1 || true)"
  else
    occ="$(grep 'rankformula_high_occupancy ' "$se" | head -n1 || true)"
  fi
  fo="$(field forward_warp_occupancy_pct "$occ")"; ro="$(field reverse_warp_occupancy_pct "$occ")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$mode" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "$high" \
    "$ug" "$um" "$mm" "${fo:-NA}" "${ro:-NA}" >>"$RESULT"
}

for mode in pair16 pair64 orbitcta64; do
  case "$mode" in
    pair16) bin="$ONEESAN_BUILD_DIR/b300_pair16_ab_n${N}" ;;
    pair64) bin="$ONEESAN_BUILD_DIR/b300_pair64_ab_n${N}" ;;
    orbitcta64) bin="$orbit_bin" ;;
  esac
  for ((r=1;r<=REPEATS;++r)); do run_one "$mode" "$bin" "$r"; done
done

cat "$RESULT"
echo '--- ptxas resources ---'
cat "$RESOURCE"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
q={}
for mode in ('pair16','pair64','orbitcta64'):
    g=[r for r in rows if r['mode']==mode]
    def med(k):
        x=[float(r[k]) for r in g if r[k]!='NA']
        return statistics.median(x) if x else None
    q[mode]={k:med(k) for k in ('wall_s','high_s','avg_memctrl_util_pct','forward_occupancy_pct','reverse_occupancy_pct')}
    print(mode,q[mode])
base=q['pair16']['high_s']
for mode in ('pair64','orbitcta64'):
    if base and q[mode]['high_s']:
        print(f'{mode}_high_speedup_vs_pair16={base/q[mode]["high_s"]:.6f}x')
best=min((m for m in q if q[m]['high_s'] is not None),key=lambda m:q[m]['high_s'])
print(f'BEST_HIGH={best} high_s={q[best]["high_s"]:.6f} wall_s={q[best]["wall_s"]:.6f} memctrl={q[best]["avg_memctrl_util_pct"]}')
PY

echo "b300-pair64-orbitcta64-ab OK n=$N repeats=$REPEATS pair_grid=${PAIR_GX}x${PAIR_GY} orbit_gy=$ORBIT_GY result=$RESULT" >&2
