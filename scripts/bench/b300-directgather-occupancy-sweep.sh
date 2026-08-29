#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; REPEATS="${REPEATS:-1}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"
GX="${GX:-32}"; GY="${GY:-8}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }
fi
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather_occupancy_n${N}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
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

# label window4 pair maxr threads
CASES=(
  "full256 0 0 0 256"
  "win4_256 1 0 0 256"
  "pair256 1 1 0 256"
  "pair128 1 1 0 128"
  "pair_r160 1 1 160 256"
  "pair_r128 1 1 128 256"
)

printf 'mode\trepeat\tthreads\twindow4\tpair\tmaxrregcount\tresidue\twall_s\tforward_high_s\treverse_high_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\thigh_max_registers\thigh_spill_store_bytes\thigh_spill_load_bytes\n' >"$RESULT"

declare -A BIN_FOR BUILDERR_FOR
for spec in "${CASES[@]}"; do
  read -r label window4 pair maxr threads <<<"$spec"
  key="w${window4}_p${pair}_r${maxr}"
  if [[ -z "${BIN_FOR[$key]+x}" ]]; then
    bin="$ONEESAN_BUILD_DIR/occ_directgather_${key}_n${N}"
    berr="$LOGDIR/${key}.build.err"
    N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP=2 DEPTHMAJOR=1 FORCE7=0 \
      MLP_WINDOW4="$window4" PAIR_MLP="$pair" PREFETCH_NEXT=0 PM_ACCUM=1 \
      MAXRREGCOUNT="$maxr" PTXAS_VERBOSE=1 \
      bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
      >"$LOGDIR/${key}.build.out" 2>"$berr"
    BIN_FOR[$key]="$bin"; BUILDERR_FOR[$key]="$berr"
  fi

  resource="$LOGDIR/${key}.ptxas.tsv"
  printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$resource"
  python3 "$PARSER" "${BUILDERR_FOR[$key]}" --label "$label" >>"$resource"
  read -r regs ss sl < <(python3 - "$resource" "$label" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
g=[r for r in rows if r['backend']==sys.argv[2] and 'high' in r['kernel'].lower()]
regs=[int(r['registers']) for r in g if r['registers']!='NA']
ss=[int(r['spill_store_bytes']) for r in g if r['spill_store_bytes']!='NA']
sl=[int(r['spill_load_bytes']) for r in g if r['spill_load_bytes']!='NA']
print(max(regs) if regs else 'NA',sum(ss) if ss else 'NA',sum(sl) if sl else 'NA')
PY
)

  for ((rep=1;rep<=REPEATS;++rep)); do
    so="$LOGDIR/${label}_r${rep}.out"; se="$LOGDIR/${label}_r${rep}.err"; util="$LOGDIR/${label}_r${rep}.util"
    BUCKET_THREADS="$threads" BUCKET_GRID_X="$GX" BUCKET_GRID_Y="$GY" \
      "${BIN_FOR[$key]}" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
    pid=$!; sample_process "$pid" "$util" & sampler=$!
    set +e; wait "$pid"; rc=$?; set -e; wait "$sampler" || true
    (( rc == 0 )) || exit "$rc"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || exit 3
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$label residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    read -r avg_gpu avg_mem max_gpu max_mem < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$label" "$rep" "$threads" "$window4" "$pair" "$maxr" "$residue" \
      "$(field wall_s "$line")" "$(field forward_high_s "$detail")" "$(field reverse_high_s "$detail")" \
      "$avg_gpu" "$avg_mem" "$max_mem" "$regs" "$ss" "$sl" >>"$RESULT"
  done
done

cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
for mode in dict.fromkeys(r['mode'] for r in rows):
    g=[r for r in rows if r['mode']==mode]
    med=lambda k: statistics.median(float(r[k]) for r in g if r[k]!='NA')
    fh=med('forward_high_s'); rh=med('reverse_high_s')
    print(mode,
          f"wall={med('wall_s'):.6f}",
          f"high={fh+rh:.6f}",
          f"memctrl={med('avg_memctrl_util_pct'):.3f}%",
          f"gpu={med('avg_gpu_util_pct'):.3f}%",
          f"regs={g[0]['high_max_registers']}",
          f"spill_store={g[0]['high_spill_store_bytes']}",
          f"spill_load={g[0]['high_spill_load_bytes']}")
PY

echo "b300-directgather-occupancy-sweep OK result=$RESULT" >&2
