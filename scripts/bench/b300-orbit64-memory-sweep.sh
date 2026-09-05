#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
MOD="${MOD:-4294967291}"
NGPU="${NGPU:-8}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
REPEATS="${REPEATS:-1}"
PM_ACCUM="${PM_ACCUM:-1}"
WINDOW4="${WINDOW4:-1}"
LOW_GRID_X="${LOW_GRID_X:-16}"
LOW_GRID_Y="${LOW_GRID_Y:-8}"
CONFIGS="${CONFIGS:-128:64 128:128 256:32 256:64 256:128 256:256 512:32 512:64 512:128}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.10}"
if [[ -z "${EXPECT+x}" ]]; then
  if [[ "$N" == 21 && "$MOD" == 4294967291 ]]; then EXPECT=998035516
  else echo "EXPECT required for N=$N MOD=$MOD" >&2; exit 2
  fi
fi
for x in PM_ACCUM WINDOW4; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbit64_memory_sweep_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_orbit64_memory_sweep_n${N}}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

bash "$ONEESAN_ROOT/scripts/bench/rankformula-directgather64-proof.sh" \
  >"$LOGDIR/directgather64.proof.out" 2>"$LOGDIR/directgather64.proof.err"

N="$N" ARCH="$ARCH" OUT="$BIN" PM_ACCUM="$PM_ACCUM" \
  DIRECTGATHER64=1 RANKFORMULA_MLP_WINDOW4="$WINDOW4" \
  RANKFORMULA_DIRECTGATHER_FORCE7=0 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
  >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"

printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$LOGDIR/build.err" --label orbit64 >>"$RESOURCE" || true

field() {
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

sample_process() {
  local pid="$1" out="$2"
  : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' \
      >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

printf 'threads\torbit_grid_y\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\ttotal_high_s\tforward_low_s\treverse_low_s\ttranspose_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_gpu_util_pct\tmax_memctrl_util_pct\tsamples\n' >"$RESULT"

run_case() {
  local threads="$1" gy="$2" rep="$3"
  local tag="t${threads}_y${gy}_r${rep}"
  local so="$LOGDIR/${tag}.out" se="$LOGDIR/${tag}.err" util="$LOGDIR/${tag}.util"
  BUCKET_THREADS="$threads" BUCKET_ORBITCTA_GRID_Y="$gy" \
    BUCKET_LOW_GRID_X="$LOW_GRID_X" BUCKET_LOW_GRID_Y="$LOW_GRID_Y" \
    "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  local pid=$!
  sample_process "$pid" "$util" & local sampler=$!
  set +e
  wait "$pid"; local rc=$?
  set -e
  wait "$sampler" || true
  (( rc == 0 )) || { echo "$tag failed rc=$rc" >&2; return "$rc"; }

  local line detail residue wall fh rh fl rl ts high ag am mg mm ns
  line="$(grep '^residue=' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$tag missing residue" >&2; return 3; }
  residue="$(field residue "$line")"
  [[ "$residue" == "$EXPECT" ]] || { echo "$tag residue mismatch got=$residue expected=$EXPECT" >&2; return 4; }
  wall="$(field wall_s "$line")"
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; ts="$(field transpose_s "$detail")"
  high="$(python3 - "${fh:-0}" "${rh:-0}" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  read -r ag am mg mm ns < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d %d\n",sg/n,sm/n,mg,mm,n;else print "NA NA NA NA 0"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$threads" "$gy" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "$high" \
    "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" "$ag" "$am" "$mg" "$mm" "$ns" >>"$RESULT"
}

for cfg in $CONFIGS; do
  IFS=: read -r threads gy <<<"$cfg"
  [[ "$threads" =~ ^[0-9]+$ && "$gy" =~ ^[0-9]+$ ]] || { echo "bad config $cfg" >&2; exit 2; }
  (( threads >= 32 && threads <= 1024 && threads % 32 == 0 && gy >= 1 )) || { echo "bad config $cfg" >&2; exit 2; }
  for ((r=1; r<=REPEATS; ++r)); do
    echo "=== orbit64 threads=$threads orbit_grid_y=$gy repeat=$r/$REPEATS ===" >&2
    run_case "$threads" "$gy" "$r"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
groups={}
for r in rows: groups.setdefault((r['threads'],r['orbit_grid_y']),[]).append(r)
out=[]
for (t,y),g in groups.items():
    z={'threads':t,'orbit_grid_y':y,'repeats':len(g)}
    for k in ('wall_s','forward_high_s','reverse_high_s','total_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct'):
        xs=[float(r[k]) for r in g if r[k] != 'NA']
        z[k]=statistics.median(xs) if xs else None
    out.append(z)
out.sort(key=lambda z:z['total_high_s'] if z['total_high_s'] is not None else float('inf'))
keys=('threads','orbit_grid_y','repeats','wall_s','forward_high_s','reverse_high_s','total_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct')
with open(dst,'w') as f:
    f.write('\t'.join(keys)+'\n')
    for z in out: f.write('\t'.join(str(z.get(k)) for k in keys)+'\n')
if out:
    best=out[0]
    print(f'fastest=threads{best["threads"]}_y{best["orbit_grid_y"]}')
    print(f'fastest_wall_s={best["wall_s"]}')
    print(f'fastest_total_high_s={best["total_high_s"]}')
    print(f'fastest_avg_memctrl_pct={best["avg_memctrl_util_pct"]}')
    mem=[z for z in out if z['avg_memctrl_util_pct'] is not None]
    if mem:
        m=max(mem,key=lambda z:z['avg_memctrl_util_pct'])
        print(f'highest_memctrl=threads{m["threads"]}_y{m["orbit_grid_y"]}')
        print(f'highest_memctrl_pct={m["avg_memctrl_util_pct"]}')
        print(f'highest_memctrl_total_high_s={m["total_high_s"]}')
print('scheduler=orbitcta')
print('descriptor=directgather64')
print('grid_x=1')
print(f'summary={dst}')
PY

echo "b300-orbit64-memory-sweep OK n=$N configs='$CONFIGS' result=$RESULT resource=$RESOURCE" >&2
