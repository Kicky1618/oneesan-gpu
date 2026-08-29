#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; ORBIT_GY="${BUCKET_ORBITCTA_GRID_Y:-128}"
LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
FLAT_PER_SM_LIST="${FLAT_PER_SM_LIST:-1 2 4 8}"; REPEATS="${REPEATS:-1}"
SPARSE64="${DIRECTGATHER_SPARSE64:-1}"; SORT_RANKS="${DIRECTGATHER_SORT_RANKS:-0}"
COL_ILP="${ORBITCTA_COL_ILP:-2}"; PAIR_MLP="${PAIR_MLP:-1}"; CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
WINDOW4="${RANKFORMULA_MLP_WINDOW4:-1}"; PM_ACCUM="${PM_ACCUM:-1}"
PRECTX_FORWARD="${PRECTX_FORWARD:-0}"; PRECTX_REVERSE="${PRECTX_REVERSE:-0}"; PRECTX_COMPACT="${PRECTX_COMPACT:-0}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbitcta_flat_ab_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || { echo 'default correctness gate is fixed to n=21 mod 4294967291 residue 998035516' >&2; exit 2; }
for x in SPARSE64 SORT_RANKS PAIR_MLP CPASYNC_PAIR WINDOW4 PM_ACCUM PRECTX_FORWARD PRECTX_REVERSE PRECTX_COMPACT PTXAS_VERBOSE; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
case "$COL_ILP" in 1|2|4) ;; *) echo 'ORBITCTA_COL_ILP must be 1,2,4' >&2; exit 2;; esac
[[ "$SORT_RANKS" == 0 || "$SPARSE64" == 0 ]] || { echo 'sorted ranks currently require dense64' >&2; exit 2; }
if [[ "$PAIR_MLP" == 1 ]]; then [[ "$WINDOW4" == 1 ]] || { echo 'PAIR_MLP requires WINDOW4=1' >&2; exit 2; }; [[ "$COL_ILP" == 2 || "$COL_ILP" == 4 ]] || { echo 'PAIR_MLP requires ILP2/4' >&2; exit 2; }; fi
[[ "$CPASYNC_PAIR" == 0 || "$PAIR_MLP" == 1 ]] || { echo 'CPASYNC_PAIR requires PAIR_MLP=1' >&2; exit 2; }
[[ "$PRECTX_COMPACT" == 0 || "$PRECTX_FORWARD" == 1 || "$PRECTX_REVERSE" == 1 ]] || { echo 'PRECTX_COMPACT requires prectx' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

if [[ "$CPASYNC_PAIR" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync-peer.out" 2>"$LOGDIR/cpasync-peer.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync-peer.out" || { echo 'cp.async remote-peer preflight failed' >&2; exit 5; }
fi
if [[ "$PRECTX_COMPACT" == 1 ]]; then
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >"$LOGDIR/compact-prectx.out" 2>"$LOGDIR/compact-prectx.err"
fi

COMMON=(N="$N" ARCH="$ARCH" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" DIRECTGATHER_SORT_RANKS="$SORT_RANKS" ORBITCTA_COL_ILP="$COL_ILP" PAIR_MLP="$PAIR_MLP" CPASYNC_PAIR="$CPASYNC_PAIR" RANKFORMULA_MLP_WINDOW4="$WINDOW4" PM_ACCUM="$PM_ACCUM" PRECTX_FORWARD="$PRECTX_FORWARD" PRECTX_REVERSE="$PRECTX_REVERSE" PRECTX_COMPACT="$PRECTX_COMPACT" PTXAS_VERBOSE="$PTXAS_VERBOSE")
ORD_BIN="$ONEESAN_BUILD_DIR/b300_orbitcta_flat_ab_ordinary_n${N}"
FLAT_BIN="$ONEESAN_BUILD_DIR/b300_orbitcta_flat_ab_flat_n${N}"
env "${COMMON[@]}" ORBITCTA_FLAT=0 OUT="$ORD_BIN" bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/ordinary.build.out" 2>"$LOGDIR/ordinary.build.err"
env "${COMMON[@]}" ORBITCTA_FLAT=1 OUT="$FLAT_BIN" bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/flat.build.out" 2>"$LOGDIR/flat.build.err"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }

printf 'mode\tflat\tblocks_per_sm\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_gpu_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
run_one(){
  local label="$1" flat="$2" psm="$3" bin="$4" rep="$5"
  local so="$LOGDIR/${label}_r${rep}.out" se="$LOGDIR/${label}_r${rep}.err" util="$LOGDIR/${label}_r${rep}.util"
  if [[ "$flat" == 1 ]]; then
    BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$psm" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  else
    BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  fi
  local pid=$!; sample "$pid" "$util" & local sp=$!; set +e; wait "$pid"; local rc=$?; set -e; wait "$sp" || true
  (( rc == 0 )) || { echo "$label failed rc=$rc" >&2; exit "$rc"; }
  local line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$label missing residue" >&2; exit 3; }
  local residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$label residue=$residue expected=$EXPECT" >&2; exit 4; }
  local detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; local fh="$(field forward_high_s "$detail")"; local rh="$(field reverse_high_s "$detail")"
  local high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  local ag am mg mm; read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$flat" "$psm" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mg" "$mm" >>"$RESULT"
}

for ((r=1;r<=REPEATS;++r)); do run_one ordinary 0 0 "$ORD_BIN" "$r"; done
for psm in $FLAT_PER_SM_LIST; do
  [[ "$psm" =~ ^[0-9]+$ ]] && (( psm > 0 )) || { echo "bad flat blocks/SM: $psm" >&2; exit 2; }
  for ((r=1;r<=REPEATS;++r)); do run_one "flat${psm}" 1 "$psm" "$FLAT_BIN" "$r"; done
done

python3 - "$RESULT" "$SUMMARY" "$WINNER_ENV" <<'PY'
import csv,statistics,sys
src,summary,winner=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for mode in dict.fromkeys(r['mode'] for r in rows):
 g=[r for r in rows if r['mode']==mode]
 z={'mode':mode,'flat':g[0]['flat'],'blocks_per_sm':g[0]['blocks_per_sm'],'repeats':len(g)}
 for k in ('wall_s','forward_high_s','reverse_high_s','high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct'):
  z[k]=statistics.median(float(r[k]) for r in g if r[k]!='NA')
 out.append(z)
keys=('mode','flat','blocks_per_sm','repeats','wall_s','forward_high_s','reverse_high_s','high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct')
with open(summary,'w') as f:
 f.write('\t'.join(keys)+'\n')
 for z in out:f.write('\t'.join(str(z[k]) for k in keys)+'\n')
for z in sorted(out,key=lambda z:z['high_s']):
 print('ORBIT_SCHED',z['mode'],f"wall_s={z['wall_s']:.6f}",f"high_s={z['high_s']:.6f}",f"mc={z['avg_memctrl_util_pct']:.3f}")
b=min(out,key=lambda z:z['wall_s'])
with open(winner,'w') as f:
 f.write(f'ORBITCTA_FLAT={b["flat"]}\n')
 f.write(f'BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM={b["blocks_per_sm"]}\n')
 f.write(f'ORBITCTA_SCHEDULER_PROFILE={b["mode"]}\n')
print('WINNER='+b['mode'],f"wall_s={b['wall_s']:.6f}",f"high_s={b['high_s']:.6f}",f"winner_env={winner}")
PY
cat "$RESULT"
echo "orbitcta flat A/B OK result=$RESULT summary=$SUMMARY winner_env=$WINNER_ENV" >&2
