#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS_LIST="${THREADS_LIST:-128 192 256 320 384 512}"; REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"; ADAPTIVE_WAVES="${ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES:-0}"
SPARSE64="${DIRECTGATHER_SPARSE64:-1}"; PM_ACCUM="${PM_ACCUM:-1}"; PRECTX_FLAT_BID="${PRECTX_FLAT_BID:-1}"
QUAD_OVERLAP_LOCAL="${QUAD_OVERLAP_LOCAL:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_pipe2_producer_thread_sweep_n${N}_b${BATCH}_bid${PRECTX_FLAT_BID}_qol${QUAD_OVERLAP_LOCAL}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || exit 2
for x in SPARSE64 PM_ACCUM PRECTX_FLAT_BID QUAD_OVERLAP_LOCAL; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
case "$BATCH" in 1|2|4|8|16) ;; *) exit 2;; esac
case "$ADAPTIVE_WAVES" in 0|1|2|4) ;; *) exit 2;; esac
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || exit 2
for t in $THREADS_LIST; do (( t >= 64 && t <= 1024 && t % 32 == 0 )) || { echo "bad producer threads=$t" >&2; exit 2; }; done

bash "$ONEESAN_ROOT/scripts/bench/b300-pipe2-producer-warp-coverage-proof.sh" >"$LOGDIR/coverage.out"
bash "$ONEESAN_ROOT/scripts/bench/b300-compact-prectx-meta-proof.sh" >"$LOGDIR/meta.out"
[[ "$PRECTX_FLAT_BID" == 0 ]] || ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-flat-bid-selftest.sh" >"$LOGDIR/flatbid.out" 2>"$LOGDIR/flatbid.err"

BIN="$LOGDIR/producer_best"
N="$N" ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" DIRECTGATHER_SORT_RANKS=0 \
RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR=1 ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 \
ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_BATCH="$BATCH" ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$ADAPTIVE_WAVES" \
ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 ORBITCTA_FLAT_DYNAMIC_PIPE2=1 ORBITCTA_COL_ILP=4 \
QUAD_MLP=1 QUAD_OVERLAP_LOCAL="$QUAD_OVERLAP_LOCAL" QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0 \
QUAD_OVERLAP_BYPASS_LOCAL0=0 QUAD_CPASYNC_PREFETCH_BYTES=0 QUAD_CPASYNC_GROUP_COLS=1 \
PRECTX_FORWARD=1 PRECTX_REVERSE=1 PRECTX_COMPACT=1 PRECTX_FLAT_BID="$PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED=0 PRECTX_WARPCOOP=0 \
PRODUCER_PRECTX_WARPCOOP=1 PTXAS_VERBOSE=1 OUT="$BIN" \
bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh" >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"
[[ -x "$BIN" ]] || exit 5
for marker in 'pipe2_producer_warp=1' 'pipe2_producer_prectx_warpcoop=1' 'quad_mlp=1'; do grep -q "$marker" "$LOGDIR/build.err" || exit 5; done
python3 "$PARSER" "$LOGDIR/build.err" --label producer_best >"$RESOURCE" || true

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l"|tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do
 nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null |
 awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>pg)pg=g;if(m>pm)pm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,pg,pm;else print "NA NA NA NA"}' >>"$out" || true
 sleep "$SAMPLE_INTERVAL"; done; }
printf 'threads\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_pct\tavg_memctrl_pct\tpeak_gpu_pct\tpeak_memctrl_pct\tforward_regs\treverse_regs\tforward_blocks_per_sm\treverse_blocks_per_sm\tforward_warp_occupancy_pct\treverse_warp_occupancy_pct\n' >"$RESULT"
for t in $THREADS_LIST; do
 for ((r=1;r<=REPEATS;++r)); do
  so="$LOGDIR/t${t}_r${r}.out";se="$LOGDIR/t${t}_r${r}.err";util="$LOGDIR/t${t}_r${r}.util"
  BUCKET_THREADS="$t" BUCKET_LOW_GRID_X=16 BUCKET_LOW_GRID_Y=8 "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" & pid=$!
  sample "$pid" "$util" & sp=$!; set +e;wait "$pid";rc=$?;set -e;wait "$sp"||true;((rc==0))||exit "$rc"
  line="$(grep '^residue=' "$so"|tail -n1)";res="$(field residue "$line")";[[ "$res" == "$EXPECT" ]]||{ echo "threads=$t residue=$res" >&2;exit 4;}
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1)";occ="$(grep 'rankformula_orbitcta_flat_occupancy device=0 ' "$se"|head -n1)"
  fh="$(field forward_high_s "$detail")";rh="$(field reverse_high_s "$detail")";high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(float(sys.argv[1])+float(sys.argv[2]))
PY
)"
  read -r ag am pg pm < <(awk '{sg+=$1;sm+=$2;if($3>pg)pg=$3;if($4>pm)pm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,pg,pm;else print "NA NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$r" "$res" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$pg" "$pm" "$(field forward_regs "$occ")" "$(field reverse_regs "$occ")" "$(field forward_blocks_per_sm "$occ")" "$(field reverse_blocks_per_sm "$occ")" "$(field forward_warp_occupancy_pct "$occ")" "$(field reverse_warp_occupancy_pct "$occ")" >>"$RESULT"
 done
done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); assert len({x['residue'] for x in r})==1
for t in sorted({int(x['threads']) for x in r}):
 g=[x for x in r if int(x['threads'])==t]
 med=lambda k:statistics.median(float(x[k]) for x in g if x[k]!='NA')
 print(f'threads={t} wall={med("wall_s"):.9f} high={med("high_s"):.9f} memctrl={med("avg_memctrl_pct"):.3f} gpu={med("avg_gpu_pct"):.3f} focc={med("forward_warp_occupancy_pct"):.3f} rocc={med("reverse_warp_occupancy_pct"):.3f}')
best=min({int(x['threads']) for x in r},key=lambda t:statistics.median(float(x['wall_s']) for x in r if int(x['threads'])==t))
print(f'producer_thread_winner={best}')
PY
echo "producer thread sweep OK result=$RESULT resources=$RESOURCE" >&2
