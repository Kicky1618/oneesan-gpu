#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
[[ "$N" == 21 && "$MOD" == 4294967291 ]] || { echo 'this tuner is intentionally fixed to n=21 mod 4294967291' >&2; exit 2; }
EXPECT=998035516
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; WARP_GX="${WARP_GX:-32}"; WARP_GY="${WARP_GY:-8}"
ORBIT_GY="${ORBIT_GY:-128}"; LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
PM_ACCUM="${PM_ACCUM:-1}"; REPEATS="${REPEATS:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"; RUN_CPASYNC="${RUN_CPASYNC:-1}"; RUN_PRECTX="${RUN_PRECTX:-1}"; RUN_QUAD="${RUN_QUAD:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"; PROFILE_OUT="${PROFILE_OUT:-${PREFIX}.env}"; PTXAS="${PTXAS:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
for x in PM_ACCUM RUN_CPASYNC RUN_PRECTX RUN_QUAD; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }

if [[ "$RUN_CPASYNC" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync-peer.out" 2>"$LOGDIR/cpasync-peer.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync-peer.out" || { echo 'cp.async remote-peer gate failed' >&2; exit 5; }
fi

# family:name:ilp:sparse:sorted:cpasync:localcp:overlapcp:quad:pre_fwd:pre_rev
profiles=(
  'warp:warp_dense_reg:2:0:0:0:0:0:0:0:0'
  'warp:warp_sparse_reg:2:1:0:0:0:0:0:0:0'
  'warp:warp_sparse_sorted_reg:2:1:1:0:0:0:0:0:0'
  'orbit:orbit_dense_reg:2:0:0:0:0:0:0:0:0'
  'orbit:orbit_sparse_reg:2:1:0:0:0:0:0:0:0'
)
if [[ "$RUN_QUAD" == 1 ]]; then
  profiles+=(
    'warp:warp_dense_quad:4:0:0:0:0:0:1:0:0'
    'warp:warp_sparse_quad:4:1:0:0:0:0:1:0:0'
  )
fi
if [[ "$RUN_CPASYNC" == 1 ]]; then
  profiles+=(
    'warp:warp_sparse_cpa:2:1:0:1:0:0:0:0:0'
    'warp:warp_sparse_overlap:2:1:0:1:0:1:0:0:0'
    'warp:warp_sparse_sorted_cpa:2:1:1:1:0:0:0:0:0'
    'orbit:orbit_sparse_cpa:2:1:0:1:0:0:0:0:0'
  )
fi
if [[ "$RUN_PRECTX" == 1 ]]; then
  profiles+=(
    'warp:warp_sparse_prectx:2:1:0:0:0:0:0:1:1'
    'orbit:orbit_sparse_prectx:2:1:0:0:0:0:0:1:1'
  )
fi

printf 'profile\tfamily\tcol_ilp\tsparse64\tsorted\tcpasync\tlocal_cpasync\toverlap_cpasync\tquad_mlp\tpre_fwd\tpre_rev\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tprectx_mib_per_gpu\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$PTXAS"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

for spec in "${profiles[@]}"; do
  IFS=: read -r family name ilp sparse sorted cpa localcp overlapcp quad pf pr <<<"$spec"
  bin="$ONEESAN_BUILD_DIR/b300_tune21_${name}"
  if [[ "$family" == warp ]]; then
    N=21 ARCH="$ARCH" OUT="$bin" COL_ILP="$ilp" PM_ACCUM="$PM_ACCUM" DEPTHMAJOR=1 PAIR_MLP=1 MLP_WINDOW4=1 \
      DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$sparse" SORTED="$sorted" QUAD_MLP="$quad" \
      CPASYNC_PAIR="$cpa" CPASYNC_LOCAL_PAIR="$localcp" CPASYNC_OVERLAP_LOCAL_PAIR="$overlapcp" \
      PRECTX_FORWARD="$pf" PRECTX_REVERSE="$pr" FORCE7=0 PREFETCH_NEXT=0 PTXAS_VERBOSE=1 TRANSPOSE_MODE=pipeline \
      bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" >"$LOGDIR/$name.build.out" 2>"$LOGDIR/$name.build.err"
  else
    [[ "$sorted" == 0 && "$localcp" == 0 && "$overlapcp" == 0 && "$quad" == 0 ]] || { echo "unsupported orbit profile $name" >&2; exit 6; }
    N=21 ARCH="$ARCH" OUT="$bin" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$sparse" ORBITCTA_COL_ILP="$ilp" PAIR_MLP=1 CPASYNC_PAIR="$cpa" \
      PRECTX_FORWARD="$pf" PRECTX_REVERSE="$pr" RANKFORMULA_MLP_WINDOW4=1 PM_ACCUM="$PM_ACCUM" PTXAS_VERBOSE=1 \
      bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/$name.build.out" 2>"$LOGDIR/$name.build.err"
  fi
  python3 "$PARSER" "$LOGDIR/$name.build.err" --label "$name" >>"$PTXAS" || true

  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/${name}_r${r}.out"; se="$LOGDIR/${name}_r${r}.err"; util="$LOGDIR/${name}_r${r}.util"
    if [[ "$family" == warp ]]; then
      BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
        "$bin" 21 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
    else
      BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
        "$bin" 21 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
    fi
    pid=$!; sample "$pid" "$util" & sp=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sp" || true
    ((rc==0)) || { echo "$name failed rc=$rc" >&2; exit "$rc"; }
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$name missing residue" >&2; exit 7; }
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$name residue=$residue expected=$EXPECT" >&2; exit 8; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
    high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
    preline="$(grep 'p10dc_prectx_high fixed_owner=' "$se" | head -n1 || true)"; premib=0
    if [[ -n "$preline" ]]; then
      cb="$(field closure_context_bytes "$preline")"; f0="$(field fwd_nn "$preline")"; f1="$(field fwd_nrnl "$preline")"; r0="$(field rev_nn "$preline")"; r1="$(field rev_nr "$preline")"; r2="$(field rev_nl "$preline")"
      premib="$(python3 - "${cb:-0}" "${f0:-0}" "${f1:-0}" "${r0:-0}" "${r1:-0}" "${r2:-0}" <<'PY'
import sys
cb,*n=map(int,sys.argv[1:]);print(f'{cb*sum(n)/(1<<20):.6f}')
PY
)"
    fi
    read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$family" "$ilp" "$sparse" "$sorted" "$cpa" "$localcp" "$overlapcp" "$quad" "$pf" "$pr" "$r" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$premib" "$ag" "$am" "$mm" >>"$RESULT"
  done
done

python3 - "$RESULT" "$PROFILE_OUT" <<'PY'
import csv,statistics,sys
src,out=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
profiles={}
for name in sorted({r['profile'] for r in rows}):
 g=[r for r in rows if r['profile']==name]
 z=dict(g[0]);z['wall']=statistics.median(float(r['wall_s']) for r in g);z['high']=statistics.median(float(r['high_s']) for r in g)
 vals=[float(r['avg_memctrl_util_pct']) for r in g if r['avg_memctrl_util_pct']!='NA'];z['mc']=statistics.median(vals) if vals else float('nan');profiles[name]=z
for z in sorted(profiles.values(),key=lambda z:z['wall']):
 print(f"PROFILE {z['profile']} family={z['family']} wall_s={z['wall']:.6f} high_s={z['high']:.6f} mc_avg={z['mc']:.3f} ilp={z['col_ilp']} sparse={z['sparse64']} sorted={z['sorted']} cpa={z['cpasync']} overlap={z['overlap_cpasync']} quad={z['quad_mlp']} pf={z['pre_fwd']} pr={z['pre_rev']}")
best={fam:min((z for z in profiles.values() if z['family']==fam),key=lambda z:z['wall']) for fam in ('warp','orbit')}
with open(out,'w') as f:
 f.write('# generated by b300-hbm-profile-tune21.sh\n')
 for fam,prefix in (('warp','WARP'),('orbit','ORBIT')):
  z=best[fam]
  f.write(f'{prefix}_PROFILE={z["profile"]}\n')
  f.write(f'{prefix}_COL_ILP={z["col_ilp"]}\n')
  f.write(f'{prefix}_SPARSE64={z["sparse64"]}\n')
  f.write(f'{prefix}_SORTED={z["sorted"]}\n')
  f.write(f'{prefix}_CPASYNC_PAIR={z["cpasync"]}\n')
  f.write(f'{prefix}_CPASYNC_LOCAL_PAIR={z["local_cpasync"]}\n')
  f.write(f'{prefix}_CPASYNC_OVERLAP_LOCAL_PAIR={z["overlap_cpasync"]}\n')
  f.write(f'{prefix}_QUAD_MLP={z["quad_mlp"]}\n')
  f.write(f'{prefix}_PRECTX_FORWARD={z["pre_fwd"]}\n')
  f.write(f'{prefix}_PRECTX_REVERSE={z["pre_rev"]}\n')
 f.write('CANDIDATES="forced warp_tuned orbit_tuned"\n')
print('BEST_WARP',best['warp']['profile'],f"wall_s={best['warp']['wall']:.6f}")
print('BEST_ORBIT',best['orbit']['profile'],f"wall_s={best['orbit']['wall']:.6f}")
print('profile_file='+out)
PY
cat "$RESULT"
echo "tuner OK result=$RESULT profile=$PROFILE_OUT ptxas=$PTXAS" >&2
