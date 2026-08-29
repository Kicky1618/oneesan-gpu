#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_IN" ]] || { echo "missing PROFILE_IN=$PROFILE_IN" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_IN"
N=21; MOD=4294967291; EXPECT=998035516; NGPU=8
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; WARP_GX="${WARP_GX:-32}"; WARP_GY="${WARP_GY:-8}"; ORBIT_GY="${ORBIT_GY:-128}"; LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
PM_ACCUM="${PM_ACCUM:-1}"; REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_refine_prectx21}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

need(){ local n="$1"; [[ -n "${!n+x}" ]] || { echo "profile missing $n" >&2; exit 2; }; }
for n in WARP_PROFILE WARP_COL_ILP WARP_SPARSE64 WARP_SORTED WARP_CPASYNC_PAIR WARP_CPASYNC_LOCAL_PAIR WARP_CPASYNC_OVERLAP_LOCAL_PAIR WARP_QUAD_MLP WARP_PRECTX_FORWARD WARP_PRECTX_REVERSE ORBIT_PROFILE ORBIT_COL_ILP ORBIT_SPARSE64 ORBIT_SORTED ORBIT_CPASYNC_PAIR ORBIT_CPASYNC_LOCAL_PAIR ORBIT_CPASYNC_OVERLAP_LOCAL_PAIR ORBIT_QUAD_MLP ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE; do need "$n"; done
[[ "$ORBIT_SORTED" == 0 && "$ORBIT_CPASYNC_LOCAL_PAIR" == 0 && "$ORBIT_CPASYNC_OVERLAP_LOCAL_PAIR" == 0 && "$ORBIT_QUAD_MLP" == 0 ]] || { echo 'unsupported orbit profile in input' >&2; exit 2; }

if (( WARP_CPASYNC_PAIR || ORBIT_CPASYNC_PAIR )); then
  ARCH="$ARCH" NGPU=8 THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync-peer.out" 2>"$LOGDIR/cpasync-peer.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync-peer.out" || { echo 'cp.async remote-peer gate failed' >&2; exit 5; }
fi
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }

printf 'family\tbase_profile\tmode\tpre_fwd\tpre_rev\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tprectx_mib_per_gpu\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
run_family(){
 local fam="$1" base="$2" ilp="$3" sparse="$4" sorted="$5" cpa="$6" localcp="$7" overlap="$8" quad="$9"
 for p in 'off:0:0' 'fwd:1:0' 'both:1:1'; do
  IFS=: read -r mode pf pr <<<"$p"; local bin="$ONEESAN_BUILD_DIR/b300_refine_${fam}_${base}_${mode}_n21"
  if [[ "$fam" == warp ]]; then
   N=21 ARCH="$ARCH" OUT="$bin" COL_ILP="$ilp" PM_ACCUM="$PM_ACCUM" DEPTHMAJOR=1 PAIR_MLP=1 MLP_WINDOW4=1 DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$sparse" SORTED="$sorted" QUAD_MLP="$quad" \
    CPASYNC_PAIR="$cpa" CPASYNC_LOCAL_PAIR="$localcp" CPASYNC_OVERLAP_LOCAL_PAIR="$overlap" PRECTX_FORWARD="$pf" PRECTX_REVERSE="$pr" FORCE7=0 PREFETCH_NEXT=0 PTXAS_VERBOSE=1 TRANSPOSE_MODE=pipeline \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" >"$LOGDIR/${fam}_${mode}.build.out" 2>"$LOGDIR/${fam}_${mode}.build.err"
  else
   N=21 ARCH="$ARCH" OUT="$bin" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$sparse" ORBITCTA_COL_ILP="$ilp" PAIR_MLP=1 CPASYNC_PAIR="$cpa" PRECTX_FORWARD="$pf" PRECTX_REVERSE="$pr" RANKFORMULA_MLP_WINDOW4=1 PM_ACCUM="$PM_ACCUM" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/${fam}_${mode}.build.out" 2>"$LOGDIR/${fam}_${mode}.build.err"
  fi
  for ((r=1;r<=REPEATS;++r)); do
   local so="$LOGDIR/${fam}_${mode}_r${r}.out" se="$LOGDIR/${fam}_${mode}_r${r}.err" util="$LOGDIR/${fam}_${mode}_r${r}.util"
   if [[ "$fam" == warp ]]; then BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$bin" 21 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
   else BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$bin" 21 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" & fi
   local pid=$!; sample "$pid" "$util" & local sp=$!; set +e; wait "$pid"; local rc=$?; set -e; wait "$sp" || true; ((rc==0)) || exit "$rc"
   local line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || exit 3; local residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$fam/$mode residue=$residue" >&2; exit 4; }
   local detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)" fh="$(field forward_high_s "$detail")" rh="$(field reverse_high_s "$detail")"
   local high="$(python3 - "$fh" "$rh" <<'PY'
import sys;print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)" premib=0 preline="$(grep 'p10dc_prectx_high fixed_owner=' "$se" | head -n1 || true)"
   if [[ -n "$preline" ]]; then local cb="$(field closure_context_bytes "$preline")" f0="$(field fwd_nn "$preline")" f1="$(field fwd_nrnl "$preline")" r0="$(field rev_nn "$preline")" r1="$(field rev_nr "$preline")" r2="$(field rev_nl "$preline")"; premib="$(python3 - "${cb:-0}" "${f0:-0}" "${f1:-0}" "${r0:-0}" "${r1:-0}" "${r2:-0}" <<'PY'
import sys;cb,*n=map(int,sys.argv[1:]);print(f'{cb*sum(n)/(1<<20):.6f}')
PY
)"; fi
   local am mm; read -r am mm < <(awk '{sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %d\n",sm/n,mm;else print "NA NA"}' "$util")
   printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$fam" "$base" "$mode" "$pf" "$pr" "$r" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$premib" "$am" "$mm" >>"$RESULT"
  done
 done
}
run_family warp "$WARP_PROFILE" "$WARP_COL_ILP" "$WARP_SPARSE64" "$WARP_SORTED" "$WARP_CPASYNC_PAIR" "$WARP_CPASYNC_LOCAL_PAIR" "$WARP_CPASYNC_OVERLAP_LOCAL_PAIR" "$WARP_QUAD_MLP"
run_family orbit "$ORBIT_PROFILE" "$ORBIT_COL_ILP" "$ORBIT_SPARSE64" 0 "$ORBIT_CPASYNC_PAIR" 0 0 0

python3 - "$RESULT" "$PROFILE_IN" "$PROFILE_OUT" <<'PY'
import csv,statistics,sys,re
src,inp,out=sys.argv[1:];rows=list(csv.DictReader(open(src),delimiter='\t'))
base={}
for line in open(inp):
 s=line.strip()
 if s and not s.startswith('#') and '=' in s:
  k,v=s.split('=',1);base[k]=v.strip('"')
for fam,prefix in (('warp','WARP'),('orbit','ORBIT')):
 cand=[]
 for mode in ('off','fwd','both'):
  g=[r for r in rows if r['family']==fam and r['mode']==mode]
  z={'mode':mode,'pf':g[0]['pre_fwd'],'pr':g[0]['pre_rev'],'wall':statistics.median(float(r['wall_s']) for r in g),'high':statistics.median(float(r['high_s']) for r in g)}
  cand.append(z);print('PRECTX',fam,mode,f"wall={z['wall']:.6f}",f"high={z['high']:.6f}")
 best=min(cand,key=lambda z:z['wall']);base[prefix+'_PRECTX_FORWARD']=best['pf'];base[prefix+'_PRECTX_REVERSE']=best['pr']
 root=re.sub(r'_prectx(?:fwd|both)?$','',base[prefix+'_PROFILE'])
 base[prefix+'_PROFILE']=root if best['mode']=='off' else root+'_prectx'+('fwd' if best['mode']=='fwd' else 'both')
 print('BEST_PRECTX',fam,best['mode'],f"wall={best['wall']:.6f}")
order=['WARP_PROFILE','WARP_COL_ILP','WARP_SPARSE64','WARP_SORTED','WARP_CPASYNC_PAIR','WARP_CPASYNC_LOCAL_PAIR','WARP_CPASYNC_OVERLAP_LOCAL_PAIR','WARP_QUAD_MLP','WARP_PRECTX_FORWARD','WARP_PRECTX_REVERSE','ORBIT_PROFILE','ORBIT_COL_ILP','ORBIT_SPARSE64','ORBIT_SORTED','ORBIT_CPASYNC_PAIR','ORBIT_CPASYNC_LOCAL_PAIR','ORBIT_CPASYNC_OVERLAP_LOCAL_PAIR','ORBIT_QUAD_MLP','ORBIT_PRECTX_FORWARD','ORBIT_PRECTX_REVERSE']
with open(out,'w') as f:
 f.write('# generated by b300-hbm-profile-refine-prectx21.sh\n')
 for k in order:f.write(f'{k}={base[k]}\n')
 f.write('CANDIDATES="forced warp_tuned orbit_tuned"\n')
print('profile_file='+out)
PY
cat "$RESULT"
echo "prectx refine OK input=$PROFILE_IN output=$PROFILE_OUT result=$RESULT" >&2
