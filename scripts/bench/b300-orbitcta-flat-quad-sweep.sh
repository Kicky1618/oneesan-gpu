#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; ARCH="${ARCH:-sm_103}"
EXPECT="${EXPECT:-998035516}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${THREADS:-256}"; CHUNKS="${CHUNKS:-2 4 8}"; POOLS="${POOLS:-auto 1 2}"; REPEATS="${REPEATS:-1}"
SPARSE64="${SPARSE64:-1}"; QUAD_SPARSE_DESC_MLP="${QUAD_SPARSE_DESC_MLP:-0}"
PRECTX_FORWARD="${PRECTX_FORWARD:-0}"; PRECTX_REVERSE="${PRECTX_REVERSE:-0}"; PRECTX_COMPACT="${PRECTX_COMPACT:-0}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.20}"
[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || { echo 'default exact gate is n21/mod4294967291/residue998035516' >&2; exit 2; }
for x in SPARSE64 QUAD_SPARSE_DESC_MLP PRECTX_FORWARD PRECTX_REVERSE PRECTX_COMPACT; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
[[ "$QUAD_SPARSE_DESC_MLP" == 0 || "$SPARSE64" == 1 ]] || { echo 'QUAD_SPARSE_DESC_MLP requires SPARSE64=1' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/directgather64-quad-proof.sh"
ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >/tmp/oneesan-qol-peer.out 2>/tmp/oneesan-qol-peer.err
grep -q 'cp_async_remote_peer=OK exact=OK' /tmp/oneesan-qol-peer.out || { echo 'remote peer cp.async preflight failed' >&2; exit 5; }
if [[ "$PRECTX_COMPACT" == 1 ]]; then
  ARCH="$ARCH" PM_ACCUM=1 bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >/tmp/oneesan-qol-prectx.out 2>/tmp/oneesan-qol-prectx.err
fi

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbitcta_flat_quad_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
printf 'chunk\tpool\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }

for chunk in $CHUNKS; do
  case "$chunk" in 2|4|8|16|32) ;; *) echo "bad chunk=$chunk" >&2; exit 2;; esac
  bin="$ONEESAN_BUILD_DIR/b300_orbitcta_flat_qol_qsd${QUAD_SPARSE_DESC_MLP}_chunk${chunk}_n${N}"
  N="$N" ARCH="$ARCH" OUT="$bin" PM_ACCUM=1 DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" \
    DIRECTGATHER_SORT_RANKS=0 RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR=1 \
    QUAD_MLP=1 QUAD_OVERLAP_LOCAL=1 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP="$QUAD_SPARSE_DESC_MLP" \
    ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK="$chunk" ORBITCTA_COL_ILP=4 \
    PRECTX_FORWARD="$PRECTX_FORWARD" PRECTX_REVERSE="$PRECTX_REVERSE" PRECTX_COMPACT="$PRECTX_COMPACT" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/chunk${chunk}.build.out" 2>"$LOGDIR/chunk${chunk}.build.err"
  python3 "$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py" "$LOGDIR/chunk${chunk}.build.err" --label "qol_qsd${QUAD_SPARSE_DESC_MLP}_chunk${chunk}" >>"$RESOURCE" || true

  for pool in $POOLS; do
    [[ "$pool" == auto || "$pool" =~ ^[1-9][0-9]*$ ]] || { echo "bad pool=$pool" >&2; exit 2; }
    for ((r=1;r<=REPEATS;++r)); do
      tag="c${chunk}_p${pool}_r${r}"; so="$LOGDIR/$tag.out"; se="$LOGDIR/$tag.err"; util="$LOGDIR/$tag.util"
      if [[ "$pool" == auto ]]; then
        env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM \
          BUCKET_THREADS="$THREADS" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
          "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
      else
        env -u BUCKET_ORBITCTA_FLAT_BLOCKS BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$pool" \
          BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
          "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
      fi
      pid=$!; sample "$pid" "$util" & sp=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sp" || true
      (( rc == 0 )) || { echo "$tag failed rc=$rc" >&2; exit "$rc"; }
      line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$tag missing residue" >&2; exit 3; }
      residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$tag residue=$residue expected=$EXPECT" >&2; exit 4; }
      detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
      [[ -n "$fh" && -n "$rh" ]] || { echo "$tag missing HIGH phase timing" >&2; exit 6; }
      high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
      read -r ag am mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$util")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$chunk" "$pool" "$r" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mm" >>"$RESULT"
    done
  done
done

cat "$RESULT"
python3 - "$RESULT" "$WINNER_ENV" "$QUAD_SPARSE_DESC_MLP" <<'PY'
import csv,statistics,sys
src,winner,qsd=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t')); by={}
for r in rows: by.setdefault((int(r['chunk']),r['pool']),[]).append(r)
res={k:{x['residue'] for x in g} for k,g in by.items()}
if any(len(v)!=1 for v in res.values()) or len({next(iter(v)) for v in res.values()})!=1: raise SystemExit(f'RESIDUE MISMATCH {res}')
out=[]
for (c,p),g in by.items():
 h=statistics.median(float(x['high_s']) for x in g); w=statistics.median(float(x['wall_s']) for x in g)
 mc=statistics.median(float(x['avg_memctrl_util_pct']) for x in g if x['avg_memctrl_util_pct']!='NA')
 out.append((w,h,c,p,mc))
for w,h,c,p,mc in sorted(out): print(f'chunk={c}',f'pool={p}',f'wall_s={w:.6f}',f'high_s={h:.6f}',f'mc_avg_pct={mc:.3f}')
b=min(out)
pool='0' if b[3]=='auto' else b[3]
with open(winner,'w') as f:
 f.write('ORBITCTA_FLAT=1\n')
 f.write(f'ORBITCTA_FLAT_CHUNK={b[2]}\n')
 f.write(f'ORBITCTA_FLAT_BLOCKS_PER_SM={pool}\n')
 f.write('ORBIT_QUAD_MLP=1\n')
 f.write('ORBIT_QUAD_OVERLAP_LOCAL=1\n')
 f.write('ORBIT_QUAD_LOCAL_DIRECT_MAX=0\n')
 f.write(f'ORBIT_QUAD_SPARSE_DESC_MLP={qsd}\n')
 f.write('ORBIT_CPASYNC_PAIR=1\n')
 f.write('ORBIT_COL_ILP=4\n')
 f.write(f'ORBIT_QUAD_PROFILE=quad_qsd{qsd}_chunk{b[2]}_pool{b[3]}\n')
 f.write(f'ORBIT_QUAD_WALL_S={b[0]:.9f}\n')
 f.write(f'ORBIT_QUAD_HIGH_S={b[1]:.9f}\n')
print('BEST_ORBITCTA_QUAD',f'qsd={qsd}',f'chunk={b[2]}',f'pool={b[3]}',f'wall_s={b[0]:.6f}',f'high_s={b[1]:.6f}',f'mc_avg_pct={b[4]:.3f}',f'winner_env={winner}')
PY

echo "b300-orbitcta-flat-quad-sweep OK qsd=$QUAD_SPARSE_DESC_MLP result=$RESULT resources=$RESOURCE winner_env=$WINNER_ENV" >&2
