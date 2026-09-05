#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"
THREADS="${THREADS:-256}"; ORBIT_GY="${ORBIT_GY:-128}"; LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
ORBITCTA_COL_ILP="${ORBITCTA_COL_ILP:-2}"; PAIR_MLP="${PAIR_MLP:-1}"; CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
RANKFORMULA_MLP_WINDOW4="${RANKFORMULA_MLP_WINDOW4:-1}"; PM_ACCUM="${PM_ACCUM:-1}"
REPEATS="${REPEATS:-1}"; EXPECT="${EXPECT:-}"
if [[ -z "$EXPECT" && "$N" == 21 && "$MOD" == 4294967291 ]]; then EXPECT=998035516; fi
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }
[[ "$NGPU" == 8 ]] || { echo "orbit64 requires NGPU=8" >&2; exit 2; }
for x in PAIR_MLP CPASYNC_PAIR RANKFORMULA_MLP_WINDOW4 PM_ACCUM; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
case "$ORBITCTA_COL_ILP" in 1|2|4) ;; *) echo "ORBITCTA_COL_ILP must be 1,2,4" >&2; exit 2;; esac
if [[ "$PAIR_MLP" == 1 ]]; then
  [[ "$RANKFORMULA_MLP_WINDOW4" == 1 ]] || { echo "PAIR_MLP requires WINDOW4=1" >&2; exit 2; }
  [[ "$ORBITCTA_COL_ILP" == 2 || "$ORBITCTA_COL_ILP" == 4 ]] || { echo "PAIR_MLP requires ILP2/4" >&2; exit 2; }
fi
[[ "$CPASYNC_PAIR" == 0 || "$PAIR_MLP" == 1 ]] || { echo "CPASYNC_PAIR requires PAIR_MLP=1" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbit64_sparse64_ab_n${N}_i${ORBITCTA_COL_ILP}_pair${PAIR_MLP}_cpa${CPASYNC_PAIR}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; PLAN="${PLAN:-${PREFIX}_plan.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
DENSE_BIN="$ONEESAN_BUILD_DIR/b300_orbit64_dense_i${ORBITCTA_COL_ILP}_n${N}"
SPARSE_BIN="$ONEESAN_BUILD_DIR/b300_orbit64_sparse_i${ORBITCTA_COL_ILP}_n${N}"

if [[ "$CPASYNC_PAIR" == 1 ]]; then
  CPA_LOG="$LOGDIR/cpasync_peer.out"
  ARCH="$ARCH" NGPU=8 THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$CPA_LOG" 2>&1
  grep -q 'cp_async_remote_peer=OK exact=OK' "$CPA_LOG" || { cat "$CPA_LOG" >&2; exit 6; }
fi

build_case(){
  local mode="$1" sparse="$2" bin="$3"
  N="$N" ARCH="$ARCH" OUT="$bin" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$sparse" \
    ORBITCTA_COL_ILP="$ORBITCTA_COL_ILP" PAIR_MLP="$PAIR_MLP" CPASYNC_PAIR="$CPASYNC_PAIR" \
    RANKFORMULA_MLP_WINDOW4="$RANKFORMULA_MLP_WINDOW4" PM_ACCUM="$PM_ACCUM" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
    >"$LOGDIR/${mode}.build.out" 2>"$LOGDIR/${mode}.build.err"
}
build_case dense64 0 "$DENSE_BIN"
build_case sparse64 1 "$SPARSE_BIN"

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$LOGDIR/dense64.build.err" --label dense64 >>"$RESOURCE" || true
python3 "$PARSER" "$LOGDIR/sparse64.build.err" --label sparse64 >>"$RESOURCE" || true

export BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY"
export BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
summarize_dmon(){ python3 - "$1" <<'PY'
import sys
v=[]
for line in open(sys.argv[1],encoding='utf-8',errors='replace'):
    s=line.strip()
    if not s or s.startswith('#'): continue
    a=s.split()
    if len(a)>=3:
        try:v.append(float(a[2]))
        except ValueError:pass
if len(v)>16:v=v[8:]
print('NA\tNA\t0' if not v else f'{sum(v)/len(v):.3f}\t{max(v):.3f}\t{len(v)}')
PY
}

printf 'backend\tmax_gpu_authoritative_gib\tmetadata_mib_per_gpu\tmax_device_need_gib\tpeer_gib_per_transpose\tsnake_peer_tib_per_residue\tprepare_s\n' >"$PLAN"
plan_case(){
  local mode="$1" bin="$2" pe="$LOGDIR/${mode}.plan.err"
  "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" 8 --plan-only >/dev/null 2>"$pe"
  local line="$(grep 'backend=gridfp-b300-bucket-snake-onepass-graph-batch-v0.1-plan' "$pe" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$mode missing plan line" >&2; exit 3; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" \
    "$(field max_gpu_authoritative_gib "$line")" "$(field metadata_mib_per_gpu "$line")" \
    "$(field max_device_need_gib "$line")" "$(field peer_gib_per_transpose "$line")" \
    "$(field snake_peer_tib_per_residue "$line")" "$(field prepare_s "$line")" >>"$PLAN"
}
plan_case dense64 "$DENSE_BIN"
plan_case sparse64 "$SPARSE_BIN"

printf 'backend\trepeat\tresidue\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"
run_case(){
  local mode="$1" bin="$2" rep="$3" so="$LOGDIR/${mode}_r${rep}.out" se="$LOGDIR/${mode}_r${rep}.err" dm="$LOGDIR/${mode}_r${rep}.dmon"
  nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & local dp=$!
  set +e; "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se"; local rc=$?; set -e
  kill "$dp" 2>/dev/null || true; wait "$dp" 2>/dev/null || true
  ((rc==0)) || { echo "$mode failed rc=$rc; see $se" >&2; exit "$rc"; }
  local line detail residue wall avg mx ns
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; [[ -n "$detail" ]] || { echo "$mode missing phase detail" >&2; exit 5; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; IFS=$'\t' read -r avg mx ns <<<"$(summarize_dmon "$dm")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$rep" "$residue" "$wall" \
    "$(field forward_high_s "$detail")" "$(field forward_low_s "$detail")" "$(field reverse_low_s "$detail")" \
    "$(field reverse_high_s "$detail")" "$(field transpose_s "$detail")" "$avg" "$mx" "$ns" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r)); do run_case dense64 "$DENSE_BIN" "$r"; run_case sparse64 "$SPARSE_BIN" "$r"; done

python3 - "$RESULT" "$EXPECT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));expect=sys.argv[2];by={}
for r in rows:by.setdefault(r['backend'],[]).append(r)
res={b:{r['residue'] for r in g} for b,g in by.items()}
if any(len(x)!=1 for x in res.values()):raise SystemExit(f'unstable residues {res}')
z={b:next(iter(x)) for b,x in res.items()}
if len(set(z.values()))!=1:raise SystemExit(f'RESIDUE MISMATCH {z}')
answer=next(iter(z.values()))
if expect and answer!=expect:raise SystemExit(f'EXPECTED MISMATCH got={answer} expected={expect}')
print('residue_match=OK',answer)
s={}
for b,g in by.items():
    def med(k):return statistics.median(float(r[k]) for r in g)
    av=[float(r['mc_avg_pct']) for r in g if r['mc_avg_pct']!='NA'];mx=[float(r['mc_max_pct']) for r in g if r['mc_max_pct']!='NA']
    s[b]=(med('wall_s'),med('forward_low_s')+med('reverse_low_s'),med('forward_high_s')+med('reverse_high_s'),statistics.median(av) if av else float('nan'),max(mx) if mx else float('nan'))
for b,(w,l,h,a,m) in sorted(s.items(),key=lambda x:x[1][0]):print(b,f'wall_s={w:.6f}',f'low_s={l:.6f}',f'high_s={h:.6f}',f'mc_avg_pct={a:.3f}',f'mc_max_pct={m:.3f}')
d=s['dense64'];q=s['sparse64'];print(f'sparse_wall_speedup={d[0]/q[0]:.6f}',f'sparse_low_speedup={d[1]/q[1]:.6f}',f'sparse_high_speedup={d[2]/q[2]:.6f}',f'mc_avg_delta_pp={q[3]-d[3]:.3f}')
PY

cat "$PLAN"
echo "result=$RESULT plan=$PLAN ptxas=$RESOURCE logs=$LOGDIR orbit_ilp=$ORBITCTA_COL_ILP pair_mlp=$PAIR_MLP cpasync_pair=$CPASYNC_PAIR window4=$RANKFORMULA_MLP_WINDOW4 pm_accum=$PM_ACCUM" >&2
