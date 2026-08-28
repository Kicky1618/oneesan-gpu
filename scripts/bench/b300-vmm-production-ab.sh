#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-native}"
PTX_ARCH="${PTX_ARCH:-sm_103}"
RUNS="${RUNS:-1}"
MOD="${MOD:-4294967291}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_vmm_production_ab}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"

(( RUNS >= 1 )) || { echo "RUNS must be >=1" >&2; exit 2; }
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= 8 )) || { echo "need 8 visible GPUs; got $visible" >&2; exit 2; }
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-generate-proof.sh"
ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-ptx-proof.sh" \
  >"$LOGDIR/vmm_ptx.out" 2>"$LOGDIR/vmm_ptx.err"

BIN_COMPARE="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmmab_compare"
BIN_U32="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmmab_u32"
BIN_VMM="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmmab_vmm"

N=27 SHARD_ADDRESS_MODE=1 ARCH="$ARCH" OUT="$BIN_COMPARE" \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-shard-address-mode.sh" \
  >"$LOGDIR/compare.build.out" 2>"$LOGDIR/compare.build.err"
N=27 ARCH="$ARCH" OUT="$BIN_U32" \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-shard-address-hi32-u32addr.sh" \
  >"$LOGDIR/u32.build.out" 2>"$LOGDIR/u32.build.err"
N=27 ARCH="$ARCH" OUT="$BIN_VMM" \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm.sh" \
  >"$LOGDIR/vmm.build.out" 2>"$LOGDIR/vmm.build.err"

nvidia-smi -L >&2
nvidia-smi topo -m >&2 || true
printf 'mode\trun\tresidue\twall_s\tactive_max_s\tactive_sum_s\tprepare_s\n' >"$RESULT"

run_one(){
  local mode="$1" run="$2" bin
  case "$mode" in
    compare) bin="$BIN_COMPARE" ;;
    u32) bin="$BIN_U32" ;;
    vmm) bin="$BIN_VMM" ;;
    *) return 2 ;;
  esac
  local out="$LOGDIR/${mode}_run${run}.out" err="$LOGDIR/${mode}_run${run}.err"
  echo "=== B300 W28x8 full solver mode=$mode run $run/$RUNS ===" >&2
  GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" \
    "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  local line residue wall active_max active_sum prepare
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { tail -n 120 "$err" >&2 || true; cat "$out" >&2; exit 3; }
  field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p" <<<"$line"; }
  residue="$(field residue)"; wall="$(field wall_s)"; active_max="$(field active_max_s)"; active_sum="$(field active_sum_s)"; prepare="$(field prepare_s)"
  [[ -n "$residue" && -n "$wall" && -n "$active_max" && -n "$active_sum" && -n "$prepare" ]] || { echo "$line" >&2; exit 4; }
  if [[ "$mode" == vmm ]]; then grep -Fq 'backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' <<<"$line" || exit 5; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$residue" "$wall" "$active_max" "$active_sum" "$prepare" >>"$RESULT"
}

for ((r=1;r<=RUNS;++r)); do
  case $(( (r-1)%3 )) in
    0) order=(compare u32 vmm) ;;
    1) order=(u32 vmm compare) ;;
    2) order=(vmm compare u32) ;;
  esac
  for mode in "${order[@]}"; do run_one "$mode" "$r"; done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
out=[]; residues={}
for mode in ('compare','u32','vmm'):
    rs=[r for r in rows if r['mode']==mode]
    if not rs: raise SystemExit(f'missing mode={mode}')
    rr={r['residue'] for r in rs}
    if len(rr)!=1: raise SystemExit(f'unstable residue {mode}: {rr}')
    residues[mode]=next(iter(rr))
    def med(k): return statistics.median(float(r[k]) for r in rs)
    out.append({'mode':mode,'runs':len(rs),'residue':residues[mode],
                'median_wall_s':f'{med("wall_s"):.9f}',
                'median_active_max_s':f'{med("active_max_s"):.9f}',
                'median_active_sum_s':f'{med("active_sum_s"):.9f}',
                'median_prepare_s':f'{med("prepare_s"):.9f}'})
if len(set(residues.values()))!=1: raise SystemExit(f'residue mismatch {residues}')
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t');w.writeheader();w.writerows(out)
q={r['mode']:r for r in out}; base=float(q['compare']['median_wall_s']); basea=float(q['compare']['median_active_max_s'])
for mode in ('u32','vmm'):
    w=float(q[mode]['median_wall_s']); a=float(q[mode]['median_active_max_s'])
    print(f'b300_{mode}_wall_speedup_vs_compare={base/w:.6f}x')
    print(f'b300_{mode}_wall_delta_pct_vs_compare={(w/base-1)*100:.4f}%')
    print(f'b300_{mode}_active_max_speedup_vs_compare={basea/a:.6f}x')
best=min(out,key=lambda r:float(r['median_wall_s']))
print(f'b300_vmm_ab_best_mode={best["mode"]}')
print(f'b300_vmm_ab_best_median_wall_s={best["median_wall_s"]}')
print('compare=three_compare_subtract_shard_address')
print('u32=hi32_seed_fully_u32_shard_address')
print('vmm=contiguous_multi_gpu_virtual_address_direct_global_index')
print(f'residue={residues["compare"]}')
print(f'summary={dst}')
PY

echo "b300-vmm-production-ab OK runs=$RUNS mod=$MOD target_mib=$TARGET_MIB max_window=$MAX_WINDOW result=$RESULT" >&2
