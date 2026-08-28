#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-native}"
PTX_ARCH="${PTX_ARCH:-sm_80}"
RUNS="${RUNS:-1}"
MOD="${MOD:-4294967291}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_shard_address_hi32_seed_production_ab}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"

if (( RUNS < 1 )); then echo "RUNS must be >=1" >&2; exit 2; fi
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < 8 )); then echo "need 8 visible GPUs; got $visible" >&2; exit 2; fi

bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-hi32-seed-w28-ngpu8-proof.sh"
ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-shard-address-hi32-seed-production-ptx-proof.sh"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GENERATOR="$ONEESAN_ROOT/scripts/build/gen-b300-shard-address-hi32-seed.py"
GENSRC="$LOGDIR/generated.cu"
python3 "$GENERATOR" "$SRC" "$GENSRC" >"$LOGDIR/generate.out"

BINS=()
for mode in 1 5; do
  BINS[$mode]="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_hi32ab_mode${mode}"
  TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" \
    -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 \
    -DB300_FAST_SHARD_ADDRESS8=0 -DB300_SHARD_ADDRESS_MODE="$mode" \
    "$GENSRC" -o "${BINS[$mode]}" \
    >"$LOGDIR/mode_${mode}.build.out" 2>"$LOGDIR/mode_${mode}.build.err"
done

printf 'mode\trun\tresidue\twall_s\tactive_max_s\tactive_sum_s\tprepare_s\n' >"$RESULT"
run_one(){
  local mode="$1" run="$2"
  local out="$LOGDIR/mode_${mode}_run${run}.out" err="$LOGDIR/mode_${mode}_run${run}.err"
  echo "=== full B300 W28x8 shard-address mode=$mode run $run/$RUNS ===" >&2
  GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" \
    "${BINS[$mode]}" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  local line residue wall active_max active_sum prepare
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { tail -n 100 "$err" >&2 || true; cat "$out" >&2; exit 3; }
  residue="$(sed -nE 's/.* residue=([^[:space:]]+).*/\1/p' <<<"$line")"
  wall="$(sed -nE 's/.* wall_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  active_max="$(sed -nE 's/.* active_max_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  active_sum="$(sed -nE 's/.* active_sum_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  prepare="$(sed -nE 's/.* prepare_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$residue" && -n "$wall" && -n "$active_max" && -n "$active_sum" && -n "$prepare" ]] || { echo "$line" >&2; exit 4; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$residue" "$wall" "$active_max" "$active_sum" "$prepare" >>"$RESULT"
}
for ((r=1;r<=RUNS;++r)); do
  if (( r & 1 )); then order=(1 5); else order=(5 1); fi
  for mode in "${order[@]}"; do run_one "$mode" "$r"; done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
out=[]; residues={}
for mode in ('1','5'):
    rs=[r for r in rows if r['mode']==mode]
    if not rs: raise SystemExit(f'missing mode={mode}')
    rr={r['residue'] for r in rs}
    if len(rr)!=1: raise SystemExit(f'unstable residue mode={mode}: {rr}')
    residues[mode]=next(iter(rr))
    def med(k): return statistics.median(float(r[k]) for r in rs)
    out.append({'mode':mode,'runs':len(rs),'residue':residues[mode],
                'median_wall_s':f'{med("wall_s"):.9f}',
                'median_active_max_s':f'{med("active_max_s"):.9f}',
                'median_active_sum_s':f'{med("active_sum_s"):.9f}',
                'median_prepare_s':f'{med("prepare_s"):.9f}'})
if residues['1']!=residues['5']: raise SystemExit(f'residue mismatch {residues}')
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t');w.writeheader();w.writerows(out)
q={r['mode']:r for r in out}
for key,label in [('median_wall_s','wall'),('median_active_max_s','active_max'),('median_active_sum_s','active_sum')]:
    old=float(q['1'][key]); new=float(q['5'][key])
    print(f'b300_hi32_seed_{label}_speedup={old/new:.6f}x')
    print(f'b300_hi32_seed_{label}_delta_pct={(new/old-1)*100:.4f}%')
print('mode1=three_compare_subtract_stages')
print('mode5=hi32_seed_single_correction')
print('main_seed=(hi32*365)>>12 main_seed_shiftadd=1 block_seed=hi32>>2 correction_max=1')
print(f'residue={residues["1"]}')
print(f'summary={dst}')
PY

echo "b300-shard-address-hi32-seed-production-ab OK runs=$RUNS mod=$MOD target_mib=$TARGET_MIB max_window=$MAX_WINDOW result=$RESULT" >&2
