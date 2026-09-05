#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
SCRATCH_MIB="${SCRATCH_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-27}"
GPUS=8
ARCH="${ARCH:-native}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_vmm_main_pull_row${ROWS}_ab}"
BASE_BIN="${BASE_BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_vmm_mainpull0_n27}"
CAND_BIN="${CAND_BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_vmm_mainpull1_n27}"
[[ "$ROWS" =~ ^[0-9]+$ ]] && (( ROWS >= 1 && ROWS <= 28 )) || { echo "ROWS must be 1..28" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= GPUS )) || { echo "need $GPUS GPUs, visible=$visible" >&2; exit 2; }
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$BASE_BIN")"

echo "=== build baseline VMM + Mate cache ===" >&2
N="$N" ARCH="$ARCH" OUT="$BASE_BIN" MAIN_MATE_CACHE=1 MAIN_PULL=0 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm.sh" >"${PREFIX}.base.build.out" 2>"${PREFIX}.base.build.err"
echo "=== build candidate VMM + Mate cache + main pull ===" >&2
N="$N" ARCH="$ARCH" OUT="$CAND_BIN" MAIN_MATE_CACHE=1 MAIN_PULL=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm.sh" >"${PREFIX}.cand.build.out" 2>"${PREFIX}.cand.build.err"
grep -Fq 'main_mate_cache=1 main_pull=0' "${PREFIX}.base.build.out"
grep -Fq 'main_mate_cache=1 main_pull=1' "${PREFIX}.cand.build.out"
grep -Fq 'p_gt_1_main_update=destination_pull' "${PREFIX}.cand.build.out"

run_one(){
  local name="$1" bin="$2"
  local out="${PREFIX}.${name}.out" err="${PREFIX}.${name}.err" tele="${PREFIX}.${name}.gpu.csv"
  : >"$tele"
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,memory.used,memory.total \
    --format=csv,noheader,nounits -l 1 >"$tele" 2>/dev/null &
  local mon=$!
  sleep 1
  set +e
  B300_ROW_LIMIT="$ROWS" "$bin" "$N" "$MOD" "$SCRATCH_MIB" "$MAX_WINDOW" "$GPUS" >"$out" 2>"$err"
  local rc=$?
  set -e
  kill "$mon" 2>/dev/null || true
  wait "$mon" 2>/dev/null || true
  (( rc == 0 )) || { echo "$name failed rc=$rc" >&2; tail -n 100 "$err" >&2 || true; return "$rc"; }
  local line
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$name missing backend result" >&2; cat "$out" >&2; return 4; }
  local residue wall
  residue="$(sed -nE 's/.* residue=([^[:space:]]+).*/\1/p' <<<"$line")"
  wall="$(sed -nE 's/.* wall_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$residue" && -n "$wall" ]] || { echo "$name failed to parse residue/wall" >&2; return 5; }
  local stats
  stats="$(python3 - "$tele" <<'PY'
import csv,sys
m=[];s=[]
with open(sys.argv[1],newline='') as f:
    for r in csv.reader(f):
        if len(r)<4: continue
        try:s.append(float(r[2].strip()));m.append(float(r[3].strip()))
        except ValueError:pass
if not m: print('nan nan nan')
else: print(f'{sum(m)/len(m):.9f} {max(m):.9f} {sum(s)/len(s):.9f}')
PY
)"
  printf '%s\t%s\t%s\t%s\t%s\n' "$residue" "$wall" $stats
}

# Baseline first keeps the default comparison conservative for a candidate that
# may benefit from a warmed GPU. Re-run with a fresh process if the margin is tiny.
echo "=== run baseline rows=$ROWS ===" >&2
read -r BASE_RES BASE_WALL BASE_MEM BASE_MEM_MAX BASE_SM <<<"$(run_one base "$BASE_BIN")"
echo "=== run main-pull candidate rows=$ROWS ===" >&2
read -r CAND_RES CAND_WALL CAND_MEM CAND_MEM_MAX CAND_SM <<<"$(run_one cand "$CAND_BIN")"

[[ "$BASE_RES" == "$CAND_RES" ]] || {
  echo "main-pull row-limited residue mismatch baseline=$BASE_RES candidate=$CAND_RES" >&2
  exit 6
}
python3 - "$BASE_WALL" "$CAND_WALL" "$BASE_MEM" "$CAND_MEM" "$BASE_MEM_MAX" "$CAND_MEM_MAX" "$BASE_SM" "$CAND_SM" <<'PY'
import sys
bw,cw,bm,cm,bmx,cmx,bs,cs=map(float,sys.argv[1:])
print(f'b300_main_pull_wall_baseline_s={bw:.9f}')
print(f'b300_main_pull_wall_candidate_s={cw:.9f}')
print(f'b300_main_pull_wall_speedup={bw/cw:.9f}x')
print(f'b300_main_pull_wall_delta_pct={(cw/bw-1)*100:.4f}%')
print(f'b300_main_pull_memctl_baseline_avg_pct={bm:.3f}')
print(f'b300_main_pull_memctl_candidate_avg_pct={cm:.3f}')
print(f'b300_main_pull_memctl_avg_delta_pp={cm-bm:.3f}')
print(f'b300_main_pull_memctl_baseline_max_pct={bmx:.3f}')
print(f'b300_main_pull_memctl_candidate_max_pct={cmx:.3f}')
print(f'b300_main_pull_sm_baseline_avg_pct={bs:.3f}')
print(f'b300_main_pull_sm_candidate_avg_pct={cs:.3f}')
PY
printf 'b300_main_pull_rows=%s\n' "$ROWS"
printf 'b300_main_pull_residue=%s\n' "$BASE_RES"
printf 'b300_main_pull_exact_intermediate_match=1\n'
printf 'b300_main_pull_scratch_mib=%s\n' "$SCRATCH_MIB"
printf 'b300_main_pull_max_window=%s\n' "$MAX_WINDOW"
printf 'b300_main_pull_base_stdout=%s\n' "${PREFIX}.base.out"
printf 'b300_main_pull_cand_stdout=%s\n' "${PREFIX}.cand.out"
printf 'b300_main_pull_base_telemetry=%s\n' "${PREFIX}.base.gpu.csv"
printf 'b300_main_pull_cand_telemetry=%s\n' "${PREFIX}.cand.gpu.csv"
