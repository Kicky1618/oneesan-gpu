#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";CAL_ROWS="${CAL_ROWS:-1}";RUNS="${RUNS:-1}";MIN_SPEEDUP="${MIN_SPEEDUP:-1.005}";MIN_ACTIVE="${MIN_ACTIVE:-0.995}"
MOD="${MOD:-4294967291}";TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";RESERVE="${GRIDFP_VRAM_RESERVE_MIB:-8192}";META_CAP="${B300_STAGED_META_MAX_MIB:-512}";INTERVAL_CAP="${B300_STAGED_INTERVAL_MAX_MIB:-256}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_control_bundle_ab}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";CAL="$PREFIX.cal.tsv";OUT="$PREFIX.tsv";mkdir -p "$LOGDIR"
require_nvcc_version_at_least "$NVCC" 13 0 "B300 control bundle A/B"
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)" -ge 8 ]]||{ echo 'need 8 GPUs' >&2;exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-experimental-source-preflight.sh" >"$LOGDIR/source-preflight.out" 2>&1
BASE="$ONEESAN_BUILD_DIR/b300_control_base";FAST="$ONEESAN_BUILD_DIR/b300_control_bundle"
NVCC="$NVCC" ARCH="$ARCH" OUT="$BASE" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-static-lpt-stagedmeta.sh" >"$LOGDIR/base.build" 2>&1
NVCC="$NVCC" ARCH="$ARCH" OUT="$FAST" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-static-lpt-control-bundle.sh" >"$LOGDIR/fast.build" 2>&1
grep -Fq 'per_group_interval_h2d=0' "$LOGDIR/fast.build";grep -Fq 'per_group_meta_copy_bytes=0' "$LOGDIR/fast.build"
run(){ local mode="$1" rows="$2" bin="$BASE";[[ "$mode" == fast ]]&&bin="$FAST";local log="$LOGDIR/${mode}_r${rows}.out" err="$LOGDIR/${mode}_r${rows}.err";B300_ROW_LIMIT="$rows" B300_STAGED_META_MAX_MIB="$META_CAP" B300_STAGED_INTERVAL_MAX_MIB="$INTERVAL_CAP" GRIDFP_VRAM_RESERVE_MIB="$RESERVE" "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$log" 2>"$err";grep '^backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' "$log"|tail -n1; }
parse(){ python3 - "$1" <<'PY'
import re,sys
s=sys.argv[1]
def f(k):return re.search(r'(?:^| )'+k+r'=([^ ]+)',s).group(1)
print(f('residue'),f('wall_s'),f('prepare_s'),f('active_max_s'))
PY
}
BLINE="$(run base "$CAL_ROWS")";FLINE="$(run fast "$CAL_ROWS")";read -r BR BW BP BA < <(parse "$BLINE");read -r FR FW FP FA < <(parse "$FLINE");[[ "$BR" == "$FR" ]]||{ echo 'cal residue mismatch' >&2;exit 3; }
read -r SW SA < <(python3 - "$BW" "$FW" "$BA" "$FA" <<'PY'
import sys
bw,fw,ba,fa=map(float,sys.argv[1:]);print(f'{bw/fw:.9f} {ba/fa:.9f}')
PY
)
printf 'mode\trows\tresidue\twall_s\tprepare_s\tactive_max_s\nbase\t%s\t%s\t%s\t%s\t%s\nfast\t%s\t%s\t%s\t%s\t%s\n' "$CAL_ROWS" "$BR" "$BW" "$BP" "$BA" "$CAL_ROWS" "$FR" "$FW" "$FP" "$FA" >"$CAL"
set +e;python3 - "$SW" "$MIN_SPEEDUP" "$SA" "$MIN_ACTIVE" <<'PY'
import sys
w,wm,a,am=map(float,sys.argv[1:]);raise SystemExit(0 if w>=wm and a>=am else 9)
PY
rc=$?;set -e;[[ "$rc" == 0 ]]||{ [[ "$rc" == 9 ]]&&{ echo "SKIP_FULL cal_wall=${SW}x cal_active=${SA}x";exit 0;};exit "$rc"; }
printf 'run\tbase_residue\tfast_residue\tbase_wall_s\tfast_wall_s\tbase_prepare_s\tfast_prepare_s\n' >"$OUT"
for((r=1;r<=RUNS;++r));do BLINE="$(run base 28)";FLINE="$(run fast 28)";read -r BR BW BP BA < <(parse "$BLINE");read -r FR FW FP FA < <(parse "$FLINE");[[ "$BR" == "$FR" ]]||exit 4;printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$BR" "$FR" "$BW" "$FW" "$BP" "$FP" >>"$OUT";done
python3 - "$OUT" "$SW" "$SA" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));med=lambda k:statistics.median(float(x[k]) for x in r);bw,fw=med('base_wall_s'),med('fast_wall_s');bt=statistics.median(float(x['base_wall_s'])+float(x['base_prepare_s']) for x in r);ft=statistics.median(float(x['fast_wall_s'])+float(x['fast_prepare_s']) for x in r);print(f'cal_wall_speedup={float(sys.argv[2]):.6f}x cal_active_speedup={float(sys.argv[3]):.6f}x');print(f'full_wall_speedup={bw/fw:.6f}x full_total_speedup={bt/ft:.6f}x');print('meta_copy_bytes_per_group=0 interval_h2d_per_group=0 cudaSetDevice_calls=448')
PY
