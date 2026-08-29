#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";PTX_ARCH="${PTX_ARCH:-sm_103}";CAL_RUNS="${CAL_RUNS:-2}";RUNS="${RUNS:-1}";MIN_SPEEDUP="${MIN_SPEEDUP:-1.005}";MIN_ACTIVE="${MIN_ACTIVE:-0.995}"
MOD="${MOD:-4294967291}";TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";RESERVE="${GRIDFP_VRAM_RESERVE_MIB:-8192}";META_CAP="${B300_STAGED_META_MAX_MIB:-512}";INT_CAP="${B300_STAGED_INTERVAL_MAX_MIB:-256}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_static_lpt_tournament}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";CAL="$PREFIX.cal.tsv";FULL="$PREFIX.full.tsv";mkdir -p "$LOGDIR"
require_nvcc_version_at_least "$NVCC" 13 0 "B300 static LPT tournament";[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)" -ge 8 ]]||exit 2
bash "$ONEESAN_ROOT/scripts/bench/b300-experimental-source-preflight.sh" >"$LOGDIR/source-preflight.out" 2>&1
NVCC="$NVCC" GPUS=8 ARCH="$ARCH" PTX_ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-preflight.sh" >"$LOGDIR/vmm-preflight.out" 2>&1
NVCC="$NVCC" ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-control-bundle-integration-ptx-proof.sh" >"$LOGDIR/bundle-ptx.out" 2>&1
NVCC="$NVCC" ARCH="$ARCH" GPUS=8 bash "$ONEESAN_ROOT/scripts/bench/b300-cross-device-event-wait-preflight.sh" >"$LOGDIR/cross-device-event.out" 2>&1
modes=(base metaptr metaarg intervals workerbind bundle persistent concurrent async windowbatch crosswindow);declare -A build bin
build[base]=b300-hbm32-vmm-static-lpt-stagedmeta.sh;build[metaptr]=b300-hbm32-vmm-static-lpt-metaptr.sh;build[metaarg]=b300-hbm32-vmm-static-lpt-metakernelarg.sh;build[intervals]=b300-hbm32-vmm-static-lpt-staged-intervals.sh;build[workerbind]=b300-hbm32-vmm-static-lpt-workerbind.sh;build[bundle]=b300-hbm32-vmm-static-lpt-control-bundle.sh;build[persistent]=b300-hbm32-vmm-static-lpt-persistent-workers.sh;build[concurrent]=b300-hbm32-vmm-static-lpt-persistent-concurrent.sh;build[async]=b300-hbm32-vmm-static-lpt-persistent-async.sh;build[windowbatch]=b300-hbm32-vmm-static-lpt-windowbatch.sh;build[crosswindow]=b300-hbm32-vmm-static-lpt-cross-window.sh
for m in "${modes[@]}";do bin[$m]="$ONEESAN_BUILD_DIR/b300_tournament_$m";NVCC="$NVCC" ARCH="$ARCH" OUT="${bin[$m]}" bash "$ONEESAN_ROOT/scripts/build/${build[$m]}" >"$LOGDIR/$m.build" 2>&1;done
run(){ local m="$1" rows="$2" tag="$3" r="$4" out="$LOGDIR/${tag}_${r}_${m}.out" err="$LOGDIR/${tag}_${r}_${m}.err";B300_ROW_LIMIT="$rows" B300_STAGED_META_MAX_MIB="$META_CAP" B300_STAGED_INTERVAL_MAX_MIB="$INT_CAP" GRIDFP_VRAM_RESERVE_MIB="$RESERVE" "${bin[$m]}" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err";grep '^backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' "$out"|tail -n1; }
parse(){ python3 - "$1" <<'PY'
import re,sys
s=sys.argv[1]
def f(k):return re.search(r'(?:^| )'+k+r'=([^ ]+)',s).group(1)
print(f('residue'),f('wall_s'),f('prepare_s'),f('active_max_s'))
PY
}
printf 'mode\trun\tresidue\twall_s\tprepare_s\tactive_max_s\n' >"$CAL"
for((r=0;r<CAL_RUNS;++r));do for((j=0;j<${#modes[@]};++j));do m="${modes[$(((j+r)%${#modes[@]}))]}";line="$(run "$m" 1 cal "$r")";read -r rr w p a < <(parse "$line");printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$m" "$r" "$rr" "$w" "$p" "$a" >>"$CAL";done;done
read -r WIN SPEED ACTIVE < <(python3 - "$CAL" "$MIN_ACTIVE" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));amin=float(sys.argv[2]);modes=sorted({x['mode'] for x in r});res={m:{x['residue'] for x in r if x['mode']==m} for m in modes}
if len(res['base'])!=1 or any(v!=res['base'] for v in res.values()):raise SystemExit(f'residue mismatch {res}')
med=lambda m,k:statistics.median(float(x[k]) for x in r if x['mode']==m);bw,ba=med('base','wall_s'),med('base','active_max_s');ok=[m for m in modes if ba/med(m,'active_max_s')>=amin];win=min(ok,key=lambda m:med(m,'wall_s'));print(win,f'{bw/med(win,"wall_s"):.9f}',f'{ba/med(win,"active_max_s"):.9f}')
PY
)
echo "tournament_cal_winner=$WIN wall_speedup=${SPEED}x active_speedup=${ACTIVE}x cal_runs=$CAL_RUNS"
set +e;python3 - "$WIN" "$SPEED" "$MIN_SPEEDUP" <<'PY'
import sys
raise SystemExit(0 if sys.argv[1]!='base' and float(sys.argv[2])>=float(sys.argv[3]) else 9)
PY
rc=$?;set -e;[[ "$rc" == 0 ]]||{ [[ "$rc" == 9 ]]&&{ echo 'SKIP_FULL no qualifying winner';exit 0;};exit "$rc"; }
printf 'mode\trun\tresidue\twall_s\tprepare_s\tactive_max_s\n' >"$FULL"
for((r=0;r<RUNS;++r));do if((r&1));then order=($WIN base);else order=(base $WIN);fi;for m in "${order[@]}";do line="$(run "$m" 28 full "$r")";read -r rr w p a < <(parse "$line");printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$m" "$r" "$rr" "$w" "$p" "$a" >>"$FULL";done;done
python3 - "$FULL" "$WIN" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));w=sys.argv[2];med=lambda m,k:statistics.median(float(x[k]) for x in r if x['mode']==m);res={m:{x['residue'] for x in r if x['mode']==m} for m in ('base',w)}
if res['base']!=res[w]:raise SystemExit(f'full residue mismatch {res}')
bw,ww=med('base','wall_s'),med(w,'wall_s');bt=statistics.median(float(x['wall_s'])+float(x['prepare_s']) for x in r if x['mode']=='base');wt=statistics.median(float(x['wall_s'])+float(x['prepare_s']) for x in r if x['mode']==w);print(f'winner={w} full_wall_speedup={bw/ww:.6f}x full_total_speedup={bt/wt:.6f}x residue={next(iter(res["base"]))}')
PY
