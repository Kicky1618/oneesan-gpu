#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27; NGPU=8
MOD="${MOD:-4294967291}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"
THREADS="${THREADS:-256}"
SAMPLE_LOG2="${SAMPLE_LOG2:-20}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankstate_closure_warp_profile_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
ISO="${ISO:-$ONEESAN_BUILD_DIR/rankstate_closure_warp_profile_$$}"
BASE_BIN="$ONEESAN_BUILD_DIR/b300_rankstate_closure_warp_profile_base_n27"
PROF_BIN="$ONEESAN_BUILD_DIR/b300_rankstate_closure_warp_profile_n27"
mkdir -p "$LOGDIR" "$ISO" "$ISO/tmp" "$(dirname "$PREFIX")"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=1024&&THREADS%32==0)) || { echo 'THREADS must be a warp multiple 32..1024' >&2; exit 2; }
[[ "$SAMPLE_LOG2" =~ ^[0-9]+$ ]] && ((SAMPLE_LOG2>=8&&SAMPLE_LOG2<=30)) || { echo 'SAMPLE_LOG2 must be 8..30' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-warp-batch-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-warp-hybrid-proof.sh"

BASE_BUILD_OUT="$LOGDIR/base.build.out"; BASE_BUILD_ERR="$LOGDIR/base.build.err"
echo '=== build rank-state ILP4 all-warp reference ===' >&2
ONEESAN_BUILD_DIR="$ISO" ONEESAN_TMP_DIR="$ISO/tmp" \
N=27 ARCH="$ARCH" OUT="$BASE_BIN" FAST_SHARD_ADDRESS8=1 \
MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 \
RANK_STATE_ILP2=0 RANK_STATE_ILP3=0 RANK_STATE_ILP4=1 BLOCK_CLOSURE_QUAD=0 BLOCK_CLOSURE_WARP=1 \
HOT_DELTA_TABLE=0 CONCURRENT_GROUP_IO=1 MAXRREGCOUNT=0 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$BASE_BUILD_OUT" 2>"$BASE_BUILD_ERR"
[[ -x "$BASE_BIN" ]] || { echo 'reference binary missing' >&2; exit 3; }
BUILD_SRC="$(sed -nE 's/^  build_source=(.*)$/\1/p' "$BASE_BUILD_OUT" | tail -n1)"
[[ -n "$BUILD_SRC" && -f "$BUILD_SRC" ]] || { echo 'could not resolve generated all-warp source' >&2; exit 3; }

PROF_SRC="$ISO/final_closure_warp_profile.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-warp-profile.py" "$BUILD_SRC" "$PROF_SRC" >"$LOGDIR/profile.transform.out"
grep -Fq 'B300_CLOSURE_WARP_PROF[29][29][33]' "$PROF_SRC"
grep -Fq 'b300_closure_warp_profile_threshold threshold=' "$PROF_SRC"

echo '=== compile sampled profile binary ===' >&2
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
  -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=1 -DB300_BLOCK_CLOSURE_QUAD=0 \
  -DB300_CLOSURE_WARP_PROF_LOG2="$SAMPLE_LOG2" "$PROF_SRC" -o "$PROF_BIN" \
  >"$LOGDIR/profile.build.out" 2>"$LOGDIR/profile.build.err"
[[ -x "$PROF_BIN" ]] || { echo 'profile binary missing' >&2; exit 3; }

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
run_one(){
  local mode="$1" bin="$2"
  local out="$LOGDIR/$mode.out" err="$LOGDIR/$mode.err"
  echo "=== run $mode rows=$ROWS threads=$THREADS ===" >&2
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" \
    "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  local rc=$?; set -e
  ((rc==0)) || { echo "$mode failed rc=$rc" >&2; tail -n 180 "$err" >&2 || true; return "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$mode missing backend line" >&2; return 4; }
  printf '%s\t%s\n' "$(field residue "$line")" "$(field wall_s "$line")"
}
IFS=$'\t' read -r BASE_RES BASE_WALL <<<"$(run_one base "$BASE_BIN")"
IFS=$'\t' read -r PROF_RES PROF_WALL <<<"$(run_one profile "$PROF_BIN")"
[[ "$BASE_RES" == "$PROF_RES" ]] || { echo "FATAL closure profile residue mismatch base=$BASE_RES profile=$PROF_RES" >&2; exit 5; }
PROFILE_OUT="$LOGDIR/profile.out"
SUMMARY_LINE="$(grep '^b300_closure_warp_profile ' "$PROFILE_OUT" | tail -n1 || true)"
[[ -n "$SUMMARY_LINE" ]] || { echo 'profile summary line missing' >&2; exit 6; }
SAMPLES="$(field samples "$SUMMARY_LINE")"; CANDIDATES="$(field candidates "$SUMMARY_LINE")"
[[ "$SAMPLES" =~ ^[0-9]+$ ]] && ((SAMPLES>0)) || { echo "profile sampled zero closure states samples=$SAMPLES" >&2; exit 6; }

RECOMMEND="$(python3 - "$PROFILE_OUT" <<'PY'
import sys
rows=[]
for line in open(sys.argv[1],encoding='utf-8',errors='replace'):
    if not line.startswith('b300_closure_warp_profile_threshold '):continue
    d={}
    for x in line.split()[1:]:
        if '=' in x:
            k,v=x.split('=',1);d[k]=v
    try: rows.append((int(d['threshold']),float(d['warp_state_fraction']),float(d['warp_candidate_fraction'])))
    except Exception: pass
if not rows:raise SystemExit('no threshold rows in closure profile')
rows.sort()
def keep(frac):
    ok=[r for r in rows if r[2]>=frac]
    return max(ok,key=lambda r:r[0]) if ok else rows[0]
for frac,name in ((.95,'keep95'),(.90,'keep90'),(.80,'keep80')):
    t,s,c=keep(frac);print(f'{name}={t},{s:.9f},{c:.9f}')
seeds=set()
for frac in (.95,.90,.80):
    t,_,_=keep(frac)
    for q in (t-2,t,t+2):
        if 1<=q<=20:seeds.add(q)
print('thresholds='+','.join(map(str,sorted(seeds))))
PY
)"
KEEP95="$(sed -n 's/^keep95=//p' <<<"$RECOMMEND")"
KEEP90="$(sed -n 's/^keep90=//p' <<<"$RECOMMEND")"
KEEP80="$(sed -n 's/^keep80=//p' <<<"$RECOMMEND")"
THRESHOLDS_CSV="$(sed -n 's/^thresholds=//p' <<<"$RECOMMEND")"
THRESHOLDS_SPACE="${THRESHOLDS_CSV//,/ }"

printf 'b300_closure_warp_profile_exact_intermediate_match=1\n'
printf 'b300_closure_warp_profile_residue=%s\n' "$BASE_RES"
printf 'b300_closure_warp_profile_reference_wall_s=%s\n' "$BASE_WALL"
printf 'b300_closure_warp_profile_instrumented_wall_s=%s\n' "$PROF_WALL"
printf 'b300_closure_warp_profile_sample_log2=%s\n' "$SAMPLE_LOG2"
printf 'b300_closure_warp_profile_samples=%s\n' "$SAMPLES"
printf 'b300_closure_warp_profile_candidates=%s\n' "$CANDIDATES"
printf 'b300_closure_warp_profile_keep95=%s\n' "$KEEP95"
printf 'b300_closure_warp_profile_keep90=%s\n' "$KEEP90"
printf 'b300_closure_warp_profile_keep80=%s\n' "$KEEP80"
printf 'b300_closure_warp_profile_thresholds_csv=%s\n' "$THRESHOLDS_CSV"
printf 'b300_closure_warp_profile_thresholds="%s"\n' "$THRESHOLDS_SPACE"
printf 'b300_closure_warp_profile_raw=%s\n' "$PROFILE_OUT"
printf 'b300_closure_warp_profile_note=instrumented wall is not a speed metric; thresholds retain measured valid-source candidate work and require subsequent exact wall A/B\n'
