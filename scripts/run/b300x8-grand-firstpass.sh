#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'B300 grand first-pass currently targets n=27' >&2; exit 2; }

PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
ARCH="${ARCH:-native}"
MAX_WINDOW="${MAX_WINDOW:-14}"
SMOKE_PRIME="${SMOKE_PRIME:-4294967291}"
FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"
BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"
WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
RUN_NEXTSELF_STAGE="${RUN_NEXTSELF_STAGE:-1}"
RUN_HYBRID_STAGE="${RUN_HYBRID_STAGE:-1}"
RUN_HYBRID_NS_STAGE="${RUN_HYBRID_NS_STAGE:-1}"
NEXTSELF_THREADS="${NEXTSELF_THREADS:-256}"
NEXTSELF_SEARCH_ROWS="${NEXTSELF_SEARCH_ROWS:-1}"
NEXTSELF_VALIDATE_ROWS="${NEXTSELF_VALIDATE_ROWS:-4 8}"
NEXTSELF_SEARCH_REPEATS="${NEXTSELF_SEARCH_REPEATS:-1}"
NEXTSELF_VALIDATE_REPEATS="${NEXTSELF_VALIDATE_REPEATS:-1}"
NEXTSELF_MIN_SPEEDUP="${NEXTSELF_MIN_SPEEDUP:-1.01}"
HYBRID_MIN_SPEEDUP="${HYBRID_MIN_SPEEDUP:-1.01}"
HYBRID_NS_MIN_SPEEDUP="${HYBRID_NS_MIN_SPEEDUP:-1.01}"
HYBRID_NS_SEARCH_REPEATS="${HYBRID_NS_SEARCH_REPEATS:-1}"
HYBRID_NS_VALIDATE_REPEATS="${HYBRID_NS_VALIDATE_REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
RACE_PREFIX="${RACE_PREFIX:-${PREFIX}.race}"
LOG="${LOG:-${PREFIX}.log}"
META="${META:-${PREFIX}.meta}"
SELECTED_ENV="${SELECTED_ENV:-${PREFIX}.selected.env}"
RACE_RESULT="${RACE_RESULT:-${RACE_PREFIX}.tsv}"

for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE; do
  v="${!x}"
  [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
for x in MAX_WINDOW FORCED_TARGET_MIB BUCKET_TARGET_MIB; do
  v="${!x}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || { echo "$x must be positive integer" >&2; exit 2; }
done
[[ "$SMOKE_PRIME" =~ ^[1-9][0-9]*$ ]] || { echo 'SMOKE_PRIME must be positive integer' >&2; exit 2; }
[[ "$NEXTSELF_THREADS" =~ ^[0-9]+$ ]] && ((NEXTSELF_THREADS>=32 && NEXTSELF_THREADS<=768 && NEXTSELF_THREADS%32==0)) || {
  echo 'NEXTSELF_THREADS must be warp multiple 32..768' >&2; exit 2;
}
for x in NEXTSELF_SEARCH_REPEATS NEXTSELF_VALIDATE_REPEATS HYBRID_NS_SEARCH_REPEATS HYBRID_NS_VALIDATE_REPEATS; do
  v="${!x}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || { echo "$x must be >=1" >&2; exit 2; }
done
python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" <<'PY'
import sys
for name,v in zip(('NEXTSELF_MIN_SPEEDUP','HYBRID_MIN_SPEEDUP','HYBRID_NS_MIN_SPEEDUP'),map(float,sys.argv[1:])):
    if v < 1.0: raise SystemExit(f'{name} must be >=1')
PY
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v git >/dev/null || { echo 'git required' >&2; exit 2; }
command -v sha256sum >/dev/null || { echo 'sha256sum required' >&2; exit 2; }
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( GPU_COUNT >= 8 )) || { echo "need 8 visible GPUs; got $GPU_COUNT" >&2; exit 2; }

mkdir -p "$(dirname "$LOG")" "$(dirname "$META")" "$(dirname "$SELECTED_ENV")" "$WORK_ROOT"
HEAD_SHA="$(git -C "$ONEESAN_ROOT" rev-parse HEAD)"
HEAD_DIRTY=0
[[ -z "$(git -C "$ONEESAN_ROOT" status --porcelain=v1 --untracked-files=normal)" ]] || HEAD_DIRTY=1
PROFILE_SHA="$(sha256sum "$PROFILE_FILE" | awk '{print $1}')"

{
  printf 'schema=2\n'
  printf 'n=%s\n' "$N"
  printf 'head_sha=%s\n' "$HEAD_SHA"
  printf 'head_dirty=%s\n' "$HEAD_DIRTY"
  printf 'profile_file=%s\n' "$PROFILE_FILE"
  printf 'profile_sha256=%s\n' "$PROFILE_SHA"
  printf 'gpu_count=%s\n' "$GPU_COUNT"
  printf 'arch=%s\n' "$ARCH"
  printf 'max_window=%s\n' "$MAX_WINDOW"
  printf 'smoke_prime=%s\n' "$SMOKE_PRIME"
  printf 'forced_target_mib=%s\n' "$FORCED_TARGET_MIB"
  printf 'bucket_target_mib=%s\n' "$BUCKET_TARGET_MIB"
  printf 'work_root=%s\n' "$WORK_ROOT"
  printf 'race_prefix=%s\n' "$RACE_PREFIX"
  printf 'race_result=%s\n' "$RACE_RESULT"
  printf 'select_only=1\n'
  printf 'rebuild_buckets=%s\n' "$REBUILD_BUCKETS"
  printf 'run_nextself_stage=%s\n' "$RUN_NEXTSELF_STAGE"
  printf 'run_hybrid_stage=%s\n' "$RUN_HYBRID_STAGE"
  printf 'run_hybrid_ns_stage=%s\n' "$RUN_HYBRID_NS_STAGE"
  printf 'nextself_threads=%s\n' "$NEXTSELF_THREADS"
  printf 'nextself_search_rows=%s\n' "$NEXTSELF_SEARCH_ROWS"
  printf 'nextself_validate_rows=%s\n' "$NEXTSELF_VALIDATE_ROWS"
  printf 'nextself_search_repeats=%s\n' "$NEXTSELF_SEARCH_REPEATS"
  printf 'nextself_validate_repeats=%s\n' "$NEXTSELF_VALIDATE_REPEATS"
  printf 'nextself_min_speedup=%s\n' "$NEXTSELF_MIN_SPEEDUP"
  printf 'hybrid_min_speedup=%s\n' "$HYBRID_MIN_SPEEDUP"
  printf 'hybrid_ns_min_speedup=%s\n' "$HYBRID_NS_MIN_SPEEDUP"
  printf 'hybrid_ns_search_repeats=%s\n' "$HYBRID_NS_SEARCH_REPEATS"
  printf 'hybrid_ns_validate_repeats=%s\n' "$HYBRID_NS_VALIDATE_REPEATS"
  printf 'hybrid8_nextself_transform_preflight=1\n'
  printf 'grand_selector_contract_preflight=1\n'
  printf 'nvcc_version_begin=1\n'
  nvcc --version | sed 's/^/nvcc: /'
  printf 'nvcc_version_end=1\n'
  printf 'gpu_inventory_begin=1\n'
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu: /'
  printf 'gpu_inventory_end=1\n'
} >"$META"

if (( HEAD_DIRTY )); then
  echo 'WARNING: repository has uncommitted or untracked changes; provenance records head_dirty=1' >&2
fi

echo '=== B300 grand first-pass: GPU-free preflight ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid8-nextself-transform-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-joint-nextgen-hybrid8-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-grand-selector-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-grand-selector-contract-preflight.sh"

echo "=== B300 grand first-pass: n=27 head=${HEAD_SHA:0:12} GPUs=$GPU_COUNT SELECT_ONLY=1 ===" >&2
set +e
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" SMOKE_PRIME="$SMOKE_PRIME" \
  FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" WORK_ROOT="$WORK_ROOT" RACE_PREFIX="$RACE_PREFIX" \
  SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS" \
  RUN_NEXTSELF_STAGE="$RUN_NEXTSELF_STAGE" RUN_HYBRID_STAGE="$RUN_HYBRID_STAGE" RUN_HYBRID_NS_STAGE="$RUN_HYBRID_NS_STAGE" \
  NEXTSELF_THREADS="$NEXTSELF_THREADS" NEXTSELF_SEARCH_ROWS="$NEXTSELF_SEARCH_ROWS" \
  NEXTSELF_VALIDATE_ROWS="$NEXTSELF_VALIDATE_ROWS" NEXTSELF_SEARCH_REPEATS="$NEXTSELF_SEARCH_REPEATS" \
  NEXTSELF_VALIDATE_REPEATS="$NEXTSELF_VALIDATE_REPEATS" NEXTSELF_MIN_SPEEDUP="$NEXTSELF_MIN_SPEEDUP" \
  HYBRID_MIN_SPEEDUP="$HYBRID_MIN_SPEEDUP" HYBRID_NS_MIN_SPEEDUP="$HYBRID_NS_MIN_SPEEDUP" \
  HYBRID_NS_SEARCH_REPEATS="$HYBRID_NS_SEARCH_REPEATS" HYBRID_NS_VALIDATE_REPEATS="$HYBRID_NS_VALIDATE_REPEATS" \
  PREFIX="$PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh" 27 "$@" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

printf 'exit_code=%s\n' "$rc" >>"$META"
if (( rc != 0 )); then
  echo "b300x8-grand-firstpass FAILED rc=$rc log=$LOG meta=$META" >&2
  exit "$rc"
fi

grep -Fq 'SINGLE PASS SELECTED' "$LOG" || {
  echo "grand first-pass completed without SINGLE PASS SELECTED marker: $LOG" >&2
  exit 4
}
grep -Fq 'SELECT_ONLY=1: selected' "$LOG" || {
  echo "grand first-pass did not stop at SELECT_ONLY boundary: $LOG" >&2
  exit 4
}
[[ -s "$RACE_RESULT" ]] || { echo "grand first-pass race result missing: $RACE_RESULT" >&2; exit 4; }

SELECTED_LINE="$(grep -F 'SINGLE PASS SELECTED' "$LOG" | tail -n1)"
WIN="$(python3 - "$RACE_RESULT" "$SELECTED_LINE" <<'PY'
import csv,re,sys
path,line=sys.argv[1:]
r=list(csv.DictReader(open(path,encoding='utf-8'),delimiter='\t'))
ok=[x for x in r if x.get('status')=='ok']
if not ok: raise SystemExit('no successful single-pass candidate')
res={x['residue'] for x in ok}
if len(res)!=1: raise SystemExit('single-pass result residue mismatch')
b=min(ok,key=lambda x:float(x['wall_s']))
m=re.search(r'backend=([^ ]+) profile=([^ ]+) wall_s=([^ ]+) residue=([^ ]+)',line)
if not m: raise SystemExit('selected_line parse failed')
if (b['backend'],b['profile'],b['wall_s'],b['residue']) != m.groups():
    raise SystemExit('selected_line does not match TSV winner')
print('\t'.join((b['backend'],b['profile'],b['binary'],b['residue'],b['wall_s'])))
PY
)"
IFS=$'\t' read -r BEST BEST_PROFILE BEST_BIN BEST_RES BEST_WALL <<<"$WIN"
[[ -x "$BEST_BIN" ]] || { echo "selected binary missing: $BEST_BIN" >&2; exit 4; }
BEST_SHA="$(sha256sum "$BEST_BIN" | awk '{print $1}')"
SHA12="${BEST_SHA:0:12}"
BEST_WORK="$WORK_ROOT/b300_exact_singlepass_${BEST}_${BEST_PROFILE}_${SHA12}_n27"
CHECKPOINT="$BEST_WORK/checkpoint.json"
[[ -s "$CHECKPOINT" ]] || { echo "seeded checkpoint missing: $CHECKPOINT" >&2; exit 4; }
RACE_SHA="$(sha256sum "$RACE_RESULT" | awk '{print $1}')"

RUNTIME_KIND=forced
RUN_TARGET="$FORCED_TARGET_MIB"
RUN_THREADS=0
case "$BEST" in
  warp_tuned) RUNTIME_KIND=warp; RUN_TARGET="$BUCKET_TARGET_MIB" ;;
  orbit_tuned) RUNTIME_KIND=orbit; RUN_TARGET="$BUCKET_TARGET_MIB" ;;
  *)
    [[ "$BEST_PROFILE" =~ ^t([0-9]+)$ ]] || { echo "forced winner profile must encode threads: $BEST_PROFILE" >&2; exit 4; }
    RUN_THREADS="${BASH_REMATCH[1]}"
    ((RUN_THREADS>=32 && RUN_THREADS<=1024 && RUN_THREADS%32==0)) || { echo "bad selected forced threads=$RUN_THREADS" >&2; exit 4; }
    ;;
esac

python3 - "$CHECKPOINT" "$BEST_BIN" "$PROFILE_SHA" "$SMOKE_PRIME" "$BEST_RES" <<'PY'
import hashlib,json,sys
cp,binp,profile_sha,prime,residue=sys.argv[1:]
h=hashlib.sha256()
with open(binp,'rb') as f:
    for z in iter(lambda:f.read(1<<20),b''): h.update(z)
d=json.load(open(cp))
if int(d.get('n',-1)) != 27: raise SystemExit('checkpoint n mismatch')
fp=d.get('solver_fingerprint',{})
if fp.get('schema') != 3 or fp.get('binary_sha256') != h.hexdigest() or fp.get('profile_sha256') != profile_sha:
    raise SystemExit('checkpoint solver fingerprint mismatch')
r=d.get('residues',{}).get(str(int(prime)))
if not r or int(r.get('residue',-1)) != int(residue): raise SystemExit('checkpoint smoke residue missing/mismatch')
PY

{
  printf 'B300_GRAND_SELECTED_SCHEMA=1\n'
  printf 'B300_GRAND_SELECTED_VALIDATED=1\n'
  printf 'B300_GRAND_SELECTED_N=27\n'
  printf 'B300_GRAND_SELECTED_HEAD_SHA=%q\n' "$HEAD_SHA"
  printf 'B300_GRAND_SELECTED_HEAD_DIRTY=%q\n' "$HEAD_DIRTY"
  printf 'B300_GRAND_SELECTED_PROFILE_FILE=%q\n' "$PROFILE_FILE"
  printf 'B300_GRAND_SELECTED_PROFILE_SHA256=%q\n' "$PROFILE_SHA"
  printf 'B300_GRAND_SELECTED_BACKEND=%q\n' "$BEST"
  printf 'B300_GRAND_SELECTED_PROFILE=%q\n' "$BEST_PROFILE"
  printf 'B300_GRAND_SELECTED_BINARY=%q\n' "$BEST_BIN"
  printf 'B300_GRAND_SELECTED_BINARY_SHA256=%q\n' "$BEST_SHA"
  printf 'B300_GRAND_SELECTED_RESIDUE=%q\n' "$BEST_RES"
  printf 'B300_GRAND_SELECTED_WALL_S=%q\n' "$BEST_WALL"
  printf 'B300_GRAND_SELECTED_SMOKE_PRIME=%q\n' "$SMOKE_PRIME"
  printf 'B300_GRAND_SELECTED_RUNTIME_KIND=%q\n' "$RUNTIME_KIND"
  printf 'B300_GRAND_SELECTED_THREADS=%q\n' "$RUN_THREADS"
  printf 'B300_GRAND_SELECTED_TARGET_MIB=%q\n' "$RUN_TARGET"
  printf 'B300_GRAND_SELECTED_MAX_WINDOW=%q\n' "$MAX_WINDOW"
  printf 'B300_GRAND_SELECTED_WORK_DIR=%q\n' "$BEST_WORK"
  printf 'B300_GRAND_SELECTED_CHECKPOINT=%q\n' "$CHECKPOINT"
  printf 'B300_GRAND_SELECTED_RACE_PREFIX=%q\n' "$RACE_PREFIX"
  printf 'B300_GRAND_SELECTED_RACE_RESULT=%q\n' "$RACE_RESULT"
  printf 'B300_GRAND_SELECTED_RACE_RESULT_SHA256=%q\n' "$RACE_SHA"
  printf 'B300_GRAND_SELECTED_FIRSTPASS_META=%q\n' "$META"
} >"$SELECTED_ENV"

{
  printf 'selected_line=%s\n' "$SELECTED_LINE"
  printf 'selected_env=%s\n' "$SELECTED_ENV"
  printf 'selected_backend=%s\n' "$BEST"
  printf 'selected_profile=%s\n' "$BEST_PROFILE"
  printf 'selected_binary=%s\n' "$BEST_BIN"
  printf 'selected_binary_sha256=%s\n' "$BEST_SHA"
  printf 'selected_runtime_kind=%s\n' "$RUNTIME_KIND"
  printf 'selected_target_mib=%s\n' "$RUN_TARGET"
  printf 'selected_work_dir=%s\n' "$BEST_WORK"
  printf 'race_prefix=%s\n' "$RACE_PREFIX"
  printf 'race_result=%s\n' "$RACE_RESULT"
  printf 'race_result_sha256=%s\n' "$RACE_SHA"
  printf 'promotion_contract=1\n'
} >>"$META"

echo "b300x8-grand-firstpass OK log=$LOG meta=$META selected_env=$SELECTED_ENV $SELECTED_LINE" >&2
