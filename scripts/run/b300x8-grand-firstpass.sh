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
RUN_STAGEI="${RUN_STAGEI:-1}"
RUN_STAGEJ="${RUN_STAGEJ:-${RUN_STAGEH:-1}}"
RUN_STAGEK="${RUN_STAGEK:-1}"
RUN_STAGEL="${RUN_STAGEL:-1}"
RUN_STAGEM="${RUN_STAGEM:-1}"
NEXTSELF_THREADS="${NEXTSELF_THREADS:-256}"
NEXTSELF_SEARCH_ROWS="${NEXTSELF_SEARCH_ROWS:-1}"
NEXTSELF_VALIDATE_ROWS="${NEXTSELF_VALIDATE_ROWS:-4 8}"
NEXTSELF_SEARCH_REPEATS="${NEXTSELF_SEARCH_REPEATS:-1}"
NEXTSELF_VALIDATE_REPEATS="${NEXTSELF_VALIDATE_REPEATS:-1}"
NEXTSELF_MIN_SPEEDUP="${NEXTSELF_MIN_SPEEDUP:-1.01}"
HYBRID_MIN_SPEEDUP="${HYBRID_MIN_SPEEDUP:-1.01}"
HYBRID_NS_MIN_SPEEDUP="${HYBRID_NS_MIN_SPEEDUP:-1.01}"
HYBRID_NS_WIDTH_LIST="${HYBRID_NS_WIDTH_LIST:-1 2 4 8}"
HYBRID_NS_DISTANCE_LIST="${HYBRID_NS_DISTANCE_LIST:-1 2 4}"
HYBRID_NS_SEARCH_REPEATS="${HYBRID_NS_SEARCH_REPEATS:-1}"
HYBRID_NS_VALIDATE_REPEATS="${HYBRID_NS_VALIDATE_REPEATS:-1}"
STAGEI_MIN_SPEEDUP="${STAGEI_MIN_SPEEDUP:-1.002}"
STAGEJ_MIN_SPEEDUP="${STAGEJ_MIN_SPEEDUP:-${STAGEH_MIN_SPEEDUP:-1.002}}"
STAGEK_MIN_SPEEDUP="${STAGEK_MIN_SPEEDUP:-1.002}"
STAGEL_MIN_SPEEDUP="${STAGEL_MIN_SPEEDUP:-1.002}"
STAGEL_GUARD_LIST="${STAGEL_GUARD_LIST:-bb pb bp pp}"
STAGEM_MIN_SPEEDUP="${STAGEM_MIN_SPEEDUP:-1.002}"
STAGEM_POLICY_LIST="${STAGEM_POLICY_LIST:-default cg cs}"
MATE_WIDTH_LIST="${MATE_WIDTH_LIST:-1 2 4 8}"
MATE_DISTANCE_LIST="${MATE_DISTANCE_LIST:-1 2 4}"
MATE_EVICT="${MATE_EVICT:-default}"
MATE_EVICT_LIST="${MATE_EVICT_LIST:-default normal last}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
RACE_PREFIX="${RACE_PREFIX:-${PREFIX}.race}"
LOG="${LOG:-${PREFIX}.log}"
META="${META:-${PREFIX}.meta}"
SELECTED_ENV="${SELECTED_ENV:-${PREFIX}.selected.env}"
RACE_RESULT="${RACE_RESULT:-${RACE_PREFIX}.tsv}"
GRAND_SUMMARY_ENV="${GRAND_SUMMARY_ENV:-${RACE_PREFIX}_grand.env}"

for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM; do
  v="${!x}"
  [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
for x in MAX_WINDOW FORCED_TARGET_MIB BUCKET_TARGET_MIB; do
  v="${!x}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || { echo "$x must be positive integer" >&2; exit 2; }
done
[[ "$SMOKE_PRIME" =~ ^[1-9][0-9]*$ ]] || { echo 'SMOKE_PRIME must be positive integer' >&2; exit 2; }
[[ "$NEXTSELF_THREADS" =~ ^[0-9]+$ ]] && (( NEXTSELF_THREADS>=32 && NEXTSELF_THREADS<=768 && NEXTSELF_THREADS%32==0 )) || {
  echo 'NEXTSELF_THREADS must be warp multiple 32..768' >&2; exit 2;
}
for x in NEXTSELF_SEARCH_REPEATS NEXTSELF_VALIDATE_REPEATS HYBRID_NS_SEARCH_REPEATS HYBRID_NS_VALIDATE_REPEATS; do
  v="${!x}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || { echo "$x must be >=1" >&2; exit 2; }
done
normalize_widths(){
  local raw="$1" out=() w old seen
  for w in $raw; do case "$w" in 1|2|4|8) ;; *) echo "bad width=$w" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$w" ]] && seen=1; done; ((seen)) || out+=("$w"); done
  ((${#out[@]})) || { echo 'width list must not be empty' >&2; exit 2; }; printf '%s' "${out[*]}"
}
normalize_distances(){
  local raw="$1" out=() d old seen
  for d in $raw; do case "$d" in 1|2|4) ;; *) echo "bad distance=$d" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$d" ]] && seen=1; done; ((seen)) || out+=("$d"); done
  ((${#out[@]})) || { echo 'distance list must not be empty' >&2; exit 2; }; printf '%s' "${out[*]}"
}
normalize_evicts(){
  local raw="$1" out=() e old seen
  for e in $raw; do case "$e" in default|normal|last) ;; *) echo "bad eviction hint=$e" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$e" ]] && seen=1; done; ((seen)) || out+=("$e"); done
  ((${#out[@]})) || { echo 'eviction list must not be empty' >&2; exit 2; }; printf '%s' "${out[*]}"
}
normalize_guards(){
  local raw="$1" out=() g old seen
  for g in $raw; do case "$g" in bb|pb|bp|pp) ;; *) echo "bad guard profile=$g" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$g" ]] && seen=1; done; ((seen)) || out+=("$g"); done
  ((${#out[@]})) || { echo 'guard list must not be empty' >&2; exit 2; }; printf '%s' "${out[*]}"
}
normalize_mate_load_policies(){
  local raw="$1" out=() p old seen
  for p in $raw; do case "$p" in default|cg|cs) ;; *) echo "bad mate-load policy=$p" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$p" ]] && seen=1; done; ((seen)) || out+=("$p"); done
  ((${#out[@]})) || { echo 'mate-load policy list must not be empty' >&2; exit 2; }; printf '%s' "${out[*]}"
}
HYBRID_NS_WIDTH_LIST="$(normalize_widths "$HYBRID_NS_WIDTH_LIST")"
HYBRID_NS_DISTANCE_LIST="$(normalize_distances "$HYBRID_NS_DISTANCE_LIST")"
MATE_WIDTH_LIST="$(normalize_widths "$MATE_WIDTH_LIST")"
MATE_DISTANCE_LIST="$(normalize_distances "$MATE_DISTANCE_LIST")"
MATE_EVICT_LIST="$(normalize_evicts "$MATE_EVICT_LIST")"
STAGEL_GUARD_LIST="$(normalize_guards "$STAGEL_GUARD_LIST")"
STAGEM_POLICY_LIST="$(normalize_mate_load_policies "$STAGEM_POLICY_LIST")"
case "$MATE_EVICT" in default|normal|last) ;; *) echo 'MATE_EVICT must be default,normal,last' >&2; exit 2;; esac
case " $MATE_EVICT_LIST " in *" $MATE_EVICT "*) ;; *) echo 'MATE_EVICT_LIST must include MATE_EVICT' >&2; exit 2;; esac
case " $STAGEL_GUARD_LIST " in *' bb '*) ;; *) echo 'STAGEL_GUARD_LIST must include bb' >&2; exit 2;; esac
case " $STAGEM_POLICY_LIST " in *' default '*) ;; *) echo 'STAGEM_POLICY_LIST must include default' >&2; exit 2;; esac
python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" <<'PY'
import sys
names=('NEXTSELF_MIN_SPEEDUP','HYBRID_MIN_SPEEDUP','HYBRID_NS_MIN_SPEEDUP','STAGEI_MIN_SPEEDUP','STAGEJ_MIN_SPEEDUP','STAGEK_MIN_SPEEDUP','STAGEL_MIN_SPEEDUP','STAGEM_MIN_SPEEDUP')
for name,v in zip(names,map(float,sys.argv[1:])):
    if v < 1.0: raise SystemExit(f'{name} must be >=1')
PY
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v git >/dev/null || { echo 'git required' >&2; exit 2; }
command -v sha256sum >/dev/null || { echo 'sha256sum required' >&2; exit 2; }
HARDWARE_GUARD="$ONEESAN_ROOT/scripts/run/b300x8-require-b300-inventory.sh"
[[ -f "$HARDWARE_GUARD" ]] || { echo "missing B300 hardware guard=$HARDWARE_GUARD" >&2; exit 2; }
GPU_INVENTORY="$(bash "$HARDWARE_GUARD")"
GPU_COUNT="$(printf '%s\n' "$GPU_INVENTORY" | awk 'NF{n++} END{print n+0}')"
GPU_LIST="$(nvidia-smi -L)"

mkdir -p "$(dirname "$LOG")" "$(dirname "$META")" "$(dirname "$SELECTED_ENV")" "$WORK_ROOT"
HEAD_SHA="$(git -C "$ONEESAN_ROOT" rev-parse HEAD)"
HEAD_DIRTY=0
[[ -z "$(git -C "$ONEESAN_ROOT" status --porcelain=v1 --untracked-files=normal)" ]] || HEAD_DIRTY=1
PROFILE_SHA="$(sha256sum "$PROFILE_FILE" | awk '{print $1}')"

{
  printf 'schema=4\n'
  printf 'n=%s\n' "$N"
  printf 'head_sha=%s\n' "$HEAD_SHA"
  printf 'head_dirty=%s\n' "$HEAD_DIRTY"
  printf 'profile_file=%s\n' "$PROFILE_FILE"
  printf 'profile_sha256=%s\n' "$PROFILE_SHA"
  printf 'gpu_count=%s\n' "$GPU_COUNT"
  printf 'gpu_guard=b300x8_exact_model\n'
  printf 'arch=%s\n' "$ARCH"
  printf 'max_window=%s\n' "$MAX_WINDOW"
  printf 'smoke_prime=%s\n' "$SMOKE_PRIME"
  printf 'forced_target_mib=%s\n' "$FORCED_TARGET_MIB"
  printf 'bucket_target_mib=%s\n' "$BUCKET_TARGET_MIB"
  printf 'work_root=%s\n' "$WORK_ROOT"
  printf 'race_prefix=%s\n' "$RACE_PREFIX"
  printf 'race_result=%s\n' "$RACE_RESULT"
  printf 'grand_summary_env=%s\n' "$GRAND_SUMMARY_ENV"
  printf 'select_only=1\n'
  printf 'rebuild_buckets=%s\n' "$REBUILD_BUCKETS"
  printf 'run_nextself_stage=%s\n' "$RUN_NEXTSELF_STAGE"
  printf 'run_hybrid_stage=%s\n' "$RUN_HYBRID_STAGE"
  printf 'run_hybrid_ns_stage=%s\n' "$RUN_HYBRID_NS_STAGE"
  printf 'run_stagei=%s\n' "$RUN_STAGEI"
  printf 'run_stagej=%s\n' "$RUN_STAGEJ"
  printf 'run_stagek=%s\n' "$RUN_STAGEK"
  printf 'run_stagel=%s\n' "$RUN_STAGEL"
  printf 'run_stagem=%s\n' "$RUN_STAGEM"
  printf 'nextself_threads=%s\n' "$NEXTSELF_THREADS"
  printf 'nextself_search_rows=%s\n' "$NEXTSELF_SEARCH_ROWS"
  printf 'nextself_validate_rows=%s\n' "$NEXTSELF_VALIDATE_ROWS"
  printf 'nextself_search_repeats=%s\n' "$NEXTSELF_SEARCH_REPEATS"
  printf 'nextself_validate_repeats=%s\n' "$NEXTSELF_VALIDATE_REPEATS"
  printf 'nextself_min_speedup=%s\n' "$NEXTSELF_MIN_SPEEDUP"
  printf 'hybrid_min_speedup=%s\n' "$HYBRID_MIN_SPEEDUP"
  printf 'hybrid_ns_min_speedup=%s\n' "$HYBRID_NS_MIN_SPEEDUP"
  printf 'hybrid_ns_width_list=%s\n' "$HYBRID_NS_WIDTH_LIST"
  printf 'hybrid_ns_distance_list=%s\n' "$HYBRID_NS_DISTANCE_LIST"
  printf 'hybrid_ns_search_repeats=%s\n' "$HYBRID_NS_SEARCH_REPEATS"
  printf 'hybrid_ns_validate_repeats=%s\n' "$HYBRID_NS_VALIDATE_REPEATS"
  printf 'stagei_min_speedup=%s\n' "$STAGEI_MIN_SPEEDUP"
  printf 'stagej_min_speedup=%s\n' "$STAGEJ_MIN_SPEEDUP"
  printf 'stagej_mate_width_list=%s\n' "$MATE_WIDTH_LIST"
  printf 'stagej_mate_distance_list=%s\n' "$MATE_DISTANCE_LIST"
  printf 'stagej_mate_evict=%s\n' "$MATE_EVICT"
  printf 'stagek_min_speedup=%s\n' "$STAGEK_MIN_SPEEDUP"
  printf 'stagek_mate_evict_list=%s\n' "$MATE_EVICT_LIST"
  printf 'stagel_min_speedup=%s\n' "$STAGEL_MIN_SPEEDUP"
  printf 'stagel_guard_list=%s\n' "$STAGEL_GUARD_LIST"
  printf 'stagem_min_speedup=%s\n' "$STAGEM_MIN_SPEEDUP"
  printf 'stagem_policy_list=%s\n' "$STAGEM_POLICY_LIST"
  printf 'stagei_namespace_contract_preflight=1\n'
  printf 'grand_selector_contract_preflight=1\n'
  printf 'complete_prime_races_expected=1\n'
  printf 'nvcc_version_begin=1\n'
  nvcc --version | sed 's/^/nvcc: /'
  printf 'nvcc_version_end=1\n'
  printf 'gpu_list_begin=1\n'
  printf '%s\n' "$GPU_LIST" | sed 's/^/gpu-list: /'
  printf 'gpu_list_end=1\n'
  printf 'gpu_inventory_begin=1\n'
  printf '%s\n' "$GPU_INVENTORY" | sed 's/^/gpu: /'
  printf 'gpu_inventory_end=1\n'
} >"$META"

if (( HEAD_DIRTY )); then
  echo 'WARNING: repository has uncommitted or untracked changes; provenance records head_dirty=1' >&2
fi

echo '=== B300 grand first-pass: GPU-free preflight ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid8-nextself-transform-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-self-mate-independent-geometry-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-joint-nextgen-hybrid8-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-stagei-namespace-contract-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-grand-selector-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-grand-selector-contract-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-grand-stagek-contract-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-stagel-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-stagem-preflight.sh"

echo "=== B300 grand first-pass: n=27 head=${HEAD_SHA:0:12} GPUs=$GPU_COUNT SELECT_ONLY=1 self_geometry=[$HYBRID_NS_WIDTH_LIST]x[$HYBRID_NS_DISTANCE_LIST] mate_geometry=[$MATE_WIDTH_LIST]x[$MATE_DISTANCE_LIST] mate_evict=[$MATE_EVICT_LIST] guards=[$STAGEL_GUARD_LIST] mate_load=[$STAGEM_POLICY_LIST] ===" >&2
set +e
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" SMOKE_PRIME="$SMOKE_PRIME" \
  FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" WORK_ROOT="$WORK_ROOT" RACE_PREFIX="$RACE_PREFIX" \
  SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS" \
  RUN_NEXTSELF_STAGE="$RUN_NEXTSELF_STAGE" RUN_HYBRID_STAGE="$RUN_HYBRID_STAGE" RUN_HYBRID_NS_STAGE="$RUN_HYBRID_NS_STAGE" \
  RUN_STAGEI="$RUN_STAGEI" RUN_STAGEJ="$RUN_STAGEJ" RUN_STAGEK="$RUN_STAGEK" RUN_STAGEL="$RUN_STAGEL" RUN_STAGEM="$RUN_STAGEM" \
  NEXTSELF_THREADS="$NEXTSELF_THREADS" NEXTSELF_SEARCH_ROWS="$NEXTSELF_SEARCH_ROWS" NEXTSELF_VALIDATE_ROWS="$NEXTSELF_VALIDATE_ROWS" \
  NEXTSELF_SEARCH_REPEATS="$NEXTSELF_SEARCH_REPEATS" NEXTSELF_VALIDATE_REPEATS="$NEXTSELF_VALIDATE_REPEATS" NEXTSELF_MIN_SPEEDUP="$NEXTSELF_MIN_SPEEDUP" \
  HYBRID_MIN_SPEEDUP="$HYBRID_MIN_SPEEDUP" HYBRID_NS_MIN_SPEEDUP="$HYBRID_NS_MIN_SPEEDUP" \
  HYBRID_NS_WIDTH_LIST="$HYBRID_NS_WIDTH_LIST" HYBRID_NS_DISTANCE_LIST="$HYBRID_NS_DISTANCE_LIST" \
  HYBRID_NS_SEARCH_REPEATS="$HYBRID_NS_SEARCH_REPEATS" HYBRID_NS_VALIDATE_REPEATS="$HYBRID_NS_VALIDATE_REPEATS" \
  STAGEI_MIN_SPEEDUP="$STAGEI_MIN_SPEEDUP" STAGEJ_MIN_SPEEDUP="$STAGEJ_MIN_SPEEDUP" STAGEK_MIN_SPEEDUP="$STAGEK_MIN_SPEEDUP" \
  STAGEL_MIN_SPEEDUP="$STAGEL_MIN_SPEEDUP" STAGEM_MIN_SPEEDUP="$STAGEM_MIN_SPEEDUP" \
  MATE_WIDTH_LIST="$MATE_WIDTH_LIST" MATE_DISTANCE_LIST="$MATE_DISTANCE_LIST" MATE_EVICT="$MATE_EVICT" MATE_EVICT_LIST="$MATE_EVICT_LIST" \
  STAGEL_GUARD_LIST="$STAGEL_GUARD_LIST" STAGEM_POLICY_LIST="$STAGEM_POLICY_LIST" \
  PREFIX="$PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh" 27 "$@" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

printf 'exit_code=%s\n' "$rc" >>"$META"
if (( rc != 0 )); then
  echo "b300x8-grand-firstpass FAILED rc=$rc log=$LOG meta=$META" >&2
  exit "$rc"
fi

grep -Fq 'SINGLE PASS SELECTED' "$LOG" || { echo "grand first-pass completed without SINGLE PASS SELECTED marker: $LOG" >&2; exit 4; }
grep -Fq 'SELECT_ONLY=1: selected' "$LOG" || { echo "grand first-pass did not stop at SELECT_ONLY boundary: $LOG" >&2; exit 4; }
[[ -s "$RACE_RESULT" ]] || { echo "grand first-pass race result missing: $RACE_RESULT" >&2; exit 4; }
[[ -s "$GRAND_SUMMARY_ENV" ]] || { echo "grand summary env missing: $GRAND_SUMMARY_ENV" >&2; exit 4; }
# shellcheck disable=SC1090
source "$GRAND_SUMMARY_ENV"
[[ "${B300_GRAND_PREPARED:-0}" == 1 && "${B300_GRAND_STAGEJ_INTEGRATED:-0}" == 1 && \
   "${B300_GRAND_STAGEK_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEL_INTEGRATED:-0}" == 1 && \
   "${B300_GRAND_STAGEM_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1 && \
   "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {
  echo 'grand Stage-I/J/K/L/M single-race provenance markers missing' >&2; exit 4;
}
GRAND_SUMMARY_SHA="$(sha256sum "$GRAND_SUMMARY_ENV" | awk '{print $1}')"

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
if (b['backend'],b['profile'],b['wall_s'],b['residue']) != m.groups(): raise SystemExit('selected_line does not match TSV winner')
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
    (( RUN_THREADS>=32 && RUN_THREADS<=1024 && RUN_THREADS%32==0 )) || { echo "bad selected forced threads=$RUN_THREADS" >&2; exit 4; }
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
if fp.get('schema') != 3 or fp.get('binary_sha256') != h.hexdigest() or fp.get('profile_sha256') != profile_sha: raise SystemExit('checkpoint solver fingerprint mismatch')
r=d.get('residues',{}).get(str(int(prime)))
if not r or int(r.get('residue',-1)) != int(residue): raise SystemExit('checkpoint smoke residue missing/mismatch')
PY

{
  printf 'B300_GRAND_SELECTED_SCHEMA=3\n'
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
  printf 'B300_GRAND_SELECTED_GEOMETRY_WIDTH_LIST=%q\n' "$HYBRID_NS_WIDTH_LIST"
  printf 'B300_GRAND_SELECTED_GEOMETRY_DISTANCE_LIST=%q\n' "$HYBRID_NS_DISTANCE_LIST"
  printf 'B300_GRAND_SELECTED_STAGEI_ENABLED=%q\n' "$RUN_STAGEI"
  printf 'B300_GRAND_SELECTED_STAGEI_MIN_SPEEDUP=%q\n' "$STAGEI_MIN_SPEEDUP"
  printf 'B300_GRAND_SELECTED_STAGEI_ACCEPTED=%q\n' "${B300_GRAND_STAGEI_OK:-0}"
  printf 'B300_GRAND_SELECTED_STAGEI_HINT=%q\n' "${B300_GRAND_STAGEI_HINT:-default}"
  printf 'B300_GRAND_SELECTED_STAGEJ_ENABLED=%q\n' "$RUN_STAGEJ"
  printf 'B300_GRAND_SELECTED_STAGEJ_MIN_SPEEDUP=%q\n' "$STAGEJ_MIN_SPEEDUP"
  printf 'B300_GRAND_SELECTED_STAGEJ_ACCEPTED=%q\n' "${B300_GRAND_STAGEJ_OK:-0}"
  printf 'B300_GRAND_SELECTED_STAGEJ_SELF_EVICT=%q\n' "${B300_GRAND_STAGEJ_SELF_EVICT:-default}"
  printf 'B300_GRAND_SELECTED_STAGEJ_SELF_WIDTH=%q\n' "${B300_GRAND_STAGEJ_SELF_WIDTH:-0}"
  printf 'B300_GRAND_SELECTED_STAGEJ_SELF_DISTANCE=%q\n' "${B300_GRAND_STAGEJ_SELF_DISTANCE:-0}"
  printf 'B300_GRAND_SELECTED_STAGEJ_MATE_WIDTH=%q\n' "${B300_GRAND_STAGEJ_MATE_WIDTH:-0}"
  printf 'B300_GRAND_SELECTED_STAGEJ_MATE_DISTANCE=%q\n' "${B300_GRAND_STAGEJ_MATE_DISTANCE:-0}"
  printf 'B300_GRAND_SELECTED_STAGEJ_MATE_EVICT=%q\n' "${B300_GRAND_STAGEJ_MATE_EVICT:-$MATE_EVICT}"
  printf 'B300_GRAND_SELECTED_STAGEJ_STAGED_SPEEDUP=%q\n' "${B300_GRAND_STAGEJ_STAGED_SPEEDUP:-1.0}"
  printf 'B300_GRAND_SELECTED_STAGEJ_SEARCH_MATE_WIDTHS=%q\n' "$MATE_WIDTH_LIST"
  printf 'B300_GRAND_SELECTED_STAGEJ_SEARCH_MATE_DISTANCES=%q\n' "$MATE_DISTANCE_LIST"
  printf 'B300_GRAND_SELECTED_STAGEK_ENABLED=%q\n' "$RUN_STAGEK"
  printf 'B300_GRAND_SELECTED_STAGEK_MIN_SPEEDUP=%q\n' "$STAGEK_MIN_SPEEDUP"
  printf 'B300_GRAND_SELECTED_STAGEK_ACCEPTED=%q\n' "${B300_GRAND_STAGEK_OK:-0}"
  printf 'B300_GRAND_SELECTED_STAGEK_BASE_MATE_EVICT=%q\n' "${B300_GRAND_STAGEK_BASE_MATE_EVICT:-$MATE_EVICT}"
  printf 'B300_GRAND_SELECTED_STAGEK_MATE_EVICT=%q\n' "${B300_GRAND_STAGEK_MATE_EVICT:-$MATE_EVICT}"
  printf 'B300_GRAND_SELECTED_STAGEK_STAGED_SPEEDUP=%q\n' "${B300_GRAND_STAGEK_STAGED_SPEEDUP:-1.0}"
  printf 'B300_GRAND_SELECTED_STAGEK_SEARCH_EVICTS=%q\n' "$MATE_EVICT_LIST"
  printf 'B300_GRAND_SELECTED_STAGEL_ENABLED=%q\n' "$RUN_STAGEL"
  printf 'B300_GRAND_SELECTED_STAGEL_MIN_SPEEDUP=%q\n' "$STAGEL_MIN_SPEEDUP"
  printf 'B300_GRAND_SELECTED_STAGEL_ACCEPTED=%q\n' "${B300_GRAND_STAGEL_OK:-0}"
  printf 'B300_GRAND_SELECTED_STAGEL_PROFILE=%q\n' "${B300_GRAND_STAGEL_PROFILE:-bb}"
  printf 'B300_GRAND_SELECTED_STAGEL_SELF_GUARD=%q\n' "${B300_GRAND_STAGEL_SELF_GUARD:-branch}"
  printf 'B300_GRAND_SELECTED_STAGEL_MATE_GUARD=%q\n' "${B300_GRAND_STAGEL_MATE_GUARD:-branch}"
  printf 'B300_GRAND_SELECTED_STAGEL_STAGED_SPEEDUP=%q\n' "${B300_GRAND_STAGEL_STAGED_SPEEDUP:-1.0}"
  printf 'B300_GRAND_SELECTED_STAGEL_SEARCH_PROFILES=%q\n' "$STAGEL_GUARD_LIST"
  printf 'B300_GRAND_SELECTED_STAGEM_ENABLED=%q\n' "$RUN_STAGEM"
  printf 'B300_GRAND_SELECTED_STAGEM_MIN_SPEEDUP=%q\n' "$STAGEM_MIN_SPEEDUP"
  printf 'B300_GRAND_SELECTED_STAGEM_ACCEPTED=%q\n' "${B300_GRAND_STAGEM_OK:-0}"
  printf 'B300_GRAND_SELECTED_STAGEM_POLICY=%q\n' "${B300_GRAND_STAGEM_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGEM_STAGED_SPEEDUP=%q\n' "${B300_GRAND_STAGEM_STAGED_SPEEDUP:-1.0}"
  printf 'B300_GRAND_SELECTED_STAGEM_SEARCH_POLICIES=%q\n' "$STAGEM_POLICY_LIST"
  printf 'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1\n'
  printf 'B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=%q\n' "$GRAND_SUMMARY_ENV"
  printf 'B300_GRAND_SELECTED_GRAND_SUMMARY_SHA256=%q\n' "$GRAND_SUMMARY_SHA"
  printf 'B300_GRAND_SELECTED_WORK_DIR=%q\n' "$BEST_WORK"
  printf 'B300_GRAND_SELECTED_CHECKPOINT=%q\n' "$CHECKPOINT"
  printf 'B300_GRAND_SELECTED_RACE_PREFIX=%q\n' "$RACE_PREFIX"
  printf 'B300_GRAND_SELECTED_RACE_RESULT=%q\n' "$RACE_RESULT"
  printf 'B300_GRAND_SELECTED_RACE_RESULT_SHA256=%q\n' "$RACE_SHA"
  printf 'B300_GRAND_SELECTED_FIRSTPASS_META=%q\n' "$META"
  printf 'B300_GRAND_SELECTED_STAGEH_ENABLED=%q\n' "$RUN_STAGEJ"
  printf 'B300_GRAND_SELECTED_STAGEH_MIN_SPEEDUP=%q\n' "$STAGEJ_MIN_SPEEDUP"
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
  printf 'grand_summary_env=%s\n' "$GRAND_SUMMARY_ENV"
  printf 'grand_summary_sha256=%s\n' "$GRAND_SUMMARY_SHA"
  printf 'geometry_width_list=%s\n' "$HYBRID_NS_WIDTH_LIST"
  printf 'geometry_distance_list=%s\n' "$HYBRID_NS_DISTANCE_LIST"
  printf 'stagei_enabled=%s\n' "$RUN_STAGEI"
  printf 'stagei_min_speedup=%s\n' "$STAGEI_MIN_SPEEDUP"
  printf 'stagei_accepted=%s\n' "${B300_GRAND_STAGEI_OK:-0}"
  printf 'stagei_hint=%s\n' "${B300_GRAND_STAGEI_HINT:-default}"
  printf 'stagej_enabled=%s\n' "$RUN_STAGEJ"
  printf 'stagej_min_speedup=%s\n' "$STAGEJ_MIN_SPEEDUP"
  printf 'stagej_accepted=%s\n' "${B300_GRAND_STAGEJ_OK:-0}"
  printf 'stagej_self_geometry=w%sd%s\n' "${B300_GRAND_STAGEJ_SELF_WIDTH:-0}" "${B300_GRAND_STAGEJ_SELF_DISTANCE:-0}"
  printf 'stagej_mate_geometry=w%sd%s\n' "${B300_GRAND_STAGEJ_MATE_WIDTH:-0}" "${B300_GRAND_STAGEJ_MATE_DISTANCE:-0}"
  printf 'stagej_self_evict=%s\n' "${B300_GRAND_STAGEJ_SELF_EVICT:-default}"
  printf 'stagej_mate_evict=%s\n' "${B300_GRAND_STAGEJ_MATE_EVICT:-$MATE_EVICT}"
  printf 'stagej_staged_speedup=%s\n' "${B300_GRAND_STAGEJ_STAGED_SPEEDUP:-1.0}"
  printf 'stagek_enabled=%s\n' "$RUN_STAGEK"
  printf 'stagek_min_speedup=%s\n' "$STAGEK_MIN_SPEEDUP"
  printf 'stagek_accepted=%s\n' "${B300_GRAND_STAGEK_OK:-0}"
  printf 'stagek_base_mate_evict=%s\n' "${B300_GRAND_STAGEK_BASE_MATE_EVICT:-$MATE_EVICT}"
  printf 'stagek_mate_evict=%s\n' "${B300_GRAND_STAGEK_MATE_EVICT:-$MATE_EVICT}"
  printf 'stagek_staged_speedup=%s\n' "${B300_GRAND_STAGEK_STAGED_SPEEDUP:-1.0}"
  printf 'stagel_enabled=%s\n' "$RUN_STAGEL"
  printf 'stagel_min_speedup=%s\n' "$STAGEL_MIN_SPEEDUP"
  printf 'stagel_accepted=%s\n' "${B300_GRAND_STAGEL_OK:-0}"
  printf 'stagel_profile=%s\n' "${B300_GRAND_STAGEL_PROFILE:-bb}"
  printf 'stagel_guards=%s/%s\n' "${B300_GRAND_STAGEL_SELF_GUARD:-branch}" "${B300_GRAND_STAGEL_MATE_GUARD:-branch}"
  printf 'stagel_staged_speedup=%s\n' "${B300_GRAND_STAGEL_STAGED_SPEEDUP:-1.0}"
  printf 'stagem_enabled=%s\n' "$RUN_STAGEM"
  printf 'stagem_min_speedup=%s\n' "$STAGEM_MIN_SPEEDUP"
  printf 'stagem_accepted=%s\n' "${B300_GRAND_STAGEM_OK:-0}"
  printf 'stagem_policy=%s\n' "${B300_GRAND_STAGEM_POLICY:-default}"
  printf 'stagem_staged_speedup=%s\n' "${B300_GRAND_STAGEM_STAGED_SPEEDUP:-1.0}"
  printf 'complete_prime_races=1\n'
  printf 'promotion_contract=3\n'
} >>"$META"

echo "b300x8-grand-firstpass OK log=$LOG meta=$META selected_env=$SELECTED_ENV $SELECTED_LINE" >&2