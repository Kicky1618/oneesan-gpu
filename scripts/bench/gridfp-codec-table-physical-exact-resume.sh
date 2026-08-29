#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W28_SUMMARY="${W28_SUMMARY:-${1:-}}"
W28_MANIFEST="${W28_MANIFEST:-${2:-}}"
NGPU="${NGPU:-2}"
BLOCKS="${BLOCKS:-256}"
REPEATS="${REPEATS:-7}"
WARMUP="${WARMUP:-1}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
MIN_EXACT_SPEEDUP="${MIN_EXACT_SPEEDUP:-1.002}"
MIN_W28_SPEEDUP="${MIN_W28_SPEEDUP:-1.002}"
REQUIRE_ALL_PAIRS="${REQUIRE_ALL_PAIRS:-1}"
REQUIRE_SAME_HEAD="${REQUIRE_SAME_HEAD:-1}"
[[ -f "$W28_SUMMARY" && -f "$W28_MANIFEST" ]] || {
  echo "usage: $0 <physical-w28-summary.tsv> <physical-decision-manifest.txt>" >&2; exit 2; }
if (( NGPU < 2 || NGPU > 8 || BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then echo "invalid physical exact resume dimensions" >&2; exit 2; fi
[[ "$REQUIRE_ALL_PAIRS" == 0 || "$REQUIRE_ALL_PAIRS" == 1 ]] || { echo "REQUIRE_ALL_PAIRS must be 0 or 1" >&2; exit 2; }
[[ "$REQUIRE_SAME_HEAD" == 0 || "$REQUIRE_SAME_HEAD" == 1 ]] || { echo "REQUIRE_SAME_HEAD must be 0 or 1" >&2; exit 2; }
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

prior_head="$(sed -nE 's/^head_sha=(.*)$/\1/p' "$W28_MANIFEST" | tail -n1)"
prior_mode="$(sed -nE 's/^candidate_mode_resolved=([1-5])$/\1/p' "$W28_MANIFEST" | tail -n1)"
[[ -n "$prior_head" && -n "$prior_mode" ]] || { echo "W28 manifest is missing head_sha or candidate_mode_resolved" >&2; exit 3; }
current_head=unknown
if command -v git >/dev/null && git -C "$ONEESAN_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then current_head="$(git -C "$ONEESAN_ROOT" rev-parse HEAD)"; fi
if [[ "$REQUIRE_SAME_HEAD" == 1 && "$current_head" != "$prior_head" ]]; then
  echo "W28 result HEAD mismatch: prior=$prior_head current=$current_head" >&2; exit 4
fi

# Verify the stored W28 result still meets the requested threshold and matches
# the candidate encoded by the manifest before spending multi-GPU time.
python3 - "$W28_SUMMARY" "$prior_mode" "$MIN_W28_SPEEDUP" "$REPEATS" "$REQUIRE_ALL_PAIRS" <<'PY'
import csv,sys
path,mode,mins,reps,all_s=sys.argv[1:]
rows=list(csv.DictReader(open(path),delimiter='\t')); cand=[r for r in rows if r['mode']!='0']
if len(cand)!=1: raise SystemExit('W28 summary must contain one candidate')
r=cand[0]
if r['mode']!=mode: raise SystemExit(f'candidate mode mismatch summary={r["mode"]} manifest={mode}')
ok=int(r['repeats'])>=int(reps) and float(r['paired_speedup_median'])>=float(mins)
if int(all_s): ok=ok and float(r['paired_speedup_min'])>1.0
if not ok: raise SystemExit('stored W28 result no longer satisfies resume threshold')
PY

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_codec_table_physical_resume_m${prior_mode}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.txt}"; MANIFEST="${MANIFEST:-${PREFIX}_manifest.txt}"
mkdir -p "$LOGDIR" "$(dirname "$SUMMARY")" "$(dirname "$MANIFEST")"
{
  echo "current_head=$current_head"
  echo "source_w28_head=$prior_head"
  echo "candidate_mode=$prior_mode"
  echo "source_w28_summary=$W28_SUMMARY"
  echo "source_w28_manifest=$W28_MANIFEST"
  echo "ngpu=$NGPU"
  echo "repeats=$REPEATS"
  echo "min_exact_speedup=$MIN_EXACT_SPEEDUP"
  echo "min_w28_speedup=$MIN_W28_SPEEDUP"
  echo "require_all_pairs=$REQUIRE_ALL_PAIRS"
  nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  nvcc --version | tail -n1 | sed 's/^/nvcc=/'
} >"$MANIFEST"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-physical-replacement-proof.sh" >"$LOGDIR/physical-proof.out" 2>"$LOGDIR/physical-proof.err"
env ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  PREFIX="$ONEESAN_BUILD_DIR/gridfp_codec_table_physical_resume_compile" LOGDIR="$LOGDIR/compile-matrix" \
  bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-physical-compile-matrix.sh" >"$LOGDIR/compile-matrix.out" 2>"$LOGDIR/compile-matrix.err"

EXACT_PREFIX="${PREFIX}_exact"
EXACT_OUT="$LOGDIR/exact.out"; EXACT_ERR="$LOGDIR/exact.err"
env CANDIDATE_MODE="$prior_mode" PREFIX="$EXACT_PREFIX" W=10 NGPU="$NGPU" BLOCKS="$BLOCKS" \
  REPEATS="$REPEATS" WARMUP="$WARMUP" ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  bash "$ONEESAN_ROOT/scripts/bench/gridfp-reduced-runtime-codec-tables-physical-ab.sh" >"$EXACT_OUT" 2>"$EXACT_ERR"
EXACT_SUMMARY="$(sed -nE 's/^summary=(.*)$/\1/p' "$EXACT_OUT" | tail -n1)"
[[ -n "$EXACT_SUMMARY" && -f "$EXACT_SUMMARY" ]] || { echo "missing resumed exact summary" >&2; exit 5; }

GATE_OUT="$LOGDIR/physical-consensus.out"
MIN_EXACT_SPEEDUP="$MIN_EXACT_SPEEDUP" MIN_W28_SPEEDUP="$MIN_W28_SPEEDUP" \
  MIN_EXACT_PAIRS="$REPEATS" MIN_W28_PAIRS="$REPEATS" REQUIRE_ALL_PAIRS="$REQUIRE_ALL_PAIRS" \
  bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-physical-consensus-gate.sh" \
    "$EXACT_SUMMARY" "$W28_SUMMARY" >"$GATE_OUT"
{
  echo '[stored_w28_physical]'
  python3 - "$W28_SUMMARY" <<'PY'
import csv,sys
for r in csv.DictReader(open(sys.argv[1]),delimiter='\t'):
    if r['mode']!='0':
        print('mode='+r['mode']); print('name='+r['name']); print('paired_speedup_median='+r['paired_speedup_median']+'x'); print('paired_speedup_min='+r['paired_speedup_min']+'x')
PY
  echo
  echo '[exact_physical]'; grep -E '(physical_codec_|summary=)' "$EXACT_OUT" || true; echo
  echo '[physical_consensus]'; cat "$GATE_OUT"; echo
  echo "manifest=$MANIFEST"
} >"$SUMMARY"
cat "$SUMMARY"
echo "gridfp-codec-table-physical-exact-resume OK candidate_mode=$prior_mode exact_summary=$EXACT_SUMMARY w28_summary=$W28_SUMMARY summary=$SUMMARY" >&2
