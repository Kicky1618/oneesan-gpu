#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CANDIDATE_MODE="${CANDIDATE_MODE:-auto}"
PROXY_EXACT_SUMMARY="${PROXY_EXACT_SUMMARY:-${1:-}}"
PROXY_W28_SUMMARY="${PROXY_W28_SUMMARY:-${2:-}}"
NGPU="${NGPU:-2}"
EXACT_BLOCKS="${EXACT_BLOCKS:-256}"
W28_BLOCKS="${W28_BLOCKS:-256}"
W28_THREADS="${W28_THREADS:-256}"
W28_ITERS="${W28_ITERS:-16}"
W28_OWNERS="${W28_OWNERS:-0}"
W28_DIRECTIONS="${W28_DIRECTIONS:-0 1}"
REPEATS="${REPEATS:-7}"
EXACT_WARMUP="${EXACT_WARMUP:-1}"
W28_WARMUP="${W28_WARMUP:-1}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
MIN_EXACT_SPEEDUP="${MIN_EXACT_SPEEDUP:-1.002}"
MIN_W28_SPEEDUP="${MIN_W28_SPEEDUP:-1.002}"
REQUIRE_ALL_PAIRS="${REQUIRE_ALL_PAIRS:-1}"
REQUIRE_CLEAN_TREE="${REQUIRE_CLEAN_TREE:-1}"
if (( NGPU < 2 || NGPU > 8 || EXACT_BLOCKS < 1 || W28_BLOCKS < 1 ||
      W28_THREADS < 1 || W28_THREADS > 1024 || W28_ITERS < 1 || REPEATS < 1 ||
      EXACT_WARMUP < 0 || W28_WARMUP < 0 )); then
  echo "invalid physical codec decision dimensions" >&2; exit 2
fi
[[ "$REQUIRE_ALL_PAIRS" == 0 || "$REQUIRE_ALL_PAIRS" == 1 ]] || { echo "REQUIRE_ALL_PAIRS must be 0 or 1" >&2; exit 2; }
[[ "$REQUIRE_CLEAN_TREE" == 0 || "$REQUIRE_CLEAN_TREE" == 1 ]] || { echo "REQUIRE_CLEAN_TREE must be 0 or 1" >&2; exit 2; }
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs for exact phase, visible=$visible" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_codec_table_physical_decision}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.txt}"; MANIFEST="${MANIFEST:-${PREFIX}_manifest.txt}"
mkdir -p "$LOGDIR" "$(dirname "$SUMMARY")" "$(dirname "$MANIFEST")"
: >"$SUMMARY"

head_sha=unknown; dirty=unknown
if command -v git >/dev/null && git -C "$ONEESAN_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  head_sha="$(git -C "$ONEESAN_ROOT" rev-parse HEAD)"
  if git -C "$ONEESAN_ROOT" diff --quiet -- . && git -C "$ONEESAN_ROOT" diff --cached --quiet -- .; then dirty=0; else dirty=1; fi
  if [[ "$REQUIRE_CLEAN_TREE" == 1 && "$dirty" != 0 ]]; then
    echo "physical codec decision requires a clean tracked tree; set REQUIRE_CLEAN_TREE=0 to override" >&2
    exit 2
  fi
fi
{
  echo "head_sha=$head_sha"
  echo "tracked_dirty=$dirty"
  echo "candidate_mode_requested=$CANDIDATE_MODE"
  echo "ngpu=$NGPU"
  echo "repeats=$REPEATS"
  echo "exact_blocks=$EXACT_BLOCKS"
  echo "w28_blocks=$W28_BLOCKS"
  echo "w28_threads=$W28_THREADS"
  echo "w28_iters=$W28_ITERS"
  echo "w28_owners=$W28_OWNERS"
  echo "w28_directions=$W28_DIRECTIONS"
  echo "min_exact_speedup=$MIN_EXACT_SPEEDUP"
  echo "min_w28_speedup=$MIN_W28_SPEEDUP"
  echo "require_all_pairs=$REQUIRE_ALL_PAIRS"
  echo "arch=$ARCH"
  nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  nvcc --version | tail -n1 | sed 's/^/nvcc=/'
} >"$MANIFEST"

for proof in \
  gridfp-runtime-ab-env-proof.sh \
  gridfp-codec-table-proxy-coverage-proof.sh \
  gridfp-codec-table-physical-replacement-proof.sh \
  gridfp-codec-table-physical-consensus-gate-proof.sh; do
  bash "$ONEESAN_ROOT/scripts/bench/$proof" >"$LOGDIR/${proof%.sh}.out" 2>"$LOGDIR/${proof%.sh}.err"
done

if [[ "$CANDIDATE_MODE" == auto ]]; then
  [[ -f "$PROXY_EXACT_SUMMARY" && -f "$PROXY_W28_SUMMARY" ]] || {
    echo "CANDIDATE_MODE=auto requires proxy exact and W28 summary TSVs" >&2; exit 2; }
  PROXY_GATE_OUT="$LOGDIR/proxy-consensus.out"
  MIN_EXACT_SPEEDUP="$MIN_EXACT_SPEEDUP" MIN_W28_SPEEDUP="$MIN_W28_SPEEDUP" \
    MIN_EXACT_PAIRS="$REPEATS" MIN_W28_PAIRS="$REPEATS" REQUIRE_ALL_PAIRS="$REQUIRE_ALL_PAIRS" \
    bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-consensus-gate.sh" \
      "$PROXY_EXACT_SUMMARY" "$PROXY_W28_SUMMARY" >"$PROXY_GATE_OUT"
  grep -Fq 'codec_consensus_physical_replacement_ready=1' "$PROXY_GATE_OUT" || {
    cat "$PROXY_GATE_OUT" >"$SUMMARY"
    echo 'physical_decision_proxy_ready=0' >>"$SUMMARY"
    echo 'physical_decision_next_step=KEEP_PROXY_ONLY' >>"$SUMMARY"
    cat "$SUMMARY"
    echo "gridfp-codec-table-physical-decision-suite OK proxy_rejected=1 summary=$SUMMARY" >&2
    exit 0
  }
  CANDIDATE_MODE="$(sed -nE 's/^codec_consensus_candidate_mode=([0-9]+)$/\1/p' "$PROXY_GATE_OUT" | tail -n1)"
fi
[[ "$CANDIDATE_MODE" =~ ^[1-5]$ ]] || { echo "CANDIDATE_MODE must be auto or 1..5" >&2; exit 2; }
echo "candidate_mode_resolved=$CANDIDATE_MODE" >>"$MANIFEST"

# Stage 1: one physical GPU, W28 production rank path. Reject early before the
# multi-GPU exact run if physical placement itself loses.
W28_PREFIX="${PREFIX}_w28"
W28_OUT="$LOGDIR/w28.out"; W28_ERR="$LOGDIR/w28.err"
env CANDIDATE_MODE="$CANDIDATE_MODE" PREFIX="$W28_PREFIX" \
  LOGICAL_NGPU=8 OWNERS="$W28_OWNERS" DIRECTIONS="$W28_DIRECTIONS" \
  BLOCKS="$W28_BLOCKS" THREADS="$W28_THREADS" ITERS="$W28_ITERS" \
  REPEATS="$REPEATS" WARMUP="$W28_WARMUP" ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-w28-rank-physical-ab.sh" >"$W28_OUT" 2>"$W28_ERR"
W28_SUMMARY="$(sed -nE 's/^summary=(.*)$/\1/p' "$W28_OUT" | tail -n1)"
[[ -n "$W28_SUMMARY" && -f "$W28_SUMMARY" ]] || { echo "missing physical W28 summary" >&2; exit 3; }

if ! python3 - "$W28_SUMMARY" "$MIN_W28_SPEEDUP" "$REPEATS" "$REQUIRE_ALL_PAIRS" <<'PY'
import csv,sys
path,mins,reps,all_s=sys.argv[1:]
rows=list(csv.DictReader(open(path),delimiter='\t')); cand=[r for r in rows if r['mode']!='0']
if len(cand)!=1: raise SystemExit(2)
r=cand[0]; ok=int(r['repeats'])>=int(reps) and float(r['paired_speedup_median'])>=float(mins)
if int(all_s): ok=ok and float(r['paired_speedup_min'])>1.0
raise SystemExit(0 if ok else 1)
PY
then
  {
    echo '[w28_physical]'; cat "$W28_OUT"; echo
    echo 'physical_decision_w28_ready=0'
    echo 'physical_decision_exact_skipped=1'
    echo 'physical_decision_production_promotion_ready=0'
    echo 'physical_decision_next_step=KEEP_EXPERIMENTAL'
    echo "manifest=$MANIFEST"
  } >"$SUMMARY"
  cat "$SUMMARY"
  echo "gridfp-codec-table-physical-decision-suite OK early_reject=1 candidate_mode=$CANDIDATE_MODE summary=$SUMMARY" >&2
  exit 0
fi

# Stage 2: exact two-row multi-GPU run only for the W28 survivor.
EXACT_PREFIX="${PREFIX}_exact"
EXACT_OUT="$LOGDIR/exact.out"; EXACT_ERR="$LOGDIR/exact.err"
env CANDIDATE_MODE="$CANDIDATE_MODE" PREFIX="$EXACT_PREFIX" W=10 NGPU="$NGPU" BLOCKS="$EXACT_BLOCKS" \
  REPEATS="$REPEATS" WARMUP="$EXACT_WARMUP" ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  bash "$ONEESAN_ROOT/scripts/bench/gridfp-reduced-runtime-codec-tables-physical-ab.sh" >"$EXACT_OUT" 2>"$EXACT_ERR"
EXACT_SUMMARY="$(sed -nE 's/^summary=(.*)$/\1/p' "$EXACT_OUT" | tail -n1)"
[[ -n "$EXACT_SUMMARY" && -f "$EXACT_SUMMARY" ]] || { echo "missing physical exact summary" >&2; exit 4; }

GATE_OUT="$LOGDIR/physical-consensus.out"
MIN_EXACT_SPEEDUP="$MIN_EXACT_SPEEDUP" MIN_W28_SPEEDUP="$MIN_W28_SPEEDUP" \
  MIN_EXACT_PAIRS="$REPEATS" MIN_W28_PAIRS="$REPEATS" REQUIRE_ALL_PAIRS="$REQUIRE_ALL_PAIRS" \
  bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-physical-consensus-gate.sh" \
    "$EXACT_SUMMARY" "$W28_SUMMARY" >"$GATE_OUT"

{
  echo '[w28_physical]'; grep -E '(w28_physical_|summary=)' "$W28_OUT" || true; echo
  echo '[exact_physical]'; grep -E '(physical_codec_|summary=)' "$EXACT_OUT" || true; echo
  echo '[physical_consensus]'; cat "$GATE_OUT"; echo
  echo "manifest=$MANIFEST"
} >"$SUMMARY"
cat "$SUMMARY"
echo "gridfp-codec-table-physical-decision-suite OK candidate_mode=$CANDIDATE_MODE exact_summary=$EXACT_SUMMARY w28_summary=$W28_SUMMARY summary=$SUMMARY" >&2
