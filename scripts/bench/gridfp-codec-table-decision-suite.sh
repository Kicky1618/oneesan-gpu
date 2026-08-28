#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

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
REQUIRE_CLEAN_TREE="${REQUIRE_CLEAN_TREE:-1}"
if (( NGPU < 2 || NGPU > 8 || EXACT_BLOCKS < 1 || W28_BLOCKS < 1 ||
      W28_THREADS < 1 || W28_THREADS > 1024 || W28_ITERS < 1 || REPEATS < 1 ||
      EXACT_WARMUP < 0 || W28_WARMUP < 0 )); then
  echo "invalid codec table decision suite dimensions" >&2; exit 2
fi
[[ "$REQUIRE_CLEAN_TREE" == 0 || "$REQUIRE_CLEAN_TREE" == 1 ]] || { echo "REQUIRE_CLEAN_TREE must be 0 or 1" >&2; exit 2; }
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null || ! command -v git >/dev/null; then
  echo "nvcc, nvidia-smi, and git are required" >&2; exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU GPUs for exact phase, visible=$visible" >&2; exit 2; }
REPO_HEAD="$(git -C "$ONEESAN_ROOT" rev-parse HEAD 2>/dev/null || true)"
[[ -n "$REPO_HEAD" ]] || { echo "ONEESAN_ROOT is not a git checkout" >&2; exit 2; }
if [[ "$REQUIRE_CLEAN_TREE" == 1 ]]; then
  git -C "$ONEESAN_ROOT" diff --quiet --ignore-submodules -- || { echo "tracked worktree changes present; commit/stash or set REQUIRE_CLEAN_TREE=0" >&2; exit 2; }
  git -C "$ONEESAN_ROOT" diff --cached --quiet --ignore-submodules -- || { echo "staged changes present; commit/stash or set REQUIRE_CLEAN_TREE=0" >&2; exit 2; }
fi

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_codec_table_decision}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.txt}"
MANIFEST="${MANIFEST:-${PREFIX}_manifest.txt}"
mkdir -p "$LOGDIR" "$(dirname "$SUMMARY")" "$(dirname "$MANIFEST")"
: >"$SUMMARY"
{
  echo "repo_head=$REPO_HEAD"
  echo "tracked_tree_clean=$(git -C "$ONEESAN_ROOT" diff --quiet --ignore-submodules -- && git -C "$ONEESAN_ROOT" diff --cached --quiet --ignore-submodules -- && echo 1 || echo 0)"
  echo "arch=$ARCH"
  echo "logical_exact_ngpu=$NGPU"
  echo "exact_blocks=$EXACT_BLOCKS"
  echo "w28_blocks=$W28_BLOCKS"
  echo "w28_threads=$W28_THREADS"
  echo "w28_iters=$W28_ITERS"
  echo "w28_owners=$W28_OWNERS"
  echo "w28_directions=$W28_DIRECTIONS"
  echo "repeats=$REPEATS"
  echo "min_exact_speedup=$MIN_EXACT_SPEEDUP"
  echo "min_w28_speedup=$MIN_W28_SPEEDUP"
  echo '--- gpu ---'
  nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id --format=csv,noheader
  echo '--- nvcc ---'
  nvcc --version
} >"$MANIFEST"

# Cheap CPU-only invariants first. Fail before compiling six CUDA variants if
# the baseline or proxy coverage drifted.
for proof in \
  gridfp-runtime-ab-env-proof.sh \
  gridfp-codec-table-proxy-coverage-proof.sh \
  gridfp-choose-sym-u32-table-proof.sh \
  gridfp-primitive-sym-u32-table-proof.sh \
  gridfp-codec-table-consensus-gate-proof.sh; do
  bash "$ONEESAN_ROOT/scripts/bench/$proof" \
    >"$LOGDIR/${proof%.sh}.out" 2>"$LOGDIR/${proof%.sh}.err"
done

W28_PREFIX="${PREFIX}_w28_rank"
W28_OUT="$LOGDIR/w28-rank.out"; W28_ERR="$LOGDIR/w28-rank.err"
echo '=== codec decision: W28 rank ===' >&2
env PREFIX="$W28_PREFIX" BLOCKS="$W28_BLOCKS" THREADS="$W28_THREADS" ITERS="$W28_ITERS" \
  OWNERS="$W28_OWNERS" DIRECTIONS="$W28_DIRECTIONS" REPEATS="$REPEATS" WARMUP="$W28_WARMUP" \
  ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-w28-rank-microprobe.sh" >"$W28_OUT" 2>"$W28_ERR"
W28_SUMMARY="$(sed -nE 's/^summary=(.*)$/\1/p' "$W28_OUT" | tail -n1)"
[[ -n "$W28_SUMMARY" && -f "$W28_SUMMARY" ]] || { echo "missing W28 rank summary" >&2; exit 3; }

EXACT_PREFIX="${PREFIX}_exact"
EXACT_OUT="$LOGDIR/exact.out"; EXACT_ERR="$LOGDIR/exact.err"
echo '=== codec decision: W10 exact ===' >&2
env PREFIX="$EXACT_PREFIX" W=10 NGPU="$NGPU" BLOCKS="$EXACT_BLOCKS" \
  REPEATS="$REPEATS" WARMUP="$EXACT_WARMUP" ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  bash "$ONEESAN_ROOT/scripts/bench/gridfp-reduced-runtime-codec-tables-sym-u32-ab.sh" >"$EXACT_OUT" 2>"$EXACT_ERR"
EXACT_SUMMARY="$(sed -nE 's/^summary=(.*)$/\1/p' "$EXACT_OUT" | tail -n1)"
[[ -n "$EXACT_SUMMARY" && -f "$EXACT_SUMMARY" ]] || { echo "missing exact summary" >&2; exit 4; }

GATE_OUT="$LOGDIR/consensus-gate.out"; GATE_ERR="$LOGDIR/consensus-gate.err"
MIN_EXACT_SPEEDUP="$MIN_EXACT_SPEEDUP" MIN_W28_SPEEDUP="$MIN_W28_SPEEDUP" \
  MIN_EXACT_PAIRS="$REPEATS" MIN_W28_PAIRS="$REPEATS" REQUIRE_ALL_PAIRS=1 \
  bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-consensus-gate.sh" \
    "$EXACT_SUMMARY" "$W28_SUMMARY" >"$GATE_OUT" 2>"$GATE_ERR"

{
  echo '[manifest]'
  cat "$MANIFEST"
  echo
  echo '[w28_rank]'
  grep -E '(paired_speedup|winner_|checksum_exact|summary=)' "$W28_OUT" || true
  echo
  echo '[exact]'
  grep -E '(paired_speedup|winner_|candidate_bytes|exact=|summary=)' "$EXACT_OUT" || true
  echo
  echo '[consensus]'
  cat "$GATE_OUT"
  cat "$GATE_ERR"
} >"$SUMMARY"

cat "$SUMMARY"
echo "gridfp-codec-table-decision-suite OK repo_head=$REPO_HEAD exact_summary=$EXACT_SUMMARY w28_summary=$W28_SUMMARY manifest=$MANIFEST summary=$SUMMARY" >&2
