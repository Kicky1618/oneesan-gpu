#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
MOD="${MOD:-4294967291}"
EXPECT="${EXPECT:-998035516}"
NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
RESULT="${RESULT:-$ONEESAN_ROOT/work/b300_bucket_depth8_graph_highctx_ab_n${N}.tsv}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300_bucket_depth8_graph_highctx_ab_n${N}_logs}"

if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then
  echo "PM_ACCUM must be 0 or 1" >&2
  exit 2
fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then
  echo "TERNARY_KEY4 must be 0 or 1" >&2
  exit 2
fi
if (( NGPU != 8 )); then
  echo "depth8 graph HIGH-context A/B currently requires NGPU=8" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null; then
  echo "nvcc not found" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found" >&2
  exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < NGPU )); then
  echo "requested $NGPU GPUs, but only $visible are visible" >&2
  exit 2
fi

mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

field() {
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

run_one() {
  local mode="$1" high_ctx="$2"
  local tag="${high_ctx}_${mode}"
  local suffix="_${high_ctx}_${mode}"
  if [[ "$PM_ACCUM" == 1 ]]; then suffix="${suffix}_pm"; fi
  local bin="$ONEESAN_BUILD_DIR/ab_depth8_graph${suffix}_key4${TERNARY_KEY4}_n${N}"
  local so="$LOGDIR/${tag}.out"
  local se="$LOGDIR/${tag}.err"

  echo "=== build $tag ===" >&2
  N="$N" \
  OUT="$bin" \
  HIGH_CTX="$high_ctx" \
  TRANSPOSE_MODE="$mode" \
  PM_ACCUM="$PM_ACCUM" \
  TERNARY_KEY4="$TERNARY_KEY4" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depth8-graph-batch.sh"

  echo "=== run $tag ===" >&2
  "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"

  local summary detail residue wall fh fl rl rh ts
  summary="$(grep 'exact_batch_result' "$so" | tail -n1 || true)"
  if [[ -z "$summary" ]]; then
    summary="$(grep 'residue=' "$so" | tail -n1 || true)"
  fi
  detail="$(grep 'snake_onepass_batch modulus=' "$se" | tail -n1 || true)"

  residue="$(field residue "$summary")"
  wall="$(field wall_s "$summary")"
  if [[ -z "$residue" ]]; then
    echo "$tag produced no residue; see $so and $se" >&2
    exit 3
  fi
  if [[ "$residue" != "$EXPECT" ]]; then
    echo "$tag residue mismatch got=$residue expected=$EXPECT" >&2
    exit 4
  fi

  fh="$(field forward_high_s "$detail")"
  fl="$(field forward_low_s "$detail")"
  rl="$(field reverse_low_s "$detail")"
  rh="$(field reverse_high_s "$detail")"
  ts="$(field transpose_s "$detail")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$high_ctx" "$mode" "$TERNARY_KEY4" "$residue" "${wall:-NA}" \
    "${fh:-NA}" "${fl:-NA}" "${rl:-NA}" "${rh:-NA}" "${ts:-NA}" "$bin" >>"$RESULT"
}

printf 'high_ctx\ttranspose_mode\tternary_key4\tresidue\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\tbinary\n' >"$RESULT"

for mode in sync events pipeline; do
  run_one "$mode" thread
  run_one "$mode" shared
done

cat "$RESULT"
echo "depth8-graph-highctx-ab OK n=$N modulus=$MOD expected=$EXPECT pm_accum=$PM_ACCUM ternary_key4=$TERNARY_KEY4 result=$RESULT" >&2
