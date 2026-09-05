#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
MOD="${MOD:-4294967291}"
EXPECT="${EXPECT:-998035516}"
NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_pattern10_depth_decode_factorial_n${N}_${TRANSPOSE_MODE}_pm${PM_ACCUM}_key4${TERNARY_KEY4}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"

if (( NGPU != 8 )); then
  echo "depth/decode factorial currently requires NGPU=8" >&2
  exit 2
fi
if (( REPEATS < 1 )); then
  echo "REPEATS must be >=1" >&2
  exit 2
fi
case "$TRANSPOSE_MODE" in
  sync|events|pipeline) ;;
  *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2 ;;
esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then
  echo "PM_ACCUM must be 0 or 1" >&2
  exit 2
fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then
  echo "TERNARY_KEY4 must be 0 or 1" >&2
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

build_one() {
  local tag="$1" depth="$2" decode="$3"
  local build_script bin
  if [[ "$depth" == 8 ]]; then
    build_script="scripts/build/b300-bucket-snake-pattern10-depth8-graph-batch.sh"
    bin="$ONEESAN_BUILD_DIR/factorial_${tag}_n${N}"
    N="$N" OUT="$bin" P10_DECODE="$decode" HIGH_CTX=thread \
      TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
      bash "$ONEESAN_ROOT/$build_script"
  else
    build_script="scripts/build/b300-bucket-snake-pattern10-depth4-graph-batch.sh"
    bin="$ONEESAN_BUILD_DIR/factorial_${tag}_n${N}"
    N="$N" OUT="$bin" P10_DECODE="$decode" \
      TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
      bash "$ONEESAN_ROOT/$build_script"
  fi
  printf '%s' "$bin"
}

run_one() {
  local tag="$1" depth="$2" decode="$3" bin="$4" rep="$5"
  local so="$LOGDIR/${tag}_r${rep}.out"
  local se="$LOGDIR/${tag}_r${rep}.err"
  local line detail residue wall fh fl rl rh ts

  echo "=== run $tag repeat=$rep/$REPEATS ===" >&2
  "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"

  line="$(grep '^residue=' "$so" | tail -n1 || true)"
  if [[ -z "$line" ]]; then
    echo "$tag repeat=$rep missing residue line; see $so and $se" >&2
    exit 3
  fi
  residue="$(field residue "$line")"
  wall="$(field wall_s "$line")"
  if [[ "$residue" != "$EXPECT" ]]; then
    echo "$tag repeat=$rep residue mismatch got=$residue expected=$EXPECT" >&2
    exit 4
  fi

  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"
  fl="$(field forward_low_s "$detail")"
  rl="$(field reverse_low_s "$detail")"
  rh="$(field reverse_high_s "$detail")"
  ts="$(field transpose_s "$detail")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$depth" "$decode" "$rep" "$residue" "$wall" \
    "${fh:-NA}" "${fl:-NA}" "${rl:-NA}" "${rh:-NA}" "${ts:-NA}" >>"$RESULT"
}

printf 'backend\tdepth_bits\tdecode\trepeat\tresidue\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\n' >"$RESULT"

configs=(
  'depth8_unrank 8 unrank'
  'depth8_lut 8 lut'
  'depth4_unrank 4 unrank'
  'depth4_lut 4 lut'
)

for config in "${configs[@]}"; do
  read -r tag depth decode <<<"$config"
  echo "=== build $tag ===" >&2
  bin="$(build_one "$tag" "$depth" "$decode")"
  for ((rep=1; rep<=REPEATS; ++rep)); do
    run_one "$tag" "$depth" "$decode" "$bin" "$rep"
  done
done

cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv
import statistics
import sys

result_path, summary_path = sys.argv[1:]
metrics = [
    "wall_s",
    "forward_high_s",
    "forward_low_s",
    "reverse_low_s",
    "reverse_high_s",
    "transpose_s",
]

with open(result_path, newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

by_backend = {}
for row in rows:
    by_backend.setdefault(row["backend"], []).append(row)

summary = {}
for backend, group in by_backend.items():
    out = {
        "backend": backend,
        "depth_bits": group[0]["depth_bits"],
        "decode": group[0]["decode"],
        "repeats": str(len(group)),
    }
    for metric in metrics:
        values = [float(row[metric]) for row in group if row[metric] != "NA"]
        out[metric] = f"{statistics.median(values):.9f}" if values else "NA"
    summary[backend] = out

with open(summary_path, "w", newline="") as f:
    fields = ["backend", "depth_bits", "decode", "repeats", *metrics]
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for name in ("depth8_unrank", "depth8_lut", "depth4_unrank", "depth4_lut"):
        writer.writerow(summary[name])

w8u = float(summary["depth8_unrank"]["wall_s"])
w8l = float(summary["depth8_lut"]["wall_s"])
w4u = float(summary["depth4_unrank"]["wall_s"])
w4l = float(summary["depth4_lut"]["wall_s"])

lut8 = w8u / w8l
lut4 = w4u / w4l
depth_unrank = w8u / w4u
depth_lut = w8l / w4l
full = w8u / w4l
interaction = full / (lut8 * depth_unrank)

print(f"lut_speedup_at_depth8={lut8:.6f}x")
print(f"lut_speedup_at_depth4={lut4:.6f}x")
print(f"depth4_speedup_with_unrank={depth_unrank:.6f}x")
print(f"depth4_speedup_with_lut={depth_lut:.6f}x")
print(f"full_depth4_lut_speedup_vs_depth8_unrank={full:.6f}x")
print(f"depth_decode_interaction_ratio={interaction:.6f}x")
print(f"summary={summary_path}")
PY

echo "depth/decode factorial OK n=$N repeats=$REPEATS transpose=$TRANSPOSE_MODE pm_accum=$PM_ACCUM ternary_key4=$TERNARY_KEY4 result=$RESULT" >&2
