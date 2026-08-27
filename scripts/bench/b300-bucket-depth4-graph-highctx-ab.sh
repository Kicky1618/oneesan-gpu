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
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depth4_highctx_ab_n${N}_${TRANSPOSE_MODE}_pm${PM_ACCUM}_key4${TERNARY_KEY4}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"

if (( NGPU != 8 )); then
  echo "depth4 HIGH-context A/B currently requires NGPU=8" >&2
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
  local decode="$1" ctx="$2" bin="$3"
  N="$N" OUT="$bin" P10_DECODE="$decode" HIGH_CTX="$ctx" \
    TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depth4-graph-batch.sh"
}

run_one() {
  local decode="$1" ctx="$2" bin="$3" rep="$4"
  local tag="${decode}_${ctx}"
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
    "$decode" "$ctx" "$rep" "$residue" "$wall" \
    "${fh:-NA}" "${fl:-NA}" "${rl:-NA}" "${rh:-NA}" "${ts:-NA}" "$bin" >>"$RESULT"
}

printf 'decode\thigh_ctx\trepeat\tresidue\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\tbinary\n' >"$RESULT"

for decode in unrank lut; do
  for ctx in thread shared warp; do
    tag="${decode}_${ctx}"
    bin="$ONEESAN_BUILD_DIR/ab_depth4_${tag}_${TRANSPOSE_MODE}_n${N}"
    echo "=== build $tag ===" >&2
    build_one "$decode" "$ctx" "$bin"
    for ((rep=1; rep<=REPEATS; ++rep)); do
      run_one "$decode" "$ctx" "$bin" "$rep"
    done
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

by_key = {}
for row in rows:
    by_key.setdefault((row["decode"], row["high_ctx"]), []).append(row)

summary_rows = []
for decode in ("unrank", "lut"):
    for ctx in ("thread", "shared", "warp"):
        group = by_key[(decode, ctx)]
        out = {"decode": decode, "high_ctx": ctx, "repeats": str(len(group))}
        for metric in metrics:
            values = [float(row[metric]) for row in group if row[metric] != "NA"]
            out[metric] = f"{statistics.median(values):.9f}" if values else "NA"
        summary_rows.append(out)

with open(summary_path, "w", newline="") as f:
    fields = ["decode", "high_ctx", "repeats", *metrics]
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(summary_rows)

lookup = {(row["decode"], row["high_ctx"]): row for row in summary_rows}
for decode in ("unrank", "lut"):
    base = float(lookup[(decode, "thread")]["wall_s"])
    for ctx in ("shared", "warp"):
        wall = float(lookup[(decode, ctx)]["wall_s"])
        print(f"{decode}_{ctx}_speedup_vs_thread={base / wall:.6f}x")
        for phase in ("forward_high_s", "reverse_high_s"):
            a = lookup[(decode, "thread")][phase]
            b = lookup[(decode, ctx)][phase]
            if a != "NA" and b != "NA":
                print(f"{decode}_{ctx}_{phase}_speedup={float(a) / float(b):.6f}x")
print(f"summary={summary_path}")
PY

echo "depth4-highctx-ab OK n=$N repeats=$REPEATS transpose=$TRANSPOSE_MODE pm_accum=$PM_ACCUM ternary_key4=$TERNARY_KEY4 result=$RESULT" >&2
