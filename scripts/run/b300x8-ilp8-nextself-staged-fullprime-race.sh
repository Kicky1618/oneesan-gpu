#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'next-self staged full-prime race targets n=27' >&2; exit 2; }

PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${GRIDFP_THREADS:-256}"
RANDOM_CG="${RANDOM_CG:-1}"
WARP_SCAN="${WARP_SCAN:-1}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
WAIT_VARIANTS="${WAIT_VARIANTS:-pairfirst blockfirst}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"
VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
SEARCH_REPEATS="${SEARCH_REPEATS:-1}"
VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.01}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
RUN_STAGED="${RUN_STAGED:-1}"
SELECT_ONLY="${SELECT_ONLY:-1}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
STAGED_PREFIX="${STAGED_PREFIX:-$ONEESAN_ROOT/work/b300x8_ilp8_nextself_staged}"
WINNER_ENV="${WINNER_ENV:-${STAGED_PREFIX}_winner.env}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300x8_ilp8_nextself_staged_fullprime_n27}"
PREPARE_ENV="${PREPARE_ENV:-${RACE_PREFIX}_prepared.env}"

for x in RANDOM_CG WARP_SCAN RUN_STAGED SELECT_ONLY REBUILD_BUCKETS PREPARE_ONLY; do
  v="${!x}"
  [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32 && THREADS<=768 && THREADS%32==0)) || { echo 'GRIDFP_THREADS must be warp multiple 32..768' >&2; exit 2; }
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32 && MAXRREGCOUNT<=255))) || { echo 'MAXRREGCOUNT must be 0 or 32..255' >&2; exit 2; }

if [[ "$RUN_STAGED" == 1 ]]; then
  echo '=== next-self staged calibration: ROWS=1 search -> ROWS=4/8 validation ===' >&2
  N=27 ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    GRIDFP_THREADS="$THREADS" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" MAXRREGCOUNT="$MAXRREGCOUNT" \
    WAIT_VARIANTS="$WAIT_VARIANTS" SEARCH_ROWS="$SEARCH_ROWS" VALIDATE_ROWS="$VALIDATE_ROWS" \
    SEARCH_REPEATS="$SEARCH_REPEATS" VALIDATE_REPEATS="$VALIDATE_REPEATS" MIN_SPEEDUP="$MIN_SPEEDUP" \
    SAMPLE_INTERVAL="$SAMPLE_INTERVAL" PREFIX="$STAGED_PREFIX" WINNER_ENV="$WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300x8-ilp8-nextself-staged-ab.sh"
fi

[[ -f "$WINNER_ENV" ]] || { echo "missing staged WINNER_ENV=$WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
[[ "${B300_NEXTSELF_STAGED_VALIDATED:-0}" == 1 ]] || {
  echo 'next-self did not survive staged validation; refusing full-prime promotion' >&2
  exit 4
}
for key in \
  B300_NEXTSELF_VARIANT B300_NEXTSELF_RESIDUE B300_NEXTSELF_CONTROL_WALL_S B300_NEXTSELF_WALL_S \
  B300_NEXTSELF_SPEEDUP B300_NEXTSELF_CONTROL_SPILL_FREE B300_NEXTSELF_SPILL_FREE \
  B300_NEXTSELF_CONTROL_BIN B300_NEXTSELF_BIN; do
  [[ -n "${!key+x}" ]] || { echo "winner env missing $key" >&2; exit 3; }
done
case "$B300_NEXTSELF_VARIANT" in pairfirst|blockfirst) ;; *) echo "bad staged variant=$B300_NEXTSELF_VARIANT" >&2; exit 3;; esac
[[ "$B300_NEXTSELF_CONTROL_SPILL_FREE" == 1 && "$B300_NEXTSELF_SPILL_FREE" == 1 ]] || {
  echo 'refusing full-prime promotion with unknown/nonzero main-kernel spills' >&2
  exit 4
}
[[ -x "$B300_NEXTSELF_CONTROL_BIN" && -x "$B300_NEXTSELF_BIN" ]] || { echo 'staged control/next-self binary missing' >&2; exit 3; }
python3 - "$B300_NEXTSELF_SPEEDUP" "$MIN_SPEEDUP" <<'PY'
import sys
speed=float(sys.argv[1]); need=float(sys.argv[2])
if speed < need:
    raise SystemExit(f'next-self staged speedup {speed:.9f}x is below required {need:.9f}x')
PY

mkdir -p "${RACE_PREFIX}_adapters"
make_adapter(){
  local src="$1" label="$2" out="${RACE_PREFIX}_adapters/${label}.sh" sha
  sha="$(sha256sum "$src" | awk '{print $1}')"
  cat >"$out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
# saturation_binary_sha256=$sha
SRC=$(printf '%q' "$src")
THREADS=$(printf '%q' "$THREADS")
PLAN_MIB=$(printf '%q' "$PLAN_MIB")
[[ \$# -ge 5 ]] || { echo 'usage: adapter N TARGET_MIB MAX_WINDOW NGPU MOD' >&2; exit 2; }
N=\$1
TARGET=\$2
MAXW=\$3
NGPU=\$4
MOD=\$5
export B300_ROW_LIMIT=28 GRIDFP_THREADS="\$THREADS" GRIDFP_PLAN_TARGET_MIB="\$PLAN_MIB"
"\$SRC" "\$N" "\$MOD" "\$TARGET" "\$MAXW" "\$NGPU" | awk '
  BEGIN{normalized=0}
  /^backend=gridfp-b300-hbm32/ && /residue=/ && /wall_s=/ && !normalized {
    sub(/^backend=[^ ]+ /,"backend=gridfp-b300-hbm32-forced2window-opt-batch ")
    normalized=1
  }
  {print}
  END{if(!normalized)exit 42}
'
EOF
  chmod +x "$out"
  bash -n "$out"
  printf '%s\n' "$out"
}

CONTROL_ADAPTER="$(make_adapter "$B300_NEXTSELF_CONTROL_BIN" "control_${B300_NEXTSELF_VARIANT}")"
NEXTSELF_ADAPTER="$(make_adapter "$B300_NEXTSELF_BIN" "nextself_${B300_NEXTSELF_VARIANT}")"
CONTROL_SHA="$(sha256sum "$B300_NEXTSELF_CONTROL_BIN" | awk '{print $1}')"
NEXTSELF_SHA="$(sha256sum "$B300_NEXTSELF_BIN" | awk '{print $1}')"
WINNER_ENV_SHA="$(sha256sum "$WINNER_ENV" | awk '{print $1}')"

cat >"${RACE_PREFIX}_promotion.env" <<EOF
B300_NEXTSELF_PROMOTION_VALIDATED=1
B300_NEXTSELF_PROMOTION_VARIANT=$(printf '%q' "$B300_NEXTSELF_VARIANT")
B300_NEXTSELF_PROMOTION_THREADS=$(printf '%q' "$THREADS")
B300_NEXTSELF_PROMOTION_CONTROL_BIN=$(printf '%q' "$B300_NEXTSELF_CONTROL_BIN")
B300_NEXTSELF_PROMOTION_BIN=$(printf '%q' "$B300_NEXTSELF_BIN")
B300_NEXTSELF_PROMOTION_CONTROL_ADAPTER=$(printf '%q' "$CONTROL_ADAPTER")
B300_NEXTSELF_PROMOTION_ADAPTER=$(printf '%q' "$NEXTSELF_ADAPTER")
B300_NEXTSELF_PROMOTION_CONTROL_SHA256=$CONTROL_SHA
B300_NEXTSELF_PROMOTION_SHA256=$NEXTSELF_SHA
B300_NEXTSELF_PROMOTION_WINNER_ENV_SHA256=$WINNER_ENV_SHA
B300_NEXTSELF_PROMOTION_PARTIAL_RESIDUE=$(printf '%q' "$B300_NEXTSELF_RESIDUE")
B300_NEXTSELF_PROMOTION_STAGED_SPEEDUP=$(printf '%q' "$B300_NEXTSELF_SPEEDUP")
EOF

label="nextself_${B300_NEXTSELF_VARIANT}"
base_label="staged_${B300_NEXTSELF_VARIANT}_control"

if [[ "$PREPARE_ONLY" == 1 ]]; then
  mkdir -p "$(dirname "$PREPARE_ENV")"
  {
    printf 'B300_NEXTSELF_PREPARED=1\n'
    printf 'B300_NEXTSELF_PREPARED_BIN=%q\n' "$NEXTSELF_ADAPTER"
    printf 'B300_NEXTSELF_PREPARED_LABEL=%q\n' "$label"
    printf 'B300_NEXTSELF_PREPARED_THREADS=%q\n' "$THREADS"
    printf 'B300_NEXTSELF_PREPARED_CONTROL_BIN=%q\n' "$CONTROL_ADAPTER"
    printf 'B300_NEXTSELF_PREPARED_CONTROL_LABEL=%q\n' "$base_label"
    printf 'B300_NEXTSELF_PREPARED_CONTROL_THREADS=%q\n' "$THREADS"
    printf 'B300_NEXTSELF_PREPARED_STAGED_SPEEDUP=%q\n' "$B300_NEXTSELF_SPEEDUP"
    printf 'B300_NEXTSELF_PREPARED_PARTIAL_RESIDUE=%q\n' "$B300_NEXTSELF_RESIDUE"
    printf 'B300_NEXTSELF_PREPARED_PROFILE_FILE=%q\n' "$PROFILE_FILE"
    printf 'B300_NEXTSELF_PREPARED_PROMOTION_ENV=%q\n' "${RACE_PREFIX}_promotion.env"
  } >"$PREPARE_ENV"
  cat "$PREPARE_ENV"
  echo "NEXTSELF PREPARED env=$PREPARE_ENV label=$label control=$base_label staged_speedup=${B300_NEXTSELF_SPEEDUP}x" >&2
  exit 0
fi

echo "=== full-prime race: $label vs $base_label vs profiled warp/orbit ===" >&2
echo "staged_speedup=${B300_NEXTSELF_SPEEDUP}x control_sha=${CONTROL_SHA:0:12} nextself_sha=${NEXTSELF_SHA:0:12}" >&2

exec env \
  PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
  FORCED_OVERRIDE_BIN="$NEXTSELF_ADAPTER" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$THREADS" \
  FORCED_BASE_BIN="$CONTROL_ADAPTER" FORCED_BASE_LABEL="$base_label" FORCED_BASE_THREADS="$THREADS" \
  REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
