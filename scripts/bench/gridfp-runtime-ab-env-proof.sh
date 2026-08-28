#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BUILD="$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh"
ENVFILE="$ONEESAN_ROOT/scripts/lib/gridfp-runtime-ab-env.sh"
[[ -f "$BUILD" && -f "$ENVFILE" ]] || { echo "missing runtime A/B audit input" >&2; exit 2; }

tmp_build="$(mktemp)"
tmp_env="$(mktemp)"
trap 'rm -f "$tmp_build" "$tmp_env"' EXIT

# The build script may put several assignments on one physical line. Extract
# every RUNTIME_* assignment token rather than only line-leading assignments.
# HASH_BUCKETS is derived from storage bytes / associativity and is not an
# independent experiment input.
grep -oE 'RUNTIME_[A-Z0-9_]+=' "$BUILD" |
  sed 's/=$//' |
  grep -v '^RUNTIME_FIND_INDEX_HASH_BUCKETS$' |
  sort -u >"$tmp_build"
grep -oE 'RUNTIME_[A-Z0-9_]+=' "$ENVFILE" |
  sed 's/=$//' |
  sort -u >"$tmp_env"

if ! diff -u "$tmp_build" "$tmp_env"; then
  echo "runtime A/B environment is out of sync with build runtime knobs" >&2
  exit 3
fi

count="$(wc -l <"$tmp_build" | tr -d ' ')"
[[ "$count" -ge 48 ]] || {
  echo "runtime A/B audit extracted suspiciously few knobs: $count" >&2
  exit 4
}
for required in \
  RUNTIME_TURN_DIRECT_COMPRESS_INVERSE \
  RUNTIME_TURN_DIRECT_HIGH_COMPRESS_STEP \
  RUNTIME_TURN_DIRECT_LOW_COMPRESS_STEP \
  RUNTIME_TURN_DIRECT_HIGH_COMPRESS_INVERSE \
  RUNTIME_TURN_DIRECT_LOW_EXPAND_INVERSE \
  RUNTIME_FIND_INDEX_BUCKETS \
  RUNTIME_FIND_INDEX_WAYS \
  RUNTIME_FIND_INDEX_CACHE; do
  grep -Fxq "$required" "$tmp_env" || {
    echo "runtime A/B environment missing required knob: $required" >&2
    exit 5
  }
done

echo "gridfp-runtime-ab-env-proof OK runtime_knobs=$count exact=1"
