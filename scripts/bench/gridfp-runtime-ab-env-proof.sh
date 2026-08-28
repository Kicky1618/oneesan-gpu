#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BUILD="$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh"
ENVFILE="$ONEESAN_ROOT/scripts/lib/gridfp-runtime-ab-env.sh"
[[ -f "$BUILD" && -f "$ENVFILE" ]] || { echo "missing runtime A/B audit input" >&2; exit 2; }

tmp_build="$(mktemp)"
tmp_env="$(mktemp)"
trap 'rm -f "$tmp_build" "$tmp_env"' EXIT

sed -nE 's/^(RUNTIME_[A-Z0-9_]+)=.*/\1/p' "$BUILD" | sort -u >"$tmp_build"
sed -nE 's/^  (RUNTIME_[A-Z0-9_]+)=.*/\1/p' "$ENVFILE" | sort -u >"$tmp_env"

if ! diff -u "$tmp_build" "$tmp_env"; then
  echo "runtime A/B environment is out of sync with build runtime knobs" >&2
  exit 3
fi

count="$(wc -l <"$tmp_build" | tr -d ' ')"
grep -Fxq 'RUNTIME_TURN_DIRECT_COMPRESS_INVERSE' "$tmp_env"
grep -Fxq 'RUNTIME_FIND_INDEX_BUCKETS' "$tmp_env"
grep -Fxq 'RUNTIME_FIND_INDEX_WAYS' "$tmp_env"
grep -Fxq 'RUNTIME_FIND_INDEX_CACHE' "$tmp_env"

echo "gridfp-runtime-ab-env-proof OK runtime_knobs=$count exact=1"
