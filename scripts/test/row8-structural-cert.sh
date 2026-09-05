#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CERT10="$ONEESAN_ROOT/formal/certificates/row8_gridfp_structural_w10.json"
CERT22="$ONEESAN_ROOT/formal/certificates/row8_gridfp_structural_w22.json"
TOOL="$ONEESAN_ROOT/scripts/tools/row8_gridfp_structural_cert.py"
EXACT="$ONEESAN_ROOT/scripts/run/b300x8-exact.sh"
TMP="$ONEESAN_BUILD_DIR/row8-structural-cert-test"
mkdir -p "$TMP"

expect_rc2() {
  local name="$1"; shift
  set +e
  "$@" >"$TMP/$name.out" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" -ne 2 ]]; then
    echo "$name: expected rc=2, got rc=$rc" >&2
    cat "$TMP/$name.out" >&2
    exit 40
  fi
}

echo '== Row-8 structural integer certificates =='
python3 "$TOOL" --verify "$CERT10"
python3 "$TOOL" --verify "$CERT22"

# The certificate must be fail-closed under metadata/fingerprint corruption.
cp "$CERT10" "$TMP/corrupt.json"
python3 - "$TMP/corrupt.json" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p))
d['critical_files'][0]['sha256']='0'*64
open(p,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
set +e
python3 "$TOOL" --verify "$TMP/corrupt.json" >"$TMP/corrupt.out" 2>&1
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then
  echo 'corrupted row8 structural certificate was accepted' >&2
  cat "$TMP/corrupt.out" >&2
  exit 41
fi

# Row-7 remains benchmark-only.
expect_rc2 row7-reject env ROW7_TENSOR=1 NGPU=1 "$EXACT" 9

# Dense/pivot row8 alone is not sufficient for exact admission.
expect_rc2 row8-no-structural env ROW8_TENSOR=1 ROW8_STRUCTURAL=0 NGPU=1 "$EXACT" 9

# Structural row8 is width-specific and must fail before GPU probing if the
# requested width has no deterministic Grid-FP/structural certificate.
expect_rc2 row8-missing-cert env ROW8_TENSOR=1 ROW8_STRUCTURAL=1 \
  ROW8_GRIDFP_CERT="$TMP/does-not-exist.json" NGPU=1 "$EXACT" 10

# A valid certificate for another width must not be reusable.
expect_rc2 row8-target-mismatch env ROW8_TENSOR=1 ROW8_STRUCTURAL=1 \
  ROW8_GRIDFP_CERT="$CERT10" NGPU=1 "$EXACT" 10

# Explicitly malformed flag combinations remain rejected.
expect_rc2 structural-without-row8 env ROW8_TENSOR=0 ROW8_STRUCTURAL=1 NGPU=1 "$EXACT" 9

rm -f "$TMP"/*.out "$TMP/corrupt.json"
echo 'row8 structural certificate gate: PASS'
