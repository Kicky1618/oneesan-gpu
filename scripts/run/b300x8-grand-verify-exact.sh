#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'grand exact verification currently targets n=27' >&2; exit 2; }

FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
SELECTED_ENV="${SELECTED_ENV:-${FIRSTPASS_PREFIX}.selected.env}"
readonly SELECTED_ENV_PATH="$SELECTED_ENV"
readonly CERTIFICATE_REQUESTED="${CERTIFICATE:-}"
readonly ONEESAN_ROOT_PATH="$ONEESAN_ROOT"
[[ -s "$SELECTED_ENV_PATH" ]] || { echo "missing first-pass selection contract: $SELECTED_ENV_PATH" >&2; exit 2; }
# shellcheck disable=SC1090
source "$SELECTED_ENV_PATH"

for key in \
  B300_GRAND_SELECTED_SCHEMA B300_GRAND_SELECTED_VALIDATED B300_GRAND_SELECTED_N \
  B300_GRAND_SELECTED_PROFILE_FILE B300_GRAND_SELECTED_PROFILE_SHA256 \
  B300_GRAND_SELECTED_BINARY B300_GRAND_SELECTED_BINARY_SHA256 \
  B300_GRAND_SELECTED_WORK_DIR B300_GRAND_SELECTED_CHECKPOINT; do
  [[ -n "${!key+x}" ]] || { echo "selection contract missing $key" >&2; exit 3; }
done
[[ "$B300_GRAND_SELECTED_SCHEMA" =~ ^[1-9][0-9]*$ ]] || { echo "bad selection schema=$B300_GRAND_SELECTED_SCHEMA" >&2; exit 3; }
SELECTION_SCHEMA="$B300_GRAND_SELECTED_SCHEMA"
(( SELECTION_SCHEMA >= 1 && SELECTION_SCHEMA <= 3 )) || { echo "unsupported grand selection schema=$SELECTION_SCHEMA" >&2; exit 3; }
[[ "$B300_GRAND_SELECTED_VALIDATED" == 1 && "$B300_GRAND_SELECTED_N" == 27 ]] || {
  echo 'invalid grand selection contract' >&2; exit 3;
}
[[ -x "$B300_GRAND_SELECTED_BINARY" ]] || { echo "selected binary missing: $B300_GRAND_SELECTED_BINARY" >&2; exit 3; }
[[ -s "$B300_GRAND_SELECTED_PROFILE_FILE" ]] || { echo "selected profile missing: $B300_GRAND_SELECTED_PROFILE_FILE" >&2; exit 3; }
[[ -s "$B300_GRAND_SELECTED_CHECKPOINT" ]] || { echo "selected checkpoint missing: $B300_GRAND_SELECTED_CHECKPOINT" >&2; exit 3; }
EXACT="$B300_GRAND_SELECTED_WORK_DIR/exact.txt"
[[ -s "$EXACT" ]] || { echo "exact result missing: $EXACT" >&2; exit 3; }

command -v sha256sum >/dev/null || { echo 'sha256sum required' >&2; exit 2; }
BIN_SHA="$(sha256sum "$B300_GRAND_SELECTED_BINARY" | awk '{print $1}')"
PROFILE_SHA="$(sha256sum "$B300_GRAND_SELECTED_PROFILE_FILE" | awk '{print $1}')"
[[ "$BIN_SHA" == "$B300_GRAND_SELECTED_BINARY_SHA256" ]] || { echo 'selected binary fingerprint mismatch' >&2; exit 4; }
[[ "$PROFILE_SHA" == "$B300_GRAND_SELECTED_PROFILE_SHA256" ]] || { echo 'selected profile fingerprint mismatch' >&2; exit 4; }

if [[ -n "$CERTIFICATE_REQUESTED" ]]; then
  CERTIFICATE="$CERTIFICATE_REQUESTED"
else
  CERTIFICATE="$B300_GRAND_SELECTED_WORK_DIR/exact.verify.json"
fi
VERIFY_LOG="$B300_GRAND_SELECTED_WORK_DIR/exact.verify.log"

python3 "$ONEESAN_ROOT_PATH/scripts/solve/verify_b300_exact_result.py" 27 \
  --checkpoint "$B300_GRAND_SELECTED_CHECKPOINT" \
  --exact "$EXACT" \
  --binary "$B300_GRAND_SELECTED_BINARY" \
  --profile-sha256 "$B300_GRAND_SELECTED_PROFILE_SHA256" \
  --certificate "$CERTIFICATE" "$@" | tee "$VERIFY_LOG"

grep -Fq 'B300_EXACT_VERIFY_OK ' "$VERIFY_LOG" || { echo 'verification success marker missing' >&2; exit 4; }
[[ -s "$CERTIFICATE" ]] || { echo "verification certificate missing: $CERTIFICATE" >&2; exit 4; }
python3 - "$CERTIFICATE" "$BIN_SHA" "$PROFILE_SHA" "$B300_GRAND_SELECTED_CHECKPOINT" "$EXACT" <<'PY'
import hashlib,json,sys
from pathlib import Path
cert,bsha,psha,checkpoint,exact=sys.argv[1:]
d=json.load(open(cert))
if d.get('schema') != 1 or d.get('verified') is not True or int(d.get('n',-1)) != 27:
    raise SystemExit('bad verification certificate header')
if d.get('solver_binary_sha256') != bsha: raise SystemExit('certificate binary SHA mismatch')
if d.get('solver_profile_sha256') != psha: raise SystemExit('certificate profile SHA mismatch')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for z in iter(lambda:f.read(1<<20),b''): h.update(z)
 return h.hexdigest()
if d.get('checkpoint_sha256') != sha(checkpoint): raise SystemExit('certificate checkpoint SHA mismatch')
if d.get('exact_txt_sha256') != sha(exact): raise SystemExit('certificate exact.txt SHA mismatch')
PY

CERT_SHA="$(sha256sum "$CERTIFICATE" | awk '{print $1}')"
EXACT_SHA="$(sha256sum "$EXACT" | awk '{print $1}')"
echo "B300 GRAND VERIFY COMPLETE schema=$SELECTION_SCHEMA exact=$EXACT exact_sha256=$EXACT_SHA certificate=$CERTIFICATE certificate_sha256=$CERT_SHA" >&2
