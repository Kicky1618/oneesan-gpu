#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
FIRST="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-latest.sh"; EXACT="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-latest.sh"; ENV_PRE="$ONEESAN_ROOT/scripts/run/b300x8-grand-latest-preflight.sh"
for f in "$FIRST" "$EXACT" "$ENV_PRE"; do [[ -f "$f" ]] || exit 2; bash -n "$f"; done
grep -Fq 'b300x8-grand-firstpass-stager.sh' "$FIRST" || { echo 'latest firstpass is not Stage R' >&2; exit 3; }
grep -Fq 'b300x8-grand-promote-exact-stager.sh' "$EXACT" || { echo 'latest exact promoter is not Stage R' >&2; exit 3; }
grep -Fq 'RUN_LATEST_PREFLIGHT="${RUN_LATEST_PREFLIGHT:-1}"' "$FIRST" || { echo 'latest firstpass does not enable environment preflight by default' >&2; exit 3; }
grep -Fq 'b300x8-grand-latest-preflight.sh' "$FIRST" || { echo 'latest firstpass does not call environment preflight' >&2; exit 3; }
[[ "$(grep -c '^exec bash ' "$FIRST")" == 1 && "$(grep -c '^exec bash ' "$EXACT")" == 1 ]] || { echo 'latest entrypoints must tail-exec Stage R exactly once' >&2; exit 3; }
python3 - "$FIRST" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text(); p=s.find('b300x8-grand-latest-preflight.sh'); e=s.find('exec bash "$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-stager.sh"')
if p<0 or e<0 or p>=e: raise SystemExit('environment preflight must run before Stage-R firstpass exec')
PY
echo 'b300-grand-latest-entrypoint-preflight OK latest_stage=R firstpass=1 exact_promotion=1 environment_preflight=1 preflight_default_on=1 tail_exec=1 gpu_work=0'
