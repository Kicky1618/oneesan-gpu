#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

RUNNER="$ONEESAN_ROOT/scripts/run/b300x8-ilp8-nextself-staged-fullprime-race.sh"
COMMON="$ONEESAN_ROOT/scripts/lib/common.sh"
[[ -f "$RUNNER" ]] || { echo "missing runner=$RUNNER" >&2; exit 2; }
[[ -f "$COMMON" ]] || { echo "missing common=$COMMON" >&2; exit 2; }
bash -n "$RUNNER"

for marker in \
  'B300_NEXTSELF_STAGED_VALIDATED' \
  'B300_NEXTSELF_CONTROL_SPILL_FREE' \
  'B300_NEXTSELF_SPILL_FREE' \
  'B300_NEXTSELF_CONTROL_BIN' \
  'B300_NEXTSELF_BIN' \
  'B300_ROW_LIMIT=28' \
  'exec "$SRC" "$N" "$MOD" "$TARGET" "$MAXW" "$NGPU"' \
  'FORCED_OVERRIDE_BIN="$NEXTSELF_ADAPTER"' \
  'FORCED_BASE_BIN="$CONTROL_ADAPTER"'; do
  grep -Fq "$marker" "$RUNNER" || { echo "missing next-self promotion marker: $marker" >&2; exit 3; }
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-nextself-fullprime-preflight.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/work"

make_fake_sat(){
  local out="$1" tag="$2"
  cat >"$out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  '$tag' "\${1:-}" "\${2:-}" "\${3:-}" "\${4:-}" "\${5:-}" \
  "\${B300_ROW_LIMIT:-}" "\${GRIDFP_THREADS:-}" "\${GRIDFP_PLAN_TARGET_MIB:-}" >>$(printf '%q' "$tmp/calls.tsv")
EOF
  chmod +x "$out"
}
CONTROL="$tmp/control.bin"
NEXTSELF="$tmp/nextself.bin"
make_fake_sat "$CONTROL" control
make_fake_sat "$NEXTSELF" nextself

WINNER="$tmp/winner.env"
cat >"$WINNER" <<EOF
B300_NEXTSELF_STAGED_VALIDATED=1
B300_NEXTSELF_VARIANT=blockfirst
B300_NEXTSELF_RESIDUE=123456789
B300_NEXTSELF_CONTROL_WALL_S=10.000000000
B300_NEXTSELF_WALL_S=9.000000000
B300_NEXTSELF_SPEEDUP=1.111111111
B300_NEXTSELF_CONTROL_MC_AVG_PCT=40.000
B300_NEXTSELF_MC_AVG_PCT=44.000
B300_NEXTSELF_MC_DELTA_PP=4.000
B300_NEXTSELF_CONTROL_SPILL_FREE=1
B300_NEXTSELF_SPILL_FREE=1
B300_NEXTSELF_STAGE_VALID=1
B300_NEXTSELF_CONTROL_BIN=$(printf '%q' "$CONTROL")
B300_NEXTSELF_BIN=$(printf '%q' "$NEXTSELF")
B300_NEXTSELF_MIN_SPEEDUP=1.010000000
EOF
PROFILE="$tmp/profile.env"
printf 'ORBIT_PROFILE=preflight\n' >"$PROFILE"

CAPTURE="$tmp/capture.sh"
cat >"$CAPTURE" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "${FORCED_OVERRIDE_LABEL:-}" == nextself_blockfirst ]]
[[ "${FORCED_BASE_LABEL:-}" == staged_blockfirst_control ]]
[[ -x "${FORCED_OVERRIDE_BIN:-}" && -x "${FORCED_BASE_BIN:-}" ]]
"$FORCED_OVERRIDE_BIN" 27 65536 14 8 4294967291
"$FORCED_BASE_BIN" 27 65536 14 8 4294967291
printf 'capture_ok=1\n'
EOF
chmod +x "$CAPTURE"

PATCHED="$tmp/runner.sh"
python3 - "$RUNNER" "$PATCHED" "$CAPTURE" "$COMMON" <<'PY'
from pathlib import Path
import shlex,sys
src=Path(sys.argv[1]).read_text()
out=Path(sys.argv[2])
capture=sys.argv[3]
common=sys.argv[4]
source_line='source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"'
if src.count(source_line)!=1:
    raise SystemExit(f'common source anchor expected once, got {src.count(source_line)}')
src=src.replace(source_line,'source '+shlex.quote(common),1)
needle='"$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"'
if src.count(needle)!=1:
    raise SystemExit(f'external runner anchor expected once, got {src.count(needle)}')
src=src.replace(needle,shlex.quote(capture)+' 27 "$@"',1)
out.write_text(src)
PY
chmod +x "$PATCHED"
bash -n "$PATCHED"
grep -Fq "source $(printf '%q' "$COMMON")" "$PATCHED" || { echo 'patched runner common path mismatch' >&2; exit 3; }

PROFILE_FILE="$PROFILE" RUN_STAGED=0 WINNER_ENV="$WINNER" RACE_PREFIX="$tmp/work/race" \
  GRIDFP_THREADS=256 GRIDFP_PLAN_TARGET_MIB=16384 MIN_SPEEDUP=1.01 \
  bash "$PATCHED" 27 >"$tmp/runner.out" 2>"$tmp/runner.err"
grep -q '^capture_ok=1$' "$tmp/runner.out"

python3 - "$tmp/calls.tsv" <<'PY'
import csv,sys
rows=list(csv.reader(open(sys.argv[1]),delimiter='\t'))
assert len(rows)==2,rows
for row,tag in zip(rows,('nextself','control')):
    assert row[0]==tag,row
    # forced adapter input: N TARGET WINDOW NGPU MOD
    # saturation binary must receive: N MOD TARGET WINDOW NGPU
    assert row[1:6]==['27','4294967291','65536','14','8'],row
    assert row[6:]==['28','256','16384'],row
print('nextself_adapter_argument_order=OK')
PY

PROMO="$tmp/work/race_promotion.env"
[[ -s "$PROMO" ]] || { echo 'promotion env missing' >&2; exit 4; }
grep -q '^B300_NEXTSELF_PROMOTION_VALIDATED=1$' "$PROMO"
grep -q '^B300_NEXTSELF_PROMOTION_VARIANT=blockfirst$' "$PROMO"
grep -q '^B300_NEXTSELF_PROMOTION_PARTIAL_RESIDUE=123456789$' "$PROMO"

echo 'b300_nextself_fullprime_preflight=OK shell_syntax=OK staged_gate=OK spill_gate=OK adapter_order=OK fingerprint=OK gpu_work=0'
