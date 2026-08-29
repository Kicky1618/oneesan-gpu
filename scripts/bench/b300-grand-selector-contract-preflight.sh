#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

REAL_GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
[[ -f "$REAL_GRAND" ]] || { echo "missing grand selector=$REAL_GRAND" >&2; exit 2; }
bash -n "$REAL_GRAND"

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-grand-contract.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
root="$tmp/root"
mkdir -p "$root/scripts/run" "$root/scripts/lib" "$root/work" "$root/bin"
cp "$REAL_GRAND" "$root/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
chmod +x "$root/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"

cat >"$root/scripts/lib/common.sh" <<'EOF'
#!/usr/bin/env bash
ONEESAN_ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export ONEESAN_ROOT
EOF

make_bin(){
  local name="$1"
  cat >"$root/bin/$name" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
  chmod +x "$root/bin/$name"
}
for name in joint-primary joint-base sat-nextself sat-control hybrid hybrid-base composed stagef-control; do make_bin "$name"; done

cat >"$root/scripts/run/b300x8-joint-calibrated-select.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${PREPARE_ENV:?}"
cat >"$PREPARE_ENV" <<EOT
B300_JOINT_PREPARED=1
FORCED_OVERRIDE_BIN=$ONEESAN_ROOT/bin/joint-primary
FORCED_OVERRIDE_LABEL=joint_primary
FORCED_OVERRIDE_THREADS=256
FORCED_BASE_BIN=$ONEESAN_ROOT/bin/joint-base
FORCED_BASE_LABEL=joint_base
FORCED_BASE_THREADS=128
FORCED_TARGET_MIB=65536
SMOKE_PRIME=4294967291
EOT
EOF
chmod +x "$root/scripts/run/b300x8-joint-calibrated-select.sh"

cat >"$root/scripts/run/b300x8-ilp8-nextself-staged-fullprime-race.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
rc="${FAKE_NEXTSELF_RC:-0}"
((rc==0)) || exit "$rc"
: "${PREPARE_ENV:?}"
cat >"$PREPARE_ENV" <<EOT
B300_NEXTSELF_PREPARED=1
B300_NEXTSELF_PREPARED_BIN=$ONEESAN_ROOT/bin/sat-nextself
B300_NEXTSELF_PREPARED_LABEL=sat_nextself
B300_NEXTSELF_PREPARED_THREADS=256
B300_NEXTSELF_PREPARED_CONTROL_BIN=$ONEESAN_ROOT/bin/sat-control
B300_NEXTSELF_PREPARED_CONTROL_LABEL=sat_control
B300_NEXTSELF_PREPARED_CONTROL_THREADS=256
B300_NEXTSELF_PREPARED_STAGED_SPEEDUP=1.02
EOT
EOF
chmod +x "$root/scripts/run/b300x8-ilp8-nextself-staged-fullprime-race.sh"

cat >"$root/scripts/run/b300x8-nextgen-hybrid8-staged-fullprime-race.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
rc="${FAKE_HYBRID_RC:-0}"
((rc==0)) || exit "$rc"
: "${PREPARE_ENV:?}"
cat >"$PREPARE_ENV" <<EOT
B300_HYBRID8_PREPARED=1
B300_HYBRID8_PREPARED_BIN=$ONEESAN_ROOT/bin/hybrid
B300_HYBRID8_PREPARED_LABEL=hybrid
B300_HYBRID8_PREPARED_THREADS=512
B300_HYBRID8_PREPARED_BASE_BIN=$ONEESAN_ROOT/bin/hybrid-base
B300_HYBRID8_PREPARED_BASE_LABEL=hybrid_base
B300_HYBRID8_PREPARED_BASE_THREADS=256
B300_HYBRID8_PREPARED_STAGED_SPEEDUP=1.03
EOT
EOF
chmod +x "$root/scripts/run/b300x8-nextgen-hybrid8-staged-fullprime-race.sh"

cat >"$root/scripts/run/b300x8-nextgen-hybrid8-nextself-staged-fullprime-race.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
rc="${FAKE_STAGEF_RC:-0}"
((rc==0)) || exit "$rc"
: "${PREPARE_ENV:?}"
cat >"$PREPARE_ENV" <<EOT
B300_HYBRID8_NEXTSELF_PREPARED=1
B300_HYBRID8_NEXTSELF_PREPARED_BIN=$ONEESAN_ROOT/bin/composed
B300_HYBRID8_NEXTSELF_PREPARED_LABEL=composed
B300_HYBRID8_NEXTSELF_PREPARED_THREADS=512
B300_HYBRID8_NEXTSELF_PREPARED_CONTROL_BIN=$ONEESAN_ROOT/bin/stagef-control
B300_HYBRID8_NEXTSELF_PREPARED_CONTROL_LABEL=stagef_control
B300_HYBRID8_NEXTSELF_PREPARED_CONTROL_THREADS=256
B300_HYBRID8_NEXTSELF_PREPARED_STAGED_SPEEDUP=1.04
B300_HYBRID8_NEXTSELF_PREPARED_MANIFEST=$ONEESAN_ROOT/work/fake-stagef.sha256
EOT
EOF
chmod +x "$root/scripts/run/b300x8-nextgen-hybrid8-nextself-staged-fullprime-race.sh"

cat >"$root/scripts/run/b300x8-race-external-forced-profiled-once.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${FAKE_CAPTURE:?}"
cat >"$FAKE_CAPTURE" <<EOT
P_BIN=${FORCED_OVERRIDE_BIN:-}
P_LABEL=${FORCED_OVERRIDE_LABEL:-}
P_THREADS=${FORCED_OVERRIDE_THREADS:-}
B_BIN=${FORCED_BASE_BIN:-}
B_LABEL=${FORCED_BASE_LABEL:-}
B_THREADS=${FORCED_BASE_THREADS:-}
E1_BIN=${FORCED_EXTRA_BIN:-}
E1_LABEL=${FORCED_EXTRA_LABEL:-}
E1_THREADS=${FORCED_EXTRA_THREADS:-}
E2_BIN=${FORCED_EXTRA2_BIN:-}
E2_LABEL=${FORCED_EXTRA2_LABEL:-}
E2_THREADS=${FORCED_EXTRA2_THREADS:-}
E3_BIN=${FORCED_EXTRA3_BIN:-}
E3_LABEL=${FORCED_EXTRA3_LABEL:-}
E3_THREADS=${FORCED_EXTRA3_THREADS:-}
PROFILE_FILE=${PROFILE_FILE:-}
SMOKE_PRIME=${SMOKE_PRIME:-}
TARGET_MIB=${FORCED_TARGET_MIB:-}
MAX_WINDOW=${MAX_WINDOW:-}
SELECT_ONLY=${SELECT_ONLY:-}
EOT
printf 'fake_race=OK\n'
EOF
chmod +x "$root/scripts/run/b300x8-race-external-forced-profiled-once.sh"

profile="$root/work/profile.env"
printf 'PROFILE=contract\n' >"$profile"
grand="$root/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"

run_case(){
  local name="$1" next_rc="$2" hybrid_rc="$3" stagef_rc="$4" expected_mode="$5"
  shift 5
  local prefix="$root/work/$name" capture="$root/work/$name.capture" out="$root/work/$name.out" err="$root/work/$name.err"
  FAKE_NEXTSELF_RC="$next_rc" FAKE_HYBRID_RC="$hybrid_rc" FAKE_STAGEF_RC="$stagef_rc" FAKE_CAPTURE="$capture" \
    PROFILE_FILE="$profile" PREFIX="$prefix" RUN_NEXTSELF_STAGE=1 RUN_HYBRID_STAGE=1 RUN_HYBRID_NS_STAGE=1 \
    bash "$grand" 27 >"$out" 2>"$err"
  [[ -s "$capture" ]] || { echo "$name missing fake race capture" >&2; exit 4; }
  local summary="${prefix}.race_grand.env"
  [[ -s "$summary" ]] || { echo "$name missing grand summary" >&2; exit 4; }
  # shellcheck disable=SC1090
  source "$summary"
  [[ "$B300_GRAND_MODE" == "$expected_mode" ]] || { echo "$name mode=$B300_GRAND_MODE expected=$expected_mode" >&2; exit 4; }
  python3 - "$name" "$capture" "$root" "$@" <<'PY'
from pathlib import Path
import sys
name,capture,root,*expect=sys.argv[1:]
got={}
for line in Path(capture).read_text().splitlines():
    k,v=line.split('=',1);got[k]=v
keys=('P_BIN','B_BIN','E1_BIN','E2_BIN','E3_BIN')
if len(expect)!=5: raise SystemExit('internal expected candidate count mismatch')
for k,want in zip(keys,expect):
    if want:
        want=f'{root}/bin/{want}'
    if got.get(k,'')!=want:
        raise SystemExit(f'{name}: {k}={got.get(k)!r} expected={want!r}')
if got.get('SELECT_ONLY')!='1': raise SystemExit(f'{name}: SELECT_ONLY not forced/preserved')
if got.get('SMOKE_PRIME')!='4294967291': raise SystemExit(f'{name}: smoke prime contract lost')
if got.get('TARGET_MIB')!='65536': raise SystemExit(f'{name}: target contract lost')
print(f'grand_contract_case={name} OK')
PY
}

# P, B, E1, E2, E3 expected binary basenames.
run_case composed_all 0 0 0 hybrid8_nextself_composed_grand \
  composed stagef-control hybrid-base sat-nextself joint-primary
run_case stagef_rejected 0 0 4 nextself_hybrid8_joint \
  sat-nextself sat-control hybrid hybrid-base joint-primary
run_case nextself_rejected 4 0 0 hybrid8_nextself_composed_joint \
  composed stagef-control hybrid-base joint-primary joint-base
run_case hybrid_rejected 0 4 0 nextself_joint \
  sat-nextself sat-control joint-primary joint-base ''
run_case transforms_rejected 4 4 0 joint_fallback \
  joint-primary joint-base '' '' ''

echo 'b300_grand_selector_contract_preflight=OK cases=5 composed=OK stagef_reject=OK nextself_reject=OK hybrid_reject=OK joint_fallback=OK candidate_env=OK select_only=OK gpu_work=0'
