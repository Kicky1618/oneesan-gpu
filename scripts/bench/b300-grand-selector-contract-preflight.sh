#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
REAL_GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
[[ -f "$REAL_GRAND" ]] || exit 2; bash -n "$REAL_GRAND"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-grand-contract.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
root="$tmp/root"; mkdir -p "$root/scripts/run" "$root/scripts/lib" "$root/work" "$root/bin"; cp "$REAL_GRAND" "$root/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"; chmod +x "$root/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
cat >"$root/scripts/lib/common.sh" <<'EOF'
#!/usr/bin/env bash
ONEESAN_ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; export ONEESAN_ROOT
EOF
make_bin(){ local name="$1"; printf '#!/usr/bin/env bash\nexit 0\n' >"$root/bin/$name"; chmod +x "$root/bin/$name"; }
for name in joint-primary joint-base sat-nextself sat-control hybrid hybrid-base composed stagef-control stagei-hint stagei-control stageh-mate stageh-self; do make_bin "$name"; done
cat >"$root/scripts/run/b300x8-joint-calibrated-select.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
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
cat >"$root/scripts/run/b300x8-ilp8-nextself-staged-fullprime-race.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
rc="${FAKE_NEXTSELF_RC:-0}"; ((rc==0)) || exit "$rc"
cat >"$PREPARE_ENV" <<EOT
B300_NEXTSELF_PREPARED=1
B300_NEXTSELF_PREPARED_BIN=$ONEESAN_ROOT/bin/sat-nextself
B300_NEXTSELF_PREPARED_LABEL=sat_nextself
B300_NEXTSELF_PREPARED_THREADS=256
B300_NEXTSELF_PREPARED_CONTROL_BIN=$ONEESAN_ROOT/bin/sat-control
B300_NEXTSELF_PREPARED_CONTROL_LABEL=sat_control
B300_NEXTSELF_PREPARED_CONTROL_THREADS=256
EOT
EOF
cat >"$root/scripts/run/b300x8-nextgen-hybrid8-staged-fullprime-race.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
rc="${FAKE_HYBRID_RC:-0}"; ((rc==0)) || exit "$rc"
cat >"$PREPARE_ENV" <<EOT
B300_HYBRID8_PREPARED=1
B300_HYBRID8_PREPARED_BIN=$ONEESAN_ROOT/bin/hybrid
B300_HYBRID8_PREPARED_LABEL=hybrid
B300_HYBRID8_PREPARED_THREADS=512
B300_HYBRID8_PREPARED_BASE_BIN=$ONEESAN_ROOT/bin/hybrid-base
B300_HYBRID8_PREPARED_BASE_LABEL=hybrid_base
B300_HYBRID8_PREPARED_BASE_THREADS=256
EOT
EOF
cat >"$root/scripts/run/b300x8-nextgen-hybrid8-nextself-staged-fullprime-race.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
rc="${FAKE_STAGEF_RC:-0}"; ((rc==0)) || exit "$rc"
cat >"$PREPARE_ENV" <<EOT
B300_HYBRID8_NEXTSELF_PREPARED=1
B300_HYBRID8_NEXTSELF_PREPARED_WIDTH=4
B300_HYBRID8_NEXTSELF_PREPARED_DISTANCE=2
B300_HYBRID8_NEXTSELF_PREPARED_BIN=$ONEESAN_ROOT/bin/composed
B300_HYBRID8_NEXTSELF_PREPARED_LABEL=composed_w4_d2
B300_HYBRID8_NEXTSELF_PREPARED_THREADS=512
B300_HYBRID8_NEXTSELF_PREPARED_CONTROL_BIN=$ONEESAN_ROOT/bin/stagef-control
B300_HYBRID8_NEXTSELF_PREPARED_CONTROL_LABEL=stagef_control
B300_HYBRID8_NEXTSELF_PREPARED_CONTROL_THREADS=256
B300_HYBRID8_NEXTSELF_PREPARED_MANIFEST=$ONEESAN_ROOT/work/fake-stagef.sha256
EOT
EOF
cat >"$root/scripts/run/b300x8-nextgen-hybrid8-nextself-evict-staged-fullprime-race.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
rc="${FAKE_STAGEI_RC:-0}"; ((rc==0)) || exit "$rc"
cat >"$PREPARE_ENV" <<EOT
B300_STAGEI_PREPARED=1
B300_STAGEI_PREPARED_HINT=last
B300_STAGEI_PREPARED_WIDTH=4
B300_STAGEI_PREPARED_DISTANCE=2
B300_STAGEI_PREPARED_BIN=$ONEESAN_ROOT/bin/stagei-hint
B300_STAGEI_PREPARED_LABEL=stagei_last_w4_d2
B300_STAGEI_PREPARED_THREADS=512
B300_STAGEI_PREPARED_CONTROL_BIN=$ONEESAN_ROOT/bin/stagei-control
B300_STAGEI_PREPARED_CONTROL_LABEL=stagei_default_w4_d2
B300_STAGEI_PREPARED_CONTROL_THREADS=512
B300_STAGEI_PREPARED_MANIFEST=$ONEESAN_ROOT/work/fake-stagei.sha256
EOT
EOF
cat >"$root/scripts/run/b300x8-nextgen-hybrid8-nextmate-staged-fullprime-race.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
rc="${FAKE_STAGEH_RC:-0}"; ((rc==0)) || exit "$rc"
ev="${SELF_EVICT:-default}"
cat >"$PREPARE_ENV" <<EOT
B300_STAGEH_PREPARED=1
B300_STAGEH_PREPARED_WIDTH=4
B300_STAGEH_PREPARED_DISTANCE=2
B300_STAGEH_PREPARED_SELF_EVICT=$ev
B300_STAGEH_PREPARED_BIN=$ONEESAN_ROOT/bin/stageh-mate
B300_STAGEH_PREPARED_LABEL=stageh_mate_w4_d2_sev$ev
B300_STAGEH_PREPARED_THREADS=512
B300_STAGEH_PREPARED_CONTROL_BIN=$ONEESAN_ROOT/bin/stageh-self
B300_STAGEH_PREPARED_CONTROL_LABEL=stageh_self_w4_d2_ev$ev
B300_STAGEH_PREPARED_CONTROL_THREADS=512
B300_STAGEH_PREPARED_SPEEDUP=1.01
B300_STAGEH_PREPARED_HIGH_S=0.5
B300_STAGEH_PREPARED_MANIFEST=$ONEESAN_ROOT/work/fake-stageh.sha256
EOT
EOF
cat >"$root/scripts/run/b300x8-race-external-forced-profiled-once.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cat >"$FAKE_CAPTURE" <<EOT
P_BIN=${FORCED_OVERRIDE_BIN:-}
B_BIN=${FORCED_BASE_BIN:-}
E1_BIN=${FORCED_EXTRA_BIN:-}
E2_BIN=${FORCED_EXTRA2_BIN:-}
E3_BIN=${FORCED_EXTRA3_BIN:-}
SMOKE_PRIME=${SMOKE_PRIME:-}
TARGET_MIB=${FORCED_TARGET_MIB:-}
SELECT_ONLY=${SELECT_ONLY:-}
EOT
EOF
chmod +x "$root/scripts/run/"*.sh
profile="$root/work/profile.env"; printf 'PROFILE=contract\n' >"$profile"; grand="$root/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
run_case(){
 local name="$1" nr="$2" hr="$3" fr="$4" ir="$5" mr="$6" mode="$7"; shift 7
 local prefix="$root/work/$name" capture="$root/work/$name.capture"
 FAKE_NEXTSELF_RC="$nr" FAKE_HYBRID_RC="$hr" FAKE_STAGEF_RC="$fr" FAKE_STAGEI_RC="$ir" FAKE_STAGEH_RC="$mr" FAKE_CAPTURE="$capture" PROFILE_FILE="$profile" PREFIX="$prefix" RUN_NEXTSELF_STAGE=1 RUN_HYBRID_STAGE=1 RUN_HYBRID_NS_STAGE=1 RUN_STAGEI=1 RUN_STAGEH=1 HYBRID_NS_WIDTH_LIST='1 2 4 8' HYBRID_NS_DISTANCE_LIST='1 2 4' bash "$grand" 27 >/dev/null 2>"$root/work/$name.err"
 local summary="${prefix}.race_grand.env"; [[ -s "$summary" && -s "$capture" ]] || { cat "$root/work/$name.err" >&2; exit 4; }; source "$summary"
 [[ "$B300_GRAND_MODE" == "$mode" ]] || { echo "$name mode=$B300_GRAND_MODE expected=$mode" >&2; exit 4; }
 [[ "$B300_GRAND_HYBRID8_NEXTSELF_SEARCH_WIDTHS" == '1 2 4 8' && "$B300_GRAND_HYBRID8_NEXTSELF_SEARCH_DISTANCES" == '1 2 4' ]] || exit 4
 if [[ "$mode" == stageh_nextmate_* ]]; then
   [[ "$B300_GRAND_STAGEH_OK" == 1 ]] || exit 4
   if ((ir==0)); then [[ "$B300_GRAND_STAGEI_OK" == 1 && "$B300_GRAND_STAGEI_HINT" == last && "$B300_GRAND_STAGEH_SELF_EVICT" == last ]] || exit 4; else [[ "$B300_GRAND_STAGEI_OK" == 0 && "$B300_GRAND_STAGEH_SELF_EVICT" == default ]] || exit 4; fi
 fi
 python3 - "$name" "$capture" "$root" "$@" <<'PY'
from pathlib import Path
import sys
name,capture,root,*exp=sys.argv[1:]; got=dict(x.split('=',1) for x in Path(capture).read_text().splitlines()); keys=('P_BIN','B_BIN','E1_BIN','E2_BIN','E3_BIN')
for k,w in zip(keys,exp):
    want=f'{root}/bin/{w}' if w else ''
    if got.get(k,'')!=want: raise SystemExit(f'{name}: {k}={got.get(k)!r} expected={want!r}')
if got.get('SELECT_ONLY')!='1' or got.get('SMOKE_PRIME')!='4294967291' or got.get('TARGET_MIB')!='65536': raise SystemExit(f'{name}: race contract lost')
print('grand_contract_case='+name+' OK')
PY
}
run_case all_composed 0 0 0 0 0 stageh_nextmate_grand stageh-mate stageh-self hybrid-base sat-nextself joint-primary
run_case stageh_rejected 0 0 0 0 4 stagei_selfevict_grand stagei-hint stagei-control hybrid-base sat-nextself joint-primary
run_case stagei_rejected 0 0 0 4 0 stageh_nextmate_grand stageh-mate stageh-self hybrid-base sat-nextself joint-primary
run_case stagei_stageh_rejected 0 0 0 4 4 hybrid8_nextself_composed_grand composed stagef-control hybrid-base sat-nextself joint-primary
run_case nextself_rejected 4 0 0 0 0 stageh_nextmate_joint stageh-mate stageh-self hybrid-base joint-primary joint-base
run_case stagef_rejected 0 0 4 0 0 nextself_hybrid8_joint sat-nextself sat-control hybrid hybrid-base joint-primary
run_case hybrid_rejected 0 4 0 0 0 nextself_joint sat-nextself sat-control joint-primary joint-base ''
run_case transforms_rejected 4 4 0 0 0 joint_fallback joint-primary joint-base '' '' ''
echo 'b300_grand_selector_contract_preflight=OK cases=8 stagei=OK stageh_on_stagei=OK stageh_default_fallback=OK stagei_fallback=OK geometry_fallback=OK candidate_env=OK select_only=OK gpu_work=0'
