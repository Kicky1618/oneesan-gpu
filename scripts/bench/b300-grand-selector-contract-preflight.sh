#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

REAL_GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
[[ -f "$REAL_GRAND" ]] || exit 2
bash -n "$REAL_GRAND"

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-grand-contract.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
root="$tmp/root"
mkdir -p "$root/scripts/run" "$root/scripts/lib" "$root/work" "$root/bin" "$root/build"
cp "$REAL_GRAND" "$root/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
chmod +x "$root/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
cat >"$root/scripts/lib/common.sh" <<'EOF'
#!/usr/bin/env bash
ONEESAN_ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ONEESAN_BUILD_DIR="$ONEESAN_ROOT/build"
export ONEESAN_ROOT ONEESAN_BUILD_DIR
EOF
make_bin(){ local name="$1"; printf '#!/usr/bin/env bash\nexit 0\n' >"$root/bin/$name"; chmod +x "$root/bin/$name"; }
for name in joint-primary joint-base sat-nextself sat-control hybrid hybrid-base composed stagef-control stagei-evict stagei-control stagej-mategeo stagej-control stagek-hint stagek-control; do make_bin "$name"; done

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
manifest="$ONEESAN_ROOT/work/fake-stagef.sha256"; printf 'fake\n' >"$manifest"
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
B300_HYBRID8_NEXTSELF_PREPARED_MANIFEST=$manifest
EOT
EOF
cat >"$root/scripts/run/b300x8-nextgen-hybrid8-selfevict-prepare.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
rc="${FAKE_STAGEI_RC:-0}"; ((rc==0)) || exit "$rc"
manifest="$ONEESAN_ROOT/work/fake-stagei.sha256"; printf 'fake\n' >"$manifest"
cat >"$PREPARE_ENV" <<EOT
B300_EVICT_PREPARED=1
B300_EVICT_HINT=last
B300_EVICT_WIDTH=4
B300_EVICT_DISTANCE=2
B300_EVICT_BIN=$ONEESAN_ROOT/bin/stagei-evict
B300_EVICT_LABEL=stagei_last_w4_d2
B300_EVICT_THREADS=512
B300_EVICT_CONTROL_BIN=$ONEESAN_ROOT/bin/stagei-control
B300_EVICT_CONTROL_LABEL=stagei_default_w4_d2
B300_EVICT_CONTROL_THREADS=512
B300_EVICT_MANIFEST=$manifest
EOT
EOF
cat >"$root/scripts/run/b300x8-nextgen-hybrid8-nextmate-geometry-stagej-staged-fullprime-race.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
rc="${FAKE_STAGEJ_RC:-0}"; ((rc==0)) || exit "$rc"
sev="${SELF_EVICT:-default}"; mev="${MATE_EVICT:-default}"
manifest="$ONEESAN_ROOT/work/fake-stagej.sha256"; printf 'fake\n' >"$manifest"
cat >"$PREPARE_ENV" <<EOT
B300_STAGEJ_PREPARED=1
B300_STAGEJ_PREPARED_MOD=${MOD:-4294967291}
B300_STAGEJ_PREPARED_SELF_WIDTH=4
B300_STAGEJ_PREPARED_SELF_DISTANCE=2
B300_STAGEJ_PREPARED_MATE_WIDTH=2
B300_STAGEJ_PREPARED_MATE_DISTANCE=1
B300_STAGEJ_PREPARED_SELF_EVICT=$sev
B300_STAGEJ_PREPARED_MATE_EVICT=$mev
B300_STAGEJ_PREPARED_BIN=$ONEESAN_ROOT/bin/stagej-mategeo
B300_STAGEJ_PREPARED_LABEL=stagej_mate_w2_d1_sev${sev}_mev${mev}
B300_STAGEJ_PREPARED_THREADS=512
B300_STAGEJ_PREPARED_CONTROL_BIN=$ONEESAN_ROOT/bin/stagej-control
B300_STAGEJ_PREPARED_CONTROL_LABEL=stagej_control_sev${sev}
B300_STAGEJ_PREPARED_CONTROL_THREADS=512
B300_STAGEJ_PREPARED_STAGED_SPEEDUP=1.010000000
B300_STAGEJ_PREPARED_MANIFEST=$manifest
EOT
EOF
cat >"$root/scripts/run/b300x8-nextgen-hybrid8-mate-evict-stagek-staged-fullprime-race.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
rc="${FAKE_STAGEK_RC:-0}"; ((rc==0)) || exit "$rc"
manifest="$ONEESAN_ROOT/work/fake-stagek.sha256"; printf 'fake\n' >"$manifest"
cat >"$PREPARE_ENV" <<EOT
B300_STAGEK_PREPARED=1
B300_STAGEK_PREPARED_MOD=${MOD:-4294967291}
B300_STAGEK_PREPARED_BIN=$ONEESAN_ROOT/bin/stagek-hint
B300_STAGEK_PREPARED_LABEL=stagek_mateevict_normal
B300_STAGEK_PREPARED_THREADS=512
B300_STAGEK_PREPARED_CONTROL_BIN=$ONEESAN_ROOT/bin/stagek-control
B300_STAGEK_PREPARED_CONTROL_LABEL=stagek_mateevict_default_control
B300_STAGEK_PREPARED_CONTROL_THREADS=512
B300_STAGEK_PREPARED_SELF_WIDTH=4
B300_STAGEK_PREPARED_SELF_DISTANCE=2
B300_STAGEK_PREPARED_SELF_EVICT=${FAKE_STAGEK_SELF_EVICT:-last}
B300_STAGEK_PREPARED_MATE_WIDTH=2
B300_STAGEK_PREPARED_MATE_DISTANCE=1
B300_STAGEK_PREPARED_BASE_MATE_EVICT=default
B300_STAGEK_PREPARED_MATE_EVICT=normal
B300_STAGEK_PREPARED_STAGED_SPEEDUP=1.007000000
B300_STAGEK_PREPARED_MANIFEST=$manifest
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
WORK_ROOT=${WORK_ROOT:-}
SELECT_ONLY=${SELECT_ONLY:-}
EOT
EOF
chmod +x "$root/scripts/run/"*.sh

profile="$root/work/profile.env"; printf 'PROFILE=contract\n' >"$profile"
grand="$root/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
run_case(){
  local name="$1" nr="$2" hr="$3" fr="$4" ir="$5" jr="$6" kr="$7" mode="$8"; shift 8
  local prefix="$root/work/$name" capture="$root/work/$name.capture" workroot="$root/work/custom-$name"
  mkdir -p "$workroot"
  FAKE_NEXTSELF_RC="$nr" FAKE_HYBRID_RC="$hr" FAKE_STAGEF_RC="$fr" FAKE_STAGEI_RC="$ir" FAKE_STAGEJ_RC="$jr" FAKE_STAGEK_RC="$kr" FAKE_CAPTURE="$capture" \
    PROFILE_FILE="$profile" WORK_ROOT="$workroot" PREFIX="$prefix" RUN_NEXTSELF_STAGE=1 RUN_HYBRID_STAGE=1 RUN_HYBRID_NS_STAGE=1 RUN_STAGEI=1 RUN_STAGEJ=1 RUN_STAGEK=1 \
    HYBRID_NS_WIDTH_LIST='1 2 4 8' HYBRID_NS_DISTANCE_LIST='1 2 4' MATE_WIDTH_LIST='1 2 4 8' MATE_DISTANCE_LIST='1 2 4' MATE_EVICT=default MATE_EVICT_LIST='default normal last' \
    bash "$grand" 27 >/dev/null 2>"$root/work/$name.err"
  local summary="${prefix}.race_grand.env"
  [[ -s "$summary" && -s "$capture" ]] || { echo "contract case failed before capture: $name" >&2; cat "$root/work/$name.err" >&2; exit 4; }
  # shellcheck disable=SC1090
  source "$summary"
  [[ "$B300_GRAND_MODE" == "$mode" ]] || { echo "$name mode=$B300_GRAND_MODE expected=$mode" >&2; exit 4; }
  [[ "$B300_GRAND_STAGEI_NAMESPACE_ISOLATED" == 1 && "$B300_GRAND_STAGEJ_INTEGRATED" == 1 && "$B300_GRAND_STAGEK_INTEGRATED" == 1 && "$B300_GRAND_COMPLETE_PRIME_RACES" == 1 ]] || exit 4
  [[ "$B300_GRAND_WORK_ROOT" == "$workroot" ]] || { echo "$name work_root provenance lost" >&2; exit 4; }
  if [[ "$mode" == stagek_mateevict_* ]]; then
    [[ "$B300_GRAND_STAGEK_OK" == 1 && "$B300_GRAND_STAGEK_BASE_MATE_EVICT" == default && "$B300_GRAND_STAGEK_MATE_EVICT" == normal && "$B300_GRAND_STAGEK_STAGED_SPEEDUP" == 1.007000000 ]] || exit 4
  fi
  if [[ "$mode" == stagej_mategeo_* || "$mode" == stagek_mateevict_* ]]; then
    [[ "$B300_GRAND_STAGEJ_OK" == 1 && "$B300_GRAND_STAGEJ_SELF_WIDTH" == 4 && "$B300_GRAND_STAGEJ_SELF_DISTANCE" == 2 && "$B300_GRAND_STAGEJ_MATE_WIDTH" == 2 && "$B300_GRAND_STAGEJ_MATE_DISTANCE" == 1 ]] || exit 4
    if ((ir==0)); then [[ "$B300_GRAND_STAGEI_OK" == 1 && "$B300_GRAND_STAGEI_HINT" == last && "$B300_GRAND_STAGEJ_SELF_EVICT" == last ]] || exit 4; else [[ "$B300_GRAND_STAGEI_OK" == 0 && "$B300_GRAND_STAGEJ_SELF_EVICT" == default ]] || exit 4; fi
  fi
  python3 - "$name" "$capture" "$root" "$workroot" "$@" <<'PY'
from pathlib import Path
import sys
name,capture,root,workroot,*exp=sys.argv[1:]
got=dict(x.split('=',1) for x in Path(capture).read_text().splitlines())
for k,w in zip(('P_BIN','B_BIN','E1_BIN','E2_BIN','E3_BIN'),exp):
    want=f'{root}/bin/{w}' if w else ''
    if got.get(k,'') != want: raise SystemExit(f'{name}: {k}={got.get(k)!r} expected={want!r}')
if got.get('SELECT_ONLY')!='1' or got.get('SMOKE_PRIME')!='4294967291' or got.get('TARGET_MIB')!='65536' or got.get('WORK_ROOT')!=workroot:
    raise SystemExit(f'{name}: race contract lost')
print('grand_contract_case='+name+' OK')
PY
}

run_case all_refinements         0 0 0 0 0 0 stagek_mateevict_grand            stagek-hint stagek-control hybrid-base sat-nextself joint-primary
run_case stagek_rejected         0 0 0 0 0 4 stagej_mategeo_grand              stagej-mategeo stagej-control hybrid-base sat-nextself joint-primary
run_case stagej_rejected         0 0 0 0 4 0 stagei_selfevict_grand            stagei-evict stagei-control hybrid-base sat-nextself joint-primary
run_case stagei_rejected         0 0 0 4 0 0 stagek_mateevict_grand            stagek-hint stagek-control hybrid-base sat-nextself joint-primary
run_case stagei_k_rejected       0 0 0 4 0 4 stagej_mategeo_grand              stagej-mategeo stagej-control hybrid-base sat-nextself joint-primary
run_case all_late_rejected       0 0 0 4 4 4 hybrid8_nextself_composed_grand   composed stagef-control hybrid-base sat-nextself joint-primary
run_case nextself_rejected       4 0 0 0 0 0 stagek_mateevict_joint            stagek-hint stagek-control hybrid-base joint-primary joint-base
run_case stagef_rejected         0 0 4 0 0 0 nextself_hybrid8_joint            sat-nextself sat-control hybrid hybrid-base joint-primary
run_case hybrid_rejected         0 4 0 0 0 0 nextself_joint                    sat-nextself sat-control joint-primary joint-base ''
run_case transforms_rejected     4 4 0 0 0 0 joint_fallback                    joint-primary joint-base '' '' ''

echo 'b300_grand_selector_contract_preflight=OK cases=10 stagei_namespace=OK stagej_geometry=OK stagek_mate_evict=OK stagek_fallback=OK work_root=OK complete_prime_races=1 candidate_env=OK select_only=OK gpu_work=0'