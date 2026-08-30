#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

if len(sys.argv) != 3:
    raise SystemExit('usage: gen-b300-grand-selector-stagem.py INPUT.sh OUTPUT.sh')
src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
s = src.read_text()


def rep(old: str, new: str, label: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f'{label}: expected one anchor, got {n}')
    s = s.replace(old, new, 1)


# Generated selectors live under the build tree, not scripts/run, so bind the
# common helper through the exported repository root supplied by the wrapper.
rep(
    'source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"',
    'source "${ONEESAN_ROOT:?ONEESAN_ROOT must be exported}/scripts/lib/common.sh"',
    'common source',
)

anchor = 'STAGEL_RACE_PREFIX="${STAGEL_RACE_PREFIX:-${PREFIX}.stagel-guard.promote}"\n'
rep(anchor, anchor + '''STAGEM_PREFIX="${STAGEM_PREFIX:-${PREFIX}.stagem-mateload}"
STAGEM_WINNER_ENV="${STAGEM_WINNER_ENV:-${STAGEM_PREFIX}_winner.env}"
STAGEM_PREPARE_ENV="${STAGEM_PREPARE_ENV:-${PREFIX}.stagem-mateload.prepared.env}"
STAGEM_RACE_PREFIX="${STAGEM_RACE_PREFIX:-${PREFIX}.stagem-mateload.promote}"
''', 'Stage-M paths')

anchor = 'RUN_STAGEL="${RUN_STAGEL:-1}"\n'
rep(anchor, anchor + 'RUN_STAGEM="${RUN_STAGEM:-1}"\n', 'RUN_STAGEM')
anchor = 'STAGEL_GUARD_LIST="${STAGEL_GUARD_LIST:-bb pb bp pp}"\n'
rep(anchor, anchor + '''STAGEM_MIN_SPEEDUP="${STAGEM_MIN_SPEEDUP:-1.002}"
STAGEM_POLICY_LIST="${STAGEM_POLICY_LIST:-default cg cs}"
''', 'Stage-M knobs')

rep(
    'for x in SELECT_ONLY REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL; do',
    'for x in SELECT_ONLY REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM; do',
    'boolean validation',
)
rep(
    'python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY',
    'python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY',
    'speedup validation',
)

anchor = 'STAGEL_GUARD_LIST="$(normalize_guards "$STAGEL_GUARD_LIST")"\n'
rep(anchor, anchor + '''normalize_mate_load_policies(){
  local raw="$1"; local out=() p old seen
  for p in $raw; do case "$p" in default|cg|cs) ;; *) echo "bad mate-load policy=$p" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$p" ]] && seen=1; done; ((seen)) || out+=("$p"); done
  ((${#out[@]})) || { echo 'mate-load policy list must not be empty' >&2; exit 2; }; printf '%s' "${out[*]}"
}
STAGEM_POLICY_LIST="$(normalize_mate_load_policies "$STAGEM_POLICY_LIST")"
case " $STAGEM_POLICY_LIST " in *' default '*) ;; *) echo 'STAGEM_POLICY_LIST must include default baseline' >&2; exit 2;; esac
''', 'policy normalization')

rep(
    '  "$(dirname "$STAGEK_PREPARE_ENV")" "$(dirname "$STAGEL_PREPARE_ENV")" "$(dirname "$RACE_PREFIX")" "$WORK_ROOT"',
    '  "$(dirname "$STAGEK_PREPARE_ENV")" "$(dirname "$STAGEL_PREPARE_ENV")" "$(dirname "$STAGEM_PREPARE_ENV")" "$(dirname "$RACE_PREFIX")" "$WORK_ROOT"',
    'mkdir Stage M',
)

stage_m = r'''
STAGEM_OK=0; STAGEM_RC=0
if ((STAGEL_OK)); then
  set +e
  PROFILE_FILE="$PROFILE_FILE" STAGE_F_ENV="$HYBRID_NS_WINNER_ENV" STAGEL_WINNER_ENV="$STAGEL_WINNER_ENV" STAGEL_PREPARE_ENV="$STAGEL_PREPARE_ENV" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$PRIME" NGPU=8 RUN_STAGED="$RUN_STAGEM" PREPARE_ONLY=1 MIN_SPEEDUP="$STAGEM_MIN_SPEEDUP" POLICY_LIST="$STAGEM_POLICY_LIST" STAGED_PREFIX="$STAGEM_PREFIX" WINNER_ENV="$STAGEM_WINNER_ENV" RACE_PREFIX="$STAGEM_RACE_PREFIX" PREPARE_ENV="$STAGEM_PREPARE_ENV" bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-mate-load-stagem-staged-fullprime-race.sh" 27
  STAGEM_RC=$?; set -e
  if ((STAGEM_RC==0)); then
    source "$STAGEM_PREPARE_ENV"
    [[ "${B300_STAGEM_PREPARED:-0}" == 1 && -x "$B300_STAGEM_PREPARED_BIN" && -x "$B300_STAGEM_PREPARED_CONTROL_BIN" ]] || exit 3
    [[ "$B300_STAGEM_PREPARED_MOD" == "$PRIME" && "$B300_STAGEM_PREPARED_NGPU" == 8 ]] || { echo 'Stage-M modulus/GPU drift' >&2; exit 3; }
    [[ "$B300_STAGEM_PREPARED_CONTROL_BIN" == "$B300_STAGEL_PREPARED_BIN" ]] || { echo 'Stage-M control is not Stage-L winner' >&2; exit 3; }
    [[ "$B300_STAGEM_PREPARED_SELF_GUARD" == "$B300_STAGEL_PREPARED_SELF_GUARD" && "$B300_STAGEM_PREPARED_MATE_GUARD" == "$B300_STAGEL_PREPARED_MATE_GUARD" ]] || { echo 'Stage-M guard drift' >&2; exit 3; }
    case "$B300_STAGEM_PREPARED_POLICY" in cg|cs) ;; *) exit 3;; esac
    STAGEM_OK=1
  elif ((STAGEM_RC==4)); then
    echo 'grand selector: Stage-M mate-load policy rejected; retaining Stage L' >&2
  else
    exit "$STAGEM_RC"
  fi
fi
'''
rep('\nP_BIN=""; P_LABEL=""; P_THREADS=256\n', '\n' + stage_m + '\nP_BIN=""; P_LABEL=""; P_THREADS=256\n', 'Stage-M preparation insertion')

branches = r'''if ((STAGEM_OK && NEXTSELF_OK)); then
  MODE=stagem_mateload_grand
  P_BIN="$B300_STAGEM_PREPARED_BIN"; P_LABEL="$B300_STAGEM_PREPARED_LABEL"; P_THREADS="$B300_STAGEM_PREPARED_THREADS"
  B_BIN="$B300_STAGEM_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGEM_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGEM_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$B300_NEXTSELF_PREPARED_BIN"; E2_LABEL="$B300_NEXTSELF_PREPARED_LABEL"; E2_THREADS="$B300_NEXTSELF_PREPARED_THREADS"
  E3_BIN="$JOINT_PRIMARY_BIN"; E3_LABEL="$JOINT_PRIMARY_LABEL"; E3_THREADS="$JOINT_PRIMARY_THREADS"
elif ((STAGEM_OK)); then
  MODE=stagem_mateload_joint
  P_BIN="$B300_STAGEM_PREPARED_BIN"; P_LABEL="$B300_STAGEM_PREPARED_LABEL"; P_THREADS="$B300_STAGEM_PREPARED_THREADS"
  B_BIN="$B300_STAGEM_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGEM_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGEM_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$JOINT_PRIMARY_BIN"; E2_LABEL="$JOINT_PRIMARY_LABEL"; E2_THREADS="$JOINT_PRIMARY_THREADS"
  if [[ -n "$JOINT_BASE_BIN" && "$JOINT_BASE_BIN" != "$JOINT_PRIMARY_BIN" ]]; then E3_BIN="$JOINT_BASE_BIN"; E3_LABEL="$JOINT_BASE_LABEL"; E3_THREADS="$JOINT_BASE_THREADS"; fi
elif ((STAGEL_OK && NEXTSELF_OK)); then'''
rep('if ((STAGEL_OK && NEXTSELF_OK)); then', branches, 'Stage-M branch pair')

rep(
    'stagej_mategeo_grand|stagek_mateevict_grand|stagel_guard_grand)',
    'stagej_mategeo_grand|stagek_mateevict_grand|stagel_guard_grand|stagem_mateload_grand)',
    'drop joint base mode',
)
# The string above appears twice, once in each case; replacement must cover both.
# Restore the second occurrence explicitly if it remains.
if 'stagej_mategeo_grand|stagek_mateevict_grand|stagel_guard_grand)' in s:
    s = s.replace('stagej_mategeo_grand|stagek_mateevict_grand|stagel_guard_grand)', 'stagej_mategeo_grand|stagek_mateevict_grand|stagel_guard_grand|stagem_mateload_grand)', 1)

anchor = "  printf 'B300_GRAND_STAGEL_OK=%q\\n' \"$STAGEL_OK\"; printf 'B300_GRAND_STAGEL_MIN_SPEEDUP=%q\\n' \"$STAGEL_MIN_SPEEDUP\"; printf 'B300_GRAND_STAGEL_UPSTREAM_KIND=%q\\n' \"$STAGEL_UPSTREAM_KIND\"; printf 'B300_GRAND_STAGEL_PROFILE=%q\\n' \"${B300_STAGEL_PREPARED_PROFILE:-bb}\"; printf 'B300_GRAND_STAGEL_SELF_GUARD=%q\\n' \"${B300_STAGEL_PREPARED_SELF_GUARD:-branch}\"; printf 'B300_GRAND_STAGEL_MATE_GUARD=%q\\n' \"${B300_STAGEL_PREPARED_MATE_GUARD:-branch}\"; printf 'B300_GRAND_STAGEL_STAGED_SPEEDUP=%q\\n' \"${B300_STAGEL_PREPARED_STAGED_SPEEDUP:-1.0}\"; printf 'B300_GRAND_STAGEL_MANIFEST=%q\\n' \"${B300_STAGEL_PREPARED_MANIFEST:-}\"; printf 'B300_GRAND_STAGEL_SEARCH_PROFILES=%q\\n' \"$STAGEL_GUARD_LIST\"\n"
rep(anchor, anchor + "  printf 'B300_GRAND_STAGEM_OK=%q\\n' \"$STAGEM_OK\"; printf 'B300_GRAND_STAGEM_MIN_SPEEDUP=%q\\n' \"$STAGEM_MIN_SPEEDUP\"; printf 'B300_GRAND_STAGEM_POLICY=%q\\n' \"${B300_STAGEM_PREPARED_POLICY:-default}\"; printf 'B300_GRAND_STAGEM_STAGED_SPEEDUP=%q\\n' \"${B300_STAGEM_PREPARED_STAGED_SPEEDUP:-1.0}\"; printf 'B300_GRAND_STAGEM_MANIFEST=%q\\n' \"${B300_STAGEM_PREPARED_MANIFEST:-}\"; printf 'B300_GRAND_STAGEM_SEARCH_POLICIES=%q\\n' \"$STAGEM_POLICY_LIST\"\n", 'Stage-M summary')

rep(
    "  printf 'B300_GRAND_STAGEJ_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEK_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEL_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1\\n'; printf 'B300_GRAND_COMPLETE_PRIME_RACES=1\\n'",
    "  printf 'B300_GRAND_STAGEJ_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEK_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEL_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEM_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1\\n'; printf 'B300_GRAND_COMPLETE_PRIME_RACES=1\\n'",
    'Stage-M integrated marker',
)

for marker in (
    'RUN_STAGEM=', 'STAGEM_MIN_SPEEDUP=', 'STAGEM_POLICY_LIST=',
    'b300x8-nextgen-hybrid8-mate-load-stagem-staged-fullprime-race.sh',
    'MODE=stagem_mateload_grand', 'MODE=stagem_mateload_joint',
    'B300_GRAND_STAGEM_OK=', 'B300_GRAND_STAGEM_POLICY=',
    'B300_GRAND_STAGEM_INTEGRATED=1', 'B300_GRAND_COMPLETE_PRIME_RACES=1',
):
    if marker not in s:
        raise SystemExit(f'missing generated Stage-M artifact: {marker}')
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('generated selector must retain exactly one complete-prime race')

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(s)
print(f'generated {out} from {src}: b300_grand_stagem_integrated=1 stage_m_after_l=1 prepare_only=1 fallback_l=1 single_complete_prime=1')
