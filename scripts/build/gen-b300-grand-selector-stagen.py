#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

if len(sys.argv) != 3:
    raise SystemExit('usage: gen-b300-grand-selector-stagen.py INPUT.sh OUTPUT.sh')
src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
s = src.read_text()


def rep(old: str, new: str, label: str) -> None:
    global s
    if new in s:
        return
    n = s.count(old)
    if n != 1:
        raise SystemExit(f'{label}: expected one anchor, got {n}')
    s = s.replace(old, new, 1)


# Generated selectors live in the build tree.
rep(
    'source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"',
    'source "${ONEESAN_ROOT:?ONEESAN_ROOT must be exported}/scripts/lib/common.sh"',
    'common source',
)

anchor = 'STAGEM_RACE_PREFIX="${STAGEM_RACE_PREFIX:-${PREFIX}.stagem-mateload.promote}"\n'
rep(anchor, anchor + '''STAGEN_PREFIX="${STAGEN_PREFIX:-${PREFIX}.stagen-pairblock}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-${STAGEN_PREFIX}_winner.env}"
STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-${PREFIX}.stagen-pairblock.prepared.env}"
STAGEN_RACE_PREFIX="${STAGEN_RACE_PREFIX:-${PREFIX}.stagen-pairblock.promote}"
''', 'Stage-N paths')

anchor = 'RUN_STAGEM="${RUN_STAGEM:-1}"\n'
rep(anchor, anchor + 'RUN_STAGEN="${RUN_STAGEN:-1}"\n', 'RUN_STAGEN')
anchor = 'STAGEM_POLICY_LIST="${STAGEM_POLICY_LIST:-default cg cs}"\n'
rep(anchor, anchor + '''STAGEN_MIN_SPEEDUP="${STAGEN_MIN_SPEEDUP:-1.002}"
STAGEN_PAIR_POLICY_LIST="${STAGEN_PAIR_POLICY_LIST:-default cg cs}"
STAGEN_BLOCK_POLICY_LIST="${STAGEN_BLOCK_POLICY_LIST:-default cg cs}"
''', 'Stage-N knobs')

rep(
    'for x in SELECT_ONLY REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM; do',
    'for x in SELECT_ONLY REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN; do',
    'boolean validation',
)
rep(
    'python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY',
    'python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY',
    'speedup validation',
)

anchor = 'STAGEM_POLICY_LIST="$(normalize_policies "$STAGEM_POLICY_LIST")"\n'
rep(anchor, anchor + '''STAGEN_PAIR_POLICY_LIST="$(normalize_policies "$STAGEN_PAIR_POLICY_LIST")"
STAGEN_BLOCK_POLICY_LIST="$(normalize_policies "$STAGEN_BLOCK_POLICY_LIST")"
''', 'Stage-N policy normalization')

rep(
    '  "$(dirname "$STAGEK_PREPARE_ENV")" "$(dirname "$STAGEL_PREPARE_ENV")" "$(dirname "$STAGEM_PREPARE_ENV")" \\\n  "$(dirname "$RACE_PREFIX")" "$WORK_ROOT"',
    '  "$(dirname "$STAGEK_PREPARE_ENV")" "$(dirname "$STAGEL_PREPARE_ENV")" "$(dirname "$STAGEM_PREPARE_ENV")" "$(dirname "$STAGEN_PREPARE_ENV")" \\\n  "$(dirname "$RACE_PREFIX")" "$WORK_ROOT"',
    'mkdir Stage N',
)

stage_n = r'''
STAGEN_OK=0; STAGEN_RC=0; STAGEN_UPSTREAM_KIND=""
if ((STAGEM_OK)); then STAGEN_UPSTREAM_KIND=stagem
elif ((STAGEL_OK)); then STAGEN_UPSTREAM_KIND=stagel
fi
if [[ -n "$STAGEN_UPSTREAM_KIND" ]]; then
  set +e
  PROFILE_FILE="$PROFILE_FILE" STAGE_F_ENV="$HYBRID_NS_WINNER_ENV" \
    STAGEL_WINNER_ENV="$STAGEL_WINNER_ENV" STAGEL_PREPARE_ENV="$STAGEL_PREPARE_ENV" \
    STAGEM_WINNER_ENV="$STAGEM_WINNER_ENV" STAGEM_PREPARE_ENV="$STAGEM_PREPARE_ENV" \
    UPSTREAM_KIND="$STAGEN_UPSTREAM_KIND" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$PRIME" NGPU=8 \
    RUN_STAGED="$RUN_STAGEN" PREPARE_ONLY=1 MIN_SPEEDUP="$STAGEN_MIN_SPEEDUP" \
    PAIR_POLICY_LIST="$STAGEN_PAIR_POLICY_LIST" BLOCK_POLICY_LIST="$STAGEN_BLOCK_POLICY_LIST" \
    STAGED_PREFIX="$STAGEN_PREFIX" WINNER_ENV="$STAGEN_WINNER_ENV" RACE_PREFIX="$STAGEN_RACE_PREFIX" PREPARE_ENV="$STAGEN_PREPARE_ENV" \
    bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-pair-block-load-stagen-staged-fullprime-race.sh" 27
  STAGEN_RC=$?; set -e
  if ((STAGEN_RC==0)); then
    source "$STAGEN_PREPARE_ENV"
    [[ "${B300_STAGEN_PREPARED:-0}" == 1 && -x "$B300_STAGEN_PREPARED_BIN" && -x "$B300_STAGEN_PREPARED_CONTROL_BIN" ]] || exit 3
    [[ "$B300_STAGEN_PREPARED_MOD" == "$PRIME" && "$B300_STAGEN_PREPARED_NGPU" == 8 ]] || { echo 'Stage-N modulus/GPU drift' >&2; exit 3; }
    [[ "$B300_STAGEN_PREPARED_UPSTREAM_KIND" == "$STAGEN_UPSTREAM_KIND" ]] || { echo 'Stage-N upstream kind drift' >&2; exit 3; }
    if [[ "$STAGEN_UPSTREAM_KIND" == stagem ]]; then
      [[ "$B300_STAGEN_PREPARED_CONTROL_BIN" == "$B300_STAGEM_PREPARED_BIN" ]] || { echo 'Stage-N control is not exact Stage-M winner' >&2; exit 3; }
    else
      [[ "$B300_STAGEN_PREPARED_CONTROL_BIN" == "$B300_STAGEL_PREPARED_BIN" ]] || { echo 'Stage-N control is not exact Stage-L winner' >&2; exit 3; }
    fi
    for p in "$B300_STAGEN_PREPARED_PAIR_POLICY" "$B300_STAGEN_PREPARED_BLOCK_POLICY" "$B300_STAGEN_PREPARED_BASE_COUNT_POLICY"; do case "$p" in default|cg|cs) ;; *) exit 3;; esac; done
    [[ "$B300_STAGEN_PREPARED_PAIR_POLICY" != "$B300_STAGEN_PREPARED_BASE_COUNT_POLICY" || "$B300_STAGEN_PREPARED_BLOCK_POLICY" != "$B300_STAGEN_PREPARED_BASE_COUNT_POLICY" ]] || { echo 'Stage-N returned inherited baseline as winner' >&2; exit 3; }
    STAGEN_OK=1
  elif ((STAGEN_RC==4)); then
    echo 'grand selector: Stage-N pair/block load policy rejected; retaining Stage M/L' >&2
  else
    exit "$STAGEN_RC"
  fi
fi
'''
rep('\nP_BIN=""; P_LABEL=""; P_THREADS=256\n', '\n' + stage_n + '\nP_BIN=""; P_LABEL=""; P_THREADS=256\n', 'Stage-N preparation insertion')

branches = r'''if ((STAGEN_OK && NEXTSELF_OK)); then
  MODE=stagen_pairblock_grand
  P_BIN="$B300_STAGEN_PREPARED_BIN"; P_LABEL="$B300_STAGEN_PREPARED_LABEL"; P_THREADS="$B300_STAGEN_PREPARED_THREADS"
  B_BIN="$B300_STAGEN_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGEN_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGEN_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$B300_NEXTSELF_PREPARED_BIN"; E2_LABEL="$B300_NEXTSELF_PREPARED_LABEL"; E2_THREADS="$B300_NEXTSELF_PREPARED_THREADS"
  E3_BIN="$JOINT_PRIMARY_BIN"; E3_LABEL="$JOINT_PRIMARY_LABEL"; E3_THREADS="$JOINT_PRIMARY_THREADS"
elif ((STAGEN_OK)); then
  MODE=stagen_pairblock_joint
  P_BIN="$B300_STAGEN_PREPARED_BIN"; P_LABEL="$B300_STAGEN_PREPARED_LABEL"; P_THREADS="$B300_STAGEN_PREPARED_THREADS"
  B_BIN="$B300_STAGEN_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGEN_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGEN_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$JOINT_PRIMARY_BIN"; E2_LABEL="$JOINT_PRIMARY_LABEL"; E2_THREADS="$JOINT_PRIMARY_THREADS"
  if [[ -n "$JOINT_BASE_BIN" && "$JOINT_BASE_BIN" != "$JOINT_PRIMARY_BIN" ]]; then E3_BIN="$JOINT_BASE_BIN"; E3_LABEL="$JOINT_BASE_LABEL"; E3_THREADS="$JOINT_BASE_THREADS"; fi
elif ((STAGEM_OK && NEXTSELF_OK)); then'''
rep('if ((STAGEM_OK && NEXTSELF_OK)); then', branches, 'Stage-N branch pair')

suffix = 'stagek_mateevict_grand|stagel_guard_grand|stagem_mateload_grand)'
if s.count(suffix) != 2:
    raise SystemExit(f'drop-mode suffix: expected two anchors, got {s.count(suffix)}')
s = s.replace(suffix, 'stagek_mateevict_grand|stagel_guard_grand|stagem_mateload_grand|stagen_pairblock_grand)', 2)

anchor = "  printf 'B300_GRAND_STAGEM_OK=%q\\n' \"$STAGEM_OK\"; printf 'B300_GRAND_STAGEM_MIN_SPEEDUP=%q\\n' \"$STAGEM_MIN_SPEEDUP\"; printf 'B300_GRAND_STAGEM_POLICY=%q\\n' \"${B300_STAGEM_PREPARED_POLICY:-default}\"; printf 'B300_GRAND_STAGEM_STAGED_SPEEDUP=%q\\n' \"${B300_STAGEM_PREPARED_STAGED_SPEEDUP:-1.0}\"; printf 'B300_GRAND_STAGEM_MANIFEST=%q\\n' \"${B300_STAGEM_PREPARED_MANIFEST:-}\"; printf 'B300_GRAND_STAGEM_SEARCH_POLICIES=%q\\n' \"$STAGEM_POLICY_LIST\"\n"
rep(anchor, anchor + "  printf 'B300_GRAND_STAGEN_OK=%q\\n' \"$STAGEN_OK\"; printf 'B300_GRAND_STAGEN_MIN_SPEEDUP=%q\\n' \"$STAGEN_MIN_SPEEDUP\"; printf 'B300_GRAND_STAGEN_UPSTREAM_KIND=%q\\n' \"$STAGEN_UPSTREAM_KIND\"; printf 'B300_GRAND_STAGEN_PAIR_POLICY=%q\\n' \"${B300_STAGEN_PREPARED_PAIR_POLICY:-default}\"; printf 'B300_GRAND_STAGEN_BLOCK_POLICY=%q\\n' \"${B300_STAGEN_PREPARED_BLOCK_POLICY:-default}\"; printf 'B300_GRAND_STAGEN_BASE_COUNT_POLICY=%q\\n' \"${B300_STAGEN_PREPARED_BASE_COUNT_POLICY:-default}\"; printf 'B300_GRAND_STAGEN_STAGED_SPEEDUP=%q\\n' \"${B300_STAGEN_PREPARED_STAGED_SPEEDUP:-1.0}\"; printf 'B300_GRAND_STAGEN_MANIFEST=%q\\n' \"${B300_STAGEN_PREPARED_MANIFEST:-}\"; printf 'B300_GRAND_STAGEN_SEARCH_PAIR_POLICIES=%q\\n' \"$STAGEN_PAIR_POLICY_LIST\"; printf 'B300_GRAND_STAGEN_SEARCH_BLOCK_POLICIES=%q\\n' \"$STAGEN_BLOCK_POLICY_LIST\"\n", 'Stage-N summary')

rep(
    "  printf 'B300_GRAND_STAGEJ_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEK_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEL_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEM_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1\\n'; printf 'B300_GRAND_COMPLETE_PRIME_RACES=1\\n'",
    "  printf 'B300_GRAND_STAGEJ_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEK_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEL_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEM_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEN_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1\\n'; printf 'B300_GRAND_COMPLETE_PRIME_RACES=1\\n'",
    'Stage-N integrated marker',
)

for marker in (
    'RUN_STAGEN=', 'STAGEN_MIN_SPEEDUP=', 'STAGEN_PAIR_POLICY_LIST=', 'STAGEN_BLOCK_POLICY_LIST=',
    'b300x8-nextgen-hybrid8-pair-block-load-stagen-staged-fullprime-race.sh',
    'MODE=stagen_pairblock_grand', 'MODE=stagen_pairblock_joint',
    'B300_GRAND_STAGEN_OK=', 'B300_GRAND_STAGEN_PAIR_POLICY=', 'B300_GRAND_STAGEN_BLOCK_POLICY=',
    'B300_GRAND_STAGEN_INTEGRATED=1', 'B300_GRAND_COMPLETE_PRIME_RACES=1',
):
    if marker not in s:
        raise SystemExit(f'missing generated Stage-N artifact: {marker}')
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('generated selector must retain exactly one complete-prime race')

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(s)
print(f'generated {out} from {src}: b300_grand_stagen_integrated=1 stage_n_after_m_l=1 prepare_only=1 fallback_ml=1 single_complete_prime=1')
