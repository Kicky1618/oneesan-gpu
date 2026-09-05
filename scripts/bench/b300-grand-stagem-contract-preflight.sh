#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BASE_GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
GRAND_WRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagem.sh"
GEN_GRAND="$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-stagem.py"
BASE_FIRST="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh"
FIRST_WRAP="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-stagem.sh"
GEN_FIRST="$ONEESAN_ROOT/scripts/build/gen-b300-grand-firstpass-stagem.py"
PROMOTE_M="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-mate-load-stagem-staged-fullprime-race.sh"
PRE_M="$ONEESAN_ROOT/scripts/bench/b300-stagem-preflight.sh"
HW_GUARD="$ONEESAN_ROOT/scripts/run/b300x8-require-b300-inventory.sh"
for f in "$BASE_GRAND" "$GRAND_WRAP" "$GEN_GRAND" "$BASE_FIRST" "$FIRST_WRAP" "$GEN_FIRST" "$PROMOTE_M" "$PRE_M" "$HW_GUARD"; do
  [[ -f "$f" ]] || { echo "missing Stage-M grand dependency=$f" >&2; exit 2; }
  case "$f" in *.py) python3 -m py_compile "$f";; *) bash -n "$f";; esac
done
bash "$PRE_M"

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stagem-grand.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

# Hardware guard is GPU-free testable: a fake nvidia-smi supplies inventory
# rows, proving the canonical first-pass accepts only exactly eight B300s.
mkdir -p "$tmp/fakebin"
cat >"$tmp/fakebin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ "$*" == '--query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader' ]] || exit 90
case "${FAKE_GPU_CASE:-}" in
  b300x8) count=8; bad=-1; bad_name='' ;;
  b300x7) count=7; bad=-1; bad_name='' ;;
  b300x9) count=9; bad=-1; bad_name='' ;;
  mixed) count=8; bad=7; bad_name='NVIDIA H200' ;;
  gb300) count=8; bad=7; bad_name='NVIDIA GB300' ;;
  *) exit 91 ;;
esac
for ((i=0; i<count; i++)); do
  name='NVIDIA B300'
  (( i == bad )) && name="$bad_name"
  printf '%d, %s, GPU-%012d, 280000 MiB, 590.00\n' "$i" "$name" "$i"
done
SH
chmod +x "$tmp/fakebin/nvidia-smi"

guard_case(){
  local case_name="$1" expect="$2" needle="$3" rc
  set +e
  PATH="$tmp/fakebin:$PATH" FAKE_GPU_CASE="$case_name" bash "$HW_GUARD" >"$tmp/$case_name.out" 2>"$tmp/$case_name.err"
  rc=$?
  set -e
  if [[ "$expect" == pass ]]; then
    (( rc == 0 )) || { cat "$tmp/$case_name.err" >&2; echo "hardware guard case unexpectedly failed: $case_name" >&2; exit 3; }
    [[ "$(wc -l <"$tmp/$case_name.out")" == 8 ]] || { echo "hardware guard case emitted wrong row count: $case_name" >&2; exit 3; }
  else
    (( rc != 0 )) || { echo "hardware guard case unexpectedly passed: $case_name" >&2; exit 3; }
    grep -Fq "$needle" "$tmp/$case_name.err" || { cat "$tmp/$case_name.err" >&2; echo "hardware guard rejection marker missing: $case_name" >&2; exit 3; }
  fi
}
guard_case b300x8 pass ''
guard_case b300x7 fail 'need exactly 8 visible GPUs; got 7'
guard_case b300x9 fail 'need exactly 8 visible GPUs; got 9'
guard_case mixed fail 'is not NVIDIA B300: name=NVIDIA H200'
guard_case gb300 fail 'is not NVIDIA B300: name=NVIDIA GB300'

# The canonical grand selector now carries Stage M directly. Keep the generated
# selector path only as a backwards-compatibility proof for an older L-only base;
# never patch an already integrated selector a second time.
if grep -Fq 'B300_GRAND_STAGEM_INTEGRATED=1' "$BASE_GRAND"; then
  GRAND="$BASE_GRAND"
  GRAND_SOURCE=canonical
else
  STAGEM_SELECTOR_BUILD_DIR="$tmp/selector" PATCH_ONLY=1 bash "$GRAND_WRAP" 27 >"$tmp/selector.env"
  # shellcheck disable=SC1090
  source "$tmp/selector.env"
  [[ "${B300_STAGEM_SELECTOR_PATCHED:-0}" == 1 && -s "$B300_STAGEM_SELECTOR_GENERATED" ]] || exit 3
  GRAND="$B300_STAGEM_SELECTOR_GENERATED"
  GRAND_SOURCE=generated
fi
bash -n "$GRAND"

# First-pass may still be on the compatibility generator until its canonical
# entrypoint absorbs L/M. This branch is intentionally independent of GRAND_SOURCE.
if grep -Fq 'B300_GRAND_SELECTED_STAGEM_ACCEPTED' "$BASE_FIRST"; then
  FIRST="$BASE_FIRST"
  FIRST_SOURCE=canonical
else
  STAGEM_FIRSTPASS_BUILD_DIR="$tmp/firstpass" PATCH_ONLY=1 bash "$FIRST_WRAP" 27 >"$tmp/firstpass.env"
  # shellcheck disable=SC1090
  source "$tmp/firstpass.env"
  [[ "${B300_STAGEM_FIRSTPASS_PATCHED:-0}" == 1 && -s "$B300_STAGEM_FIRSTPASS_GENERATED" ]] || exit 3
  FIRST="$B300_STAGEM_FIRSTPASS_GENERATED"
  FIRST_SOURCE=generated
fi
bash -n "$FIRST"

need(){ local f="$1" s="$2"; grep -Fq "$s" "$f" || { echo "Stage-M grand marker missing in $f: $s" >&2; exit 3; }; }
for s in \
  'STAGEM_PREFIX=' \
  'STAGEM_WINNER_ENV=' \
  'STAGEM_PREPARE_ENV=' \
  'RUN_STAGEM=' \
  'STAGEM_MIN_SPEEDUP=' \
  'STAGEM_POLICY_LIST=' \
  'b300x8-nextgen-hybrid8-mate-load-stagem-staged-fullprime-race.sh' \
  'PREPARE_ONLY=1' \
  'B300_STAGEM_PREPARED' \
  'B300_STAGEM_PREPARED_BIN' \
  'B300_STAGEM_PREPARED_CONTROL_BIN' \
  'Stage-M mate-load policy rejected; retaining Stage L' \
  'MODE=stagem_mateload_grand' \
  'MODE=stagem_mateload_joint' \
  'B300_GRAND_STAGEM_OK' \
  'B300_GRAND_STAGEM_POLICY' \
  'B300_GRAND_STAGEM_INTEGRATED=1'; do need "$GRAND" "$s"; done

python3 - "$GRAND" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('grand selector must own exactly one complete-prime race')
mi=s.find('b300x8-nextgen-hybrid8-mate-load-stagem-staged-fullprime-race.sh')
ri=s.find('b300x8-race-external-forced-profiled-once.sh')
if mi < 0 or ri < 0 or mi >= ri:
    raise SystemExit('Stage M must be prepared before the one complete-prime race')
for marker in ('P_BIN="$B300_STAGEM_PREPARED_BIN"','B_BIN="$B300_STAGEM_PREPARED_CONTROL_BIN"'):
    if marker not in s: raise SystemExit('Stage-M candidate replacement missing: '+marker)
if 'STAGEM_OK && NEXTSELF_OK' not in s or 'elif ((STAGEM_OK))' not in s:
    raise SystemExit('Stage-M grand/joint branch pair missing')
if '[[ "$B300_STAGEM_PREPARED_CONTROL_BIN" == "$B300_STAGEL_PREPARED_BIN" ]]' not in s:
    raise SystemExit('Stage-M exact Stage-L control binding missing')
print('stagem_single_prime_contract=OK')
PY

for s in \
  'RUN_STAGEL=' 'RUN_STAGEM=' \
  'stagel_min_speedup=' 'stagel_guard_list=' \
  'stagem_min_speedup=' 'stagem_policy_list=' \
  'B300_GRAND_SELECTED_STAGEL_ACCEPTED' \
  'B300_GRAND_SELECTED_STAGEM_ACCEPTED' \
  'B300_GRAND_SELECTED_STAGEM_POLICY'; do need "$FIRST" "$s"; done

python3 - "$FIRST" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
# A generated compatibility first-pass calls the Stage-M selector wrapper;
# canonical first-pass is allowed to call the integrated selector directly.
if 'b300x8-joint-nextself-hybrid8-select-stagem.sh' not in s and 'b300x8-joint-nextself-hybrid8-select.sh" 27' not in s:
    raise SystemExit('firstpass has no Stage-M-aware selector path')
for marker in (
    'b300x8-require-b300-inventory.sh',
    'GPU_INVENTORY="$(bash "$HARDWARE_GUARD")"',
    'GPU_LIST="$(nvidia-smi -L)"',
    'gpu_guard=b300x8_exact_model',
):
    if marker not in s:
        raise SystemExit('firstpass B300 hardware guard marker missing: '+marker)
if 'nvidia-smi --query-gpu=index --format=csv,noheader | wc -l' in s:
    raise SystemExit('firstpass still uses count-only GPU admission')
for forbidden in ('RUN_STAGEN=', 'RUN_STAGEO=', 'RUN_STAGEP=', 'RUN_STAGEQ=', 'RUN_STAGER=', 'RUN_STAGES=', 'RUN_STAGET=', 'RUN_STAGEU='):
    if forbidden in s:
        raise SystemExit('canonical firstpass must stop at Stage M before B300 measurement: '+forbidden)
m=re.search(r'B300_GRAND_SELECTED_SCHEMA=([0-9]+)',s)
if not m or int(m.group(1)) < 3:
    raise SystemExit('Stage-M firstpass must preserve hardened selected schema >=3')
print('stagem_firstpass_contract=OK schema='+m.group(1))
PY

echo "b300-grand-stagem-contract-preflight OK grand_source=$GRAND_SOURCE firstpass_source=$FIRST_SOURCE stage_m_after_l=1 prepare_only=1 fallback_l=1 forced_slots_replaced=1 single_complete_prime=1 exact_b300x8_guard=1 canonical_stops_at_m=1 gpu_work=0"
