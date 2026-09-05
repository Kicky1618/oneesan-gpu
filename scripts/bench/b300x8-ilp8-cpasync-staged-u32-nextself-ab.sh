#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
BASE="$ONEESAN_ROOT/scripts/bench/b300x8-ilp8-cpasync-staged-wait-ab.sh"
[[ -f "$BASE" ]] || { echo "missing base A/B script: $BASE" >&2; exit 2; }
TMP="$(mktemp "$ONEESAN_ROOT/scripts/bench/.b300_staged_nextself.XXXXXX.sh")"
cleanup(){ rm -f "$TMP"; }
trap cleanup EXIT INT TERM
python3 - "$BASE" "$TMP" <<'PY'
import pathlib,sys
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
s=s.replace('b300x8_ilp8_cpasync_staged_wait_ab_','b300x8_ilp8_cpasync_staged_u32_nextself_ab_')
old='python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-rankstate-ilp8-cpasync-staged-wait.py" "$BASE_SRC" "$STAGED_SRC" | tee "$LOGDIR/staged.transform.log"'
new='''STAGE1="$LOGDIR/staged_u32_before_nextself.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-rankstate-ilp8-cpasync-staged-wait-u32.py" "$BASE_SRC" "$STAGE1" | tee "$LOGDIR/staged.transform.log"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-rankstate-ilp8-next-self-prefetch.py" "$STAGE1" "$STAGED_SRC" | tee -a "$LOGDIR/staged.transform.log"'''
if s.count(old)!=1: raise SystemExit(f'base A/B transform anchor expected one match got {s.count(old)}')
s=s.replace(old,new,1)
out.write_text(s)
PY
chmod +x "$TMP"
grep -Fq 'gen-b300-main-rankstate-ilp8-cpasync-staged-wait-u32.py' "$TMP"
grep -Fq 'gen-b300-main-rankstate-ilp8-next-self-prefetch.py' "$TMP"
echo 'staged_wait_variant=u32_pairfirst_nextself wait_pair_group=1 next_self_prefetches=8 cache=L2' >&2
exec bash "$TMP" "$@"
