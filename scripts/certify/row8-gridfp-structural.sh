#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
ARCH="${ARCH:-native}"
RESET="${RESET:-0}"
SAFETY_MIB="${ROW8_CERT_SAFETY_MIB:-1024}"
for spec in "N:$N" "NGPU:$NGPU" "TARGET_MIB:$TARGET_MIB" "MAX_WINDOW:$MAX_WINDOW" "GRIDFP_VRAM_RESERVE_MIB:$GRIDFP_VRAM_RESERVE_MIB" "RESET:$RESET" "ROW8_CERT_SAFETY_MIB:$SAFETY_MIB"; do
  require_uint "${spec%%:*}" "${spec#*:}" || exit 2
done
if (( N < 8 || N > 27 )); then
  echo "N must be in 8..27 for the row8 structural certificate" >&2
  exit 2
fi
if (( NGPU < 1 || NGPU > 8 )); then
  echo "NGPU must be in 1..8" >&2
  exit 2
fi
if (( RESET != 0 && RESET != 1 )); then
  echo "RESET must be 0 or 1" >&2
  exit 2
fi

W=$((N + 1))
TOOL="$ONEESAN_ROOT/scripts/tools/row8_gridfp_structural_cert.py"
MEM_TOOL="$ONEESAN_ROOT/scripts/tools/row8_cert_memory_plan.py"
CERT_DIR="$ONEESAN_ROOT/formal/certificates"
LOG="$CERT_DIR/row8_gridfp_structural_w${W}.log"
CERT="$CERT_DIR/row8_gridfp_structural_w${W}.json"
BIN="${BIN:-$ONEESAN_BUILD_DIR/row8_gridfp_structural_cert_n${N}}"
PROV="${BIN}.provenance.json"
WORK="${ROW8_CERT_WORK_DIR:-$ONEESAN_BUILD_DIR/row8-gridfp-structural-w${W}}"
PARTS="$WORK/parts"
STATE="$WORK/state.json"
LOCK="$WORK/.lock"
mkdir -p "$CERT_DIR" "$WORK" "$PARTS"

# Prevent two hosts/processes from appending evidence for the same width at once.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "row8 certificate work directory is already in use: $WORK" >&2
  exit 2
fi

mapfile -t PRIMES < <(
  python3 - "$TOOL" "$W" <<'PY'
import json, subprocess, sys
raw = subprocess.check_output(
    [sys.executable, sys.argv[1], "--show-required", sys.argv[2]], text=True
)
data = json.loads(raw)
for p in data["primes"]:
    print(p)
PY
)
if (( ${#PRIMES[@]} == 0 )); then
  echo "certificate tool returned no primes" >&2
  exit 2
fi

needs_build=0
if [[ ! -x "$BIN" || ! -f "$PROV" ]]; then
  needs_build=1
elif ! python3 "$ONEESAN_ROOT/scripts/solve/build_provenance.py" verify \
    "$PROV" --binary "$BIN" --root "$ONEESAN_ROOT" --verify-sources \
    --expect-compile-arg="-DTARGET_W=$W" >/dev/null 2>&1; then
  needs_build=1
fi
if (( needs_build )); then
  echo "building row8 certificate comparator for n=$N width=$W" >&2
  N="$N" ROW8_TENSOR=1 ARCH="$ARCH" OUT="$BIN" \
    "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh"
fi
python3 "$ONEESAN_ROOT/scripts/solve/build_provenance.py" verify \
  "$PROV" --binary "$BIN" --root "$ONEESAN_ROOT" --verify-sources \
  --expect-compile-arg="-DTARGET_W=$W" >/dev/null

BIN_SHA="$(sha256sum "$BIN" | awk '{print $1}')"
PROV_SHA="$(sha256sum "$PROV" | awk '{print $1}')"
PRIME_CSV="$(IFS=,; echo "${PRIMES[*]}")"

# The resume state is content-addressed by the comparator binary/provenance and
# the exact required prime prefix. Never silently mix evidence across builds.
if (( RESET )); then
  rm -rf "$PARTS" "$STATE"
  mkdir -p "$PARTS"
fi
python3 - "$STATE" "$N" "$W" "$BIN_SHA" "$PROV_SHA" "$PRIME_CSV" <<'PY'
import json, os, sys
from pathlib import Path
state=Path(sys.argv[1])
expected={
  "schema":"oneesan-row8-gridfp-structural-resume-v1",
  "n":int(sys.argv[2]), "width":int(sys.argv[3]),
  "binary_sha256":sys.argv[4], "provenance_sha256":sys.argv[5],
  "primes":[int(x) for x in sys.argv[6].split(',') if x],
  "completed":{},
}
if state.exists():
    try: old=json.loads(state.read_text())
    except Exception as e: raise SystemExit(f"invalid resume state {state}: {e}")
    for k in ("schema","n","width","binary_sha256","provenance_sha256","primes"):
        if old.get(k)!=expected[k]:
            raise SystemExit(
              f"row8 certificate resume fingerprint changed at {k}; "
              "rerun with RESET=1 to discard stale partial evidence"
            )
    if not isinstance(old.get("completed"),dict):
        raise SystemExit("invalid resume state completed map")
else:
    tmp=state.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(expected,indent=2,sort_keys=True)+'\n')
    os.replace(tmp,state)
PY

validate_part() {
  local part="$1" prime="$2"
  python3 - "$part" "$N" "$W" "$prime" <<'PY'
import re,sys
p=sys.argv[1]; n=int(sys.argv[2]); w=int(sys.argv[3]); prime=int(sys.argv[4])
text=open(p,encoding='utf-8').read()
pat=r'row8_cert_compare n=(\d+) width=(\d+) modulus=(\d+) gpus=(\d+) main_states=(\d+) mismatch=(\d+) exact=(\d+) wall_s=([0-9.eE+-]+)'
rows=re.findall(pat,text)
if len(rows)!=1:
    raise SystemExit(f"{p}: expected exactly one row8_cert_compare record, got {len(rows)}")
rn,rw,rp,rg,states,mis,ex,wall=rows[0]
if (int(rn),int(rw),int(rp))!=(n,w,prime):
    raise SystemExit(f"{p}: target mismatch n={rn} w={rw} p={rp}")
if int(states)<=0 or int(rg)<=0 or int(mis)!=0 or int(ex)!=1:
    raise SystemExit(f"{p}: comparison did not prove equality: {rows[0]}")
PY
}

record_part() {
  local part="$1" prime="$2"
  local part_sha
  part_sha="$(sha256sum "$part" | awk '{print $1}')"
  python3 - "$STATE" "$prime" "$part_sha" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); prime=sys.argv[2]; sha=sys.argv[3]
d=json.loads(p.read_text())
d.setdefault("completed",{})[prime]=sha
t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

need_gpu=0
for p in "${PRIMES[@]}"; do
  if [[ ! -f "$PARTS/${p}.log" ]]; then
    need_gpu=1
    break
  fi
done
if (( need_gpu )); then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; missing prime evidence still requires GPU execution" >&2
    exit 2
  fi
  gpu_selector="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "$gpu_selector" ]]; then
    IFS=',' read -r -a visible_ids <<< "$gpu_selector"
    visible=${#visible_ids[@]}
  else
    visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  fi
  if (( visible < NGPU )); then
    echo "requested $NGPU GPUs, but only $visible are CUDA-visible" >&2
    exit 2
  fi

  PLAN_JSON="$WORK/memory_plan.json"
  python3 "$MEM_TOOL" "$W" --gpus "$NGPU" --reserve-mib "$GRIDFP_VRAM_RESERVE_MIB" \
    --safety-mib "$SAFETY_MIB" --json > "$PLAN_JSON"
  read -r required_total_mib required_scratch_mib < <(python3 - "$PLAN_JSON" <<'PY'
import json,math,sys
d=json.load(open(sys.argv[1]))
print(math.ceil(d['minimum_total_mib']), math.ceil(d['max_forced_scratch_mib']))
PY
)
  if (( TARGET_MIB < required_scratch_mib )); then
    echo "TARGET_MIB=$TARGET_MIB is below forced-window requirement ${required_scratch_mib} MiB for width $W" >&2
    exit 2
  fi
  if [[ -n "$gpu_selector" ]]; then
    selected="$(IFS=,; echo "${visible_ids[*]:0:NGPU}")"
    gpu_totals="$(nvidia-smi -i "$selected" --query-gpu=memory.total --format=csv,noheader,nounits)"
  else
    gpu_totals="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n "$NGPU")"
  fi
  min_gpu_total_mib="$(printf '%s\n' "$gpu_totals" | awk 'NR==1{m=$1} $1<m{m=$1} END{print int(m)}')"
  if [[ -z "$min_gpu_total_mib" ]] || (( min_gpu_total_mib < required_total_mib )); then
    echo "insufficient physical HBM for row8 certificate preflight: min_total_mib=${min_gpu_total_mib:-unknown} required_mib=$required_total_mib" >&2
    echo "plan=$PLAN_JSON (set ROW8_CERT_SAFETY_MIB lower only after reviewing the runtime allocation margin)" >&2
    exit 2
  fi
  echo "row8 certificate HBM preflight: min_total_mib=$min_gpu_total_mib required_mib=$required_total_mib max_scratch_mib=$required_scratch_mib reserve_mib=$GRIDFP_VRAM_RESERVE_MIB safety_mib=$SAFETY_MIB" >&2
fi

# The comparator constructs ordinary production Grid-FP through row 8, then
# compares every coordinate with the structural initializer.  Missing moduli are
# sent in one process so the ~2 TiB authoritative allocation, factor tables, and
# prepared schedule are built only once.  If a later modulus fails, every earlier
# exact=1 summary already emitted by that batch is salvaged into an immutable
# per-prime part; the next invocation resumes from the remaining prime prefix.
export GRIDFP_ROW8_CERT_COMPARE=1
export GRIDFP_VRAM_RESERVE_MIB
completed=0
MISSING=()
for p in "${PRIMES[@]}"; do
  part="$PARTS/${p}.log"
  if [[ -f "$part" ]]; then
    validate_part "$part" "$p" || {
      echo "existing prime evidence is invalid: $part; rerun with RESET=1" >&2
      exit 2
    }
    expected_sha="$(python3 - "$STATE" "$p" <<'PY2'
import json,sys
print(json.load(open(sys.argv[1])).get('completed',{}).get(sys.argv[2],''))
PY2
)"
    actual_sha="$(sha256sum "$part" | awk '{print $1}')"
    if [[ -n "$expected_sha" && "$expected_sha" != "$actual_sha" ]]; then
      echo "prime evidence hash changed for p=$p; rerun with RESET=1" >&2
      exit 2
    fi
    if [[ -z "$expected_sha" ]]; then record_part "$part" "$p"; fi
    ((completed+=1))
    echo "row8 certificate reuse p=$p ($completed/${#PRIMES[@]})" >&2
  else
    MISSING+=("$p")
  fi
done

if (( ${#MISSING[@]} )); then
  BATCHES="$WORK/batches"
  mkdir -p "$BATCHES"
  batch_id="$(date -u +%Y%m%dT%H%M%SZ)-${MISSING[0]}-${#MISSING[@]}"
  batch_tmp="$BATCHES/.${batch_id}.log.tmp"
  batch_log="$BATCHES/${batch_id}.log"
  rm -f "$batch_tmp"
  echo "row8 certificate batch n=$N width=$W gpus=$NGPU missing=${#MISSING[@]}/${#PRIMES[@]} primes=$(IFS=,; echo "${MISSING[*]}")" >&2
  set +e
  "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "${MISSING[@]}" 2>&1 | tee "$batch_tmp"
  batch_rc=${PIPESTATUS[0]}
  set -e
  mv "$batch_tmp" "$batch_log"

  # Extract only the authoritative machine-readable summary for each completed
  # modulus.  The full batch transcript is retained separately for diagnostics;
  # the immutable part is deliberately self-contained so it can be copied to a
  # GPU-less verification host.
  python3 - "$batch_log" "$PARTS" "$N" "$W" "$(IFS=,; echo "${MISSING[*]}")" <<'PY2'
import os,re,sys
from pathlib import Path
log=Path(sys.argv[1]); parts=Path(sys.argv[2]); n=int(sys.argv[3]); w=int(sys.argv[4])
wanted=[int(x) for x in sys.argv[5].split(',') if x]
pat=re.compile(r'^row8_cert_compare n=(\d+) width=(\d+) modulus=(\d+) gpus=(\d+) main_states=(\d+) mismatch=(\d+) exact=(\d+) wall_s=([0-9.eE+-]+)$')
seen={}
for line in log.read_text(errors='replace').splitlines():
    m=pat.match(line.strip())
    if not m: continue
    rn,rw,p,g,states,mis,ex,wall=m.groups(); p=int(p)
    if p not in wanted: continue
    if p in seen: raise SystemExit(f'duplicate row8_cert_compare for modulus {p}')
    if (int(rn),int(rw)) != (n,w): raise SystemExit(f'batch target mismatch for modulus {p}')
    if int(g)<=0 or int(states)<=0 or int(mis)!=0 or int(ex)!=1: continue
    seen[p]=line.strip()+'\n'
for p,text in seen.items():
    dst=parts/f'{p}.log'
    if dst.exists():
        if dst.read_text()!=text: raise SystemExit(f'existing immutable part differs for modulus {p}')
        continue
    tmp=parts/f'.{p}.log.tmp'
    tmp.write_text(text)
    os.replace(tmp,dst)
print('salvaged_primes='+','.join(map(str,seen)))
PY2

  # Bind every newly salvaged part into the resume state before deciding whether
  # the batch as a whole succeeded.
  for p in "${MISSING[@]}"; do
    part="$PARTS/${p}.log"
    if [[ -f "$part" ]]; then
      validate_part "$part" "$p"
      expected_sha="$(python3 - "$STATE" "$p" <<'PY2'
import json,sys
print(json.load(open(sys.argv[1])).get('completed',{}).get(sys.argv[2],''))
PY2
)"
      if [[ -z "$expected_sha" ]]; then record_part "$part" "$p"; fi
    fi
  done
  if (( batch_rc != 0 )); then
    echo "row8 certificate batch exited rc=$batch_rc; completed summaries were salvaged, rerun to resume remaining primes" >&2
    exit "$batch_rc"
  fi
  for p in "${MISSING[@]}"; do
    [[ -f "$PARTS/${p}.log" ]] || {
      echo "row8 certificate batch returned success without evidence for modulus $p" >&2
      exit 2
    }
  done
fi

# Revalidate every immutable part immediately before publishing the aggregate
# evidence log. Concatenation order is the production prime order expected by
# the deterministic CRT-bound verifier.
LOG_TMP="${LOG}.tmp"
: > "$LOG_TMP"
for p in "${PRIMES[@]}"; do
  part="$PARTS/${p}.log"
  validate_part "$part" "$p"
  cat "$part" >> "$LOG_TMP"
done
mv "$LOG_TMP" "$LOG"

python3 "$TOOL" --generate \
  --log "$LOG" \
  --provenance "$PROV" \
  --out "$CERT"
python3 "$TOOL" --verify "$CERT"
sha256sum "$CERT" "$LOG"
echo "row8 structural exact admission certificate ready: $CERT" >&2
