#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

SUMMARY="${1:-${SUMMARY:-}}"
MIN_PAIRED_SPEEDUP="${MIN_PAIRED_SPEEDUP:-1.002}"
MIN_REPEATS="${MIN_REPEATS:-7}"
REQUIRE_ALL_PAIRS="${REQUIRE_ALL_PAIRS:-1}"
[[ -n "$SUMMARY" && -f "$SUMMARY" ]] || {
  echo "usage: $0 <combined-summary.tsv>" >&2
  exit 2
}
[[ "$REQUIRE_ALL_PAIRS" == 0 || "$REQUIRE_ALL_PAIRS" == 1 ]] || {
  echo "REQUIRE_ALL_PAIRS must be 0 or 1" >&2; exit 2; }

python3 - "$SUMMARY" "$MIN_PAIRED_SPEEDUP" "$MIN_REPEATS" "$REQUIRE_ALL_PAIRS" <<'PY'
import csv, math, sys
path, min_speedup_s, min_repeats_s, require_all_s = sys.argv[1:]
min_speedup=float(min_speedup_s); min_repeats=int(min_repeats_s); require_all=int(require_all_s)
if min_speedup < 1.0: raise SystemExit('MIN_PAIRED_SPEEDUP must be >= 1')
if min_repeats < 1: raise SystemExit('MIN_REPEATS must be >= 1')
rows=list(csv.DictReader(open(path), delimiter='\t'))
required={'mode','name','candidate_physical_bytes','repeats','paired_speedup_median','paired_speedup_min','paired_speedup_max'}
if not rows or not required.issubset(rows[0]):
    raise SystemExit(f'summary missing required fields: {sorted(required-set(rows[0]) if rows else required)}')
base=[r for r in rows if r['mode']=='0']
if len(base)!=1: raise SystemExit('summary must contain exactly one baseline mode=0')

eligible=[]
for r in rows:
    if r['mode']=='0': continue
    repeats=int(r['repeats']); median=float(r['paired_speedup_median']); lo=float(r['paired_speedup_min']); hi=float(r['paired_speedup_max'])
    enough=repeats >= min_repeats
    speed=median >= min_speedup
    all_faster=lo > 1.0
    ok=enough and speed and (all_faster if require_all else True)
    if ok: eligible.append(r)
    print(f'codec_layout_gate_mode{r["mode"]}_name={r["name"]}')
    print(f'codec_layout_gate_mode{r["mode"]}_paired_speedup_median={median:.9f}x')
    print(f'codec_layout_gate_mode{r["mode"]}_paired_speedup_min={lo:.9f}x')
    print(f'codec_layout_gate_mode{r["mode"]}_repeats={repeats}')
    print(f'codec_layout_gate_mode{r["mode"]}_all_pairs_faster={int(all_faster)}')
    print(f'codec_layout_gate_mode{r["mode"]}_eligible={int(ok)}')

print(f'codec_layout_gate_min_paired_speedup={min_speedup:.9f}x')
print(f'codec_layout_gate_min_repeats={min_repeats}')
print(f'codec_layout_gate_require_all_pairs={require_all}')
if not eligible:
    print('codec_layout_gate_candidate=NONE')
    print('codec_layout_gate_next_step=KEEP_PROXY_ONLY')
    print('codec_layout_gate_physical_replacement_ready=0')
    raise SystemExit(0)

# Prefer paired performance. Break exact ties by the smaller candidate table.
winner=max(eligible, key=lambda r: (float(r['paired_speedup_median']), -int(r['candidate_physical_bytes'])))
n=int(winner['repeats'])
# With REQUIRE_ALL_PAIRS=1, every paired observation beats baseline. Under a
# symmetric no-effect sign null, the one-sided probability is 2^-n.
sign_p=(0.5**n) if float(winner['paired_speedup_min']) > 1.0 else float('nan')
print(f'codec_layout_gate_candidate_mode={winner["mode"]}')
print(f'codec_layout_gate_candidate={winner["name"]}')
print(f'codec_layout_gate_candidate_bytes={winner["candidate_physical_bytes"]}')
print(f'codec_layout_gate_candidate_paired_speedup_median={float(winner["paired_speedup_median"]):.9f}x')
print(f'codec_layout_gate_candidate_paired_speedup_min={float(winner["paired_speedup_min"]):.9f}x')
if math.isfinite(sign_p): print(f'codec_layout_gate_all-win_sign_p={sign_p:.9g}')
print('codec_layout_gate_next_step=PHYSICAL_REPLACEMENT_AB')
print('codec_layout_gate_physical_replacement_ready=1')
print('codec_layout_gate_production_promotion=0')
PY
