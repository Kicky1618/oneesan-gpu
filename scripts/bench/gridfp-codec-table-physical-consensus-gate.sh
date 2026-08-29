#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

EXACT_SUMMARY="${1:-}"
W28_SUMMARY="${2:-}"
MIN_EXACT_SPEEDUP="${MIN_EXACT_SPEEDUP:-1.002}"
MIN_W28_SPEEDUP="${MIN_W28_SPEEDUP:-1.002}"
MIN_EXACT_PAIRS="${MIN_EXACT_PAIRS:-7}"
MIN_W28_PAIRS="${MIN_W28_PAIRS:-7}"
REQUIRE_ALL_PAIRS="${REQUIRE_ALL_PAIRS:-1}"
[[ -f "$EXACT_SUMMARY" && -f "$W28_SUMMARY" ]] || {
  echo "usage: $0 <physical-exact-summary.tsv> <physical-w28-summary.tsv>" >&2; exit 2; }
[[ "$REQUIRE_ALL_PAIRS" == 0 || "$REQUIRE_ALL_PAIRS" == 1 ]] || { echo "REQUIRE_ALL_PAIRS must be 0 or 1" >&2; exit 2; }

python3 - "$EXACT_SUMMARY" "$W28_SUMMARY" "$MIN_EXACT_SPEEDUP" "$MIN_W28_SPEEDUP" "$MIN_EXACT_PAIRS" "$MIN_W28_PAIRS" "$REQUIRE_ALL_PAIRS" <<'PY'
import csv,sys
exact_path,w28_path,min_e_s,min_w_s,min_en_s,min_wn_s,all_s=sys.argv[1:]
min_e=float(min_e_s); min_w=float(min_w_s); min_en=int(min_en_s); min_wn=int(min_wn_s); require_all=int(all_s)
if min_e < 1 or min_w < 1 or min_en < 1 or min_wn < 1:
    raise SystemExit('invalid physical promotion threshold')

def rows(path):
    rs=list(csv.DictReader(open(path),delimiter='\t'))
    if len(rs)!=2: raise SystemExit(f'{path}: expected baseline + one candidate')
    by={r['mode']:r for r in rs}
    if '0' not in by: raise SystemExit(f'{path}: missing baseline')
    cand=[r for r in rs if r['mode']!='0']
    if len(cand)!=1: raise SystemExit(f'{path}: expected one candidate')
    return by['0'],cand[0]
ebase,e=rows(exact_path); wbase,w=rows(w28_path)
for base,path in ((ebase,exact_path),(wbase,w28_path)):
    if base['name']!='baseline': raise SystemExit(f'{path}: mode0 is not baseline')
if e['mode']!=w['mode'] or e['name']!=w['name'] or e['choose_mode']!=w['choose_mode'] or e['primitive_mode']!=w['primitive_mode']:
    raise SystemExit('physical candidate metadata mismatch')
ebytes=e.get('constant_bytes') or e.get('candidate_physical_bytes')
wbytes=w.get('constant_bytes') or w.get('candidate_physical_bytes')
if ebytes!=wbytes: raise SystemExit('physical candidate byte mismatch')
mode=int(e['mode']); cbytes=int(ebytes)
if mode<1 or mode>5 or cbytes>=13688: raise SystemExit('invalid physical candidate')
em=float(e['paired_speedup_median']); wm=float(w['paired_speedup_median'])
emin=float(e['paired_speedup_min']); wmin=float(w['paired_speedup_min'])
en=int(e['repeats']); wn=int(w['repeats'])
all_fast=emin>1.0 and wmin>1.0
ready=en>=min_en and wn>=min_wn and em>=min_e and wm>=min_w and (all_fast if require_all else True)
print(f'physical_consensus_candidate_mode={mode}')
print(f'physical_consensus_candidate={e["name"]}')
print(f'physical_consensus_choose_mode={e["choose_mode"]}')
print(f'physical_consensus_primitive_mode={e["primitive_mode"]}')
print(f'physical_consensus_constant_bytes={cbytes}')
print(f'physical_consensus_saved_constant_bytes={13688-cbytes}')
print(f'physical_consensus_exact_paired_speedup_median={em:.9f}x')
print(f'physical_consensus_exact_paired_speedup_min={emin:.9f}x')
print(f'physical_consensus_w28_paired_speedup_median={wm:.9f}x')
print(f'physical_consensus_w28_paired_speedup_min={wmin:.9f}x')
print(f'physical_consensus_exact_pairs={en}')
print(f'physical_consensus_w28_pairs={wn}')
print(f'physical_consensus_all_pairs_faster={int(all_fast)}')
print(f'physical_consensus_min_exact_speedup={min_e:.9f}x')
print(f'physical_consensus_min_w28_speedup={min_w:.9f}x')
print(f'physical_consensus_require_all_pairs={require_all}')
print(f'physical_consensus_production_promotion_ready={int(ready)}')
print('physical_consensus_legacy_constant_retained=0')
if ready:
    print('physical_consensus_next_step=PRODUCTION_DEFAULT_CANDIDATE')
else:
    print('physical_consensus_next_step=KEEP_EXPERIMENTAL')
PY
