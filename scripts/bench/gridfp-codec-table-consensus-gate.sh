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
  echo "usage: $0 <exact-summary.tsv> <w28-rank-summary.tsv>" >&2; exit 2; }
[[ "$REQUIRE_ALL_PAIRS" == 0 || "$REQUIRE_ALL_PAIRS" == 1 ]] || { echo "REQUIRE_ALL_PAIRS must be 0 or 1" >&2; exit 2; }

python3 - "$EXACT_SUMMARY" "$W28_SUMMARY" "$MIN_EXACT_SPEEDUP" "$MIN_W28_SPEEDUP" "$MIN_EXACT_PAIRS" "$MIN_W28_PAIRS" "$REQUIRE_ALL_PAIRS" <<'PY'
import csv, math, sys
exact_path,w28_path,min_e_s,min_w_s,min_en_s,min_wn_s,all_s=sys.argv[1:]
min_e=float(min_e_s); min_w=float(min_w_s); min_en=int(min_en_s); min_wn=int(min_wn_s); require_all=int(all_s)
if min_e < 1 or min_w < 1 or min_en < 1 or min_wn < 1: raise SystemExit('invalid gate threshold')

def read(path):
    rows=list(csv.DictReader(open(path),delimiter='\t'))
    need={'mode','name','candidate_physical_bytes','repeats','paired_speedup_median','paired_speedup_min'}
    if not rows or not need.issubset(rows[0]): raise SystemExit(f'{path}: missing required fields')
    return {r['mode']:r for r in rows}
e=read(exact_path); w=read(w28_path)
if '0' not in e or '0' not in w: raise SystemExit('missing baseline mode 0')
modes=sorted((set(e)&set(w))-{'0'}, key=int)
eligible=[]
for m in modes:
    er,wr=e[m],w[m]
    if er['name'] != wr['name'] or er['candidate_physical_bytes'] != wr['candidate_physical_bytes']:
        raise SystemExit(f'layout metadata mismatch mode={m}')
    em=float(er['paired_speedup_median']); wm=float(wr['paired_speedup_median'])
    emin=float(er['paired_speedup_min']); wmin=float(wr['paired_speedup_min'])
    en=int(er['repeats']); wn=int(wr['repeats'])
    all_fast=emin>1.0 and wmin>1.0
    ok=en>=min_en and wn>=min_wn and em>=min_e and wm>=min_w and (all_fast if require_all else True)
    geo=math.sqrt(em*wm)
    if ok: eligible.append((geo,-int(er['candidate_physical_bytes']),m,er,wr))
    print(f'codec_consensus_mode{m}_name={er["name"]}')
    print(f'codec_consensus_mode{m}_exact_paired_speedup_median={em:.9f}x')
    print(f'codec_consensus_mode{m}_w28_paired_speedup_median={wm:.9f}x')
    print(f'codec_consensus_mode{m}_geomean_speedup={geo:.9f}x')
    print(f'codec_consensus_mode{m}_all_pairs_faster={int(all_fast)}')
    print(f'codec_consensus_mode{m}_eligible={int(ok)}')
print(f'codec_consensus_min_exact_speedup={min_e:.9f}x')
print(f'codec_consensus_min_w28_speedup={min_w:.9f}x')
print(f'codec_consensus_require_all_pairs={require_all}')
if not eligible:
    print('codec_consensus_candidate=NONE')
    print('codec_consensus_physical_replacement_ready=0')
    print('codec_consensus_next_step=KEEP_PROXY_ONLY')
    raise SystemExit(0)
geo,neg_bytes,m,er,wr=max(eligible)
print(f'codec_consensus_candidate_mode={m}')
print(f'codec_consensus_candidate={er["name"]}')
print(f'codec_consensus_candidate_bytes={er["candidate_physical_bytes"]}')
print(f'codec_consensus_candidate_geomean_speedup={geo:.9f}x')
print(f'codec_consensus_candidate_exact_speedup={float(er["paired_speedup_median"]):.9f}x')
print(f'codec_consensus_candidate_w28_speedup={float(wr["paired_speedup_median"]):.9f}x')
print('codec_consensus_physical_replacement_ready=1')
print('codec_consensus_next_step=PHYSICAL_REPLACEMENT_AB')
print('codec_consensus_production_promotion=0')
PY
