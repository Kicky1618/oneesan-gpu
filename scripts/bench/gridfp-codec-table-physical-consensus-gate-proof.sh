#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GATE="$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-physical-consensus-gate.sh"
[[ -f "$GATE" ]] || { echo "missing physical consensus gate" >&2; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

cat >"$tmp/exact.tsv" <<'EOF'
mode	name	choose_mode	primitive_mode	constant_bytes	repeats	wall_ms_median	paired_speedup_median	paired_speedup_min	paired_speedup_max
0	baseline	0	0	13688	7	100.0	1.000000000	1.000000000	1.000000000
1	max_compact	1	1	1800	7	99.0	1.010101010	1.003000000	1.020000000
EOF
cat >"$tmp/w28.tsv" <<'EOF'
mode	name	choose_mode	primitive_mode	candidate_physical_bytes	repeats	paired_speedup_median	paired_speedup_min	paired_speedup_max	ns_per_sample_median
0	baseline	0	0	13688	7	1.000000000	1.000000000	1.000000000	12.0
1	max_compact	1	1	1800	7	1.008000000	1.001000000	1.015000000	11.9
EOF
out="$(MIN_EXACT_SPEEDUP=1.002 MIN_W28_SPEEDUP=1.002 MIN_EXACT_PAIRS=7 MIN_W28_PAIRS=7 REQUIRE_ALL_PAIRS=1 bash "$GATE" "$tmp/exact.tsv" "$tmp/w28.tsv")"
grep -Fq 'physical_consensus_candidate_mode=1' <<<"$out"
grep -Fq 'physical_consensus_production_promotion_ready=1' <<<"$out"
grep -Fq 'physical_consensus_next_step=PRODUCTION_DEFAULT_CANDIDATE' <<<"$out"

# One slower paired sample must block promotion when REQUIRE_ALL_PAIRS=1.
sed 's/1.001000000/0.999000000/' "$tmp/w28.tsv" >"$tmp/w28-slow.tsv"
out="$(MIN_EXACT_SPEEDUP=1.002 MIN_W28_SPEEDUP=1.002 MIN_EXACT_PAIRS=7 MIN_W28_PAIRS=7 REQUIRE_ALL_PAIRS=1 bash "$GATE" "$tmp/exact.tsv" "$tmp/w28-slow.tsv")"
grep -Fq 'physical_consensus_all_pairs_faster=0' <<<"$out"
grep -Fq 'physical_consensus_production_promotion_ready=0' <<<"$out"
grep -Fq 'physical_consensus_next_step=KEEP_EXPERIMENTAL' <<<"$out"

# Mismatched layout metadata must hard-fail rather than compare unrelated runs.
sed 's/max_compact/full_shape_both/' "$tmp/w28.tsv" >"$tmp/w28-mismatch.tsv"
if bash "$GATE" "$tmp/exact.tsv" "$tmp/w28-mismatch.tsv" >/dev/null 2>&1; then
  echo "physical consensus gate accepted mismatched metadata" >&2
  exit 3
fi

echo "gridfp-codec-table-physical-consensus-gate-proof OK ready_case=1 slower_pair_blocks=1 metadata_mismatch_fails=1 exact=1"
