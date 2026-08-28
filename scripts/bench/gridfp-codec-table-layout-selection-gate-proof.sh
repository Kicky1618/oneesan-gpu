#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GATE="$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-layout-selection-gate.sh"
[[ -f "$GATE" ]] || { echo "missing gate: $GATE" >&2; exit 2; }
tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT

cat >"$tmp" <<'TSV'
mode	name	choose_mode	primitive_mode	candidate_physical_bytes	repeats	wall_ms_median	wall_ms_min	wall_ms_max	paired_speedup_median	paired_speedup_min	paired_speedup_max
0	baseline	0	0	13688	7	100.0	99.0	101.0	1.000000000	1.000000000	1.000000000
1	max_compact	1	1	1800	7	99.2	98.4	100.0	1.008000000	1.003000000	1.012000000
2	tri_choose_compact_primitive	2	1	2640	7	99.0	98.0	100.0	1.010000000	1.004000000	1.016000000
3	full_choose_compact_primitive	3	1	4264	7	98.9	98.0	100.1	1.011000000	0.999000000	1.020000000
TSV
out="$(MIN_PAIRED_SPEEDUP=1.005 MIN_REPEATS=7 REQUIRE_ALL_PAIRS=1 bash "$GATE" "$tmp")"
grep -Fq 'codec_layout_gate_mode1_eligible=1' <<<"$out"
grep -Fq 'codec_layout_gate_mode2_eligible=1' <<<"$out"
grep -Fq 'codec_layout_gate_mode3_eligible=0' <<<"$out"
grep -Fq 'codec_layout_gate_candidate_mode=2' <<<"$out"
grep -Fq 'codec_layout_gate_candidate=tri_choose_compact_primitive' <<<"$out"
grep -Fq 'codec_layout_gate_physical_replacement_ready=1' <<<"$out"
grep -Fq 'codec_layout_gate_production_promotion=0' <<<"$out"
grep -Fq 'codec_layout_gate_all-win_sign_p=0.0078125' <<<"$out"

out2="$(MIN_PAIRED_SPEEDUP=1.02 MIN_REPEATS=7 REQUIRE_ALL_PAIRS=1 bash "$GATE" "$tmp")"
grep -Fq 'codec_layout_gate_candidate=NONE' <<<"$out2"
grep -Fq 'codec_layout_gate_next_step=KEEP_PROXY_ONLY' <<<"$out2"
grep -Fq 'codec_layout_gate_physical_replacement_ready=0' <<<"$out2"

echo "gridfp-codec-table-layout-selection-gate-proof OK winner_selection=1 all_pairs_gate=1 threshold_gate=1 no_auto_promotion=1 exact=1"
