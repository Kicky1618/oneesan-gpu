#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GATE="$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-consensus-gate.sh"
[[ -f "$GATE" ]] || { echo "missing gate: $GATE" >&2; exit 2; }
e="$(mktemp)"; w="$(mktemp)"; trap 'rm -f "$e" "$w"' EXIT
header='mode	name	choose_mode	primitive_mode	candidate_physical_bytes	repeats	wall_ms_median	wall_ms_min	wall_ms_max	paired_speedup_median	paired_speedup_min	paired_speedup_max'
printf '%b\n' "$header" >"$e"; printf '%b\n' "$header" >"$w"
printf '%b\n' '0	baseline	0	0	13688	7	100	99	101	1.0	1.0	1.0' >>"$e"
printf '%b\n' '1	max_compact	1	1	1800	7	99	98	100	1.008	1.003	1.012' >>"$e"
printf '%b\n' '2	tri_choose_compact_primitive	2	1	2640	7	98.8	98	100	1.010	1.004	1.016' >>"$e"
printf '%b\n' '3	full_choose_compact_primitive	3	1	4264	7	98.5	97	101	1.012	0.999	1.020' >>"$e"
printf '%b\n' '0	baseline	0	0	13688	14	0	0	0	1.0	1.0	1.0' >>"$w"
printf '%b\n' '1	max_compact	1	1	1800	14	0	0	0	1.020	1.006	1.030' >>"$w"
printf '%b\n' '2	tri_choose_compact_primitive	2	1	2640	14	0	0	0	1.015	1.005	1.025' >>"$w"
printf '%b\n' '3	full_choose_compact_primitive	3	1	4264	14	0	0	0	1.030	1.010	1.040' >>"$w"
out="$(MIN_EXACT_SPEEDUP=1.005 MIN_W28_SPEEDUP=1.005 bash "$GATE" "$e" "$w")"
# Mode3 is rejected because exact has one losing pair. Mode1 wins the geometric
# mean among candidates that are consistently faster in both measurements.
grep -Fq 'codec_consensus_mode3_eligible=0' <<<"$out"
grep -Fq 'codec_consensus_mode1_eligible=1' <<<"$out"
grep -Fq 'codec_consensus_mode2_eligible=1' <<<"$out"
grep -Fq 'codec_consensus_candidate_mode=1' <<<"$out"
grep -Fq 'codec_consensus_candidate=max_compact' <<<"$out"
grep -Fq 'codec_consensus_physical_replacement_ready=1' <<<"$out"
grep -Fq 'codec_consensus_production_promotion=0' <<<"$out"
out2="$(MIN_EXACT_SPEEDUP=1.02 MIN_W28_SPEEDUP=1.02 bash "$GATE" "$e" "$w")"
grep -Fq 'codec_consensus_candidate=NONE' <<<"$out2"
grep -Fq 'codec_consensus_physical_replacement_ready=0' <<<"$out2"
grep -Fq 'codec_consensus_next_step=KEEP_PROXY_ONLY' <<<"$out2"
echo 'gridfp-codec-table-consensus-gate-proof OK agreement=1 all_pairs=1 no_auto_promotion=1 exact=1'
