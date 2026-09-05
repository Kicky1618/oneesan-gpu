#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

scripts=(
  "$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilpcg-calibrate-staged.sh"
  "$ONEESAN_ROOT/scripts/run/b300x8-mainrec-ilpcg-staged-fullprime-race.sh"
  "$ONEESAN_ROOT/scripts/bench/b300x8-saturate-ilp8-ab.sh"
  "$ONEESAN_ROOT/scripts/bench/b300x8-saturate-hybrid-ilp8-threshold-ab.sh"
  "$ONEESAN_ROOT/scripts/run/b300x8-hybrid-ilp8-staged-fullprime-race.sh"
)
for script in "${scripts[@]}"; do
  bash -n "$script"
done

selector="$ONEESAN_ROOT/scripts/bench/b300-hybrid-ilp8-export-winner.py"
python3 - "$selector" <<'PY'
import pathlib,sys
path=pathlib.Path(sys.argv[1])
compile(path.read_text(),str(path),'exec')
print('hybrid_selector_python_syntax=OK')
PY

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-staged-ilp-preflight.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
build="$tmp/build"
mkdir -p "$build"
for tag in \
  n27_ilp4warp_dualmask_closuretab_cg_warpscan \
  n27_mainhybrid8_t0_warp_dualmask_closuretab_cg_warpscan \
  n27_mainhybrid8_t100_warp_dualmask_closuretab_cg_warpscan; do
  file="$build/oneesan_cuda_gridfp_b300_hbm32_${tag}"
  printf '#!/usr/bin/env bash\nexit 0\n' >"$file"
  chmod +x "$file"
done

result="$tmp/results.tsv"
cat >"$result" <<'EOF'
mode	threshold	repeat	residue	wall_s	active_max_s	active_sum_s	mc_avg_pct	mc_max_pct	mc_samples	regs_max	spill_store_max_bytes	spill_load_max_bytes
ilp4	NA	1	12345	10.000000000	9.0	9.0	20.0	22.0	8	nan	nan	nan
hybrid	0	1	12345	8.000000000	7.0	7.0	28.0	31.0	8	160	8	0
hybrid	100	1	12345	9.000000000	8.0	8.0	26.0	29.0	8	144	0	0
EOF
winner="$tmp/winner.env"
python3 "$selector" "$result" "$winner" \
  --build-dir "$build" --threads 256 --random-cg 1 --warp-scan 1 >"$tmp/select.out"
# shellcheck disable=SC1090
source "$winner"
[[ "$B300_HYBRID_WINNER_MODE" == hybrid ]]
[[ "$B300_HYBRID_WINNER_THRESHOLD" == 100 ]]
[[ "$B300_HYBRID_WINNER_TRANSFORMED" == 1 ]]
[[ "$B300_HYBRID_WINNER_SPILL_FREE" == 1 ]]
[[ "$B300_HYBRID_RESIDUE" == 12345 ]]
python3 - "$B300_HYBRID_WINNER_SPEEDUP_VS_ILP4" <<'PY'
import sys
v=float(sys.argv[1])
assert 1.11 < v < 1.12, v
PY

grep -q '^b300_hybrid_winner_mode=hybrid$' "$tmp/select.out"
grep -q '^b300_hybrid_winner_threshold=100$' "$tmp/select.out"

echo 'b300_staged_ilp_preflight=OK shell_syntax=OK selector_spill_gate=OK selector_exact_gate=OK gpu_work=0'
