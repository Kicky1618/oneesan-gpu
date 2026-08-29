#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
W=$((N + 1))
ARCH="${ARCH:-sm_80}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"
HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"

if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then
  echo "invalid factor split" >&2
  exit 2
fi
if (( LOW_LUT_K > 14 || HIGH_LUT_K > 14 )); then
  echo "rankchunk32 exact traffic requires half widths <=14" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null; then
  echo "nvcc is required to compile the host-only CUDA translation unit" >&2
  exit 2
fi

SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/rankchunk32_exact_rankmask_traffic.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/rankchunk32_exact_rankmask_traffic_w${W}}"
mkdir -p "$(dirname "$BIN")"

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  "$SRC" -o "$BIN"

out="$($BIN)"
printf '%s\n' "$out"

grep -Fq "rankchunk32-exact-rankmask-traffic OK W=$W" <<<"$out"
grep -Fq 'physical_cols_exact=1' <<<"$out"
grep -Fq 'count_value_independent=1' <<<"$out"
grep -Fq 'modulus_independent=1' <<<"$out"
grep -Fq 'host_exact=1' <<<"$out"
grep -Fq 'runtime_gpu_required=0' <<<"$out"
grep -Fq 'scope=forward_high_sweep' <<<"$out"
grep -Fq 'scope=reverse_high_sweep' <<<"$out"
grep -Fq 'scope=one_residue' <<<"$out"
grep -Fq 'forward_depth_entries=' <<<"$out"
grep -Fq 'reverse_depth_entries=' <<<"$out"

for line in \
  "$(grep '^rankchunk32_exact_rankmask_traffic scope=forward_high_sweep ' <<<"$out")" \
  "$(grep '^rankchunk32_exact_rankmask_traffic scope=reverse_high_sweep ' <<<"$out")" \
  "$(grep '^rankchunk32_exact_rankmask_traffic scope=one_residue ' <<<"$out")"; do
  [[ -n "$line" ]] || { echo "missing exact traffic line" >&2; exit 3; }
  grep -Fq 'disallowed=0' <<<"$line" || { echo "rankmask shape violation: $line" >&2; exit 4; }
  chunk_calls="$(sed -nE 's/.* chunk_calls=([0-9]+).*/\1/p' <<<"$line")"
  [[ -n "$chunk_calls" && "$chunk_calls" -gt 0 ]] || { echo "empty exact traffic: $line" >&2; exit 5; }
done

residue_line="$(grep '^rankchunk32_exact_rankmask_traffic scope=one_residue ' <<<"$out")"
zero_frac="$(sed -nE 's/.* zero_frac=([^ ]+).*/\1/p' <<<"$residue_line")"
avg_popcount="$(sed -nE 's/.* avg_popcount=([^ ]+).*/\1/p' <<<"$residue_line")"
chunk_calls="$(sed -nE 's/.* chunk_calls=([0-9]+).*/\1/p' <<<"$residue_line")"
resolved_calls="$(sed -nE 's/.* resolved_calls=([0-9]+).*/\1/p' <<<"$residue_line")"

printf 'rankchunk32-exact-rankmask-traffic summary n=%s W=%s chunk_calls=%s resolved_calls=%s exact_zero_frac=%s exact_avg_popcount=%s\n' \
  "$N" "$W" "$chunk_calls" "$resolved_calls" "$zero_frac" "$avg_popcount"
echo "rankchunk32-exact-rankmask-traffic done n=$N arch=$ARCH host_exact=1 gpu_runtime=0" >&2
