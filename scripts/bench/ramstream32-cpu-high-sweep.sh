#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

N="${N:-27}"
ARCH="${ARCH:-native}"
MODULUS="${MODULUS:-4294967291}"
GPU_TARGET_MIB="${GPU_TARGET_MIB:-12288}"
CPU_WORKERS="${CPU_WORKERS:-32}"
CPU_HIGH_WORKERS="${CPU_HIGH_WORKERS:-$CPU_WORKERS}"
THRESHOLDS="${THRESHOLDS:-0 64 128 256 512 1024}"
REPEATS="${REPEATS:-1}"
BUILD="${BUILD:-1}"
EXPECTED_RESIDUE="${EXPECTED_RESIDUE:-}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_high_sweep}"

if [[ -z "$EXPECTED_RESIDUE" && "$N:$MODULUS" == "21:4294967291" ]]; then
  EXPECTED_RESIDUE=998035516
fi
if (( N < 2 || N > 27 || GPU_TARGET_MIB <= 0 || CPU_WORKERS <= 0 || CPU_HIGH_WORKERS <= 0 || REPEATS <= 0 )); then
  echo "invalid benchmark parameters" >&2
  exit 2
fi

read -r -a thresholds <<<"$THRESHOLDS"
if ((${#thresholds[@]} == 0)); then
  echo "THRESHOLDS must contain at least one value" >&2
  exit 2
fi
for x in "${thresholds[@]}"; do
  awk -v x="$x" 'BEGIN { exit !(x >= 0) }' || { echo "invalid threshold: $x" >&2; exit 2; }
done

if [[ "$BUILD" != 0 ]]; then
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-factorized-hybrid-sparse.sh
fi
bin="$ROOT/build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/cpu-high-sweep-n${N}-${ts}.tsv"
meta="$OUT_DIR/cpu-high-sweep-n${N}-${ts}.meta"

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
field() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then printf '%s\n' "${token#*=}"; return 0; fi
  done
  return 1
}

cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null | paste -sd ';' - || echo unknown)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)
n=$N
arch=$ARCH
modulus=$MODULUS
gpu_target_mib=$GPU_TARGET_MIB
cpu_workers=$CPU_WORKERS
cpu_high_workers=$CPU_HIGH_WORKERS
thresholds=$THRESHOLDS
repeats=$REPEATS
expected_residue=${EXPECTED_RESIDUE:-unknown}
binary_sha256=$(file_sha256 "$bin")
EOF

printf 'repeat\torder\tthreshold_mib\tresidue\twall_s\th2d_s\tgpu_kernel_s\td2h_s\tcpu_high_wall_s\tcpu_high_kernel_sum_s\tcpu_low_wall_s\tpcie_removed_tib\tpcie_remaining_tib\tcpu_high_groups\traw\n' >"$out"

run_one() {
  local repeat="$1" order="$2" threshold="$3"
  local line residue
  line="$(CPU_HIGH_MAX_MIB="$threshold" CPU_HIGH_WORKERS="$CPU_HIGH_WORKERS" \
    "$bin" "$N" "$MODULUS" "$GPU_TARGET_MIB" "$CPU_WORKERS" | tail -n1)"
  residue="$(field "$line" residue)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repeat" "$order" "$threshold" "$residue" \
    "$(field "$line" wall_s)" "$(field "$line" h2d_s)" \
    "$(field "$line" gpu_kernel_s)" "$(field "$line" d2h_s)" \
    "$(field "$line" cpu_high_wall_s)" "$(field "$line" cpu_high_kernel_sum_s)" \
    "$(field "$line" cpu_low_wall_s)" "$(field "$line" pcie_removed_tib_per_residue)" \
    "$(field "$line" pcie_remaining_tib_per_residue)" "$(field "$line" cpu_high_groups)" \
    "$line" >>"$out"
  printf '%s\n' "$residue"
}

reference_residue=""
for ((r=1; r<=REPEATS; ++r)); do
  if ((r % 2 == 1)); then
    order=ascending
    run_thresholds=("${thresholds[@]}")
  else
    order=descending
    run_thresholds=()
    for ((i=${#thresholds[@]}-1; i>=0; --i)); do run_thresholds+=("${thresholds[i]}"); done
  fi
  echo "repeat $r/$REPEATS ($order)" >&2
  for threshold in "${run_thresholds[@]}"; do
    echo "  CPU_HIGH_MAX_MIB=$threshold" >&2
    residue="$(run_one "$r" "$order" "$threshold")"
    if [[ -z "$reference_residue" ]]; then reference_residue="$residue"; fi
    if [[ "$residue" != "$reference_residue" ]]; then
      echo "residue mismatch threshold=$threshold got=$residue reference=$reference_residue" >&2
      exit 5
    fi
    if [[ -n "$EXPECTED_RESIDUE" && "$residue" != "$EXPECTED_RESIDUE" ]]; then
      echo "unexpected residue threshold=$threshold got=$residue expected=$EXPECTED_RESIDUE" >&2
      exit 6
    fi
  done
done

awk -F '\t' '
  NR==1 {next}
  { n[$3]++; wall[$3]+=$5; high[$3]+=$9; low[$3]+=$11; rem[$3]+=$13 }
  END {
    for (t in n)
      printf("summary threshold_mib=%s runs=%d mean_wall_s=%.9f mean_cpu_high_wall_s=%.9f mean_cpu_low_wall_s=%.9f mean_pcie_remaining_tib=%.9f\n",
             t,n[t],wall[t]/n[t],high[t]/n[t],low[t]/n[t],rem[t]/n[t]);
  }
' "$out" | sort -t= -k2,2n

best="$(awk -F '\t' 'NR>1 {s[$3]+=$5;n[$3]++} END {for(t in n){m=s[t]/n[t]; if(best==""||m<best){best=m;bt=t}} printf "%s %.9f",bt,best}' "$out")"
echo "best_threshold_mib=${best%% *} mean_wall_s=${best#* }"
echo "results=$out"
echo "metadata=$meta"
