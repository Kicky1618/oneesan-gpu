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
CPU_HIGH_MODE="${CPU_HIGH_MODE:-scratch}"
CPU_HIGH_OVERLAP="${CPU_HIGH_OVERLAP:-0}"
CPU_HIGH_CPU_LIST="${CPU_HIGH_CPU_LIST:-}"
CPU_LOW_CPU_LIST="${CPU_LOW_CPU_LIST:-}"
CPU_LOW_SCHEDULE="${CPU_LOW_SCHEDULE:-dynamic}"
CPU_LOW_DOMAIN_SIZE="${CPU_LOW_DOMAIN_SIZE:-}"
THRESHOLDS="${THRESHOLDS:-0 64 128 256 512 1024}"
REPEATS="${REPEATS:-1}"
BUILD="${BUILD:-1}"
ANALYZE="${ANALYZE:-1}"
COST_PLAN="${COST_PLAN:-auto}"
GENERATE_POLICY="${GENERATE_POLICY:-1}"
EXPECTED_RESIDUE="${EXPECTED_RESIDUE:-}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_high_sweep}"

if [[ -z "$EXPECTED_RESIDUE" && "$N:$MODULUS" == "21:4294967291" ]]; then
  EXPECTED_RESIDUE=998035516
fi
if (( N < 2 || N > 27 || GPU_TARGET_MIB <= 0 || CPU_WORKERS <= 0 || CPU_HIGH_WORKERS <= 0 || REPEATS <= 0 )); then
  echo "invalid benchmark parameters" >&2
  exit 2
fi
if [[ "$CPU_HIGH_MODE" != scratch && "$CPU_HIGH_MODE" != direct ]]; then
  echo "CPU_HIGH_MODE must be scratch or direct" >&2
  exit 2
fi
if [[ "$CPU_HIGH_OVERLAP" != 0 && "$CPU_HIGH_OVERLAP" != 1 ]]; then
  echo "CPU_HIGH_OVERLAP must be 0 or 1" >&2
  exit 2
fi
if [[ "$CPU_LOW_SCHEDULE" != dynamic && "$CPU_LOW_SCHEDULE" != sticky && "$CPU_LOW_SCHEDULE" != contiguous && "$CPU_LOW_SCHEDULE" != domain ]]; then
  echo "CPU_LOW_SCHEDULE must be dynamic, sticky, contiguous, or domain" >&2
  exit 2
fi
if [[ "$CPU_LOW_SCHEDULE" == domain ]]; then
  if [[ ! "$CPU_LOW_DOMAIN_SIZE" =~ ^[1-9][0-9]*$ ]] || (( CPU_LOW_DOMAIN_SIZE > CPU_WORKERS )); then
    echo "CPU_LOW_DOMAIN_SIZE must be in 1..CPU_WORKERS for domain schedule" >&2
    exit 2
  fi
fi
if [[ "$ANALYZE" != 0 && "$ANALYZE" != 1 ]]; then
  echo "ANALYZE must be 0 or 1" >&2
  exit 2
fi
if [[ "$GENERATE_POLICY" != 0 && "$GENERATE_POLICY" != 1 ]]; then
  echo "GENERATE_POLICY must be 0 or 1" >&2
  exit 2
fi

read -r -a thresholds <<<"$THRESHOLDS"
if ((${#thresholds[@]} == 0)); then
  echo "THRESHOLDS must contain at least one value" >&2
  exit 2
fi
for x in "${thresholds[@]}"; do
  [[ "$x" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || {
    echo "invalid threshold: $x" >&2; exit 2;
  }
done

if [[ "$BUILD" != 0 ]]; then
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-factorized-hybrid-sparse.sh
fi
bin="$ROOT/build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/cpu-high-${CPU_HIGH_MODE}-overlap${CPU_HIGH_OVERLAP}-n${N}-${ts}.tsv"
meta="$OUT_DIR/cpu-high-${CPU_HIGH_MODE}-overlap${CPU_HIGH_OVERLAP}-n${N}-${ts}.meta"
analysis_out="$OUT_DIR/cpu-high-${CPU_HIGH_MODE}-overlap${CPU_HIGH_OVERLAP}-n${N}-${ts}.analysis.txt"
cost_plan_path=""
cost_plan_log=""
policy_out=""
policy_log=""
numa_analysis_out=""

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
analysis_field() {
  local prefix="$1" name="$2"
  awk -v o="$CPU_HIGH_OVERLAP" -v p="$prefix" -v n="$name" '
    $1==p && $2=="mode=direct" && $3==("overlap=" o) {
      for (i=4; i<=NF; ++i) if ($i ~ ("^" n "=")) {
        sub("^" n "=", "", $i); print $i; exit
      }
    }' "$analysis_out"
}

if [[ "$CPU_HIGH_MODE" == direct && "$COST_PLAN" != none && -n "$COST_PLAN" ]]; then
  if [[ "$COST_PLAN" == auto ]]; then
    if [[ "$BUILD" != 0 ]]; then
      N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-high-cost-plan.sh
    fi
    cost_bin="$ROOT/build/ramstream32_cpu_high_cost_plan_n${N}"
    [[ -x "$cost_bin" ]] || { echo "missing executable: $cost_bin" >&2; exit 3; }
    cost_plan_path="$OUT_DIR/cpu-high-cost-plan-n${N}-${ts}.tsv"
    cost_plan_log="$OUT_DIR/cpu-high-cost-plan-n${N}-${ts}.log"
    "$cost_bin" "$N" >"$cost_plan_path" 2>"$cost_plan_log"
  else
    cost_plan_path="$COST_PLAN"
    [[ -f "$cost_plan_path" ]] || { echo "missing COST_PLAN: $cost_plan_path" >&2; exit 3; }
  fi
fi

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
cpu_high_mode=$CPU_HIGH_MODE
cpu_high_overlap=$CPU_HIGH_OVERLAP
cpu_high_cpu_list=${CPU_HIGH_CPU_LIST:-none}
cpu_low_cpu_list=${CPU_LOW_CPU_LIST:-none}
cpu_low_schedule=$CPU_LOW_SCHEDULE
cpu_low_domain_size=${CPU_LOW_DOMAIN_SIZE:-none}
thresholds=$THRESHOLDS
repeats=$REPEATS
analyze=$ANALYZE
cost_plan=${cost_plan_path:-none}
generate_policy=$GENERATE_POLICY
expected_residue=${EXPECTED_RESIDUE:-unknown}
binary_sha256=$(file_sha256 "$bin")
EOF
if [[ -n "$cost_plan_path" ]]; then
  echo "cost_plan_sha256=$(file_sha256 "$cost_plan_path")" >>"$meta"
fi

printf 'repeat\torder\tmode\toverlap\tthreshold_mib\tresidue\twall_s\th2d_s\tgpu_kernel_s\td2h_s\tcpu_high_wall_s\tcpu_high_kernel_sum_s\tcpu_low_wall_s\tpcie_removed_tib\tpcie_remaining_tib\tcpu_high_groups\traw\n' >"$out"

run_one() {
  local repeat="$1" order="$2" threshold="$3"
  local line residue
  line="$(CPU_HIGH_MAX_MIB="$threshold" CPU_HIGH_WORKERS="$CPU_HIGH_WORKERS" \
    CPU_HIGH_MODE="$CPU_HIGH_MODE" CPU_HIGH_OVERLAP="$CPU_HIGH_OVERLAP" \
    CPU_HIGH_CPU_LIST="$CPU_HIGH_CPU_LIST" CPU_LOW_CPU_LIST="$CPU_LOW_CPU_LIST" \
    CPU_LOW_SCHEDULE="$CPU_LOW_SCHEDULE" CPU_LOW_DOMAIN_SIZE="$CPU_LOW_DOMAIN_SIZE" \
    "$bin" "$N" "$MODULUS" "$GPU_TARGET_MIB" "$CPU_WORKERS" | tail -n1)"
  residue="$(field "$line" residue)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repeat" "$order" "$CPU_HIGH_MODE" "$CPU_HIGH_OVERLAP" "$threshold" "$residue" \
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
  echo "repeat $r/$REPEATS mode=$CPU_HIGH_MODE overlap=$CPU_HIGH_OVERLAP low_schedule=$CPU_LOW_SCHEDULE ($order)" >&2
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
  { n[$5]++; wall[$5]+=$7; high[$5]+=$11; low[$5]+=$13; rem[$5]+=$15 }
  END {
    for (t in n)
      printf("summary threshold_mib=%s runs=%d mean_wall_s=%.9f mean_cpu_high_wall_s=%.9f mean_cpu_low_wall_s=%.9f mean_pcie_remaining_tib=%.9f\n",
             t,n[t],wall[t]/n[t],high[t]/n[t],low[t]/n[t],rem[t]/n[t]);
  }
' "$out" | sort -t= -k2,2n

best="$(awk -F '\t' 'NR>1 {s[$5]+=$7;n[$5]++} END {for(t in n){m=s[t]/n[t]; if(best==""||m<best){best=m;bt=t}} printf "%s %.9f",bt,best}' "$out")"
echo "mode=$CPU_HIGH_MODE overlap=$CPU_HIGH_OVERLAP low_schedule=$CPU_LOW_SCHEDULE best_threshold_mib=${best%% *} mean_wall_s=${best#* }"

if [[ "$ANALYZE" != 0 ]]; then
  analyze_args=(scripts/tools/analyze_cpu_high_sweep.py "$out")
  if [[ -n "$cost_plan_path" ]]; then analyze_args+=(--cost-plan "$cost_plan_path"); fi
  python3 "${analyze_args[@]}" | tee "$analysis_out"

  if [[ "$CPU_HIGH_MODE" == direct && "$GENERATE_POLICY" != 0 && -n "$cost_plan_path" ]]; then
    pcie_rate="$(analysis_field affine_calibration pcie_gib_s)"
    pcie_copy_overhead="$(analysis_field affine_calibration pcie_copy_overhead_us)"
    cpu_rate="$(analysis_field affine_calibration cpu_gcell_s)"
    cpu_group_overhead="$(analysis_field affine_calibration cpu_group_overhead_us)"
    gpu_rate="$(analysis_field affine_calibration gpu_gstate_s)"
    gpu_group_overhead="$(analysis_field affine_calibration gpu_group_overhead_us)"

    [[ -n "$pcie_rate" ]] || pcie_rate="$(analysis_field calibration pcie_gib_s)"
    [[ -n "$cpu_rate" ]] || cpu_rate="$(analysis_field calibration cpu_gcell_s)"
    [[ -n "$gpu_rate" ]] || gpu_rate="$(analysis_field calibration gpu_gstate_s)"

    if [[ -n "$pcie_rate" && -n "$cpu_rate" ]]; then
      policy_out="$OUT_DIR/cpu-high-cost-policy-overlap${CPU_HIGH_OVERLAP}-n${N}-${ts}.groups"
      policy_log="$OUT_DIR/cpu-high-cost-policy-overlap${CPU_HIGH_OVERLAP}-n${N}-${ts}.log"
      planner_args=(
        scripts/tools/plan_cpu_high_groups.py "$cost_plan_path"
        --pcie-gib-s "$pcie_rate" --cpu-gcell-s "$cpu_rate"
        --gpu-target-mib "$GPU_TARGET_MIB"
      )
      if [[ -n "$pcie_copy_overhead" ]]; then
        planner_args+=(--pcie-copy-overhead-us "$pcie_copy_overhead")
      fi
      if [[ -n "$gpu_rate" ]]; then planner_args+=(--gpu-gstate-s "$gpu_rate"); fi
      if [[ -n "$cpu_group_overhead" ]]; then
        planner_args+=(--group-overhead-us "$cpu_group_overhead")
      fi
      if [[ -n "$gpu_group_overhead" ]]; then
        planner_args+=(--gpu-group-overhead-us "$gpu_group_overhead")
      fi
      if [[ "$CPU_HIGH_OVERLAP" == 1 ]]; then planner_args+=(--overlap); fi
      python3 "${planner_args[@]}" >"$policy_out" 2>"$policy_log"
      numa_analysis_out="$OUT_DIR/cpu-high-cost-policy-overlap${CPU_HIGH_OVERLAP}-n${N}-${ts}.numa.txt"
      python3 scripts/tools/analyze_cpu_high_numa.py "$cost_plan_path" \
        --groups-file "$policy_out" >"$numa_analysis_out"
      echo "policy=$policy_out"
      echo "policy_log=$policy_log"
      echo "numa_analysis=$numa_analysis_out"
    else
      echo "cost-model policy not generated: calibration rates unavailable" >&2
    fi
  fi
fi

echo "results=$out"
echo "metadata=$meta"
if [[ -n "$cost_plan_path" ]]; then echo "cost_plan=$cost_plan_path"; fi
if [[ -n "$cost_plan_log" ]]; then echo "cost_plan_log=$cost_plan_log"; fi
if [[ "$ANALYZE" != 0 ]]; then echo "analysis=$analysis_out"; fi
