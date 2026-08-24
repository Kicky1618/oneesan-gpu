#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$ROOT"

N="${N:-27}"; ARCH="${ARCH:-native}"; MODULUS="${MODULUS:-4294967291}"; GPU_TARGET_MIB="${GPU_TARGET_MIB:-12288}"
CPU_WORKERS="${CPU_WORKERS:-32}"; CPU_HIGH_WORKERS="${CPU_HIGH_WORKERS:-$CPU_WORKERS}"; CPU_HIGH_OVERLAP="${CPU_HIGH_OVERLAP:-0}"
CPU_HIGH_CPU_LIST="${CPU_HIGH_CPU_LIST:-}"; CPU_LOW_CPU_LIST="${CPU_LOW_CPU_LIST:-}"; CPU_LOW_SCHEDULE="${CPU_LOW_SCHEDULE:-dynamic}"
CPU_LOW_DOMAIN_SIZE="${CPU_LOW_DOMAIN_SIZE:-}"; CPU_LOW_DOMAIN_REFINE="${CPU_LOW_DOMAIN_REFINE:-1}"
THRESHOLD_MIB="${THRESHOLD_MIB:-256}"; GROUPS_FILE="${GROUPS_FILE:-}"; REPEATS="${REPEATS:-2}"; BUILD="${BUILD:-1}"
EXPECTED_RESIDUE="${EXPECTED_RESIDUE:-}"; OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_high_policy_ab}"

[[ -n "$GROUPS_FILE" ]] || { echo "GROUPS_FILE is required" >&2; exit 2; }
[[ -f "$GROUPS_FILE" ]] || { echo "missing GROUPS_FILE: $GROUPS_FILE" >&2; exit 2; }
[[ "$THRESHOLD_MIB" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || { echo "THRESHOLD_MIB must be non-negative" >&2; exit 2; }
(( N >= 2 && N <= 27 && GPU_TARGET_MIB > 0 && CPU_WORKERS > 0 && CPU_HIGH_WORKERS > 0 && REPEATS > 0 )) || { echo "invalid benchmark parameters" >&2; exit 2; }
[[ "$CPU_HIGH_OVERLAP" == 0 || "$CPU_HIGH_OVERLAP" == 1 ]] || { echo "CPU_HIGH_OVERLAP must be 0 or 1" >&2; exit 2; }
[[ "$CPU_LOW_SCHEDULE" == dynamic || "$CPU_LOW_SCHEDULE" == sticky || "$CPU_LOW_SCHEDULE" == contiguous || "$CPU_LOW_SCHEDULE" == domain ]] || { echo "CPU_LOW_SCHEDULE must be dynamic, sticky, contiguous, or domain" >&2; exit 2; }
[[ "$CPU_LOW_DOMAIN_REFINE" == 0 || "$CPU_LOW_DOMAIN_REFINE" == 1 ]] || { echo "CPU_LOW_DOMAIN_REFINE must be 0 or 1" >&2; exit 2; }
if [[ "$CPU_LOW_SCHEDULE" == domain ]]; then [[ "$CPU_LOW_DOMAIN_SIZE" =~ ^[1-9][0-9]*$ ]] && (( CPU_LOW_DOMAIN_SIZE <= CPU_WORKERS )) || { echo "CPU_LOW_DOMAIN_SIZE must be in 1..CPU_WORKERS for domain schedule" >&2; exit 2; }; fi
if [[ -z "$EXPECTED_RESIDUE" && "$N:$MODULUS" == "21:4294967291" ]]; then EXPECTED_RESIDUE=998035516; fi

if [[ "$BUILD" != 0 ]]; then N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-factorized-hybrid-sparse.sh; fi
bin="$ROOT/build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n${N}"; [[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }
mkdir -p "$OUT_DIR"; ts="$(date -u +%Y%m%dT%H%M%SZ)"; out="$OUT_DIR/policy-ab-n${N}-${ts}.tsv"; meta="$OUT_DIR/policy-ab-n${N}-${ts}.meta"

file_sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
field() { local line="$1" key="$2" token; for token in $line; do if [[ "$token" == "$key="* ]]; then printf '%s\n' "${token#*=}"; return 0; fi; done; return 1; }

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
cpu_high_overlap=$CPU_HIGH_OVERLAP
cpu_high_cpu_list=${CPU_HIGH_CPU_LIST:-none}
cpu_low_cpu_list=${CPU_LOW_CPU_LIST:-none}
cpu_low_schedule=$CPU_LOW_SCHEDULE
cpu_low_domain_size=${CPU_LOW_DOMAIN_SIZE:-none}
cpu_low_domain_refine=$CPU_LOW_DOMAIN_REFINE
threshold_mib=$THRESHOLD_MIB
groups_file=$GROUPS_FILE
groups_file_sha256=$(file_sha256 "$GROUPS_FILE")
groups_file_entries=$(awk '!/^[[:space:]]*(#|$)/{n++} END{print n+0}' "$GROUPS_FILE")
repeats=$REPEATS
expected_residue=${EXPECTED_RESIDUE:-unknown}
binary_sha256=$(file_sha256 "$bin")
EOF

printf 'repeat\torder\tvariant\tresidue\twall_s\th2d_s\tgpu_kernel_s\td2h_s\tcpu_high_wall_s\tcpu_low_wall_s\tpcie_removed_tib\tcpu_high_groups\tselection_hash\traw\n' >"$out"
run_one() {
  local repeat="$1" order="$2" variant="$3" line residue got_schedule got_domain got_refine
  if [[ "$variant" == threshold ]]; then
    line="$(CPU_HIGH_MODE=direct CPU_HIGH_MAX_MIB="$THRESHOLD_MIB" CPU_HIGH_GROUPS_FILE= CPU_HIGH_WORKERS="$CPU_HIGH_WORKERS" CPU_HIGH_OVERLAP="$CPU_HIGH_OVERLAP" CPU_HIGH_CPU_LIST="$CPU_HIGH_CPU_LIST" CPU_LOW_CPU_LIST="$CPU_LOW_CPU_LIST" CPU_LOW_SCHEDULE="$CPU_LOW_SCHEDULE" CPU_LOW_DOMAIN_SIZE="$CPU_LOW_DOMAIN_SIZE" CPU_LOW_DOMAIN_REFINE="$CPU_LOW_DOMAIN_REFINE" "$bin" "$N" "$MODULUS" "$GPU_TARGET_MIB" "$CPU_WORKERS" | tail -n1)"
  else
    line="$(CPU_HIGH_MODE=direct CPU_HIGH_MAX_MIB=0 CPU_HIGH_GROUPS_FILE="$GROUPS_FILE" CPU_HIGH_WORKERS="$CPU_HIGH_WORKERS" CPU_HIGH_OVERLAP="$CPU_HIGH_OVERLAP" CPU_HIGH_CPU_LIST="$CPU_HIGH_CPU_LIST" CPU_LOW_CPU_LIST="$CPU_LOW_CPU_LIST" CPU_LOW_SCHEDULE="$CPU_LOW_SCHEDULE" CPU_LOW_DOMAIN_SIZE="$CPU_LOW_DOMAIN_SIZE" CPU_LOW_DOMAIN_REFINE="$CPU_LOW_DOMAIN_REFINE" "$bin" "$N" "$MODULUS" "$GPU_TARGET_MIB" "$CPU_WORKERS" | tail -n1)"
  fi
  residue="$(field "$line" residue)"; got_schedule="$(field "$line" cpu_low_schedule)"
  [[ "$got_schedule" == "$CPU_LOW_SCHEDULE" ]] || { echo "LOW schedule provenance mismatch requested=$CPU_LOW_SCHEDULE got=$got_schedule" >&2; exit 7; }
  if [[ "$CPU_LOW_SCHEDULE" == domain ]]; then got_domain="$(field "$line" cpu_low_domain_size)"; [[ "$got_domain" == "$CPU_LOW_DOMAIN_SIZE" ]] || { echo "LOW domain provenance mismatch requested=$CPU_LOW_DOMAIN_SIZE got=$got_domain" >&2; exit 7; }; fi
  got_refine="$(field "$line" cpu_low_domain_refine)"
  [[ "$got_refine" == "$CPU_LOW_DOMAIN_REFINE" ]] || { echo "LOW refine provenance mismatch requested=$CPU_LOW_DOMAIN_REFINE got=$got_refine" >&2; exit 7; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$repeat" "$order" "$variant" "$residue" "$(field "$line" wall_s)" "$(field "$line" h2d_s)" "$(field "$line" gpu_kernel_s)" "$(field "$line" d2h_s)" "$(field "$line" cpu_high_wall_s)" "$(field "$line" cpu_low_wall_s)" "$(field "$line" pcie_removed_tib_per_residue)" "$(field "$line" cpu_high_groups)" "$(field "$line" cpu_high_selection_hash)" "$line" >>"$out"
  printf '%s\n' "$residue"
}

reference_residue=""
for ((r=1; r<=REPEATS; ++r)); do
  if ((r % 2 == 1)); then order=threshold-first; variants=(threshold policy); else order=policy-first; variants=(policy threshold); fi
  echo "repeat $r/$REPEATS low_schedule=$CPU_LOW_SCHEDULE low_domain_size=${CPU_LOW_DOMAIN_SIZE:-none} low_domain_refine=$CPU_LOW_DOMAIN_REFINE ($order)" >&2
  for variant in "${variants[@]}"; do echo "  $variant" >&2; residue="$(run_one "$r" "$order" "$variant")"; if [[ -z "$reference_residue" ]]; then reference_residue="$residue"; fi; [[ "$residue" == "$reference_residue" ]] || { echo "residue mismatch variant=$variant got=$residue reference=$reference_residue" >&2; exit 5; }; if [[ -n "$EXPECTED_RESIDUE" && "$residue" != "$EXPECTED_RESIDUE" ]]; then echo "unexpected residue variant=$variant got=$residue expected=$EXPECTED_RESIDUE" >&2; exit 6; fi; done
done

awk -F '\t' 'NR==1 {next} { n[$3]++; wall[$3]+=$5; dma[$3]+=$6+$8; gpu[$3]+=$7; high[$3]+=$9; low[$3]+=$10; removed[$3]+=$11 } END { for (v in n) printf("summary variant=%s runs=%d mean_wall_s=%.9f mean_dma_s=%.9f mean_gpu_kernel_s=%.9f mean_cpu_high_wall_s=%.9f mean_cpu_low_wall_s=%.9f mean_pcie_removed_tib=%.9f\n",v,n[v],wall[v]/n[v],dma[v]/n[v],gpu[v]/n[v],high[v]/n[v],low[v]/n[v],removed[v]/n[v]); if (n["threshold"] && n["policy"]) { tw=wall["threshold"]/n["threshold"]; pw=wall["policy"]/n["policy"]; printf("comparison threshold_mean_wall_s=%.9f policy_mean_wall_s=%.9f policy_speedup=%.9fx policy_saved_s=%.9f\n",tw,pw,tw/pw,tw-pw); } }' "$out" | sort

echo "results=$out"; echo "metadata=$meta"
