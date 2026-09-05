#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$ROOT"

N="${N:-27}"
ARCH="${ARCH:-native}"
MODULUS="${MODULUS:-4294967291}"
GPU_TARGET_MIB="${GPU_TARGET_MIB:-12288}"
CPU_WORKERS="${CPU_WORKERS:-32}"
CPU_HIGH_WORKERS="${CPU_HIGH_WORKERS:-$CPU_WORKERS}"
CPU_HIGH_MODE="${CPU_HIGH_MODE:-direct}"
CPU_HIGH_OVERLAP="${CPU_HIGH_OVERLAP:-1}"
CPU_HIGH_MAX_MIB="${CPU_HIGH_MAX_MIB:-256}"
CPU_HIGH_GROUPS_FILE="${CPU_HIGH_GROUPS_FILE:-}"
CPU_HIGH_CPU_LIST="${CPU_HIGH_CPU_LIST:-}"
CPU_LOW_CPU_LIST="${CPU_LOW_CPU_LIST:-}"
CPU_LOW_DOMAIN_SIZE="${CPU_LOW_DOMAIN_SIZE:-}"
CPU_LOW_DOMAIN_REFINE="${CPU_LOW_DOMAIN_REFINE:-1}"
CPU_LOW_DOMAIN_PAGE_TIEBREAK="${CPU_LOW_DOMAIN_PAGE_TIEBREAK:-0}"
REPEATS="${REPEATS:-8}"
BUILD="${BUILD:-1}"
EXPECTED_RESIDUE="${EXPECTED_RESIDUE:-}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_schedule_compare}"

if (( N < 2 || N > 27 || GPU_TARGET_MIB <= 0 || CPU_WORKERS <= 0 || CPU_HIGH_WORKERS <= 0 || REPEATS <= 0 )); then echo "invalid benchmark parameters" >&2; exit 2; fi
if [[ "$CPU_HIGH_MODE" != scratch && "$CPU_HIGH_MODE" != direct ]]; then echo "CPU_HIGH_MODE must be scratch or direct" >&2; exit 2; fi
if [[ "$CPU_HIGH_OVERLAP" != 0 && "$CPU_HIGH_OVERLAP" != 1 ]]; then echo "CPU_HIGH_OVERLAP must be 0 or 1" >&2; exit 2; fi
if [[ ! "$CPU_LOW_DOMAIN_SIZE" =~ ^[1-9][0-9]*$ ]] || (( CPU_LOW_DOMAIN_SIZE > CPU_WORKERS )); then echo "CPU_LOW_DOMAIN_SIZE must be a positive integer <= CPU_WORKERS" >&2; exit 2; fi
if [[ "$CPU_LOW_DOMAIN_REFINE" != 0 && "$CPU_LOW_DOMAIN_REFINE" != 1 ]]; then echo "CPU_LOW_DOMAIN_REFINE must be 0 or 1" >&2; exit 2; fi
if [[ "$CPU_LOW_DOMAIN_PAGE_TIEBREAK" != 0 && "$CPU_LOW_DOMAIN_PAGE_TIEBREAK" != 1 ]]; then echo "CPU_LOW_DOMAIN_PAGE_TIEBREAK must be 0 or 1" >&2; exit 2; fi
if [[ "$CPU_LOW_DOMAIN_PAGE_TIEBREAK" == 1 && "$CPU_LOW_DOMAIN_REFINE" != 1 ]]; then echo "CPU_LOW_DOMAIN_PAGE_TIEBREAK=1 requires CPU_LOW_DOMAIN_REFINE=1" >&2; exit 2; fi
if [[ -n "$CPU_HIGH_GROUPS_FILE" && ! -f "$CPU_HIGH_GROUPS_FILE" ]]; then echo "missing CPU_HIGH_GROUPS_FILE: $CPU_HIGH_GROUPS_FILE" >&2; exit 2; fi
if [[ -z "$EXPECTED_RESIDUE" && "$N:$MODULUS" == "21:4294967291" ]]; then EXPECTED_RESIDUE=998035516; fi

if [[ "$BUILD" != 0 ]]; then N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-factorized-hybrid-sparse.sh; fi
bin="$ROOT/build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/low-schedule-compare-n${N}-${ts}.tsv"
meta="$OUT_DIR/low-schedule-compare-n${N}-${ts}.meta"

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
cpu_high_mode=$CPU_HIGH_MODE
cpu_high_overlap=$CPU_HIGH_OVERLAP
cpu_high_max_mib=$CPU_HIGH_MAX_MIB
cpu_high_groups_file=${CPU_HIGH_GROUPS_FILE:-none}
cpu_high_cpu_list=${CPU_HIGH_CPU_LIST:-none}
cpu_low_cpu_list=${CPU_LOW_CPU_LIST:-none}
cpu_low_domain_size=$CPU_LOW_DOMAIN_SIZE
cpu_low_domain_refine=$CPU_LOW_DOMAIN_REFINE
cpu_low_domain_page_tiebreak=$CPU_LOW_DOMAIN_PAGE_TIEBREAK
repeats=$REPEATS
schedules=dynamic,sticky,contiguous,domain
order_design=cyclic-latin-4
numa_sampling=disabled
expected_residue=${EXPECTED_RESIDUE:-unknown}
binary_sha256=$(file_sha256 "$bin")
EOF
if [[ -n "$CPU_HIGH_GROUPS_FILE" ]]; then echo "cpu_high_groups_file_sha256=$(file_sha256 "$CPU_HIGH_GROUPS_FILE")" >>"$meta"; fi

printf 'repeat\torder\tposition\tschedule\tdomain_page_tiebreak\tresidue\twall_s\tcpu_low_wall_s\tcpu_low_kernel_sum_s\tcpu_low_schedule_build_s\tcpu_low_contiguous_optimal_cap\tcpu_low_domain_size\tcpu_low_domain_outer_normalized_cap\tcpu_low_domain_active_domains\tcpu_low_domain_refined_boundaries\tcpu_low_domain_refined_job_moves\tcpu_low_domain_page_boundary_moves\tcpu_low_domain_page_moved_jobs\tcpu_low_worker_start_s\tcpu_high_wall_s\th2d_s\tgpu_kernel_s\td2h_s\traw\n' >"$out"

run_one() {
  local repeat="$1" order="$2" position="$3" schedule="$4"
  local page=0
  [[ "$schedule" == domain ]] && page="$CPU_LOW_DOMAIN_PAGE_TIEBREAK"
  local line residue got_schedule got_domain got_refine got_page
  line="$(CPU_HIGH_MODE="$CPU_HIGH_MODE" CPU_HIGH_OVERLAP="$CPU_HIGH_OVERLAP" CPU_HIGH_MAX_MIB="$CPU_HIGH_MAX_MIB" CPU_HIGH_GROUPS_FILE="$CPU_HIGH_GROUPS_FILE" CPU_HIGH_WORKERS="$CPU_HIGH_WORKERS" CPU_HIGH_CPU_LIST="$CPU_HIGH_CPU_LIST" CPU_LOW_CPU_LIST="$CPU_LOW_CPU_LIST" CPU_LOW_SCHEDULE="$schedule" CPU_LOW_DOMAIN_SIZE="$CPU_LOW_DOMAIN_SIZE" CPU_LOW_DOMAIN_REFINE="$CPU_LOW_DOMAIN_REFINE" CPU_LOW_DOMAIN_PAGE_TIEBREAK="$page" RAMSTREAM_NUMA_SAMPLE_MIB=0 "$bin" "$N" "$MODULUS" "$GPU_TARGET_MIB" "$CPU_WORKERS" | tail -n1)"
  residue="$(field "$line" residue)"; got_schedule="$(field "$line" cpu_low_schedule)"; got_domain="$(field "$line" cpu_low_domain_size)"; got_refine="$(field "$line" cpu_low_domain_refine)"; got_page="$(field "$line" cpu_low_domain_page_tiebreak)"
  [[ "$got_schedule" == "$schedule" ]] || { echo "schedule provenance mismatch requested=$schedule got=$got_schedule" >&2; exit 7; }
  if [[ "$schedule" == domain && "$got_domain" != "$CPU_LOW_DOMAIN_SIZE" ]]; then echo "domain provenance mismatch requested=$CPU_LOW_DOMAIN_SIZE got=$got_domain" >&2; exit 7; fi
  [[ "$got_refine" == "$CPU_LOW_DOMAIN_REFINE" ]] || { echo "refine provenance mismatch requested=$CPU_LOW_DOMAIN_REFINE got=$got_refine schedule=$schedule" >&2; exit 7; }
  [[ "$got_page" == "$page" ]] || { echo "page provenance mismatch requested=$page got=$got_page schedule=$schedule" >&2; exit 7; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repeat" "$order" "$position" "$schedule" "$page" "$residue" \
    "$(field "$line" wall_s)" "$(field "$line" cpu_low_wall_s)" "$(field "$line" cpu_low_kernel_sum_s)" "$(field "$line" cpu_low_schedule_build_s)" \
    "$(field "$line" cpu_low_contiguous_optimal_cap)" "$got_domain" "$(field "$line" cpu_low_domain_outer_normalized_cap)" "$(field "$line" cpu_low_domain_active_domains)" \
    "$(field "$line" cpu_low_domain_refined_boundaries)" "$(field "$line" cpu_low_domain_refined_job_moves)" "$(field "$line" cpu_low_domain_page_boundary_moves)" "$(field "$line" cpu_low_domain_page_moved_jobs)" \
    "$(field "$line" cpu_low_worker_start_s)" "$(field "$line" cpu_high_wall_s)" "$(field "$line" h2d_s)" "$(field "$line" gpu_kernel_s)" "$(field "$line" d2h_s)" "$line" >>"$out"
  printf '%s\n' "$residue"
}

reference_residue=""
for ((r=1; r<=REPEATS; ++r)); do
  case $(((r - 1) % 4)) in
    0) order=dynamic-sticky-contiguous-domain; schedules=(dynamic sticky contiguous domain) ;;
    1) order=sticky-contiguous-domain-dynamic; schedules=(sticky contiguous domain dynamic) ;;
    2) order=contiguous-domain-dynamic-sticky; schedules=(contiguous domain dynamic sticky) ;;
    3) order=domain-dynamic-sticky-contiguous; schedules=(domain dynamic sticky contiguous) ;;
  esac
  echo "repeat $r/$REPEATS domain_size=$CPU_LOW_DOMAIN_SIZE domain_refine=$CPU_LOW_DOMAIN_REFINE domain_page=$CPU_LOW_DOMAIN_PAGE_TIEBREAK ($order)" >&2
  position=0
  for schedule in "${schedules[@]}"; do
    ((position+=1)); echo "  position=$position schedule=$schedule" >&2
    residue="$(run_one "$r" "$order" "$position" "$schedule")"
    if [[ -z "$reference_residue" ]]; then reference_residue="$residue"; fi
    [[ "$residue" == "$reference_residue" ]] || { echo "residue mismatch schedule=$schedule got=$residue reference=$reference_residue" >&2; exit 5; }
    if [[ -n "$EXPECTED_RESIDUE" && "$residue" != "$EXPECTED_RESIDUE" ]]; then echo "unexpected residue schedule=$schedule got=$residue expected=$EXPECTED_RESIDUE" >&2; exit 6; fi
  done
done

awk -F '\t' '
  NR==1 {next}
  {
    s=$4; n[s]++; wall[s]+=$7; low[s]+=$8; kernel[s]+=$9; build[s]+=$10;
    refine_boundaries[s]+=$15; refine_moves[s]+=$16; page_moves[s]+=$17; page_jobs[s]+=$18;
    start[s]+=$19; high[s]+=$20; pos[s,$3]++;
  }
  END {
    modes[1]="dynamic"; modes[2]="sticky"; modes[3]="contiguous"; modes[4]="domain";
    for (i=1;i<=4;i++) { s=modes[i]; if (!n[s]) continue;
      printf("summary schedule=%s runs=%d mean_wall_s=%.9f mean_cpu_low_wall_s=%.9f mean_cpu_low_kernel_sum_s=%.9f mean_schedule_build_s=%.9f mean_refined_boundaries=%.3f mean_refined_job_moves=%.3f mean_page_boundary_moves=%.3f mean_page_moved_jobs=%.3f mean_worker_start_s=%.9f mean_cpu_high_wall_s=%.9f positions=%d/%d/%d/%d\n", s,n[s],wall[s]/n[s],low[s]/n[s],kernel[s]/n[s],build[s]/n[s],refine_boundaries[s]/n[s],refine_moves[s]/n[s],page_moves[s]/n[s],page_jobs[s]/n[s],start[s]/n[s],high[s]/n[s],pos[s,1]+0,pos[s,2]+0,pos[s,3]+0,pos[s,4]+0);
    }
    if (n["dynamic"] && n["sticky"] && n["contiguous"] && n["domain"]) {
      dw=wall["dynamic"]/n["dynamic"]; sw=wall["sticky"]/n["sticky"]; cw=wall["contiguous"]/n["contiguous"]; ow=wall["domain"]/n["domain"];
      dl=low["dynamic"]/n["dynamic"]; sl=low["sticky"]/n["sticky"]; cl=low["contiguous"]/n["contiguous"]; ol=low["domain"]/n["domain"];
      printf("comparison dynamic_wall_s=%.9f sticky_wall_s=%.9f contiguous_wall_s=%.9f domain_wall_s=%.9f sticky_vs_dynamic_wall=%.9fx contiguous_vs_dynamic_wall=%.9fx domain_vs_dynamic_wall=%.9fx domain_vs_sticky_wall=%.9fx domain_vs_contiguous_wall=%.9fx dynamic_low_s=%.9f sticky_low_s=%.9f contiguous_low_s=%.9f domain_low_s=%.9f sticky_vs_dynamic_low=%.9fx contiguous_vs_dynamic_low=%.9fx domain_vs_dynamic_low=%.9fx domain_vs_sticky_low=%.9fx domain_vs_contiguous_low=%.9fx\n", dw,sw,cw,ow,dw/sw,dw/cw,dw/ow,sw/ow,cw/ow,dl,sl,cl,ol,dl/sl,dl/cl,dl/ol,sl/ol,cl/ol);
    }
  }
' "$out"

echo "results=$out"
echo "metadata=$meta"
