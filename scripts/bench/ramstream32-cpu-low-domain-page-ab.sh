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
CPU_WORKERS="${CPU_WORKERS:-64}"
CPU_HIGH_WORKERS="${CPU_HIGH_WORKERS:-32}"
CPU_HIGH_MODE="${CPU_HIGH_MODE:-direct}"
CPU_HIGH_OVERLAP="${CPU_HIGH_OVERLAP:-1}"
CPU_HIGH_MAX_MIB="${CPU_HIGH_MAX_MIB:-256}"
CPU_HIGH_GROUPS_FILE="${CPU_HIGH_GROUPS_FILE:-}"
CPU_HIGH_CPU_LIST="${CPU_HIGH_CPU_LIST:-}"
CPU_LOW_CPU_LIST="${CPU_LOW_CPU_LIST:-}"
CPU_LOW_DOMAIN_SIZE="${CPU_LOW_DOMAIN_SIZE:-32}"
REPEATS="${REPEATS:-4}"
BUILD="${BUILD:-1}"
EXPECTED_RESIDUE="${EXPECTED_RESIDUE:-}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_domain_page_ab}"

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
if [[ ! "$CPU_LOW_DOMAIN_SIZE" =~ ^[1-9][0-9]*$ ]] || (( CPU_LOW_DOMAIN_SIZE > CPU_WORKERS )); then
  echo "CPU_LOW_DOMAIN_SIZE must be a positive integer <= CPU_WORKERS" >&2
  exit 2
fi
if [[ -n "$CPU_HIGH_GROUPS_FILE" && ! -f "$CPU_HIGH_GROUPS_FILE" ]]; then
  echo "missing CPU_HIGH_GROUPS_FILE: $CPU_HIGH_GROUPS_FILE" >&2
  exit 2
fi
if [[ -z "$EXPECTED_RESIDUE" && "$N:$MODULUS" == "21:4294967291" ]]; then
  EXPECTED_RESIDUE=998035516
fi

if [[ "$BUILD" != 0 ]]; then
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-factorized-hybrid-sparse.sh
fi
bin="$ROOT/build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/domain-page-ab-n${N}-${ts}.tsv"
meta="$OUT_DIR/domain-page-ab-n${N}-${ts}.meta"
log_dir="$OUT_DIR/domain-page-ab-n${N}-${ts}.logs"
mkdir -p "$log_dir"

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
field() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s\n' "${token#*=}"
      return 0
    fi
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
cpu_high_mode=$CPU_HIGH_MODE
cpu_high_overlap=$CPU_HIGH_OVERLAP
cpu_high_max_mib=$CPU_HIGH_MAX_MIB
cpu_high_groups_file=${CPU_HIGH_GROUPS_FILE:-none}
cpu_high_cpu_list=${CPU_HIGH_CPU_LIST:-none}
cpu_low_cpu_list=${CPU_LOW_CPU_LIST:-none}
cpu_low_schedule=domain
cpu_low_domain_size=$CPU_LOW_DOMAIN_SIZE
cpu_low_domain_refine=1
page_objective=max_guard-page-sum-v5.23
variants=page0,page1
order_design=alternating-pairs
repeats=$REPEATS
numa_sampling=disabled
expected_residue=${EXPECTED_RESIDUE:-unknown}
binary_sha256=$(file_sha256 "$bin")
EOF
if [[ -n "$CPU_HIGH_GROUPS_FILE" ]]; then
  echo "cpu_high_groups_file_sha256=$(file_sha256 "$CPU_HIGH_GROUPS_FILE")" >>"$meta"
fi

printf 'repeat\torder\tpage_tiebreak\tresidue\twall_s\tcpu_low_wall_s\tcpu_low_kernel_sum_s\tcpu_low_schedule_build_s\tpage_build_s\tpage_objective\tpage_candidate_evaluations\tpage_max_guard_rejections\tpage_improving_moves\tpage_tie_load_moves\tpage_improve_sum_increase_moves\tpage_boundary_moves\tpage_moved_jobs\tpage_max_worker_cells_before\tpage_max_worker_cells_after\tpage_penalty_2m_before\tpage_penalty_2m_after\tpage_penalty_4k_before\tpage_penalty_4k_after\tcpu_high_wall_s\th2d_s\tgpu_kernel_s\td2h_s\tstderr_log\traw\n' >"$out"

run_one() {
  local repeat="$1" order="$2" page="$3"
  local stderr_log="$log_dir/repeat${repeat}-${order}-page${page}.stderr.txt"
  local line residue got_schedule got_domain got_refine got_page
  local page_line="" objective=disabled evals=0 rejects=0 page_moves=0 load_moves=0 sum_increase=0
  local max_before=0 max_after=0

  line="$(CPU_HIGH_MODE="$CPU_HIGH_MODE" \
    CPU_HIGH_OVERLAP="$CPU_HIGH_OVERLAP" \
    CPU_HIGH_MAX_MIB="$CPU_HIGH_MAX_MIB" \
    CPU_HIGH_GROUPS_FILE="$CPU_HIGH_GROUPS_FILE" \
    CPU_HIGH_WORKERS="$CPU_HIGH_WORKERS" \
    CPU_HIGH_CPU_LIST="$CPU_HIGH_CPU_LIST" \
    CPU_LOW_CPU_LIST="$CPU_LOW_CPU_LIST" \
    CPU_LOW_SCHEDULE=domain \
    CPU_LOW_DOMAIN_SIZE="$CPU_LOW_DOMAIN_SIZE" \
    CPU_LOW_DOMAIN_REFINE=1 \
    CPU_LOW_DOMAIN_PAGE_TIEBREAK="$page" \
    RAMSTREAM_NUMA_SAMPLE_MIB=0 \
    "$bin" "$N" "$MODULUS" "$GPU_TARGET_MIB" "$CPU_WORKERS" \
      2>"$stderr_log" | tail -n1)"

  residue="$(field "$line" residue)"
  got_schedule="$(field "$line" cpu_low_schedule)"
  got_domain="$(field "$line" cpu_low_domain_size)"
  got_refine="$(field "$line" cpu_low_domain_refine)"
  got_page="$(field "$line" cpu_low_domain_page_tiebreak)"
  [[ "$got_schedule" == domain ]] || { echo "schedule provenance mismatch got=$got_schedule" >&2; exit 7; }
  [[ "$got_domain" == "$CPU_LOW_DOMAIN_SIZE" ]] || { echo "domain provenance mismatch requested=$CPU_LOW_DOMAIN_SIZE got=$got_domain" >&2; exit 7; }
  [[ "$got_refine" == 1 ]] || { echo "refine provenance mismatch got=$got_refine" >&2; exit 7; }
  [[ "$got_page" == "$page" ]] || { echo "page stdout provenance mismatch requested=$page got=$got_page" >&2; exit 7; }
  grep -Eq 'cpu_low_domain_schedule .* refine=1( |$)' "$stderr_log" || {
    echo "refinement stderr provenance mismatch log=$stderr_log" >&2; exit 7;
  }
  if [[ "$page" == 1 ]]; then
    page_line="$(grep -F 'cpu_low_domain_page_tiebreak ' "$stderr_log" | tail -n1)"
    [[ -n "$page_line" ]] || {
      echo "page tie-break stderr provenance missing log=$stderr_log" >&2; exit 7;
    }
    objective="$(field "$page_line" objective)"
    [[ "$objective" == max_guard-page-sum-v5.23 ]] || {
      echo "page objective mismatch got=$objective" >&2; exit 7;
    }
    evals="$(field "$page_line" candidate_evaluations)"
    rejects="$(field "$page_line" max_guard_rejections)"
    page_moves="$(field "$page_line" page_improving_moves)"
    load_moves="$(field "$page_line" page_tie_load_moves)"
    sum_increase="$(field "$page_line" page_improve_sum_increase_moves)"
    max_before="$(field "$page_line" max_worker_cells_before)"
    max_after="$(field "$page_line" max_worker_cells_after)"
    python3 - "$evals" "$rejects" "$page_moves" "$load_moves" "$sum_increase" \
      "$max_before" "$max_after" <<'PY'
import sys
evals,rejects,page_moves,load_moves,sum_increase,max_before,max_after=map(int,sys.argv[1:])
if rejects > evals:
    raise SystemExit('max-guard rejection count exceeds candidate evaluations')
if sum_increase > page_moves:
    raise SystemExit('sum-increase move count exceeds page-improving moves')
if max_after > max_before:
    raise SystemExit('page max-worker regression')
PY
    local p20 p21 p40 p41
    p20="$(field "$line" cpu_low_domain_page_penalty_2m_before)"
    p21="$(field "$line" cpu_low_domain_page_penalty_2m_after)"
    p40="$(field "$line" cpu_low_domain_page_penalty_4k_before)"
    p41="$(field "$line" cpu_low_domain_page_penalty_4k_after)"
    python3 - "$p20" "$p21" "$p40" "$p41" <<'PY'
import sys
p20,p21,p40,p41=map(int,sys.argv[1:])
if (p21,p41) > (p20,p40):
    raise SystemExit('page penalty regression')
PY
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repeat" "$order" "$page" "$residue" \
    "$(field "$line" wall_s)" "$(field "$line" cpu_low_wall_s)" \
    "$(field "$line" cpu_low_kernel_sum_s)" "$(field "$line" cpu_low_schedule_build_s)" \
    "$(field "$line" cpu_low_domain_page_build_s)" \
    "$objective" "$evals" "$rejects" "$page_moves" "$load_moves" "$sum_increase" \
    "$(field "$line" cpu_low_domain_page_boundary_moves)" \
    "$(field "$line" cpu_low_domain_page_moved_jobs)" \
    "$max_before" "$max_after" \
    "$(field "$line" cpu_low_domain_page_penalty_2m_before)" \
    "$(field "$line" cpu_low_domain_page_penalty_2m_after)" \
    "$(field "$line" cpu_low_domain_page_penalty_4k_before)" \
    "$(field "$line" cpu_low_domain_page_penalty_4k_after)" \
    "$(field "$line" cpu_high_wall_s)" "$(field "$line" h2d_s)" \
    "$(field "$line" gpu_kernel_s)" "$(field "$line" d2h_s)" \
    "$stderr_log" "$line" >>"$out"
  printf '%s\n' "$residue"
}

reference_residue=""
for ((r=1; r<=REPEATS; ++r)); do
  if ((r % 2 == 1)); then
    order=off-first
    variants=(0 1)
  else
    order=on-first
    variants=(1 0)
  fi
  echo "repeat $r/$REPEATS domain_size=$CPU_LOW_DOMAIN_SIZE ($order)" >&2
  for page in "${variants[@]}"; do
    echo "  CPU_LOW_DOMAIN_PAGE_TIEBREAK=$page" >&2
    residue="$(run_one "$r" "$order" "$page")"
    if [[ -z "$reference_residue" ]]; then reference_residue="$residue"; fi
    if [[ "$residue" != "$reference_residue" ]]; then
      echo "residue mismatch page=$page got=$residue reference=$reference_residue" >&2
      exit 5
    fi
    if [[ -n "$EXPECTED_RESIDUE" && "$residue" != "$EXPECTED_RESIDUE" ]]; then
      echo "unexpected residue page=$page got=$residue expected=$EXPECTED_RESIDUE" >&2
      exit 6
    fi
  done
done

awk -F '\t' '
  NR==1 {next}
  {
    p=$3; n[p]++;
    wall[p]+=$5; low[p]+=$6; kernel[p]+=$7; build[p]+=$8; pbuild[p]+=$9;
    evals[p]+=$11; rejects[p]+=$12; pimprove[p]+=$13; lmove[p]+=$14; sinc[p]+=$15;
    moves[p]+=$16; moved[p]+=$17; high[p]+=$24;
  }
  END {
    for (p=0; p<=1; ++p) if (n[p])
      printf("summary page_tiebreak=%d runs=%d mean_wall_s=%.9f mean_cpu_low_wall_s=%.9f mean_cpu_low_kernel_sum_s=%.9f mean_schedule_build_s=%.9f mean_page_build_s=%.9f mean_candidate_evaluations=%.3f mean_max_guard_rejections=%.3f mean_page_improving_moves=%.3f mean_page_tie_load_moves=%.3f mean_page_improve_sum_increase_moves=%.3f mean_page_boundary_moves=%.3f mean_page_moved_jobs=%.3f mean_cpu_high_wall_s=%.9f\n",
             p,n[p],wall[p]/n[p],low[p]/n[p],kernel[p]/n[p],build[p]/n[p],pbuild[p]/n[p],evals[p]/n[p],rejects[p]/n[p],pimprove[p]/n[p],lmove[p]/n[p],sinc[p]/n[p],moves[p]/n[p],moved[p]/n[p],high[p]/n[p]);
    if (n[0] && n[1]) {
      w0=wall[0]/n[0]; w1=wall[1]/n[1]; l0=low[0]/n[0]; l1=low[1]/n[1];
      b0=build[0]/n[0]; b1=build[1]/n[1];
      printf("comparison page0_wall_s=%.9f page1_wall_s=%.9f page_speedup=%.9fx page_saved_s=%.9f page0_low_s=%.9f page1_low_s=%.9f page_low_speedup=%.9fx page_low_saved_s=%.9f page0_build_s=%.9f page1_build_s=%.9f page_extra_build_s=%.9f\n",
             w0,w1,w0/w1,w0-w1,l0,l1,l0/l1,l0-l1,b0,b1,b1-b0);
    }
  }
' "$out"

echo "results=$out"
echo "metadata=$meta"
echo "logs=$log_dir"
