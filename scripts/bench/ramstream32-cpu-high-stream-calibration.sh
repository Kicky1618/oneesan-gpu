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
CPU_HIGH_OVERLAP="${CPU_HIGH_OVERLAP:-0}"
CPU_HIGH_CPU_LIST="${CPU_HIGH_CPU_LIST:-}"
CPU_LOW_CPU_LIST="${CPU_LOW_CPU_LIST:-}"
CPU_LOW_SCHEDULE="${CPU_LOW_SCHEDULE:-dynamic}"
CPU_LOW_DOMAIN_SIZE="${CPU_LOW_DOMAIN_SIZE:-}"
CPU_LOW_DOMAIN_REFINE="${CPU_LOW_DOMAIN_REFINE:-1}"
MANIFEST="${MANIFEST:-}"
REPEATS="${REPEATS:-2}"
BUILD="${BUILD:-1}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
EXPECTED_RESIDUE="${EXPECTED_RESIDUE:-}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_high_stream_calibration}"

if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  echo "MANIFEST is required and must exist" >&2
  exit 2
fi
if (( N < 2 || N > 27 || GPU_TARGET_MIB <= 0 || CPU_WORKERS <= 0 || CPU_HIGH_WORKERS <= 0 || REPEATS <= 0 )); then
  echo "invalid benchmark parameters" >&2
  exit 2
fi
if [[ "$CPU_HIGH_OVERLAP" != 0 && "$CPU_HIGH_OVERLAP" != 1 ]]; then
  echo "CPU_HIGH_OVERLAP must be 0 or 1" >&2
  exit 2
fi
if [[ "$CPU_LOW_DOMAIN_REFINE" != 0 && "$CPU_LOW_DOMAIN_REFINE" != 1 ]]; then
  echo "CPU_LOW_DOMAIN_REFINE must be 0 or 1" >&2
  exit 2
fi
case "$CPU_LOW_SCHEDULE" in
  dynamic|sticky|contiguous) ;;
  domain)
    if [[ ! "$CPU_LOW_DOMAIN_SIZE" =~ ^[1-9][0-9]*$ ]] || (( CPU_LOW_DOMAIN_SIZE > CPU_WORKERS )); then
      echo "CPU_LOW_DOMAIN_SIZE must be a positive integer <= CPU_WORKERS for domain schedule" >&2
      exit 2
    fi
    ;;
  *) echo "CPU_LOW_SCHEDULE must be dynamic, sticky, contiguous, or domain" >&2; exit 2 ;;
esac
if [[ "$RUN_VALIDATION" != 0 && "$RUN_VALIDATION" != 1 ]]; then
  echo "RUN_VALIDATION must be 0 or 1" >&2
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

manifest_dir="$(cd "$(dirname -- "$MANIFEST")" && pwd)"
mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/stream-calibration-n${N}-${ts}.tsv"
meta="$OUT_DIR/stream-calibration-n${N}-${ts}.meta"

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
cpu_high_overlap=$CPU_HIGH_OVERLAP
cpu_high_cpu_list=${CPU_HIGH_CPU_LIST:-none}
cpu_low_cpu_list=${CPU_LOW_CPU_LIST:-none}
cpu_low_schedule=$CPU_LOW_SCHEDULE
cpu_low_domain_size=${CPU_LOW_DOMAIN_SIZE:-none}
cpu_low_domain_refine=$CPU_LOW_DOMAIN_REFINE
manifest=$MANIFEST
manifest_sha256=$(file_sha256 "$MANIFEST")
repeats=$REPEATS
run_validation=$RUN_VALIDATION
expected_residue=${EXPECTED_RESIDUE:-unknown}
binary_sha256=$(file_sha256 "$bin")
EOF

printf 'repeat\torder\trole\tsample\tgroup\tgroups_file\tresidue\twall_s\tcpu_high_wall_s\tcpu_high_kernel_sum_s\tcpu_high_groups\th2d_s\tgpu_kernel_s\td2h_s\tcpu_low_wall_s\tselection_hash\traw\n' >"$out"

mapfile -t sample_rows < <(awk -F '\t' 'NR>1 {print $1 "\t" $2 "\t" $3}' "$MANIFEST")
if ((${#sample_rows[@]} < 5)); then
  echo "manifest must contain at least five calibration samples" >&2
  exit 2
fi

run_policy() {
  local repeat="$1" order="$2" role="$3" sample="$4" group="$5" groups_file="$6"
  local path="$groups_file"
  [[ "$path" = /* ]] || path="$manifest_dir/$path"
  [[ -f "$path" ]] || { echo "missing groups file: $path" >&2; exit 3; }

  local line residue got_schedule got_domain
  line="$(CPU_HIGH_MODE=direct CPU_HIGH_MAX_MIB=0 \
    CPU_HIGH_GROUPS_FILE="$path" CPU_HIGH_WORKERS="$CPU_HIGH_WORKERS" \
    CPU_HIGH_OVERLAP="$CPU_HIGH_OVERLAP" \
    CPU_HIGH_CPU_LIST="$CPU_HIGH_CPU_LIST" CPU_LOW_CPU_LIST="$CPU_LOW_CPU_LIST" \
    CPU_LOW_SCHEDULE="$CPU_LOW_SCHEDULE" CPU_LOW_DOMAIN_SIZE="$CPU_LOW_DOMAIN_SIZE" \
    CPU_LOW_DOMAIN_REFINE="$CPU_LOW_DOMAIN_REFINE" \
    "$bin" "$N" "$MODULUS" "$GPU_TARGET_MIB" "$CPU_WORKERS" | tail -n1)"
  residue="$(field "$line" residue)"
  got_schedule="$(field "$line" cpu_low_schedule)"
  if [[ "$got_schedule" != "$CPU_LOW_SCHEDULE" ]]; then
    echo "LOW schedule provenance mismatch requested=$CPU_LOW_SCHEDULE got=$got_schedule" >&2
    exit 7
  fi
  if [[ "$CPU_LOW_SCHEDULE" == domain ]]; then
    got_domain="$(field "$line" cpu_low_domain_size)"
    if [[ "$got_domain" != "$CPU_LOW_DOMAIN_SIZE" ]]; then
      echo "LOW domain provenance mismatch requested=$CPU_LOW_DOMAIN_SIZE got=$got_domain" >&2
      exit 7
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repeat" "$order" "$role" "$sample" "$group" "$groups_file" "$residue" \
    "$(field "$line" wall_s)" "$(field "$line" cpu_high_wall_s)" \
    "$(field "$line" cpu_high_kernel_sum_s)" "$(field "$line" cpu_high_groups)" \
    "$(field "$line" h2d_s)" "$(field "$line" gpu_kernel_s)" \
    "$(field "$line" d2h_s)" "$(field "$line" cpu_low_wall_s)" \
    "$(field "$line" cpu_high_selection_hash)" "$line" >>"$out"
  printf '%s\n' "$residue"
}

reference_residue=""
for ((r=1; r<=REPEATS; ++r)); do
  if ((r % 2 == 1)); then
    order=forward
    indices=()
    for ((i=0; i<${#sample_rows[@]}; ++i)); do indices+=("$i"); done
  else
    order=reverse
    indices=()
    for ((i=${#sample_rows[@]}-1; i>=0; --i)); do indices+=("$i"); done
  fi
  echo "repeat $r/$REPEATS low_schedule=$CPU_LOW_SCHEDULE low_domain_size=${CPU_LOW_DOMAIN_SIZE:-none} low_domain_refine=$CPU_LOW_DOMAIN_REFINE ($order)" >&2
  for i in "${indices[@]}"; do
    IFS=$'\t' read -r sample groups_file group <<<"${sample_rows[i]}"
    echo "  sample=$sample group=$group" >&2
    residue="$(run_policy "$r" "$order" sample "$sample" "$group" "$groups_file")"
    if [[ -z "$reference_residue" ]]; then reference_residue="$residue"; fi
    if [[ "$residue" != "$reference_residue" ]]; then
      echo "residue mismatch sample=$sample got=$residue reference=$reference_residue" >&2
      exit 5
    fi
    if [[ -n "$EXPECTED_RESIDUE" && "$residue" != "$EXPECTED_RESIDUE" ]]; then
      echo "unexpected residue sample=$sample got=$residue expected=$EXPECTED_RESIDUE" >&2
      exit 6
    fi
  done
done

if [[ "$RUN_VALIDATION" == 1 ]]; then
  validation="$manifest_dir/validation-all.groups"
  if [[ -f "$validation" ]]; then
    echo "validation-all" >&2
    residue="$(run_policy 0 validation validation 0 -1 "$validation")"
    if [[ "$residue" != "$reference_residue" ]]; then
      echo "validation residue mismatch got=$residue reference=$reference_residue" >&2
      exit 5
    fi
  fi
fi

awk -F '\t' '
  NR==1 || $3!="sample" {next}
  { n[$4]++; wall[$4]+=$8; high[$4]+=$9; groups[$4]+=$11 }
  END { for (s in n) printf("summary sample=%s runs=%d mean_wall_s=%.9f mean_cpu_high_wall_s=%.9f mean_cpu_high_groups=%.3f\n", s,n[s],wall[s]/n[s],high[s]/n[s],groups[s]/n[s]); }
' "$out" | sort -t= -k2,2n

echo "results=$out"
echo "metadata=$meta"
