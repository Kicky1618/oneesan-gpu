#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

N="${N:-21}"
W=$((N + 1))
ARCH="${ARCH:-native}"
MODULUS="${MODULUS:-4294967291}"
SCRATCH_MIB="${SCRATCH_MIB:-256}"
CPU_THREADS="${CPU_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
MAX_WINDOW="${MAX_WINDOW:-14}"
REPEATS="${REPEATS:-1}"
BUILD="${BUILD:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_ab}"
EXPECTED_RESIDUE="${EXPECTED_RESIDUE:-}"

if [[ -z "$EXPECTED_RESIDUE" && "$N:$MODULUS" == "21:4294967291" ]]; then
  EXPECTED_RESIDUE=998035516
fi

if (( N < 2 || N > 27 )); then
  echo "N must be in [2, 27]" >&2
  exit 2
fi
if (( SCRATCH_MIB <= 0 || CPU_THREADS <= 0 || MAX_WINDOW <= 0 || REPEATS <= 0 )); then
  echo "SCRATCH_MIB, CPU_THREADS, MAX_WINDOW and REPEATS must be positive" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

BASE_BIN="$ROOT/build/oneesan_cuda_gridfp_ramstream32_w${W}"
FACT_BIN="$ROOT/build/oneesan_cuda_gridfp_ramstream32_factorized_n${N}"

if [[ "$BUILD" != 0 ]]; then
  TARGET_W="$W" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32.sh
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-factorized.sh
fi

for bin in "$BASE_BIN" "$FACT_BIN"; do
  if [[ ! -x "$bin" ]]; then
    echo "missing executable: $bin" >&2
    exit 3
  fi
done

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo unavailable
  fi
}

commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if git diff --quiet --ignore-submodules HEAD -- 2>/dev/null && \
   git diff --cached --quiet --ignore-submodules HEAD -- 2>/dev/null; then
  dirty=0
else
  dirty=1
fi
host="$(hostname 2>/dev/null || echo unknown)"
gpu="$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null | paste -sd ';' - || true)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
nvcc="$(nvcc --version 2>/dev/null | tail -n1 || true)"
base_sha256="$(file_sha256 "$BASE_BIN")"
fact_sha256="$(file_sha256 "$FACT_BIN")"
[[ -n "$gpu" ]] || gpu=unknown
[[ -n "$driver" ]] || driver=unknown
[[ -n "$nvcc" ]] || nvcc=unknown

ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/ramstream32-ab-n${N}-${ts}.tsv"
meta="$OUT_DIR/ramstream32-ab-n${N}-${ts}.meta"

cat >"$meta" <<EOF
commit=$commit
dirty=$dirty
host=$host
gpu=$gpu
driver=$driver
nvcc=$nvcc
n=$N
width=$W
arch=$ARCH
modulus=$MODULUS
expected_residue=${EXPECTED_RESIDUE:-unknown}
scratch_mib=$SCRATCH_MIB
cpu_threads=$CPU_THREADS
max_window=$MAX_WINDOW
repeats=$REPEATS
baseline_binary=$BASE_BIN
baseline_sha256=$base_sha256
factorized_binary=$FACT_BIN
factorized_sha256=$fact_sha256
EOF

printf 'repeat\torder\tbackend\tresidue\twall_s\tpack_s\th2d_s\tkernel_s\td2h_s\tunpack_s\traw\n' >"$out"

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

run_one() {
  local repeat="$1" order="$2" kind="$3"
  local line backend residue wall pack h2d kernel d2h unpack
  if [[ "$kind" == baseline ]]; then
    line="$($BASE_BIN "$N" "$MODULUS" "$SCRATCH_MIB" "$MAX_WINDOW" "$CPU_THREADS" | tail -n1)"
  else
    line="$($FACT_BIN "$N" "$MODULUS" "$SCRATCH_MIB" "$CPU_THREADS" | tail -n1)"
  fi

  backend="$(field "$line" backend)"
  residue="$(field "$line" residue)"
  wall="$(field "$line" wall_s)"
  pack="$(field "$line" pack_s)"
  h2d="$(field "$line" h2d_s)"
  kernel="$(field "$line" kernel_s)"
  d2h="$(field "$line" d2h_s)"
  unpack="$(field "$line" unpack_s)"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repeat" "$order" "$backend" "$residue" "$wall" "$pack" "$h2d" "$kernel" "$d2h" "$unpack" "$line" >>"$out"
  printf '%s\n' "$residue"
}

for ((r = 1; r <= REPEATS; ++r)); do
  if (( r % 2 == 1 )); then
    order=baseline-first
    echo "repeat $r/$REPEATS: baseline -> factorized" >&2
    base_residue="$(run_one "$r" "$order" baseline)"
    fact_residue="$(run_one "$r" "$order" factorized)"
  else
    order=factorized-first
    echo "repeat $r/$REPEATS: factorized -> baseline" >&2
    fact_residue="$(run_one "$r" "$order" factorized)"
    base_residue="$(run_one "$r" "$order" baseline)"
  fi

  if [[ "$base_residue" != "$fact_residue" ]]; then
    echo "residue mismatch at repeat $r: baseline=$base_residue factorized=$fact_residue" >&2
    exit 5
  fi
  if [[ -n "$EXPECTED_RESIDUE" && "$base_residue" != "$EXPECTED_RESIDUE" ]]; then
    echo "unexpected residue at repeat $r: got=$base_residue expected=$EXPECTED_RESIDUE" >&2
    exit 6
  fi
done

awk -F '\t' '
  NR == 1 { next }
  {
    n[$3]++
    wall[$3] += $5
    pack[$3] += $6
    kernel[$3] += $8
  }
  END {
    for (b in n) {
      printf("summary backend=%s runs=%d mean_wall_s=%.9f mean_pack_s=%.9f mean_kernel_s=%.9f\n",
             b, n[b], wall[b] / n[b], pack[b] / n[b], kernel[b] / n[b])
    }
  }
' "$out"

base_mean="$(awk -F '\t' 'NR>1 && $3=="gridfp-ramstream32-v1" {s+=$5; n++} END {if(n) printf "%.12f", s/n}' "$out")"
fact_mean="$(awk -F '\t' 'NR>1 && $3 ~ /^gridfp-ramstream32-factorized-/ {s+=$5; n++} END {if(n) printf "%.12f", s/n}' "$out")"
if [[ -n "$base_mean" && -n "$fact_mean" ]]; then
  awk -v b="$base_mean" -v f="$fact_mean" 'BEGIN {
    printf("speedup_wall=%.6fx reduction_wall=%.3f%%\n", b / f, 100.0 * (1.0 - f / b));
  }'
fi

echo "results=$out"
echo "metadata=$meta"
