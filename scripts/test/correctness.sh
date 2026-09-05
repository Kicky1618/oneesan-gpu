#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

mkdir -p "$ONEESAN_BUILD_DIR"

echo '== Python safety / ZDD tests =='
python3 -m unittest discover -s "$ONEESAN_ROOT/tests" -p 'test_*.py' -v
python3 -m py_compile \
  "$ONEESAN_ROOT/scripts/solve/build_provenance.py" \
  "$ONEESAN_ROOT/scripts/solve/path_bound.py" \
  "$ONEESAN_ROOT/scripts/solve/solver_output.py" \
  "$ONEESAN_ROOT/scripts/solve/solver_output.py" \
  "$ONEESAN_ROOT/scripts/solve/exact_safety.py" \
  "$ONEESAN_ROOT/scripts/solve/solve_b300_exact.py" \
  "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" \
  "$ONEESAN_ROOT/scripts/solve/solve_exact.py" \
  "$ONEESAN_ROOT/scripts/solve/solve_multi7.py" \
  "$ONEESAN_ROOT/scripts/tools/validate_zdd.py" \
  "$ONEESAN_ROOT/scripts/tools/gen_row6_crt20.py" \
  "$ONEESAN_ROOT/scripts/tools/verify_row6_crt20.py" \
  "$ONEESAN_ROOT/scripts/tools/verify_exact_result.py" \
  "$ONEESAN_ROOT/scripts/tools/row8_gridfp_structural_cert.py" \
  "$ONEESAN_ROOT/scripts/tools/row8_cert_memory_plan.py" \
  "$ONEESAN_ROOT/scripts/tools/row8_raw_quotient_cert.py" \
  "$ONEESAN_ROOT/scripts/tools/verify_row8_cap9_overflow.py" \
  "$ONEESAN_ROOT/scripts/test/golden_cpu.py"

echo '== Arbitrary-precision CPU references =='
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  "$ONEESAN_ROOT/src/cpp/oneesan_cpu.cpp" -o "$ONEESAN_BUILD_DIR/oneesan_cpu_correctness"
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  "$ONEESAN_ROOT/src/cpp/oneesan_frontier.cpp" -o "$ONEESAN_BUILD_DIR/oneesan_frontier_correctness"
python3 "$ONEESAN_ROOT/scripts/test/golden_cpu.py" "$ONEESAN_BUILD_DIR/oneesan_frontier_correctness" 6
python3 "$ONEESAN_ROOT/scripts/test/golden_cpu.py" "$ONEESAN_BUILD_DIR/oneesan_cpu_correctness" 4

echo '== mmap persistence / corruption / crash-restart =='
for test in mmap_resume_test mmap_resume_corruption_test mmap_resume_e2e_test; do
  g++ -std=c++17 -O2 -Wall -Wextra -Werror \
    "$ONEESAN_ROOT/tests/${test}.cpp" -o "$ONEESAN_BUILD_DIR/$test"
  "$ONEESAN_BUILD_DIR/$test"
done

echo '== Shell syntax / arithmetic-input surfaces =='
bash -n \
  "$ONEESAN_ROOT/scripts/lib/common.sh" \
  "$ONEESAN_ROOT/scripts/run/b300x8.sh" \
  "$ONEESAN_ROOT/scripts/run/b300x8-exact.sh" \
  "$ONEESAN_ROOT/scripts/run/b300x8-mmap.sh" \
  "$ONEESAN_ROOT/scripts/run/local.sh" \
  "$ONEESAN_ROOT/scripts/run/multigpu-regression.sh" \
  "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" \
  "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" \
  "$ONEESAN_ROOT/scripts/certify/row8-gridfp-structural.sh" \
  "$ONEESAN_ROOT/scripts/test/gridfp-partition.sh" \
  "$ONEESAN_ROOT/scripts/test/mmap-fault-integration.sh" \
  "$ONEESAN_ROOT/scripts/test/row8-runtime.sh" \
  "$ONEESAN_ROOT/scripts/test/row8-structural-cert.sh" \
  "$ONEESAN_ROOT/scripts/test/row6-vmm.sh"

rm -f "$ONEESAN_BUILD_DIR/arithmetic-injection-marker"
set +e
"$ONEESAN_ROOT/scripts/run/b300x8.sh" '1+$(touch build/arithmetic-injection-marker)' \
  >"$ONEESAN_BUILD_DIR/arithmetic-injection.out" 2>&1
injection_rc=$?
set -e
if [[ "$injection_rc" -ne 2 || -e "$ONEESAN_BUILD_DIR/arithmetic-injection-marker" ]]; then
  echo 'arithmetic injection regression failed' >&2
  exit 20
fi
rm -f "$ONEESAN_BUILD_DIR/arithmetic-injection.out"

echo '== Tensor exact-admission certificate gates =='
"$ONEESAN_ROOT/scripts/test/row8-structural-cert.sh"

echo '== Production source closure =='
python3 "$ONEESAN_ROOT/scripts/test/source_closure.py" \
  src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu \
  src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN_batch.cu \
  src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu \
  src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_vmm_batch.cu \
  src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_batch.cu \
  src/cuda/b300/occmajor_authvmm_rank16_batch.cu \
  src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row8tensor_batch.cu

echo '== Formal source hygiene =='
if grep -REn --exclude-dir=.lake --include='*.lean' '\b(sorry|admit|axiom)\b' "$ONEESAN_ROOT/formal"; then
  echo 'formal source contains sorry/admit/axiom' >&2
  exit 21
fi

if command -v nvcc >/dev/null 2>&1; then
  echo '== Grid-FP production partition invariant =='
  "$ONEESAN_ROOT/scripts/test/gridfp-partition.sh"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && [[ -f "$ONEESAN_ROOT/src/cuda/b300/row8_pivots_w19.bin" ]]; then
    echo '== Row-8 runtime automaton / cache regression =='
    "$ONEESAN_ROOT/scripts/test/row8-runtime.sh"
  else
    echo '== Row-8 runtime automaton / cache regression: SKIP (CUDA GPU or pivot certificate unavailable) =='
  fi
else
  echo '== Grid-FP production partition invariant: SKIP (nvcc unavailable) =='
fi

if command -v git >/dev/null 2>&1 && git -C "$ONEESAN_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$ONEESAN_ROOT" diff --check
fi

echo 'correctness audit: PASS'
