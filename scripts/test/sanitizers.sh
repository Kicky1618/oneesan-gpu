#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

mkdir -p "$ONEESAN_BUILD_DIR"
CXXFLAGS=(-std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer)
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:halt_on_error=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1}"

echo '== Sanitized arbitrary-precision references =='
g++ "${CXXFLAGS[@]}" "$ONEESAN_ROOT/src/cpp/oneesan_cpu.cpp" -o "$ONEESAN_BUILD_DIR/oneesan_cpu_san"
g++ "${CXXFLAGS[@]}" "$ONEESAN_ROOT/src/cpp/oneesan_frontier.cpp" -o "$ONEESAN_BUILD_DIR/oneesan_frontier_san"
python3 "$ONEESAN_ROOT/scripts/test/golden_cpu.py" "$ONEESAN_BUILD_DIR/oneesan_cpu_san" 4
python3 "$ONEESAN_ROOT/scripts/test/golden_cpu.py" "$ONEESAN_BUILD_DIR/oneesan_frontier_san" 5

echo '== Sanitized mmap persistence tests =='
for test in mmap_resume_test mmap_resume_corruption_test mmap_resume_e2e_test; do
  g++ "${CXXFLAGS[@]}" "$ONEESAN_ROOT/tests/${test}.cpp" -o "$ONEESAN_BUILD_DIR/${test}_san"
  "$ONEESAN_BUILD_DIR/${test}_san"
done

echo 'ASan+UBSan audit: PASS'
