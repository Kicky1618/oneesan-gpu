#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
command -v python3 >/dev/null||{ echo "python3 not found" >&2;exit 2; }
command -v bash >/dev/null||exit 2
command -v "${CXX:-g++}" >/dev/null||{ echo "${CXX:-g++} not found" >&2;exit 2; }

py=(
 scripts/build/gen-b300-vmm-production.py
 scripts/build/prune-b300-vmm-stale-shard-symbols.py
 scripts/build/lower-b300-vmm-basearg.py
 scripts/build/lower-b300-packed-group-meta.py
 scripts/build/lower-b300-staged-group-meta.py
 scripts/build/lower-b300-static-lpt-staged-meta.py
 scripts/build/lower-b300-row-limit.py
 scripts/build/lower-b300-staged-meta-pointer.py
 scripts/build/lower-b300-staged-meta-kernelarg.py
 scripts/build/lower-b300-static-lpt-staged-intervals.py
 scripts/build/lower-b300-static-worker-device-binding.py
 scripts/build/lower-b300-static-persistent-workers.py
 scripts/build/lower-b300-concurrent-staged-io.py
)
sh=(
 scripts/bench/b300-staged-group-meta-plan-proof.sh
 scripts/bench/b300-static-lpt-local-meta-proof.sh
 scripts/bench/b300-static-lpt-interval-staging-proof.sh
 scripts/bench/b300-vmm-static-lpt-stagedmeta-production-generate-proof.sh
 scripts/bench/b300-vmm-static-lpt-metaptr-production-generate-proof.sh
 scripts/bench/b300-vmm-static-lpt-metakernelarg-production-generate-proof.sh
 scripts/bench/b300-vmm-static-lpt-staged-intervals-production-generate-proof.sh
 scripts/bench/b300-vmm-static-lpt-workerbind-production-generate-proof.sh
 scripts/bench/b300-vmm-static-lpt-control-bundle-production-generate-proof.sh
 scripts/bench/b300-vmm-static-lpt-persistent-workers-production-generate-proof.sh
 scripts/bench/b300-vmm-static-lpt-persistent-concurrent-production-generate-proof.sh
 scripts/bench/b300-vmm-static-lpt-control-bundle-integration-ptx-proof.sh
 scripts/bench/b300-vmm-static-lpt-stagedmeta-production-ab.sh
 scripts/bench/b300-vmm-static-lpt-meta-source-production-ab.sh
 scripts/bench/b300-vmm-static-lpt-staged-intervals-production-ab.sh
 scripts/bench/b300-vmm-static-lpt-workerbind-production-ab.sh
 scripts/bench/b300-vmm-static-lpt-control-bundle-ab.sh
 scripts/bench/b300-vmm-static-lpt-candidate-tournament.sh
 scripts/build/b300-hbm32-vmm-static-lpt-stagedmeta.sh
 scripts/build/b300-hbm32-vmm-static-lpt-metaptr.sh
 scripts/build/b300-hbm32-vmm-static-lpt-metakernelarg.sh
 scripts/build/b300-hbm32-vmm-static-lpt-staged-intervals.sh
 scripts/build/b300-hbm32-vmm-static-lpt-workerbind.sh
 scripts/build/b300-hbm32-vmm-static-lpt-control-bundle.sh
 scripts/build/b300-hbm32-vmm-static-lpt-persistent-workers.sh
 scripts/build/b300-hbm32-vmm-static-lpt-persistent-concurrent.sh
)
for f in "${py[@]}";do [[ -f "$ONEESAN_ROOT/$f" ]]||{ echo "missing $f" >&2;exit 3; };done
for f in "${sh[@]}";do [[ -f "$ONEESAN_ROOT/$f" ]]||{ echo "missing $f" >&2;exit 3; };done
mkdir -p "$ONEESAN_TMP_DIR/pycache"
for f in "${py[@]}";do PYTHONPYCACHEPREFIX="$ONEESAN_TMP_DIR/pycache" python3 -m py_compile "$ONEESAN_ROOT/$f";done
for f in "${sh[@]}";do bash -n "$ONEESAN_ROOT/$f";done

CXX="${CXX:-g++}" bash "$ONEESAN_ROOT/scripts/bench/b300-staged-group-meta-plan-proof.sh"
CXX="${CXX:-g++}" bash "$ONEESAN_ROOT/scripts/bench/b300-static-lpt-local-meta-proof.sh"
CXX="${CXX:-g++}" bash "$ONEESAN_ROOT/scripts/bench/b300-static-lpt-interval-staging-proof.sh"

bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-stagedmeta-production-generate-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-metaptr-production-generate-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-metakernelarg-production-generate-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-staged-intervals-production-generate-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-workerbind-production-generate-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-control-bundle-production-generate-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-persistent-workers-production-generate-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-persistent-concurrent-production-generate-proof.sh"

echo "b300-experimental-source-preflight OK python_lowerers=${#py[@]} bash_scripts=${#sh[@]} cpu_proofs=3 generated_variants=8 nvcc=not_required actions=not_used"
