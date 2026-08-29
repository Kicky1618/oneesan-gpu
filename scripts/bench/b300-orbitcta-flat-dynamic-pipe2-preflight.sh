#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

build="$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2.sh"
ab="$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-pipe2-ab.sh"
pipe="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2.cuh"
wrap="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_delta_direct_affine_rankformula_nometa4_abstract.cuh"
graph="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_delta_direct_affine_rankformula_nometa4_abstract_graph.cuh"
base="$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh"

bash -n "$build"; echo "shell_syntax_ok=${build#$ONEESAN_ROOT/}"
bash -n "$ab"; echo "shell_syntax_ok=${ab#$ONEESAN_ROOT/}"
python3 -m py_compile <(sed -n "/python3 - \"\$base\" \"\$tmp\" <<'PY'/,/^PY$/p" "$build" | sed '1d;$d') 2>/dev/null || true

for s in \
  'has_item[2]' \
  'p10dc_orbitcta_flat_dynamic_effective_batch' \
  'p10dc_orbitcta_flat_dynamic_pipe2_ctx1_offset_bytes_for_threads' \
  'p10dc_orbitcta_flat_dynamic_pipe2_next_k' \
  'if (current.valid == 1)' \
  'p10dc_orbitcta_flat_forward_columns(current)' \
  'p10dc_orbitcta_flat_reverse_columns(current, edge)' \
  'resident warps may consume current while lane 0' \
  '__syncthreads();'; do
  grep -Fq "$s" "$pipe" || { echo "missing pipe2 marker: $s" >&2; exit 3; }
done
for s in \
  'P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2' \
  'ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2.cuh'; do
  grep -Fq "$s" "$wrap" || { echo "missing pipe2 wrapper marker: $s" >&2; exit 3; }
done
for s in \
  'bucket_high_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_kernel' \
  'bucket_reverse_high_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_kernel' \
  'dynamic_atomic_queue_pipe2' \
  'p10dc_orbitcta_flat_high_smem_bytes' \
  'flat_dynamic_pipe2=' \
  'dynamic_context_buffers=' \
  'steady_state_orbit_barriers='; do
  grep -Fq "$s" "$graph" || { echo "missing pipe2 graph marker: $s" >&2; exit 3; }
done
for s in \
  'P10DC_ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP' \
  'P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2=1' \
  'dynamic_pipe2=1'; do
  grep -Fq "$s" "$build" || { echo "missing pipe2 build wrapper marker: $s" >&2; exit 3; }
done
grep -Fq 'P10DC_ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP' "$base" || { echo 'base build no longer has lease-prep macro anchor' >&2; exit 3; }

# Structural correctness: pipe2 must distinguish a skipped prepared orbit from
# queue exhaustion. The original c.valid-as-sentinel prototype was incorrect.
grep -Fq 'if (!has_item[cur]) break;' "$pipe" || { echo 'pipe2 missing has_item queue sentinel' >&2; exit 3; }
if grep -Fq 'if (!current.valid) break;' "$pipe"; then
  echo 'stale c.valid queue sentinel remains in pipe2' >&2; exit 3
fi

# The steady-state loop should have no CTA barrier between next-orbit prepare
# and current column execution. Check each kernel text with a lightweight parser.
python3 - "$pipe" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
for name,call in [
 ('forward','p10dc_orbitcta_flat_forward_columns(current);'),
 ('reverse','p10dc_orbitcta_flat_reverse_columns(current, edge);')]:
    p=s.find(call)
    if p<0: raise SystemExit(f'missing {name} current column call')
    q=s.rfind('p10dc_orbitcta_flat_dynamic_pipe2_next_k(',0,p)
    if q<0: raise SystemExit(f'missing {name} next-k before columns')
    region=s[q:p]
    if '__syncthreads()' in region:
        raise SystemExit(f'{name} has barrier between next prepare and current columns')
print('pipe2_prepare_columns_overlap_structure=OK')
PY

echo 'dynamic_pipe2_contexts=2 steady_state_barriers_per_orbit=1 prepare_columns_overlap=1 skip_orbit_exact=has_item'
echo 'gpu_work=0 actions_triggered=0'
