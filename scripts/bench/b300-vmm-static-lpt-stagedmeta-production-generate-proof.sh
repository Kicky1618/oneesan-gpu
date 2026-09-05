#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py"
PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py"
BASEARG="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py"
PACK="$ONEESAN_ROOT/scripts/build/lower-b300-packed-group-meta.py"
STAGE="$ONEESAN_ROOT/scripts/build/lower-b300-staged-group-meta.py"
STATIC="$ONEESAN_ROOT/scripts/build/lower-b300-static-lpt-staged-meta.py"
ROWLIMIT="$ONEESAN_ROOT/scripts/build/lower-b300-row-limit.py"
OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_vmm_static_lpt_stagedmeta_proof.cu}"

bash "$ONEESAN_ROOT/scripts/bench/b300-staged-group-meta-plan-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-static-lpt-local-meta-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-basearg-stagedmeta-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$OUT"
python3 "$PRUNE" "$OUT" "$OUT"
python3 "$BASEARG" "$OUT" "$OUT"
python3 "$PACK" "$OUT" "$OUT"
python3 "$STAGE" "$OUT" "$OUT"
python3 "$STATIC" "$OUT" "$OUT"
python3 "$ROWLIMIT" "$OUT" "$OUT"

grep -Fq 'std::vector<std::vector<int>> by_gpu;' "$OUT"
grep -Fq 'std::vector<Code> assigned_work;' "$OUT"
grep -Fq 'if(a.work!=b.work)return a.work>b.work' "$OUT"
grep -Fq 'if(a.ms.size!=b.ms.size)return a.ms.size>b.ms.size' "$OUT"
grep -Fq 'return a.g<b.g' "$OUT"
grep -Fq 'pw.by_gpu.resize(ng);pw.assigned_work.assign(ng,0);' "$OUT"
grep -Fq 'for(int q:pw.by_gpu[d])process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);' "$OUT"
grep -Fq 'ctx[d].stage_group_meta(staged_group_meta[d]);' "$OUT"
grep -Fq 'scheduler=static_lpt' "$OUT"
grep -Fq 'tie_break=work_main_block_group' "$OUT"
grep -Fq 'copy_mode=H2D_once_local_then_D2D_per_group' "$OUT"
grep -Fq 'staged_group_count!=16384' "$OUT"
grep -Fq 'static_lpt_group_min!=2044' "$OUT"
grep -Fq 'static_lpt_group_max!=2052' "$OUT"
grep -Fq 'staged_group_meta_total_bytes!=228327424ull' "$OUT"
grep -Fq 'staged_group_meta_max_bytes!=28596672ull' "$OUT"
grep -Fq 'static_lpt_work_sum!=1812909037294ull' "$OUT"
grep -Fq 'static_lpt_work_min!=226613163686ull' "$OUT"
grep -Fq 'static_lpt_work_max!=226614905940ull' "$OUT"
grep -Fq 'staged_group_meta_max_bytes>reserve' "$OUT"
grep -Fq 'B300_ROW_LIMIT' "$OUT"
grep -Fq 'for(int row=0;row<b300_row_limit;++row)' "$OUT"

for stale in 'std::atomic<int>next{0}' 'next.fetch_add(1,std::memory_order_relaxed)' 'for(auto&c:ctx)c.stage_group_meta(staged_group_meta);' 'copy_mode=H2D_once_then_D2D_per_group' 'for(int row=0;row<W;++row)'; do
  if grep -Fq "$stale" "$OUT"; then
    echo "static LPT generated source still contains dynamic/replicated/full-loop artifact: $stale" >&2
    exit 3
  fi
done

echo "b300-vmm-static-lpt-stagedmeta-production-generate-proof OK scheduler=static_lpt work_stealing=0 deterministic_tie_break=work_main_block_group metadata_replication=0 local_meta_ids_proved=1 component_robustness_proved=1 staged_h2d_once_local=1 per_group_meta_copy=D2D_sync row_limit_env=B300_ROW_LIMIT row_limit_default_full=1 expected_default_groups=16384 expected_default_group_min=2044 expected_default_group_max=2052 expected_default_max_mib_per_gpu=27.271911621 expected_default_total_h2d_gib=0.212646484375 static_vs_replicated_h2d_reduction=8x stage_within_hbm_reserve=1"
