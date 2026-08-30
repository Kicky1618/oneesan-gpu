#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys

if len(sys.argv) != 5:
    raise SystemExit('usage: gen-b300-mainrec-ilp2-pair-block-load-policy-stager.py INPUT.cu OUTPUT.cu PAIR_POLICY BLOCK_POLICY')
src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
pair_policy, block_policy = sys.argv[3], sys.argv[4]
for name, policy in (('PAIR_POLICY', pair_policy), ('BLOCK_POLICY', block_policy)):
    if policy not in ('default', 'cg', 'cs'):
        raise SystemExit(f'{name} must be default,cg,cs')

s = src.read_text()
if 'b300_mainrec_stager_ilp2_pair_block_policy=' in s:
    raise SystemExit('source already contains Stage-R ILP2 pair/block policy')
mn = re.search(r'// b300_mainrec_stagen_pair_block_policy=1 pair=(default|cg|cs) block=(default|cg|cs) cg_l2_bytes=(0|64|128|256)', s)
if not mn:
    raise SystemExit('Stage R requires Stage-N pair/block policy marker')
high_pair, high_block, high_base_l2 = mn.group(1), mn.group(2), int(mn.group(3))
mo = re.search(r'// b300_mainrec_stageo_pair_block_cg_l2=1 pair_policy=(default|cg|cs) block_policy=(default|cg|cs) pair_l2_bytes=(0|64|128|256) block_l2_bytes=(0|64|128|256) base_l2_bytes=(0|64|128|256)', s)
if mo:
    if (mo.group(1), mo.group(2), int(mo.group(6))) != (high_pair, high_block, high_base_l2):
        raise SystemExit('Stage-O marker is inconsistent with Stage-N policy')
    high_pair_l2, high_block_l2 = int(mo.group(3)), int(mo.group(4))
else:
    high_pair_l2 = high_base_l2 if high_pair == 'cg' else 0
    high_block_l2 = high_base_l2 if high_block == 'cg' else 0
mq = re.search(r'// b300_mainrec_stageq_ilp8_pair_block_cg_l2=1 pair_policy=(default|cg|cs) block_policy=(default|cg|cs) pair_l2_bytes=(0|64|128|256) block_l2_bytes=(0|64|128|256) upstream_pair_l2_bytes=(0|64|128|256) upstream_block_l2_bytes=(0|64|128|256) stagep_preserved=([01])', s)
if mq:
    if (mq.group(1), mq.group(2)) != (high_pair, high_block):
        raise SystemExit('Stage-Q marker is inconsistent with Stage-N policy')
    high_pair_l2, high_block_l2 = int(mq.group(3)), int(mq.group(4))


def function_span(text: str, name: str) -> tuple[int, int]:
    m = re.search(r'__global__\s+void\s+' + re.escape(name) + r'\s*\(', text)
    if not m:
        raise SystemExit(f'{name} definition not found')
    start = text.rfind('\n', 0, m.start()) + 1
    brace = text.find('{', m.end())
    if brace < 0:
        raise SystemExit(f'{name} opening brace not found')
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return start, i + 1
    raise SystemExit(f'{name} closing brace not found')


def rexpr(kind: str, k: int, policy: str) -> str:
    ptr = f'in+pj{k}' if kind == 'pair' else f'in_block+bj{k}'
    direct = f'in[pj{k}]' if kind == 'pair' else f'in_block[bj{k}]'
    if policy == 'default':
        return direct
    if policy == 'cg':
        return f'b300_mainrec_stager_ilp2_load_cg({ptr})'
    return f'b300_mainrec_stager_ilp2_load_cs({ptr})'

ilp2_start, ilp2_end = function_span(s, 'main_pull_kernel_ilp2')
ilp8_start, ilp8_end = function_span(s, 'main_pull_kernel_ilp8_hybrid')
ilp8_before = s[ilp8_start:ilp8_end]
body = s[ilp2_start:ilp2_end]
ids = sorted({int(x) for x in re.findall(r'const Count pair(\d+)=', body)})
if ids != [0, 1]:
    raise SystemExit(f'Stage R expects exactly ILP2 lanes 0,1; got {ids}')
for k in ids:
    pm = re.search(rf'const Count pair{k}=hp{k}\?([^;]+):Count\(0\);', body)
    bm = re.search(rf'const Count block{k}=hb{k}\?([^;]+):Count\(0\);', body)
    if not pm or not bm:
        raise SystemExit(f'ILP2 lane {k} pair/block anchor missing')
    body = body[:pm.start()] + f'const Count pair{k}=hp{k}?{rexpr("pair", k, pair_policy)}:Count(0);' + body[pm.end():]
    bm = re.search(rf'const Count block{k}=hb{k}\?([^;]+):Count\(0\);', body)
    assert bm
    body = body[:bm.start()] + f'const Count block{k}=hb{k}?{rexpr("block", k, block_policy)}:Count(0);' + body[bm.end():]
for req in ('const Count self0=', 'const MateID m0='):
    if req not in body:
        raise SystemExit(f'Stage R damaged ILP2 artifact: {req}')
s = s[:ilp2_start] + body + s[ilp2_end:]

insert_at = function_span(s, 'main_pull_kernel_ilp2')[0]
helper = ''
if 'cg' in (pair_policy, block_policy):
    helper += '''static_assert(sizeof(Count)==4,"Stage-R ILP2 CG assumes 32-bit Count");
__device__ __forceinline__ Count b300_mainrec_stager_ilp2_load_cg(const Count* p){
#if __CUDA_ARCH__
    uint32_t v; const unsigned long long a=reinterpret_cast<unsigned long long>(p);
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a));
    return Count(v);
#else
    return *p;
#endif
}

'''
if 'cs' in (pair_policy, block_policy):
    helper += '''__device__ __forceinline__ Count b300_mainrec_stager_ilp2_load_cs(const Count* p){
#if __CUDA_ARCH__
    return __ldcs(p);
#else
    return *p;
#endif
}

'''
s = s[:insert_at] + helper + s[insert_at:]

new8_start, new8_end = function_span(s, 'main_pull_kernel_ilp8_hybrid')
if s[new8_start:new8_end] != ilp8_before:
    raise SystemExit('Stage R changed ILP8 high-state kernel')
new2_start, new2_end = function_span(s, 'main_pull_kernel_ilp2')
final2 = s[new2_start:new2_end]
for k in ids:
    if final2.count(f'const Count pair{k}=hp{k}?{rexpr("pair", k, pair_policy)}:Count(0);') != 1:
        raise SystemExit(f'Stage R final pair lane {k} mismatch')
    if final2.count(f'const Count block{k}=hb{k}?{rexpr("block", k, block_policy)}:Count(0);') != 1:
        raise SystemExit(f'Stage R final block lane {k} mismatch')

s += (
    f'\n// b300_mainrec_stager_ilp2_pair_block_policy=1 pair={pair_policy} block={block_policy} '
    f'high_pair={high_pair} high_block={high_block} high_pair_l2={high_pair_l2} high_block_l2={high_block_l2} stageq_preserved={1 if mq else 0}\n'
)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(s)
print(
    f'generated {out} from {src}: b300_mainrec_stager_ilp2_pair_block_policy=1 '
    f'pair_policy={pair_policy} block_policy={block_policy} high_pair={high_pair} high_block={high_block} '
    f'high_pair_l2={high_pair_l2} high_block_l2={high_block_l2} stageq_preserved={1 if mq else 0} '
    'lanes=2 ilp8_byte_identical=1 self_unchanged=1 mate_unchanged=1 semantics_unchanged=1'
)
