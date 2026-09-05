#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

DECL_OLD = '''__constant__ Code D_MAIN_DP[MAXW+1][MAXW+2];
__constant__ Code D_BLOCK_DP[MAXW+1][MAXW+2];
__constant__ uint32_t D_MAIN_FIXED,D_MAIN_OCC,D_BLOCK_FIXED,D_BLOCK_OCC;'''
DECL_NEW = '''struct DeviceGroupMeta{
    Code main_dp[MAXW+1][MAXW+2];
    Code block_dp[MAXW+1][MAXW+2];
    uint32_t main_fixed,main_occ,block_fixed,block_occ;
};
static_assert(sizeof(DeviceGroupMeta)==13936,"unexpected packed group metadata size");
__constant__ DeviceGroupMeta D_GROUP_META;
#define D_MAIN_DP (D_GROUP_META.main_dp)
#define D_BLOCK_DP (D_GROUP_META.block_dp)
#define D_MAIN_FIXED (D_GROUP_META.main_fixed)
#define D_MAIN_OCC (D_GROUP_META.main_occ)
#define D_BLOCK_FIXED (D_GROUP_META.block_fixed)
#define D_BLOCK_OCC (D_GROUP_META.block_occ)'''

COPY_OLD = '''ck(cudaMemcpyToSymbol(D_MAIN_DP,ms.dp,sizeof(ms.dp)),"main dp");ck(cudaMemcpyToSymbol(D_BLOCK_DP,ds.dp,sizeof(ds.dp)),"block dp");
    ck(cudaMemcpyToSymbol(D_MAIN_FIXED,&pg.mf,sizeof(pg.mf)),"mf");ck(cudaMemcpyToSymbol(D_MAIN_OCC,&pg.mo,sizeof(pg.mo)),"mo");
    ck(cudaMemcpyToSymbol(D_BLOCK_FIXED,&pg.bf,sizeof(pg.bf)),"bf");ck(cudaMemcpyToSymbol(D_BLOCK_OCC,&pg.bo,sizeof(pg.bo)),"bo");'''
COPY_NEW = '''DeviceGroupMeta hmeta{};
    std::memcpy(hmeta.main_dp,ms.dp,sizeof(ms.dp));std::memcpy(hmeta.block_dp,ds.dp,sizeof(ds.dp));
    hmeta.main_fixed=pg.mf;hmeta.main_occ=pg.mo;hmeta.block_fixed=pg.bf;hmeta.block_occ=pg.bo;
    ck(cudaMemcpyToSymbol(D_GROUP_META,&hmeta,sizeof(hmeta)),"packed group meta");'''


def once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f'{label}: expected exactly one source match, got {n}')
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('out', type=Path)
    a = ap.parse_args()
    text = a.src.read_text()
    text = once(text, DECL_OLD, DECL_NEW, 'group metadata declarations')
    text = once(text, COPY_OLD, COPY_NEW, 'group metadata symbol copies')
    for stale in (
        'cudaMemcpyToSymbol(D_MAIN_DP', 'cudaMemcpyToSymbol(D_BLOCK_DP',
        'cudaMemcpyToSymbol(D_MAIN_FIXED', 'cudaMemcpyToSymbol(D_MAIN_OCC',
        'cudaMemcpyToSymbol(D_BLOCK_FIXED', 'cudaMemcpyToSymbol(D_BLOCK_OCC',
    ):
        if stale in text:
            raise SystemExit(f'unpacked group metadata copy remains: {stale}')
    if text.count('cudaMemcpyToSymbol(D_GROUP_META') != 1:
        raise SystemExit('expected exactly one packed group metadata symbol copy site')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    print(f'lowered {a.out} packed_group_meta=1 group_meta_bytes=13936 group_meta_symbol_copies_per_group=1 old_symbol_copies_per_group=6 copy_call_reduction=6x')


if __name__ == '__main__':
    main()
