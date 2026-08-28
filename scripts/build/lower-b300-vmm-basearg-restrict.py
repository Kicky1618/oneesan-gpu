#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

REPL = (
    (
        '__global__ void gather_main_kernel(Count*out,MateID*mates,Code n,const Count* auth)',
        '__global__ void gather_main_kernel(Count* __restrict__ out,MateID* __restrict__ mates,Code n,const Count* __restrict__ auth)',
        'gather main restrict',
    ),
    (
        '__global__ void gather_block_kernel(Count*out,Code n,const Count* auth)',
        '__global__ void gather_block_kernel(Count* __restrict__ out,Code n,const Count* __restrict__ auth)',
        'gather block restrict',
    ),
    (
        '__global__ void scatter_main_kernel(const Count*in,Code n,Count* auth)',
        '__global__ void scatter_main_kernel(const Count* __restrict__ in,Code n,Count* __restrict__ auth)',
        'scatter main restrict',
    ),
    (
        '__global__ void scatter_block_kernel(const Count*in,Code n,Count* auth)',
        '__global__ void scatter_block_kernel(const Count* __restrict__ in,Code n,Count* __restrict__ auth)',
        'scatter block restrict',
    ),
    (
        '__global__ void interval_io_kernel(Count*buf,const PeerInterval*iv,size_t niv,Count*auth)',
        '__global__ void interval_io_kernel(Count* __restrict__ buf,const PeerInterval* __restrict__ iv,size_t niv,Count* __restrict__ auth)',
        'interval restrict',
    ),
)


def once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f'{label}: expected exactly one basearg match, got {n}')
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('out', type=Path)
    a = ap.parse_args()
    text = a.src.read_text()
    for old, new, label in REPL:
        text = once(text, old, new, label)
    required = (
        'Count* __restrict__ out',
        'MateID* __restrict__ mates',
        'const Count* __restrict__ auth',
        'const Count* __restrict__ in',
        'Count* __restrict__ auth',
        'const PeerInterval* __restrict__ iv',
    )
    for token in required:
        if token not in text:
            raise SystemExit(f'missing restrict artifact: {token}')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    print(f'lowered {a.out} vmm_basearg_restrict=1 gather_scatter_restrict=1 interval_restrict=1 alias_contract=scratch_interval_auth_disjoint')


if __name__ == '__main__':
    main()
