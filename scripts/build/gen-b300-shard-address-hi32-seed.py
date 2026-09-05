#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "gen-b300-shard-address-mode.py"
spec = importlib.util.spec_from_file_location("b300_shard_base_generator", BASE_PATH)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

helper = base.HELPER_NEW
old_assert = 'static_assert(B300_SHARD_ADDRESS_MODE>=0&&B300_SHARD_ADDRESS_MODE<=4,"B300_SHARD_ADDRESS_MODE must be 0..4");'
new_assert = 'static_assert(B300_SHARD_ADDRESS_MODE>=0&&B300_SHARD_ADDRESS_MODE<=5,"B300_SHARD_ADDRESS_MODE must be 0..5");'
if helper.count(old_assert) != 1:
    raise SystemExit("base generator shard mode assertion changed")
helper = helper.replace(old_assert, new_assert, 1)

marker = '#if B300_SHARD_ADDRESS_MODE == 4\n'
mode5 = r'''__device__ __forceinline__ std::uint32_t shard_hi32_seed_main(Code g){std::uint32_t h=std::uint32_t(g>>32);return((h<<8)+(h<<6)+(h<<5)+(h<<3)+(h<<2)+h)>>12;}
__device__ __forceinline__ std::uint32_t shard_hi32_seed_block(Code g){return std::uint32_t(g>>32)>>2;}
__device__ __forceinline__ ShardAddress8 shard_address8_hi32_seed_main_w28_g8(Code g){constexpr Code chunk=48214938328ULL;std::uint32_t o=shard_hi32_seed_main(g);Code local=g-shard_base8_masked(int(o),chunk);std::uint32_t corr=std::uint32_t(local>=chunk);o+=corr;local-=(Code(0)-Code(corr))&chunk;return{int(o),local};}
__device__ __forceinline__ ShardAddress8 shard_address8_hi32_seed_block_w28_g8(Code g){constexpr Code chunk=16876938176ULL;std::uint32_t o=shard_hi32_seed_block(g);Code local=g-shard_base8_masked(int(o),chunk);std::uint32_t corr=std::uint32_t(local>=chunk);o+=corr;local-=(Code(0)-Code(corr))&chunk;return{int(o),local};}
#if B300_SHARD_ADDRESS_MODE == 5
static_assert(TARGET_W==28,"hi32-seed shard address is specialized for TARGET_W=28");
__device__ __forceinline__ Count global_load_main(Code g){auto a=shard_address8_hi32_seed_main_w28_g8(g);return D_MAIN_PTR[a.owner][a.local];}
__device__ __forceinline__ Count global_load_block(Code g){auto a=shard_address8_hi32_seed_block_w28_g8(g);return D_BLOCK_PTR[a.owner][a.local];}
__device__ __forceinline__ void global_store_main(Code g,Count v){auto a=shard_address8_hi32_seed_main_w28_g8(g);D_MAIN_PTR[a.owner][a.local]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){auto a=shard_address8_hi32_seed_block_w28_g8(g);D_BLOCK_PTR[a.owner][a.local]=v;}
#elif B300_SHARD_ADDRESS_MODE == 4
'''
if helper.count(marker) != 1:
    raise SystemExit("base generator mode-4 marker changed")
helper = helper.replace(marker, mode5, 1)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one source match, got {count}")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    text = args.src.read_text()
    text = replace_once(text, base.HELPER_OLD, helper, "shard helper")
    text = replace_once(text, base.NGPU_OLD, base.NGPU_NEW, "ngpu guard")
    text = replace_once(text, base.CHUNK_OLD, base.CHUNK_NEW, "chunk guard")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"generated {args.out} from {args.src} shard_address_modes=0,1,2,3,4,5 hi32_seed_mode=5 guarded_w28_ngpu8=1")


if __name__ == "__main__":
    main()
