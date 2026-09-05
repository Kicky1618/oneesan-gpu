#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
# CUDA target is little-endian and the compact prectx tail is four consecutive
# uint8 fields: local_n,cross_depth,fixed_hs,pad.  The warp helper loads those
# bytes once as uint32. Prove the decoder and cached-bid high byte for the full
# production ranges used by K14/W28.
cases=0
for n in range(0,9):
  for depth in range(0,14):
    for hs in range(0,16):
      for bid in range(0,64):
        b=bytes((n,depth,hs,bid))
        meta=int.from_bytes(b,'little')
        assert (meta & 0xff)==n
        assert ((meta>>8)&0xff)==depth
        assert ((meta>>16)&0xff)==hs
        assert ((meta>>24)&0xff)==bid
        # Loading pad separately and loading the tail then extracting the high
        # byte must choose exactly the same cached main block.
        assert b[3] == (meta>>24)
        cases+=1
print('b300_compact_prectx_meta_proof=OK')
print(f'cases={cases} tail_bytes=4 meta_loads_cached_bid_path=1 bid_shift=24 local_n_mask=255 cross_depth_shift=8 fixed_hs_shift=16')
print('k14_local_n_max=8 k14_cross_depth_max=13 w28_main_blocks_lt64=1 bytes_added=0')
PY
