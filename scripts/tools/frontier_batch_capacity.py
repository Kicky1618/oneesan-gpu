#!/usr/bin/env python3
"""Necessary admission conditions for batched frontier maps, without a GPU.

Fractions are weighted by logical states, not masks. They are UPPER bounds:
CSR columns, temporary storage and actual free device memory can reduce coverage.
"""
import argparse
import json
from math import comb


def count_states(symbols):
    heights = {1: 1}
    for occupied in symbols:
        following = {}
        for height, count in heights.items():
            for delta in ((-1, 1) if occupied else (-1, 0, 1)):
                if height + delta >= 0:
                    following[height+delta] = following.get(height+delta, 0) + count
        heights = following
    return heights.get(0, 0)


def capacity(n, scratch_mib, map_mib):
    width = n+1
    low = width//2
    high = n-low
    full_main = count_states([False]*width)
    full_block = count_states([False]*(width-1))
    result = dict(n=n, width=width, low=low, high=high, scratch_mib=scratch_mib,
                  map_mib=map_mib, main_states=full_main, blocked_states=full_block, windows=[])
    for fixed_low in (True, False):
        fixed = low if fixed_low else high
        active = high if fixed_low else low
        stages = (active+1)//2
        shapes = []
        for occupied in range(fixed+1):
            prefix = [False]*active if fixed_low else [True]*occupied
            suffix = [True]*occupied if fixed_low else [False]*active
            main = count_states(prefix+[False]+suffix)
            blocked = count_states(prefix+suffix)
            states = main+blocked
            masks = comb(fixed, occupied)
            index_ok = states < 0x7fffffff and states <= 0xffffffff//(6*width)
            offsets_min = stages*(states+1)*4
            scratch_two = states*16*2+6*255
            possible = masks > 1 and index_ok and offsets_min <= map_mib*2**20 and scratch_two <= scratch_mib*2**20
            shapes.append(dict(occupied=occupied,masks=masks,main_states=main,blocked_states=blocked,
                               states=states,scan_admits=index_ok,min_offsets_mib=offsets_min/2**20,
                               two_group_admission_mib=scratch_two/2**20,batch_possible=possible))
        assert sum(s['masks']*s['main_states'] for s in shapes) == full_main
        assert sum(s['masks']*s['blocked_states'] for s in shapes) == full_block
        eligible = sum(s['masks']*s['states'] for s in shapes if s['batch_possible'])
        result['windows'].append(dict(fixed_low=fixed_low,stages=stages,
            state_fraction_upper_bound=eligible/(full_main+full_block),shapes=shapes))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=27)
    parser.add_argument('--scratch-mib',type=int,default=16384)
    parser.add_argument('--map-mib',type=int,default=512)
    args = parser.parse_args()
    if not 3 <= args.n <= 27 or args.scratch_mib < 1 or args.map_mib < 0:
        parser.error('require 3 <= n <= 27, scratch-mib >= 1, map-mib >= 0')
    print(json.dumps(capacity(args.n,args.scratch_mib,args.map_mib),indent=2))


if __name__ == '__main__':
    main()
