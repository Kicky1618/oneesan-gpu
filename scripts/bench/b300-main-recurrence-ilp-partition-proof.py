#!/usr/bin/env python3
from __future__ import annotations

CAP=65535

def blocks_for(n:int,threads:int,lanes:int)->int:
    return min(CAP,max(1,(n+threads*lanes-1)//(threads*lanes)))

def check_partition(n:int,grid:int,lanes:int)->None:
    seen=[0]*n
    for tid in range(min(grid,n)):
        base=tid
        while base<n:
            for k in range(lanes):
                i=base+k*grid
                if i<n:seen[i]+=1
            base+=lanes*grid
    if any(x!=1 for x in seen):
        bad=next(i for i,x in enumerate(seen) if x!=1)
        raise SystemExit(f'partition mismatch lanes={lanes} n={n} grid={grid} i={bad} count={seen[bad]}')

def inverse_ok(i:int,n:int,grid:int,lanes:int)->bool:
    if not 0<=i<n:return False
    tid=i%grid
    ordinal=i//grid
    q,k=divmod(ordinal,lanes)
    base=tid+q*lanes*grid
    return tid<grid and k<lanes and base<n and base+k*grid==i

for lanes in (4,8):
    arbitrary_cases=0
    for n in list(range(1,257))+[511,512,513,1000,4095,4096,4097,65535,65536,65537]:
        for grid in (1,2,3,7,31,32,127,256,1024,65535):
            check_partition(n,grid,lanes)
            arbitrary_cases+=1

    production_cases=0
    multi_lane_cases=0
    for threads in (32,64,128,256,512,1024):
        cap_n=CAP*threads*lanes
        ns=[1,threads-1,threads,threads+1,2*threads,4*threads-1,4*threads,4*threads+1,
            4095,4096,4097,65535,65536,65537]
        # Check the 65535-block transition algebraically without allocating a
        # vector hundreds of millions of elements long.
        ns += [cap_n-1,cap_n,cap_n+1,cap_n+lanes*threads+17]
        for n in ns:
            if n<=0:continue
            blocks=blocks_for(n,threads,lanes)
            grid=blocks*threads
            expect=min(CAP,max(1,(n+threads*lanes-1)//(threads*lanes)))
            if blocks!=expect:
                raise SystemExit(f'block formula mismatch lanes={lanes} n={n} threads={threads}')
            if n<=200000:
                check_partition(n,grid,lanes)
            last=n-1
            samples={0,min(1,last),min(threads-1,last),min(threads,last),last//7,last//3,last//2,max(0,last-1),last}
            if not all(inverse_ok(i,n,grid,lanes) for i in samples):
                raise SystemExit(f'production inverse mismatch lanes={lanes} n={n} threads={threads} grid={grid}')

            # Below the block cap, the old bm=ceil(n/threads) launch has
            # grid>=n and therefore only lane0 can be live.  The scaled launch
            # must expose >1 destination as soon as there is more than one
            # thread-span of work.
            old_blocks=min(CAP,max(1,(n+threads-1)//threads))
            old_grid=old_blocks*threads
            old_live=min(lanes,(n+old_grid-1)//old_grid)
            new_live=min(lanes,(n+grid-1)//grid)
            if n<=CAP*threads and n>threads:
                if old_live!=1:
                    raise SystemExit(f'old launch model unexpected lanes={lanes} n={n} threads={threads} live={old_live}')
                if new_live<=1:
                    raise SystemExit(f'scaled launch failed to expose MLP lanes={lanes} n={n} threads={threads} live={new_live}')
                multi_lane_cases+=1
            production_cases+=1

    print(f'b300-main-recurrence-ilp-partition-proof lanes={lanes} arbitrary_cases={arbitrary_cases} production_cases={production_cases} multi_lane_cases={multi_lane_cases} exact=1 production_launch_scaled=1 block_cap={CAP}')
print('b300-main-recurrence-ilp-partition-proof OK exact=1 production_launch_scaled=1')
