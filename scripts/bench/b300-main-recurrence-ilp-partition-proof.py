#!/usr/bin/env python3
from __future__ import annotations

for lanes in (4,8):
    cases=0
    for n in list(range(1,257))+[511,512,513,1000,4095,4096,4097,65535,65536,65537]:
        for grid in (1,2,3,7,31,32,127,256,1024,65535):
            seen=[0]*n
            # GPU threads have tids 0..grid-1, and each thread advances by
            # lanes*grid while exposing lanes offsets k*grid per iteration.
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
            cases+=1
    print(f'b300-main-recurrence-ilp-partition-proof lanes={lanes} cases={cases} exact=1')
print('b300-main-recurrence-ilp-partition-proof OK exact=1')
