#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
for threads in (64,96,128,160,192,224,256,320,384,512,1024):
    assert threads % 32 == 0 and threads > 32
    workers=threads-32
    for ilp in (1,2,4):
        # Exhaust all small/tail shapes plus representative large columns.
        cases=list(range(0,4*workers+17)) + [4095,4096,4097,8191,8192,8193,65535]
        for cols in cases:
            seen=[0]*cols
            group_step=workers*ilp
            for tid in range(32,threads):
                worker=tid-32
                base=worker
                while base<cols:
                    for j in range(ilp):
                        lr=base+j*workers
                        if lr<cols: seen[lr]+=1
                    base+=group_step
            if any(x!=1 for x in seen):
                bad=next((i,x) for i,x in enumerate(seen) if x!=1)
                raise SystemExit(f'coverage mismatch threads={threads} ilp={ilp} cols={cols} bad={bad}')
print('b300_pipe2_producer_warp_coverage=OK')
print('producer_warp=32 worker_threads=blockDim-32 ilp=1,2,4 exact_once=1 warp_coalesced=1')
PY
