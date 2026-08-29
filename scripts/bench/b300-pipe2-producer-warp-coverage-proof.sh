#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
def prove_producer(threads, ilp, cols):
    workers=threads-32
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
        raise SystemExit(f'producer coverage mismatch threads={threads} ilp={ilp} cols={cols} bad={bad}')

def prove_fallback(threads, ilp, cols):
    # Ordinary flat column executor: all block threads participate, each owns
    # ILP columns separated by blockDim and advances by blockDim*ILP.
    seen=[0]*cols
    group_step=threads*ilp
    for tid in range(threads):
        base=tid
        while base<cols:
            for j in range(ilp):
                lr=base+j*threads
                if lr<cols: seen[lr]+=1
            base+=group_step
    if any(x!=1 for x in seen):
        bad=next((i,x) for i,x in enumerate(seen) if x!=1)
        raise SystemExit(f'fallback coverage mismatch threads={threads} ilp={ilp} cols={cols} bad={bad}')

aligned=(64,96,128,160,192,224,256,320,384,512,1024)
for threads in aligned:
    assert threads % 32 == 0 and threads > 32
    workers=threads-32
    for ilp in (1,2,4):
        cases=list(range(0,4*workers+17)) + [4095,4096,4097,8191,8192,8193,65535]
        for cols in cases: prove_producer(threads,ilp,cols)

# Kernel falls back to the ordinary executor for <=32 or partial final warps.
# Prove representative boundaries on both sides of every warp edge.
for threads in (1,17,31,32,33,47,63,65,95,97,127,129,255,257,511,513,1023):
    assert threads <= 32 or threads % 32 != 0
    for ilp in (1,2,4):
        cases=list(range(0,4*threads+17)) + [4095,4096,4097,8191,8192,8193,65535]
        for cols in cases: prove_fallback(threads,ilp,cols)

print('b300_pipe2_producer_warp_coverage=OK')
print('producer_warp=32 aligned_worker_warps=exact_once fallback_partial_warp=ordinary_exact_once ilp=1,2,4')
PY
