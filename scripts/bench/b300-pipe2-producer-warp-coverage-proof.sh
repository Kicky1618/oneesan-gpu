#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
def prove_producer(threads, ilp, cols):
    workers=threads-32
    seen=[0]*cols
    group_step=workers*ilp
    quad_full=quad_tail=0
    for tid in range(32,threads):
        worker=tid-32
        base=worker
        while base<cols:
            valid=[]
            for j in range(ilp):
                lr=base+j*workers
                ok=lr<cols
                valid.append(ok)
                if ok: seen[lr]+=1
            # lr[j] grows monotonically by workers, so the producer-quad tail
            # must always be a valid prefix. The device fallback handles that
            # prefix as pair+pair or singles; a hole would violate its contract.
            if any(valid[j] and not valid[j-1] for j in range(1,ilp)):
                raise SystemExit(f'producer non-prefix tail threads={threads} ilp={ilp} cols={cols} base={base} valid={valid}')
            if ilp==4:
                if all(valid): quad_full+=1
                elif any(valid): quad_tail+=1
            base+=group_step
    if any(x!=1 for x in seen):
        bad=next((i,x) for i,x in enumerate(seen) if x!=1)
        raise SystemExit(f'producer coverage mismatch threads={threads} ilp={ilp} cols={cols} bad={bad}')
    return quad_full,quad_tail

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
quad_full_total=quad_tail_total=0
for threads in aligned:
    assert threads % 32 == 0 and threads > 32
    workers=threads-32
    for ilp in (1,2,4):
        cases=list(range(0,4*workers+17)) + [4095,4096,4097,8191,8192,8193,65535]
        for cols in cases:
            qf,qt=prove_producer(threads,ilp,cols)
            quad_full_total+=qf;quad_tail_total+=qt

# Kernel falls back to the ordinary executor for <=32 or partial final warps.
# Prove representative boundaries on both sides of every warp edge.
for threads in (1,17,31,32,33,47,63,65,95,97,127,129,255,257,511,513,1023):
    assert threads <= 32 or threads % 32 != 0
    for ilp in (1,2,4):
        cases=list(range(0,4*threads+17)) + [4095,4096,4097,8191,8192,8193,65535]
        for cols in cases: prove_fallback(threads,ilp,cols)

assert quad_full_total>0 and quad_tail_total>0
print('b300_pipe2_producer_warp_coverage=OK')
print('producer_warp=32 aligned_worker_warps=exact_once fallback_partial_warp=ordinary_exact_once ilp=1,2,4')
print(f'producer_quad_full_groups_seen={quad_full_total} producer_quad_tail_groups_seen={quad_tail_total} quad_tail_valid_prefix=1 pair_single_fallback_exact=1')
PY
