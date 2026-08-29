#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
def producer_slots(threads, tid, weight):
    warp=tid//32; lane=tid%32; nwarps=threads//32
    if weight==0:
        if warp==0: return [], (nwarps-1)*32
        return [warp-1], (nwarps-1)*32
    total_slots=1+(nwarps-1)*weight
    if warp==0: slots=[0]
    else: slots=list(range(1+(warp-1)*weight,1+warp*weight))
    return slots,total_slots*32

def prove_producer(threads, ilp, cols, weight):
    seen=[0]*cols
    quad_full=quad_tail=0
    lane_work=[0]*(threads//32)
    for tid in range(threads):
        slots,workers=producer_slots(threads,tid,weight)
        lane=tid%32
        group_step=workers*ilp
        for slot in slots:
            worker=slot*32+lane
            base=worker
            while base<cols:
                valid=[]
                for j in range(ilp):
                    lr=base+j*workers
                    ok=lr<cols
                    valid.append(ok)
                    if ok:
                        seen[lr]+=1
                        lane_work[tid//32]+=1
                if any(valid[j] and not valid[j-1] for j in range(1,ilp)):
                    raise SystemExit(f'producer non-prefix tail threads={threads} ilp={ilp} cols={cols} weight={weight} base={base} valid={valid}')
                if ilp==4:
                    if all(valid): quad_full+=1
                    elif any(valid): quad_tail+=1
                base+=group_step
    if any(x!=1 for x in seen):
        bad=next((i,x) for i,x in enumerate(seen) if x!=1)
        raise SystemExit(f'producer coverage mismatch threads={threads} ilp={ilp} cols={cols} weight={weight} bad={bad}')
    return quad_full,quad_tail,lane_work

def prove_fallback(threads, ilp, cols):
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
    for weight in range(5):
        # Device macro accepts 0..4. weight=0 excludes producer warp; positive
        # weights give producer one virtual slot and every worker warp `weight`.
        for ilp in (1,2,4):
            base_workers=(threads-32) if weight==0 else (1+(threads//32-1)*weight)*32
            cases=list(range(0,4*base_workers+17)) + [4095,4096,4097,8191,8192,8193,65535]
            for cols in cases:
                qf,qt,work=prove_producer(threads,ilp,cols,weight)
                quad_full_total+=qf;quad_tail_total+=qt
                if cols >= base_workers*4 and weight==0 and work[0] != 0:
                    raise SystemExit('weight0 producer warp unexpectedly owns columns')
                if cols >= base_workers*4 and weight>0 and work[0] == 0:
                    raise SystemExit('positive-weight producer warp owns no columns')

# For 256 threads the intended asymptotic producer shares are exact in virtual
# slot space: 0, 1/8, 1/15, 1/22, 1/29 for weights 0..4.
expected={0:(0,7),1:(1,8),2:(1,15),3:(1,22),4:(1,29)}
for w,(num,den) in expected.items():
    slots,total=producer_slots(256,0,w)
    got_num=len(slots);got_den=total//32
    if (got_num,got_den)!=(num,den):
        raise SystemExit(f'weight share mismatch w={w} got={got_num}/{got_den} expected={num}/{den}')

for threads in (1,17,31,32,33,47,63,65,95,97,127,129,255,257,511,513,1023):
    assert threads <= 32 or threads % 32 != 0
    for ilp in (1,2,4):
        cases=list(range(0,4*threads+17)) + [4095,4096,4097,8191,8192,8193,65535]
        for cols in cases: prove_fallback(threads,ilp,cols)

assert quad_full_total>0 and quad_tail_total>0
print('b300_pipe2_producer_warp_coverage=OK')
print('producer_worker_weight=0,1,2,3,4 exact_once=1 ilp=1,2,4 fallback_partial_warp=ordinary_exact_once')
print('producer_share_256=w0:0,w1:1/8,w2:1/15,w3:1/22,w4:1/29')
print(f'producer_quad_full_groups_seen={quad_full_total} producer_quad_tail_groups_seen={quad_tail_total} quad_tail_valid_prefix=1 pair_single_fallback_exact=1')
PY
