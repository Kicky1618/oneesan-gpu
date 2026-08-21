#!/usr/bin/env python3
import argparse
MAXW=28
COUNT_BYTES=8

def spec_size(W,fixed,occ):
    prev=[0]*(MAXW+2); prev[0]=1
    for pos in range(W):
        cur=[0]*(MAXW+2); f=(fixed>>pos)&1; o=(occ>>pos)&1
        for h in range(MAXW+1):
            x=0
            if not f or not o: x+=prev[h]
            if not f or o:
                if h>0: x+=prev[h-1]
                if h<MAXW+1: x+=prev[h+1]
            cur[h]=x
        prev=cur
    return prev[1]

def candidates(W,hi,lo):
    return [q for q in range(W-1,-1,-1) if q<lo-1 or q>hi]

def masks(W,hi,lo,fp,g):
    mf=mo=bf=bo=0
    for i,q in enumerate(fp):
        one=(g>>i)&1
        mf|=1<<q
        if one: mo|=1<<q
        bq=q if q<lo-1 else q-1
        bf|=1<<bq
        if one: bo|=1<<bq
    return mf,mo,bf,bo

def plan_window(W,hi,lo,target,maxbits=16):
    c=candidates(W,hi,lo)
    klim=min(len(c),maxbits)
    for k in range(klim+1):
        fp=c[:k]; mx=mm=md=0
        for g in range(1<<k):
            mf,mo,bf,bo=masks(W,hi,lo,fp,g)
            m=spec_size(W,mf,mo); d=spec_size(W-1,bf,bo)
            b=(2*m+d)*COUNT_BYTES
            if b>mx: mx,mm,md=b,m,d
            if mx>target and k<klim: break
        if mx<=target or k==klim:
            return k,mx,mm,md,fp

def row_plan(n,target,max_window=999):
    W=n+1; hi=W-1; out=[]
    while hi>=1:
        found=None
        for lo in range(max(1,hi-max_window+1),hi+1):
            p=plan_window(W,hi,lo,target)
            if p and p[1]<=target:
                found=(hi,lo,*p); break
        if not found: raise RuntimeError(f'cannot fit p={hi}')
        out.append(found); hi=found[1]-1
    return out

def full_states(W): return spec_size(W,0,0),spec_size(W-1,0,0)

def main():
    global COUNT_BYTES
    ap=argparse.ArgumentParser(); ap.add_argument('n',type=int); ap.add_argument('--vram-gib',type=float,required=True); ap.add_argument('--max-window',type=int,default=999); ap.add_argument('--count-bytes',type=int,default=8,choices=[1,2,4,8])
    a=ap.parse_args(); COUNT_BYTES=a.count_bytes; W=a.n+1; target=int(a.vram_gib*2**30)
    m,d=full_states(W); ext=(m+d)*COUNT_BYTES
    p=row_plan(a.n,target,a.max_window)
    print(f'n={a.n} width={W}')
    print(f'main_states={m} blocked_states={d} external_gib={ext/2**30:.3f}')
    print(f'target_vram_gib={a.vram_gib} max_window={a.max_window} windows_per_row={len(p)}')
    for i,x in enumerate(p,1):
        hi,lo,k,mx,mm,md,fp=x
        print(f'window{i}: p={hi}..{lo} len={hi-lo+1} groups={1<<k} fixed={fp} max_working_gib={mx/2**30:.3f}')
    traffic=2*ext*len(p)*W
    print(f'ideal_host_gpu_traffic_per_modulus_tib={traffic/2**40:.3f}')
if __name__=='__main__': main()
