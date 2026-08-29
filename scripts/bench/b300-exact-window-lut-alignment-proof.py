#!/usr/bin/env python3
MAXW=28;W=28;SCRATCH_TARGET=65536<<20;PLAN_TARGET=16384<<20
H=[[0]*(MAXW+2) for _ in range(MAXW+1)]
for h in range(MAXW+2):H[0][h]=int(h==0)
for w in range(1,MAXW+1):
    for h in range(MAXW+1):H[w][h]=H[w-1][h]+(H[w-1][h-1] if h else 0)+(H[w-1][h+1] if h<MAXW+1 else 0)
def size(width,fixed,occ):
    p=[0]*(MAXW+2);p[0]=1
    for w in range(1,width+1):
        q=[0]*(MAXW+2);pos=w-1;f=(fixed>>pos)&1;o=(occ>>pos)&1
        for h in range(MAXW+1):
            if not f or not o:q[h]+=p[h]
            if not f or o:
                if h:q[h]+=p[h-1]
                if h<MAXW+1:q[h]+=p[h+1]
        p=q
    return p[1]
def cand(hi,lo):return [q for q in range(W-1,-1,-1) if q<lo-1 or q>hi]
def masks(hi,lo,fp,g):
    mf=mo=bf=bo=0
    for i,q in enumerate(fp):
        one=(g>>i)&1;mf|=1<<q;mo|=one<<q;bq=q if q<lo-1 else q-1;bf|=1<<bq;bo|=one<<bq
    return mf,mo,bf,bo
def plan(hi,lo,target):
    c=cand(hi,lo);klim=min(len(c),20)
    for k in range(klim+1):
        fp=c[:k];mx=mm=md=0
        for g in range(1<<k):
            mf,mo,bf,bo=masks(hi,lo,fp,g);m=size(W,mf,mo);d=size(W-1,bf,bo);b=8*(m+d)
            if b>mx:mx,mm,md=b,m,d
            if mx>target and k<klim:break
        if mx<=target or k==klim:return k,mx,mm,md,fp
def schedule(target,maxwin):
    hi=W-1;out=[]
    while hi>=1:
        for lo in range(max(1,hi-maxwin+1),hi+1):
            p=plan(hi,lo,target)
            if p[1] and p[1]<=target:out.append((hi,lo,*p));hi=lo-1;break
        else:raise SystemExit(f'cannot plan hi={hi}')
    return out
def low_ok(width,fixed,k):return (fixed&((1<<k)-1))==((1<<k)-1)
def high_ok(width,fixed,k):
    low=width-k;hm=((1<<k)-1)<<low;wm=(1<<width)-1
    return (fixed&hm)==hm and (fixed&(wm^hm))==0
wide=schedule(SCRATCH_TARGET,27)
aligned=schedule(PLAN_TARGET,14)
assert [(x[0],x[1],x[2]) for x in wide]==[(27,11,10),(10,1,10)],wide
assert [(x[0],x[1],x[2]) for x in aligned]==[(27,14,13),(13,1,13)],aligned
for idx,(hi,lo,k,mx,mm,md,fp) in enumerate(aligned):
    mf,_,bf,_=masks(hi,lo,fp,0)
    if idx==0:assert low_ok(28,mf,13) and low_ok(27,bf,13)
    else:assert high_ok(28,mf,13) and high_ok(27,bf,13)
count_max=max(x[3] for x in aligned)
with_both_mates=2*count_max
assert with_both_mates<SCRATCH_TARGET
print('b300-exact-window-lut-alignment-proof OK '
      'unsplit_budget_gib=64 unsplit_windows=27:11,10:1 unsplit_fixed_bits=10 lut13_fastpath=0 '
      'planner_budget_gib=16 aligned_windows=27:14,13:1 aligned_fixed_bits=13 lut13_fastpath=1 '
      f'aligned_count_scratch_gib={count_max/(1<<30):.9f} '
      f'aligned_with_main_block_mates_gib={with_both_mates/(1<<30):.9f} scratch_budget_gib={SCRATCH_TARGET/(1<<30):.1f}')
