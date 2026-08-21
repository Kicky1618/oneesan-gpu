#!/usr/bin/env python3
from functools import lru_cache
MAXW=28

def build_full():
    dp=[[0]*(MAXW+2) for _ in range(MAXW+1)]
    for h in range(MAXW+2): dp[0][h]=int(h==0)
    for w in range(1,MAXW+1):
        for h in range(MAXW+1):
            x=dp[w-1][h]
            if h>0: x+=dp[w-1][h-1]
            if h<MAXW+1: x+=dp[w-1][h+1]
            dp[w][h]=x
    return dp
H=build_full()

def spec(W,fixed,occ):
    dp=[[0]*(MAXW+2) for _ in range(MAXW+1)]
    for h in range(MAXW+2): dp[0][h]=int(h==0)
    for w in range(1,W+1):
        pos=w-1; f=(fixed>>pos)&1; o=(occ>>pos)&1
        for h in range(MAXW+1):
            x=0
            if not f or not o: x+=dp[w-1][h]
            if not f or o:
                if h>0: x+=dp[w-1][h-1]
                if h<MAXW+1: x+=dp[w-1][h+1]
            dp[w][h]=x
    return dp[W][1]

def leaf_upper(W,fixed,occ):
    @lru_cache(None)
    def rec(pos,h):
        if pos<0: return int(h==0)
        lower=(1<<(pos+1))-1
        if fixed & lower == 0:
            return int(H[pos+1][h] != 0)
        f=(fixed>>pos)&1; o=(occ>>pos)&1
        z=0
        if not f or not o: z+=rec(pos-1,h)
        if not f or o:
            if h>0: z+=rec(pos-1,h-1)
            if h<MAXW+1: z+=rec(pos-1,h+1)
        return z
    return rec(W-1,1)

def masks(W,hi,lo,fp,g):
    mf=mo=0
    for i,q in enumerate(fp):
        mf |= 1<<q
        if (g>>i)&1: mo |= 1<<q
    return mf,mo

def report(W,hi,lo,fp):
    rows=[]
    for g in range(1<<len(fp)):
        fixed,occ=masks(W,hi,lo,fp,g)
        size=spec(W,fixed,occ)
        leaves=leaf_upper(W,fixed,occ)
        rows.append((size,leaves, size/leaves if leaves else float('inf')))
    sizes=[x[0] for x in rows]; leaves=[x[1] for x in rows]; avgs=[x[2] for x in rows]
    print(f'p={hi}..{lo} fixed={fp} groups={len(rows)}')
    print(f' size min/med/max={min(sizes):,}/{sorted(sizes)[len(sizes)//2]:,}/{max(sizes):,}')
    print(f' leaves min/med/max={min(leaves):,}/{sorted(leaves)[len(leaves)//2]:,}/{max(leaves):,}')
    print(f' avg_run min/med/max={min(avgs):.1f}/{sorted(avgs)[len(avgs)//2]:.1f}/{max(avgs):.1f}')
    for th in [128,256,512,1024,4096,16384]:
        print(f'  groups avg_run>={th}: {sum(a>=th for a in avgs)}/{len(avgs)}')

W=28
report(W,27,14,[12,11,10,9,8,7,6])
report(W,13,1,[27,26,25,24,23,22,21])
