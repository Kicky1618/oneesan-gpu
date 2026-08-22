#!/usr/bin/env python3
import argparse, random, math
from pathlib import Path
from compare_zdd_families import enumerate_family

def unique_under(paths, keep):
    mask=sum(1<<e for e in keep)
    seen=set()
    for p in paths:
        x=0
        for e in p:x|=1<<e
        y=x&mask
        if y in seen:return False
        seen.add(y)
    return True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('zdd');ap.add_argument('--trials',type=int,default=2000);ap.add_argument('--seed',type=int,default=1618);a=ap.parse_args()
    fam=enumerate_family(a.zdd); paths=[]; maxe=-1
    for p in fam:
        x=0
        for e in p:x|=1<<e;maxe=max(maxe,e)
        paths.append(x)
    E=maxe+1;lb=math.ceil(math.log2(len(paths)))
    print('paths',len(paths),'E',E,'info_lb',lb)
    rng=random.Random(a.seed);best=list(range(E))
    # Greedy deletion with random edge orders. Maintain signatures from scratch; small n only.
    for tr in range(a.trials):
        keep=list(range(E));rng.shuffle(keep)
        # actual set starts all; attempt removals in shuffled priority.
        S=set(range(E))
        for e in keep:
            cand=S-{e};mask=sum(1<<q for q in cand);seen=set();ok=True
            for x in paths:
                y=x&mask
                if y in seen:ok=False;break
                seen.add(y)
            if ok:S=cand
        if len(S)<len(best):
            best=sorted(S);print('trial',tr,'best',len(best),best,flush=True)
            if len(best)==lb:break
    print('FINAL',len(best),best)
if __name__=='__main__':main()
