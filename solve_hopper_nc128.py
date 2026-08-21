#!/usr/bin/env python3
import argparse, math, re, subprocess, time

PRIMES = [2305843009213693951,2305843009213693921,2305843009213693907,2305843009213693723,2305843009213693693,2305843009213693669,2305843009213693613,2305843009213693561,2305843009213693549,2305843009213693487,2305843009213693421,2305843009213693373,2305843009213693277,2305843009213693193,2305843009213693153,2305843009213693133,2305843009213693123,2305843009213693109,2305843009213693093,2305843009213693013,2305843009213692967,2305843009213692937,2305843009213692799,2305843009213692757,2305843009213692737,2305843009213692671]

def crt(residues):
    x,m=0,1
    for r,p in zip(residues,PRIMES):
        t=((r-x)%p)*pow(m,-1,p)%p
        x += m*t; m *= p
    return x,m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('n',type=int,nargs='?',default=19)
    ap.add_argument('--binary',default='./oneesan_cuda_hopper_nc128')
    ap.add_argument('--batch',type=int,default=4)
    a=ap.parse_args()
    if a.batch < 1: raise SystemExit('batch must be >=1')
    bound=2*a.n*(a.n+1)+1
    mbits=math.prod(PRIMES).bit_length()
    if mbits <= bound: raise SystemExit(f'CRT capacity insufficient: {mbits=} need > {bound}')
    residues=[]; total_gpu=total_wall=0.0; peak=slots=alloc=0
    nb=math.ceil(len(PRIMES)/a.batch)
    for bi,base in enumerate(range(0,len(PRIMES),a.batch),1):
        active=min(a.batch,len(PRIMES)-base)
        t0=time.perf_counter()
        p=subprocess.run([a.binary,str(a.n),str(base),str(active)],text=True,capture_output=True,check=True)
        wall=time.perf_counter()-t0
        print(p.stderr,end='')
        m=re.search(r'residues=([0-9,]+).*peak_states=(\d+).*hash_slots=(\d+).*peak_alloc_bytes=(\d+).*gpu_ms=([0-9.]+)',p.stdout)
        if not m: raise SystemExit(f'parse failed: {p.stdout!r}')
        rs=[int(x) for x in m.group(1).split(',')]
        if len(rs)!=active: raise SystemExit(f'bad residue count: {len(rs)} != {active}; binary RES_BATCH may be smaller than --batch')
        residues += rs
        total_gpu += float(m.group(5)); total_wall += wall
        peak=max(peak,int(m.group(2))); slots=max(slots,int(m.group(3))); alloc=max(alloc,int(m.group(4)))
        print(f'batch {bi}/{nb}: mods {base}..{base+active-1} gpu={float(m.group(5))/1000:.3f}s mem={int(m.group(4))/2**30:.3f}GiB')
    x,mod=crt(residues)
    print(f'n={a.n}')
    print(f'paths={x}')
    print(f'decimal_digits={len(str(x))}')
    print(f'bound_bits={bound}')
    print(f'modulus_bits={mod.bit_length()}')
    print(f'peak_states={peak}')
    print(f'hash_slots={slots}')
    print(f'peak_alloc_bytes={alloc}')
    print(f'peak_alloc_gib={alloc/2**30:.3f}')
    print(f'sum_gpu_ms={total_gpu:.3f}')
    print(f'sum_wall_s={total_wall:.3f}')

if __name__=='__main__': main()
