#!/usr/bin/env python3
import math, re, subprocess, sys, time
PRIMES=[
2305843009213693951,2305843009213693921,2305843009213693907,
2305843009213693723,2305843009213693693,2305843009213693669,
2305843009213693613,2305843009213693561,2305843009213693549,
2305843009213693487,2305843009213693421,2305843009213693373,
2305843009213693277,
]
def crt(rs):
 x,m=0,1
 for r,p in zip(rs,PRIMES):
  t=((r-x)%p)*pow(m,-1,p)%p
  x+=m*t;m*=p
 return x,m
def main():
 n=int(sys.argv[1]) if len(sys.argv)>1 else 19
 bound=2*n*(n+1)+1
 mb=math.prod(PRIMES).bit_length()
 if mb<=bound: raise SystemExit(f'CRT insufficient: modulus_bits={mb}, need > {bound}')
 t0=time.perf_counter()
 p=subprocess.run(['./build/oneesan_cuda_hopper_mem',str(n)],text=True,capture_output=True,check=True)
 wall=time.perf_counter()-t0
 print(p.stderr,end='',file=sys.stderr)
 m=re.search(r'residues=([0-9,]+).*peak_states=(\d+).*hash_slots=(\d+).*peak_alloc_bytes=(\d+).*gpu_ms=([0-9.]+)',p.stdout)
 if not m: raise SystemExit('parse failed: '+repr(p.stdout))
 rs=[int(x) for x in m.group(1).split(',')]
 x,M=crt(rs)
 print(f'n={n}')
 print(f'paths={x}')
 print(f'decimal_digits={len(str(x))}')
 print(f'bound_bits={bound}')
 print(f'modulus_bits={M.bit_length()}')
 print(f'peak_states={m.group(2)}')
 print(f'hash_slots={m.group(3)}')
 b=int(m.group(4));print(f'peak_alloc_bytes={b}');print(f'peak_alloc_gib={b/2**30:.3f}')
 print(f'gpu_ms={m.group(5)}')
 print(f'wall_s={wall:.3f}')
if __name__=='__main__':main()
