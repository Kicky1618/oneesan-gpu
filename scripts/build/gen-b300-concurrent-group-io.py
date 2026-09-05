#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-concurrent-group-io.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()

def repl(old:str,new:str,label:str,min_count:int=1)->None:
    global s
    n=s.count(old)
    if n<min_count: raise SystemExit(f'{label}: expected >= {min_count} matches got {n}')
    s=s.replace(old,new)

# Gather authoritative main/block data and materialize their topology caches on
# independent nonblocking streams. They touch disjoint scratch and auth arrays.
repl('interval_io_kernel<false,false><<<interval_blocks(pg.mi.size(),threads),threads>>>(',
     'interval_io_kernel<false,false><<<interval_blocks(pg.mi.size(),threads),threads,0,c.sMain>>>(',
     'main interval gather stream')
repl('interval_io_kernel<true,false><<<interval_blocks(pg.di.size(),threads),threads>>>(',
     'interval_io_kernel<true,false><<<interval_blocks(pg.di.size(),threads),threads,0,c.sBlock>>>(',
     'block interval gather stream')
repl('gather_main_kernel<<<bm,threads>>>(', 'gather_main_kernel<<<bm,threads,0,c.sMain>>>(', 'main gather stream')
repl('gather_block_kernel<<<bd,threads>>>(', 'gather_block_kernel<<<bd,threads,0,c.sBlock>>>(', 'block gather stream')
if 'materialize_main_mates_kernel<<<bm,threads>>>' in s:
    s=s.replace('materialize_main_mates_kernel<<<bm,threads>>>','materialize_main_mates_kernel<<<bm,threads,0,c.sMain>>>')
if 'materialize_block_mates_kernel<<<bd,threads>>>' in s:
    s=s.replace('materialize_block_mates_kernel<<<bd,threads>>>','materialize_block_mates_kernel<<<bd,threads,0,c.sBlock>>>')
repl('ck(cudaGetLastError(),"doubleD gather");ck(cudaDeviceSynchronize(),"doubleD gather sync");',
     'ck(cudaGetLastError(),"doubleD gather");ck(cudaStreamSynchronize(c.sMain),"main gather sync");ck(cudaStreamSynchronize(c.sBlock),"block gather sync");',
     'gather synchronization')

# Packed rank-state initialization is also independent between main and block.
if 'b300_init_main_rank_delta_kernel<<<bm,threads>>>' in s:
    s=s.replace('b300_init_main_rank_delta_kernel<<<bm,threads>>>','b300_init_main_rank_delta_kernel<<<bm,threads,0,c.sMain>>>')
if 'b300_init_block_rank_delta_kernel<<<bd,threads>>>' in s:
    s=s.replace('b300_init_block_rank_delta_kernel<<<bd,threads>>>','b300_init_block_rank_delta_kernel<<<bd,threads,0,c.sBlock>>>')
old='ck(cudaGetLastError(),"rank delta init");ck(cudaDeviceSynchronize(),"rank delta init sync");'
if old in s:
    s=s.replace(old,'ck(cudaGetLastError(),"rank delta init");ck(cudaStreamSynchronize(c.sMain),"main rank delta init sync");ck(cudaStreamSynchronize(c.sBlock),"block rank delta init sync");')

# Scatter the two disjoint authoritative state families concurrently too.
repl('interval_io_kernel<false,true><<<interval_blocks(pg.mi.size(),threads),threads>>>(',
     'interval_io_kernel<false,true><<<interval_blocks(pg.mi.size(),threads),threads,0,c.sMain>>>(',
     'main interval scatter stream')
repl('interval_io_kernel<true,true><<<interval_blocks(pg.di.size(),threads),threads>>>(',
     'interval_io_kernel<true,true><<<interval_blocks(pg.di.size(),threads),threads,0,c.sBlock>>>(',
     'block interval scatter stream')
repl('scatter_main_kernel<<<bm,threads>>>(', 'scatter_main_kernel<<<bm,threads,0,c.sMain>>>(', 'main scatter stream')
repl('scatter_block_kernel<<<bd,threads>>>(', 'scatter_block_kernel<<<bd,threads,0,c.sBlock>>>(', 'block scatter stream')
repl('ck(cudaGetLastError(),"doubleD scatter");ck(cudaDeviceSynchronize(),"group sync");',
     'ck(cudaGetLastError(),"doubleD scatter");ck(cudaStreamSynchronize(c.sMain),"main scatter sync");ck(cudaStreamSynchronize(c.sBlock),"block scatter sync");',
     'scatter synchronization')

for stale in ('cudaDeviceSynchronize(),"doubleD gather sync"','cudaDeviceSynchronize(),"group sync"','cudaDeviceSynchronize(),"rank delta init sync"'):
    if stale in s: raise SystemExit(f'device-wide group I/O sync remains: {stale}')
for req in ('main gather sync','block gather sync','main scatter sync','block scatter sync','threads,0,c.sMain','threads,0,c.sBlock'):
    if req not in s: raise SystemExit(f'missing concurrent group-I/O artifact: {req}')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_concurrent_group_io=1 main_block_gather_overlap=1 mate_materialize_overlap=1 rank_state_init_overlap=1 main_block_scatter_overlap=1 devicewide_group_io_sync=0')
