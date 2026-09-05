#!/usr/bin/env python3
import pathlib, sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-batch-shard-address8.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
old='''__device__ __forceinline__ Count global_load_main(Code g){int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK];}
__device__ __forceinline__ Count global_load_block(Code g){int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK];}
__device__ __forceinline__ void global_store_main(Code g,Count v){int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK]=v;}'''
new='''struct BatchShardAddress8{int owner;Code local;};
__device__ __forceinline__ BatchShardAddress8 batch_shard_address8(Code g,Code chunk){
    int o=0;Code c4=chunk<<2;if(g>=c4){g-=c4;o|=4;}
    Code c2=chunk<<1;if(g>=c2){g-=c2;o|=2;}
    if(g>=chunk){g-=chunk;o|=1;}
    return{o,g};
}
__device__ __forceinline__ Count global_load_main(Code g){auto a=batch_shard_address8(g,D_MAIN_CHUNK);return D_MAIN_PTR[a.owner][a.local];}
__device__ __forceinline__ Count global_load_block(Code g){auto a=batch_shard_address8(g,D_BLOCK_CHUNK);return D_BLOCK_PTR[a.owner][a.local];}
__device__ __forceinline__ void global_store_main(Code g,Count v){auto a=batch_shard_address8(g,D_MAIN_CHUNK);D_MAIN_PTR[a.owner][a.local]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){auto a=batch_shard_address8(g,D_BLOCK_CHUNK);D_BLOCK_PTR[a.owner][a.local]=v;}'''
if s.count(old)!=1:
    raise SystemExit(f'batch shard helper expected one generic block, got {s.count(old)}')
s=s.replace(old,new,1)
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: batch_shard_address8=1 compare_stages=3 div64=0 mod64=0')
