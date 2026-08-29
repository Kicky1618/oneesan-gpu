#!/usr/bin/env python3
import pathlib, sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-low-drop-chunk.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
if 'b300_low_cached_drop_rank' not in s or 'low_window_packed_main_cache' in s:
    # The generated CUDA does not contain the transform's print line; the second
    # condition only catches accidentally feeding this Python source to itself.
    pass

def once(old,new,label):
    global s
    n=s.count(old)
    if n!=1:raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(old,new,1)

# Read-only table pointer. Layout: [chunk 0..2][entry height 0..MAXW+1][raw 8-bit
# four-symbol code]. Each u32 packs signed 24-bit delta + 8-bit exit height.
once('__constant__ uint32_t* D_HIGH_OFFSETS_BLOCK;',
     '__constant__ uint32_t* D_HIGH_OFFSETS_BLOCK;\n__constant__ uint32_t* D_LOW_DROP_CHUNK;',
     'low drop chunk symbol')

main_marker='\nint main(int argc,char**argv){'
if s.count(main_marker)!=1:raise SystemExit('main marker not unique')
host=r'''

static std::vector<uint32_t> build_low_drop_chunk_table(){
    constexpr int CHUNKS=3,HC=MAXW+2;
    std::vector<uint32_t> t(size_t(CHUNKS)*HC*256u,0xff000000u);
    for(int c=0;c<CHUNKS;++c){
        const int top=14-4*c,lo=top-3;
        for(int h0=0;h0<HC;++h0)for(uint32_t code=0;code<256u;++code){
            int h=h0;long long delta=0;bool ok=true;
            for(int pos=top;pos>=lo;--pos){
                MateValue v=MateValue((code>>(2*(pos-lo)))&3u);
                if(v==X){ok=false;break;}
                if(v==R){
                    if(h<=0){ok=false;break;}
                    delta+=static_cast<long long>(H_DP[pos-1][h])-static_cast<long long>(H_DP[pos][h]);--h;
                }else if(v==L){
                    if(h>=MAXW+1){ok=false;break;}
                    Code b=H_DP[pos-1][h]+(h?H_DP[pos-1][h-1]:0);
                    Code a=H_DP[pos][h]+(h?H_DP[pos][h-1]:0);
                    delta+=static_cast<long long>(b)-static_cast<long long>(a);++h;
                }
            }
            if(!ok||h<0||h>255||delta<-(1ll<<23)||delta>=(1ll<<23))continue;
            uint32_t d=uint32_t(int32_t(delta))&0x00ffffffu;
            t[(size_t(c)*HC+size_t(h0))*256u+code]=d|(uint32_t(h)<<24);
        }
    }
    return t;
}
'''
s=s.replace(main_marker,host+main_marker,1)

old=r'''__device__ __forceinline__ Code b300_low_cached_drop_rank(Code main_rank,MateID cached,int p){
    constexpr MateID LOW_MASK=(MateID(1)<<30)-1ULL;
    const MateID low=cached&LOW_MASK;
    long long delta=-static_cast<long long>((cached>>30)&((MateID(1)<<30)-1ULL));
    int h=int((cached>>60)&15ULL);
#pragma unroll
    for(int pos=14;pos>=0;--pos){
        if(pos<=p)continue;
        const MateValue v=mget(low,pos);
        if(v==R){
            delta+=static_cast<long long>(D_FULL_DP[pos-1][h])-static_cast<long long>(D_FULL_DP[pos][h]);
            --h;
        }else if(v==L){
            const Code bm=D_FULL_DP[pos-1][h]+(h?D_FULL_DP[pos-1][h-1]:0);
            const Code am=D_FULL_DP[pos][h]+(h?D_FULL_DP[pos][h-1]:0);
            delta+=static_cast<long long>(bm)-static_cast<long long>(am);
            ++h;
        }
    }
    return delta>=0?main_rank+Code(delta):main_rank-Code(-delta);
}'''
new=r'''__device__ __forceinline__ Code b300_low_cached_drop_rank(Code main_rank,MateID cached,int p){
    constexpr MateID LOW_MASK=(MateID(1)<<30)-1ULL;
    constexpr int HC=MAXW+2;
    const MateID low=cached&LOW_MASK;
    long long delta=-static_cast<long long>((cached>>30)&((MateID(1)<<30)-1ULL));
    int h=int((cached>>60)&15ULL);
    const int n=14-p,full=n>>2,rem=n&3;
#pragma unroll
    for(int c=0;c<3;++c){
        if(c>=full)continue;
        const int lo=11-4*c;
        const uint32_t code=uint32_t((low>>(2*lo))&0xffULL);
        const uint32_t z=__ldg(D_LOW_DROP_CHUNK+(size_t(c)*HC+size_t(h))*256u+code);
        const int32_t d=int32_t(z<<8)>>8;
        delta+=static_cast<long long>(d);h=int(z>>24);
    }
    int pos=14-(full<<2);
#pragma unroll
    for(int r=0;r<3;++r,--pos){
        if(r>=rem)continue;
        const MateValue v=mget(low,pos);
        if(v==R){
            delta+=static_cast<long long>(D_FULL_DP[pos-1][h])-static_cast<long long>(D_FULL_DP[pos][h]);--h;
        }else if(v==L){
            const Code bm=D_FULL_DP[pos-1][h]+(h?D_FULL_DP[pos-1][h-1]:0);
            const Code am=D_FULL_DP[pos][h]+(h?D_FULL_DP[pos][h-1]:0);
            delta+=static_cast<long long>(bm)-static_cast<long long>(am);++h;
        }
    }
    return delta>=0?main_rank+Code(delta):main_rank-Code(-delta);
}'''
once(old,new,'chunked low drop kernel')

# Allocate/bind the tiny universal table once per GPU. It is independent of
# group, row, residue and modulus.
alloc_marker='\n\n    Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];'
if s.count(alloc_marker)!=1:raise SystemExit('auth allocation marker not unique')
alloc=r'''

    std::vector<uint32_t> lowDropChunkHost=build_low_drop_chunk_table();
    uint32_t* lowDropChunkD[MAXGPU]{};
    for(int d=0;d<ng;++d){
        ck(cudaSetDevice(d),"set low drop chunk device");
        ck(cudaMalloc(&lowDropChunkD[d],lowDropChunkHost.size()*sizeof(uint32_t)),"low drop chunk alloc");
        ck(cudaMemcpy(lowDropChunkD[d],lowDropChunkHost.data(),lowDropChunkHost.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),"low drop chunk copy");
        ck(cudaMemcpyToSymbol(D_LOW_DROP_CHUNK,&lowDropChunkD[d],sizeof(lowDropChunkD[d])),"low drop chunk ptr");
    }
    std::cerr<<"low_drop_chunk entries="<<lowDropChunkHost.size()<<" bytes="<<lowDropChunkHost.size()*sizeof(uint32_t)<<" max_full_chunks=3 max_scalar_tail=3\n";'''
s=s.replace(alloc_marker,alloc+alloc_marker,1)

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: low_drop_chunk=1 table_bytes=92160 max_table_loads=3 max_scalar_tail=3')
