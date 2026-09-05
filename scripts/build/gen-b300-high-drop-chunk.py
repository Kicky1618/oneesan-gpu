#!/usr/bin/env python3
import pathlib,sys
if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-high-drop-chunk.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
if 'b300_low_cached_drop_rank' not in s or 'rank_lift_n_t' not in s:
    raise SystemExit('high-drop chunk requires full pull + low-drop transforms first')

def once(old,new,label):
    global s
    n=s.count(old)
    if n!=1:raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(old,new,1)

once('__constant__ uint32_t* D_HIGH_OFFSETS_BLOCK;',
     '__constant__ uint32_t* D_HIGH_OFFSETS_BLOCK;\n__constant__ unsigned long long* D_HIGH_DROP_CHUNK;',
     'high chunk symbol')

# Insert before main_pull so both main and blocked pull helpers can use it.
marker='__device__ __forceinline__ Code main_pull_direct_pair_source_rank'
if s.count(marker)!=1:raise SystemExit('main pull helper marker not unique')
helper=r'''
__global__ void b300_build_high_drop_chunk_kernel(){
    constexpr int CHUNKS=3,HC=MAXW+2,TOTAL=CHUNKS*HC*256;
    int x=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(x>=TOTAL)return;
    const uint32_t code=uint32_t(x&255);x>>=8;
    const int h0=x%HC,c=x/HC;
    const int top=27-4*c,lo=top-3;
    int h=h0;long long delta=0;bool ok=true;
#pragma unroll
    for(int pos=top;pos>=lo;--pos){
        const MateValue v=MateValue((code>>(2*(pos-lo)))&3u);
        if(v==X){ok=false;break;}
        if(v==R){
            if(h<=0){ok=false;break;}
            delta+=static_cast<long long>(D_BLOCK_DP[pos-1][h])-static_cast<long long>(D_MAIN_DP[pos][h]);--h;
        }else if(v==L){
            if(h>=MAXW+1){ok=false;break;}
            const Code b=D_BLOCK_DP[pos-1][h]+(h?D_BLOCK_DP[pos-1][h-1]:0);
            const Code a=D_MAIN_DP[pos][h]+(h?D_MAIN_DP[pos][h-1]:0);
            delta+=static_cast<long long>(b)-static_cast<long long>(a);++h;
        }
    }
    constexpr unsigned long long DMASK=(1ULL<<56)-1ULL;
    D_HIGH_DROP_CHUNK[(size_t(c)*HC+size_t(h0))*256u+code]=
        ok?((static_cast<unsigned long long>(delta)&DMASK)|(static_cast<unsigned long long>(h)<<56)):(0xffULL<<56);
}

__device__ __forceinline__ long long b300_high_chunk_drop_delta(MateID m,int p){
    constexpr int HC=MAXW+2;
    long long delta=0;int h=1;
    const int n=27-p,full=n>>2,rem=n&3;
#pragma unroll
    for(int c=0;c<3;++c){
        if(c>=full)continue;
        const int lo=24-4*c;
        const uint32_t code=uint32_t((m>>(2*lo))&0xffULL);
        const unsigned long long z=__ldg(D_HIGH_DROP_CHUNK+(size_t(c)*HC+size_t(h))*256u+code);
        const long long d=static_cast<long long>(z<<8)>>8;
        delta+=d;h=int(z>>56);
    }
    int pos=27-(full<<2);
#pragma unroll
    for(int r=0;r<3;++r,--pos){
        if(r>=rem)continue;
        const MateValue v=mget(m,pos);
        if(v==R){
            delta+=static_cast<long long>(D_BLOCK_DP[pos-1][h])-static_cast<long long>(D_MAIN_DP[pos][h]);--h;
        }else if(v==L){
            const Code b=D_BLOCK_DP[pos-1][h]+(h?D_BLOCK_DP[pos-1][h-1]:0);
            const Code a=D_MAIN_DP[pos][h]+(h?D_MAIN_DP[pos][h-1]:0);
            delta+=static_cast<long long>(b)-static_cast<long long>(a);++h;
        }
    }
    return delta;
}

__device__ __forceinline__ Code b300_high_chunk_drop_rank(Code main_rank,MateID m,int p){
    const long long d=b300_high_chunk_drop_delta(m,p);
    return d>=0?main_rank+Code(d):main_rank-Code(-d);
}
__device__ __forceinline__ Code b300_high_chunk_lift_rank(Code block_rank,MateID full,int p){
    const long long d=b300_high_chunk_drop_delta(full,p);
    return d>=0?block_rank-Code(d):block_rank+Code(-d);
}

'''
s=s.replace(marker,helper+marker,1)

# Main pull high-window fallback.
once('Code j=b300_low_window_cache_active()?b300_low_cached_drop_rank(i,m,p):rank_drop_n_t<TARGET_W>(i,m,p);',
     'Code j=b300_low_window_cache_active()?b300_low_cached_drop_rank(i,m,p):b300_high_chunk_drop_rank(i,m,p);',
     'main high chunk drop')

# Block pull high-window endpoint and closure base lifts. Low window keeps the
# generic proven lift until its block-cache compression is handled separately.
once('Code j=rank_lift_n_t<TARGET_W>(i,b,p);',
     'Code j=p>=15?b300_high_chunk_lift_rank(i,minsert(b,p,N),p):rank_lift_n_t<TARGET_W>(i,b,p);',
     'block endpoint high chunk lift')
once('Code base_rank=rank_lift_n_t<TARGET_W>(i,b,p-1);',
     'Code base_rank=p>=15?b300_high_chunk_lift_rank(i,d,p-1):rank_lift_n_t<TARGET_W>(i,b,p-1);',
     'block closure high chunk lift')

# Build the per-group table after the group DP constants are bound. Only the
# high window needs it; low-window main uses its packed universal recurrence.
once('int bm=int(std::min<Code>(65535,(ms.size+threads-1)/threads)),bd=int(std::min<Code>(65535,(ds.size+threads-1)/threads));',
     'if(wp.p_hi==W-1){constexpr int Z=3*(MAXW+2)*256;b300_build_high_drop_chunk_kernel<<< (Z+255)/256,256 >>>();ck(cudaGetLastError(),"build high drop chunk");}\n    int bm=int(std::min<Code>(65535,(ms.size+threads-1)/threads)),bd=int(std::min<Code>(65535,(ds.size+threads-1)/threads));',
     'high chunk group build')

# One tiny table allocation per GPU, reused by every group/row/residue.
alloc_marker='\n\n    Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];'
if s.count(alloc_marker)!=1:raise SystemExit('auth allocation marker not unique')
alloc=r'''

    unsigned long long* highDropChunkD[MAXGPU]{};
    constexpr size_t HIGH_DROP_CHUNK_ENTRIES=size_t(3)*(MAXW+2)*256u;
    for(int d=0;d<ng;++d){
        ck(cudaSetDevice(d),"set high drop chunk device");
        ck(cudaMalloc(&highDropChunkD[d],HIGH_DROP_CHUNK_ENTRIES*sizeof(unsigned long long)),"high drop chunk alloc");
        ck(cudaMemcpyToSymbol(D_HIGH_DROP_CHUNK,&highDropChunkD[d],sizeof(highDropChunkD[d])),"high drop chunk ptr");
    }
    std::cerr<<"high_drop_chunk entries="<<HIGH_DROP_CHUNK_ENTRIES<<" bytes="<<HIGH_DROP_CHUNK_ENTRIES*sizeof(unsigned long long)<<" max_table_loads=3 max_scalar_tail=3\n";'''
s=s.replace(alloc_marker,alloc+alloc_marker,1)

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: high_drop_chunk=1 table_bytes_per_gpu=184320 max_table_loads=3 max_scalar_tail=3 main_drop=1 block_lift=1')
