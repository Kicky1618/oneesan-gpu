#include <cuda_runtime.h>
#include "../../common/mmap_resume.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <numeric>
#include <vector>

using Count = unsigned long long;
using MateID = unsigned long long;
using Code = unsigned long long;
static constexpr int MAXW = 28;

enum MateValue : uint8_t { N=0, R=1, L=2, X=3 };
enum MateValuePair : uint8_t {
    NN=0x0, NR=0x1, NL=0x2, NX=0x3,
    RN=0x4, RR=0x5, RL=0x6, RX=0x7,
    LN=0x8, LR=0x9, LL=0xa, LX=0xb,
    XN=0xc, XR=0xd, XL=0xe, XX=0xf
};

static Code H_DP[MAXW+1][MAXW+2];
__constant__ Code D_FULL_DP[MAXW+1][MAXW+2];
__constant__ Code D_MAIN_DP[MAXW+1][MAXW+2];
__constant__ Code D_BLOCK_DP[MAXW+1][MAXW+2];
__constant__ uint32_t D_MAIN_FIXED, D_MAIN_OCC, D_BLOCK_FIXED, D_BLOCK_OCC;
__constant__ int D_MAIN_W, D_BLOCK_W;
__constant__ Count D_MOD;

__host__ __device__ static inline MateValue mget(MateID m,int k){return MateValue((m>>(2*k))&3ULL);}
__host__ __device__ static inline MateValuePair mpair(MateID m,int p){return MateValuePair((m>>(2*(p-1)))&15ULL);}
__host__ __device__ static inline MateID mset(MateID m,int k,MateValue v){MateID z=3ULL<<(2*k);return (m&~z)|(MateID(v)<<(2*k));}
__host__ __device__ static inline MateID msetpair(MateID m,int p,MateValuePair v){MateID z=15ULL<<(2*(p-1));return (m&~z)|(MateID(v)<<(2*(p-1)));}
// GGCount Mate::shrink(k), not an ordinary erase(k).
__host__ __device__ static inline MateID mshrink(MateID m,int k){
    MateID mask=(1ULL<<(2*k))-1ULL;
    return ((m&~mask)>>2)|(m&mask);
}
__host__ __device__ static inline MateID minsert(MateID m,int k,MateValue v){
    MateID lowmask=k?((1ULL<<(2*k))-1ULL):0ULL;
    MateID lo=m&lowmask, hi=m&~lowmask;
    return lo|(MateID(v)<<(2*k))|(hi<<2);
}

static void ck(cudaError_t e,const char* what){if(e!=cudaSuccess){std::cerr<<what<<": "<<cudaGetErrorString(e)<<"\n";std::exit(1);}}

static void build_full_dp(){
    for(int h=0;h<=MAXW+1;++h) H_DP[0][h]=(h==0);
    for(int w=1;w<=MAXW;++w) for(int h=0;h<=MAXW;++h){
        Code x=H_DP[w-1][h];
        if(h>0)x+=H_DP[w-1][h-1];
        if(h<MAXW+1)x+=H_DP[w-1][h+1];
        H_DP[w][h]=x;
    }
}

struct GroupSpec {
    int width=0;
    uint32_t fixed=0, occ=0;
    Code dp[MAXW+1][MAXW+2]{};
    Code size=0;
};

static GroupSpec make_spec(int width,uint32_t fixed,uint32_t occ){
    GroupSpec s; s.width=width; s.fixed=fixed; s.occ=occ;
    for(int h=0;h<=MAXW+1;++h)s.dp[0][h]=(h==0);
    for(int w=1;w<=width;++w){
        int pos=w-1; bool f=(fixed>>pos)&1u, o=(occ>>pos)&1u;
        for(int h=0;h<=MAXW;++h){
            Code x=0;
            if(!f||!o) x+=s.dp[w-1][h];
            if(!f||o){ if(h>0)x+=s.dp[w-1][h-1]; if(h<MAXW+1)x+=s.dp[w-1][h+1]; }
            s.dp[w][h]=x;
        }
    }
    s.size=s.dp[width][1];
    return s;
}

struct Interval { Code global, local, len; };

static void add_interval(std::vector<Interval>& out,Code g,Code l,Code n){
    if(!n)return;
    if(!out.empty() && out.back().global+out.back().len==g && out.back().local+out.back().len==l) out.back().len+=n;
    else out.push_back({g,l,n});
}

static void intervals_rec(const GroupSpec& s,int pos,int h,Code gbase,Code lbase,std::vector<Interval>& out){
    if(pos<0){ if(h==0)add_interval(out,gbase,lbase,1); return; }
    uint32_t lowerMask = (pos==31)?0xffffffffu:((1u<<(pos+1))-1u);
    if((s.fixed & lowerMask)==0){
        add_interval(out,gbase,lbase,H_DP[pos+1][h]);
        return;
    }
    bool f=(s.fixed>>pos)&1u, o=(s.occ>>pos)&1u;
    // N branch
    Code gsz=H_DP[pos][h];
    if(!f||!o){ Code lsz=s.dp[pos][h]; intervals_rec(s,pos-1,h,gbase,lbase,out); lbase+=lsz; }
    gbase+=gsz;
    // R branch
    if(h>0){
        gsz=H_DP[pos][h-1];
        if(!f||o){ Code lsz=s.dp[pos][h-1]; intervals_rec(s,pos-1,h-1,gbase,lbase,out); lbase+=lsz; }
        gbase+=gsz;
    }
    // L branch
    if(h<MAXW+1){
        gsz=H_DP[pos][h+1];
        if(!f||o){ intervals_rec(s,pos-1,h+1,gbase,lbase,out); }
    }
}

static std::vector<Interval> make_intervals(const GroupSpec& s){
    std::vector<Interval> v; v.reserve(1024);
    intervals_rec(s,s.width-1,1,0,0,v);
    Code sum=0;for(auto const&i:v)sum+=i.len;
    if(sum!=s.size){std::cerr<<"interval size mismatch "<<sum<<" != "<<s.size<<"\n";std::exit(2);}
    return v;
}

static std::vector<int> window_candidates(int W,int p_hi,int p_lo){
    // Across p_hi..p_lo, only positions [p_lo-1, p_hi] can change occupancy.
    // Any position outside that interval is an invariant partition bit.
    std::vector<int> v;
    for(int q=W-1;q>=0;--q) if(q < p_lo-1 || q > p_hi) v.push_back(q);
    return v;
}

static void window_masks(int W,int p_hi,int p_lo,const std::vector<int>& fixed_pos,
                         uint32_t group,uint32_t& mf,uint32_t& mo,
                         uint32_t& bf,uint32_t& bo){
    mf=mo=bf=bo=0;
    for(size_t i=0;i<fixed_pos.size();++i){
        int q=fixed_pos[i];
        bool one=(group>>i)&1u;
        mf|=1u<<q; if(one)mo|=1u<<q;
        // q lies outside [p_lo-1,p_hi], so its compressed blocked position
        // is the same for every p in this window.
        int bq = (q < p_lo-1) ? q : q-1;
        bf|=1u<<bq; if(one)bo|=1u<<bq;
    }
}

struct WindowPlan {
    int p_hi=0,p_lo=0;
    std::vector<int> fixed_pos;
    size_t max_bytes=0;
    Code max_main=0,max_block=0;
};

static WindowPlan plan_window(int W,int p_hi,int p_lo,size_t target_bytes,int max_group_bits=16){
    WindowPlan best; best.p_hi=p_hi; best.p_lo=p_lo;
    auto cand=window_candidates(W,p_hi,p_lo);
    int klim=std::min<int>(cand.size(),max_group_bits);
    for(int k=0;k<=klim;++k){
        std::vector<int> fp(cand.begin(),cand.begin()+k);
        uint64_t ng=1ull<<k;
        size_t mx=0; Code mm=0,md=0;
        bool too_many = ng > (1ull<<max_group_bits);
        if(too_many) break;
        for(uint64_t g=0;g<ng;++g){
            uint32_t mf,mo,bf,bo; window_masks(W,p_hi,p_lo,fp,(uint32_t)g,mf,mo,bf,bo);
            GroupSpec ms=make_spec(W,mf,mo), ds=make_spec(W-1,bf,bo);
            size_t b=size_t(2*ms.size+ds.size)*sizeof(Count);
            if(b>mx){mx=b;mm=ms.size;md=ds.size;}
            if(mx>target_bytes && k<klim) break;
        }
        if(mx<=target_bytes || k==klim){
            best.fixed_pos=std::move(fp);best.max_bytes=mx;best.max_main=mm;best.max_block=md;
            return best;
        }
    }
    return best;
}

__device__ __forceinline__ bool allowed(uint32_t fixed,uint32_t occ,int pos,MateValue v){
    if(!((fixed>>pos)&1u))return v!=X;
    bool o=(occ>>pos)&1u;
    return o?(v==R||v==L):(v==N);
}

__device__ __forceinline__ MateID unrank_group(Code rank,int width,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    MateID m=0; int h=1;
    #pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){
        if(pos>=width)continue;
        if(allowed(fixed,occ,pos,N)){
            Code z=dp[pos][h]; if(rank<z)continue; rank-=z;
        }
        if(h>0 && allowed(fixed,occ,pos,R)){
            Code z=dp[pos][h-1]; if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;} rank-=z;
        }
        // remaining valid branch is L
        m|=MateID(L)<<(2*pos); ++h;
    }
    return m;
}

__device__ __forceinline__ Code rank_group(MateID m,int width,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    Code rank=0;int h=1;
    #pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){
        if(pos>=width)continue;
        MateValue s=mget(m,pos);
        if(s>N && allowed(fixed,occ,pos,N)) rank+=dp[pos][h];
        if(s>R && h>0 && allowed(fixed,occ,pos,R)) rank+=dp[pos][h-1];
        if(s==R)--h; else if(s==L)++h;
    }
    return rank;
}

__device__ __forceinline__ void atomic_add_mod(Count* p,Count v){
    if(!v)return; Count mod=D_MOD; Count old=atomicAdd(p,0ULL);
    for(;;){Count neu=(old>=mod-v)?old-(mod-v):old+v;Count seen=atomicCAS(p,old,neu);if(seen==old)return;old=seen;}
}

__global__ void blocked_group_kernel(const Count* in,Code n,Count* out_main,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x, stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID sm=unrank_group(i,D_BLOCK_W,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);MateID t=minsert(sm,p,N);Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);}
}

__global__ void main_group_kernel(const Count* in,Code n,Count* out_main,Count* out_block,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x, stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        Count c=in[i];if(!c)continue;MateID m=unrank_group(i,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);MateValuePair w=mpair(m,p);
        switch(w){
        case NN:{MateID t=msetpair(m,p,LR);Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);break;}
        case NR:case NL:{if(p==1){MateID t=msetpair(m,p,w==NR?RN:LN);Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);}else{MateID t=mshrink(m,p);Code j=rank_group(t,D_BLOCK_W,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);atomic_add_mod(out_block+j,c);}break;}
        case RN:{MateID t=msetpair(m,p,NR);Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);break;}
        case LN:{MateID t=msetpair(m,p,NL);Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);break;}
        case LL:{MateID t=msetpair(m,p,NN);int q=p-1,s=1;while(s){--q;if(q<0)break;auto v=mget(t,q);if(v==L)++s;else if(v==R)--s;}if(s)break;t=mset(t,q,L);if(p==1){Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);}else{t=mshrink(t,p-1);Code j=rank_group(t,D_BLOCK_W,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);atomic_add_mod(out_block+j,c);}break;}
        case RR:{MateID t=msetpair(m,p,NN);int q=p,s=1;while(s){++q;if(q>=D_MAIN_W)break;auto v=mget(t,q);if(v==L)--s;else if(v==R)++s;}if(s)break;t=mset(t,q,R);if(p==1){Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);}else{t=mshrink(t,p-1);Code j=rank_group(t,D_BLOCK_W,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);atomic_add_mod(out_block+j,c);}break;}
        case RL:{MateID t=msetpair(m,p,NN);if(p==1){Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);}else{t=mshrink(t,p-1);Code j=rank_group(t,D_BLOCK_W,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);atomic_add_mod(out_block+j,c);}break;}
        default:break;
        }
    }
}

static Code rank_full(MateID m,int width){Code r=0;int h=1;for(int pos=width-1;pos>=0;--pos){auto s=mget(m,pos);if(s>N)r+=H_DP[pos][h];if(s>R&&h>0)r+=H_DP[pos][h-1];if(s==R)--h;else if(s==L)++h;}return r;}



struct MappedCounts {
    int fd=-1; Count* p=nullptr; size_t n=0, bytes=0; std::string path;
    void open_file(const std::string& fn,size_t count,bool fresh){
        path=fn;n=count;bytes=n*sizeof(Count);
        const int flags=fresh?(O_RDWR|O_CREAT|O_TRUNC):O_RDWR;
        fd=::open(path.c_str(),flags,0644);
        if(fd<0){
            int e=errno;
            throw std::runtime_error(std::string("open mmap file: ")+std::strerror(e));
        }
        auto fail_opened=[&](const char* what,int e,bool remove_file){
            ::close(fd); fd=-1;
            if(remove_file)::unlink(path.c_str());
            throw std::runtime_error(std::string(what)+": "+std::strerror(e));
        };
        if(fresh){
            if(ftruncate(fd,(off_t)bytes)!=0)fail_opened("ftruncate",errno,true);
            int falloc_rc=posix_fallocate(fd,0,(off_t)bytes);
            if(falloc_rc!=0)fail_opened("posix_fallocate",falloc_rc,true);
        }else{
            struct stat st{};
            if(fstat(fd,&st)!=0)fail_opened("fstat",errno,false);
            if(static_cast<uint64_t>(st.st_size)!=static_cast<uint64_t>(bytes)){
                ::close(fd);fd=-1;
                throw std::runtime_error("external-store file size mismatch for resume: "+path);
            }
        }
        void* q=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        if(q==MAP_FAILED)fail_opened("mmap",errno,fresh);
        p=(Count*)q;
        madvise(p,bytes,MADV_RANDOM);
    }
    void close_file(){
        if(p){msync(p,bytes,MS_ASYNC);munmap(p,bytes);p=nullptr;}
        if(fd>=0){close(fd);fd=-1;}
    }
    ~MappedCounts(){close_file();}
    Count& operator[](size_t i){return p[i];}
};

static void gather_raw(const Count* global,const std::vector<Interval>& iv,std::vector<Count>& local){
    for(auto const& x:iv)std::memcpy(local.data()+x.local,global+x.global,size_t(x.len)*sizeof(Count));
}
static void scatter_raw(Count* global,const std::vector<Interval>& iv,const std::vector<Count>& local){
    for(auto const& x:iv)std::memcpy(global+x.global,local.data()+x.local,size_t(x.len)*sizeof(Count));
}

static unsigned long long parse_u64_arg(const char* text,const char* name){
    if(!text||!*text||text[0]=='-')throw std::invalid_argument(std::string("invalid ")+name+": "+(text?text:""));
    char* end=nullptr;errno=0;
    unsigned long long v=std::strtoull(text,&end,10);
    if(errno==ERANGE||end==text||*end!='\0')throw std::invalid_argument(std::string("invalid ")+name+": "+text);
    return v;
}

static int parse_int_arg(const char* text,const char* name){
    const auto v=parse_u64_arg(text,name);
    if(v>static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string(name)+" is too large: "+text);
    return static_cast<int>(v);
}

static bool env_flag(const char* name,bool default_value){
    const char* v=std::getenv(name);
    if(!v)return default_value;
    if(std::strcmp(v,"1")==0||std::strcmp(v,"true")==0||std::strcmp(v,"yes")==0)return true;
    if(std::strcmp(v,"0")==0||std::strcmp(v,"false")==0||std::strcmp(v,"no")==0)return false;
    throw std::runtime_error(std::string(name)+" must be one of 0/1/false/true/no/yes");
}

static void maybe_fault_inject(uint32_t group,const char* phase){
    const char* gp=std::getenv("GRIDFP_FAULT_GROUP");
    const char* pp=std::getenv("GRIDFP_FAULT_PHASE");
    if(!gp||!pp)return;
    char* end=nullptr;errno=0;
    unsigned long long want=std::strtoull(gp,&end,10);
    if(errno==ERANGE||end==gp||*end!='\0'||want>0xffffffffULL)return;
    if(static_cast<uint32_t>(want)!=group||std::strcmp(pp,phase)!=0)return;
    std::cerr<<"FAULT_INJECT group="<<group<<" phase="<<phase<<" exit=86\n"<<std::flush;
    ::_exit(86);
}

struct ResumeCoordinator {
    using Checkpoint=oneesan::mmap_resume::Checkpoint;
    bool enabled=false;
    int n=0;
    std::filesystem::path store_dir, checkpoint_path, undo_dir;
    Checkpoint cp;
    bool has_checkpoint=false;
    std::mutex mu;

    ResumeCoordinator()=default;
    ResumeCoordinator(bool use_resume,int n_,const std::filesystem::path& store)
        :enabled(use_resume),n(n_),store_dir(store),checkpoint_path(store/"checkpoint.state"),undo_dir(store/"undo"){}

    bool checkpoint_exists()const{return enabled&&std::filesystem::exists(checkpoint_path);}

    void load(){
        if(!checkpoint_exists())return;
        cp=oneesan::mmap_resume::load_checkpoint(checkpoint_path);
        has_checkpoint=true;
    }

    void validate_identity(Count mod,int target_mib,int max_window,uint64_t fingerprint,Code mainN,Code blockN)const{
        if(!has_checkpoint)return;
        oneesan::mmap_resume::validate_checkpoint_identity(
            cp,n,static_cast<uint64_t>(mod),target_mib,max_window,fingerprint,
            static_cast<uint64_t>(mainN),static_cast<uint64_t>(blockN));
    }

    void validate_position(int W)const{
        if(!has_checkpoint)return;
        if(cp.row<0||cp.row>=W)throw std::runtime_error("checkpoint row outside grid");
        if(cp.p_hi<1||cp.p_hi>=W||cp.p_lo<1||cp.p_lo>cp.p_hi)
            throw std::runtime_error("checkpoint window outside valid p range");
        if(cp.groups==0)throw std::runtime_error("checkpoint has zero groups");
    }

    void begin_window(int row,const WindowPlan& wp,uint32_t groups,Count mod,int target_mib,int max_window,
                      uint64_t fingerprint,Code mainN,Code blockN){
        if(!enabled)return;
        std::lock_guard<std::mutex> lock(mu);
        if(has_checkpoint&&cp.row==row&&cp.p_hi==wp.p_hi&&cp.p_lo==wp.p_lo&&cp.groups==groups)return;
        if(has_checkpoint&&!cp.complete&&cp.groups!=0&&!cp.all_done()){
            throw std::runtime_error("attempted to advance past an incomplete mmap checkpoint window");
        }
        cp={};
        cp.n=n;
        cp.modulus=static_cast<uint64_t>(mod);
        cp.target_mib=target_mib;
        cp.max_window=max_window;
        cp.executable_fingerprint=fingerprint;
        cp.main_count=static_cast<uint64_t>(mainN);
        cp.block_count=static_cast<uint64_t>(blockN);
        cp.row=row;
        cp.p_hi=wp.p_hi;
        cp.p_lo=wp.p_lo;
        cp.groups=groups;
        cp.done.assign((groups+7)/8,0);
        cp.complete=false;
        oneesan::mmap_resume::save_checkpoint_atomic(checkpoint_path,cp);
        has_checkpoint=true;
    }

    bool done(uint32_t g){
        if(!enabled)return false;
        std::lock_guard<std::mutex> lock(mu);
        return cp.is_done(g);
    }

    std::filesystem::path journal_path(uint32_t g)const{
        return oneesan::mmap_resume::journal_name(undo_dir,cp.row,cp.p_hi,cp.p_lo,g);
    }

    void prepare_journal(uint32_t g,Code main_count,Code block_count,
                         const std::vector<Count>& main_data,const std::vector<Count>& block_data){
        if(!enabled)return;
        const auto h=oneesan::mmap_resume::make_journal_header(
            n,cp.row,cp.p_hi,cp.p_lo,g,static_cast<uint64_t>(main_count),static_cast<uint64_t>(block_count));
        oneesan::mmap_resume::write_journal_atomic(journal_path(g),h,
            main_data.data(),static_cast<size_t>(main_count),
            block_data.data(),static_cast<size_t>(block_count));
    }

    void mark_done(uint32_t g){
        if(!enabled)return;
        std::lock_guard<std::mutex> lock(mu);
        if(cp.is_done(g))return;
        cp.mark_done(g);
        oneesan::mmap_resume::save_checkpoint_atomic(checkpoint_path,cp);
    }

    void commit_group(uint32_t g,Count* main_base,const std::vector<Interval>& mi,
                      Count* block_base,const std::vector<Interval>& di){
        if(!enabled)return;
        oneesan::mmap_resume::sync_intervals(main_base,mi);
        oneesan::mmap_resume::sync_intervals(block_base,di);
        maybe_fault_inject(g,"scatter");
        mark_done(g);
        maybe_fault_inject(g,"commit");
        oneesan::mmap_resume::durable_unlink(journal_path(g));
    }

    void commit_empty(uint32_t g){
        if(!enabled)return;
        mark_done(g);
    }

    void recover_group(uint32_t g,int W,const WindowPlan& wp,Count* main_base,Count* block_base){
        if(!enabled||done(g)){
            if(enabled){
                const auto path=journal_path(g);
                if(std::filesystem::exists(path))oneesan::mmap_resume::durable_unlink(path);
            }
            return;
        }
        const auto path=journal_path(g);
        if(!std::filesystem::exists(path))return;
        uint32_t mf,mo,bf,bo;
        window_masks(W,wp.p_hi,wp.p_lo,wp.fixed_pos,g,mf,mo,bf,bo);
        GroupSpec ms=make_spec(W,mf,mo),ds=make_spec(W-1,bf,bo);
        auto mi=make_intervals(ms),di=make_intervals(ds);
        const auto expected=oneesan::mmap_resume::make_journal_header(
            n,cp.row,cp.p_hi,cp.p_lo,g,static_cast<uint64_t>(ms.size),static_cast<uint64_t>(ds.size));
        std::cerr<<"recovering incomplete group g="<<g<<" from "<<path<<"\n";
        oneesan::mmap_resume::restore_journal(path,expected,main_base,mi,block_base,di);
        oneesan::mmap_resume::durable_unlink(path);
    }

    void cleanup_temps(){
        if(!enabled||!std::filesystem::exists(undo_dir))return;
        for(const auto& e:std::filesystem::directory_iterator(undo_dir)){
            if(!e.is_regular_file())continue;
            const auto name=e.path().filename().string();
            if(name.size()>=4&&name.substr(name.size()-4)==".tmp"){
                ::unlink(e.path().c_str());
            }
        }
        oneesan::mmap_resume::fsync_directory(undo_dir);
    }

    void finish(){
        if(!enabled)return;
        std::lock_guard<std::mutex> lock(mu);
        if(cp.groups!=0&&!cp.all_done())throw std::runtime_error("cannot mark mmap run complete with unfinished groups");
        cp.complete=true;
        oneesan::mmap_resume::save_checkpoint_atomic(checkpoint_path,cp);
    }
};

static int partition_selftest(){
    build_full_dp();
    uint64_t cases=0,groups_checked=0;
    for(int W=3;W<=10;++W){
        for(int p_hi=W-1;p_hi>=1;--p_hi){
            for(int p_lo=1;p_lo<=p_hi;++p_lo){
                const auto cand=window_candidates(W,p_hi,p_lo);
                for(size_t k=0;k<=cand.size();++k){
                    std::vector<int> fp(cand.begin(),cand.begin()+static_cast<std::ptrdiff_t>(k));
                    if(k>=31)throw std::runtime_error("partition selftest group-bit overflow");
                    const uint32_t ng=1u<<k;
                    std::vector<int> main_owner(static_cast<size_t>(H_DP[W][1]),-1);
                    std::vector<int> block_owner(static_cast<size_t>(H_DP[W-1][1]),-1);
                    for(uint32_t g=0;g<ng;++g){
                        uint32_t mf,mo,bf,bo;
                        window_masks(W,p_hi,p_lo,fp,g,mf,mo,bf,bo);
                        const auto ms=make_spec(W,mf,mo),ds=make_spec(W-1,bf,bo);
                        const auto mi=make_intervals(ms),di=make_intervals(ds);
                        auto mark=[&](std::vector<int>& owner,const std::vector<Interval>& iv,const char* which){
                            for(const auto& x:iv){
                                for(Code j=0;j<x.len;++j){
                                    const size_t at=static_cast<size_t>(x.global+j);
                                    if(at>=owner.size())throw std::runtime_error(std::string("partition interval out of range: ")+which);
                                    if(owner[at]!=-1)throw std::runtime_error(std::string("partition overlap: ")+which);
                                    owner[at]=static_cast<int>(g);
                                }
                            }
                        };
                        mark(main_owner,mi,"main");
                        mark(block_owner,di,"blocked");
                        ++groups_checked;
                    }
                    if(std::find(main_owner.begin(),main_owner.end(),-1)!=main_owner.end())
                        throw std::runtime_error("partition gap: main");
                    if(std::find(block_owner.begin(),block_owner.end(),-1)!=block_owner.end())
                        throw std::runtime_error("partition gap: blocked");
                    ++cases;
                }
            }
        }
    }
    std::cout<<"gridfp partition selftest: PASS cases="<<cases<<" groups="<<groups_checked<<"\n";
    return 0;
}

struct DeviceCtx {
    int dev = -1;
    Count *dA=nullptr, *dB=nullptr, *dD=nullptr;
    Code capM=0, capD=0;
    std::vector<Count> hM, hD;
    Code maxGM=0, maxGD=0;
    size_t maxIntervals=0;
    double transferred=0.0;
    double active_s=0.0;
    uint64_t groups_done=0;
    unsigned long long state_slots_done=0;

    void init(int device, Count mod) {
        dev=device;
        ck(cudaSetDevice(dev), "cudaSetDevice init");
        ck(cudaMemcpyToSymbol(D_FULL_DP,H_DP,sizeof(H_DP)),"full dp");
        ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"mod");
    }
    void ensure(Code main_cap, Code block_cap) {
        ck(cudaSetDevice(dev), "cudaSetDevice ensure");
        if(main_cap>capM){
            if(dA){cudaFree(dA);cudaFree(dB);}
            capM=main_cap;
            ck(cudaMalloc(&dA,size_t(capM)*sizeof(Count)),"dA");
            ck(cudaMalloc(&dB,size_t(capM)*sizeof(Count)),"dB");
            hM.resize(capM);
        }
        if(block_cap>capD){
            if(dD)cudaFree(dD);
            capD=block_cap;
            ck(cudaMalloc(&dD,size_t(capD)*sizeof(Count)),"dD");
            hD.resize(capD);
        }
    }
    void destroy(){
        if(dev<0)return;
        cudaSetDevice(dev);
        if(dA){cudaFree(dA);cudaFree(dB);}
        if(dD)cudaFree(dD);
        dA=dB=dD=nullptr;
    }
};

static void process_group(DeviceCtx& c,
                          int W,const WindowPlan& wp,int g,
                          Count* mainv,Count* blockv,
                          int threads,ResumeCoordinator& resume){
    auto t0=std::chrono::steady_clock::now();
    ck(cudaSetDevice(c.dev),"cudaSetDevice worker");
    uint32_t mf,mo,bf,bo;
    window_masks(W,wp.p_hi,wp.p_lo,wp.fixed_pos,(uint32_t)g,mf,mo,bf,bo);
    GroupSpec ms=make_spec(W,mf,mo), ds=make_spec(W-1,bf,bo);
    if(!ms.size && !ds.size){resume.commit_empty(static_cast<uint32_t>(g));return;}
    auto mi=make_intervals(ms), di=make_intervals(ds);
    c.maxGM=std::max(c.maxGM,ms.size);
    c.maxGD=std::max(c.maxGD,ds.size);
    c.maxIntervals=std::max({c.maxIntervals,mi.size(),di.size()});
    c.ensure(ms.size,ds.size);

    // Window groups are transition-closed and disjoint, so simultaneous
    // gather/scatter by different GPU workers touches disjoint external ranges.
    gather_raw(mainv,mi,c.hM);
    gather_raw(blockv,di,c.hD);
    resume.prepare_journal(static_cast<uint32_t>(g),ms.size,ds.size,c.hM,c.hD);
    maybe_fault_inject(static_cast<uint32_t>(g),"journal");
    c.transferred += double(ms.size+ds.size)*sizeof(Count);
    if(ms.size)ck(cudaMemcpy(c.dA,c.hM.data(),size_t(ms.size)*sizeof(Count),cudaMemcpyHostToDevice),"H2D main");
    if(ds.size)ck(cudaMemcpy(c.dD,c.hD.data(),size_t(ds.size)*sizeof(Count),cudaMemcpyHostToDevice),"H2D block");

    ck(cudaMemcpyToSymbol(D_MAIN_DP,ms.dp,sizeof(ms.dp)),"main gdp");
    ck(cudaMemcpyToSymbol(D_BLOCK_DP,ds.dp,sizeof(ds.dp)),"block gdp");
    ck(cudaMemcpyToSymbol(D_MAIN_FIXED,&mf,sizeof(mf)),"mf");
    ck(cudaMemcpyToSymbol(D_MAIN_OCC,&mo,sizeof(mo)),"mo");
    ck(cudaMemcpyToSymbol(D_BLOCK_FIXED,&bf,sizeof(bf)),"bf");
    ck(cudaMemcpyToSymbol(D_BLOCK_OCC,&bo,sizeof(bo)),"bo");
    int mw=W,bw=W-1;
    ck(cudaMemcpyToSymbol(D_MAIN_W,&mw,sizeof(mw)),"mw");
    ck(cudaMemcpyToSymbol(D_BLOCK_W,&bw,sizeof(bw)),"bw");

    Count* cur=c.dA; Count* nxt=c.dB;
    int bm=int(std::min<Code>(65535,(ms.size+threads-1)/threads));
    int bd=int(std::min<Code>(65535,(ds.size+threads-1)/threads));
    for(int p=wp.p_hi;p>=wp.p_lo;--p){
        if(ms.size)ck(cudaMemcpy(nxt,cur,size_t(ms.size)*sizeof(Count),cudaMemcpyDeviceToDevice),"identity");
        if(ds.size)blocked_group_kernel<<<bd,threads>>>(c.dD,ds.size,nxt,p);
        if(ds.size)ck(cudaMemset(c.dD,0,size_t(ds.size)*sizeof(Count)),"clear new block");
        if(ms.size)main_group_kernel<<<bm,threads>>>(cur,ms.size,nxt,c.dD,p);
        ck(cudaGetLastError(),"group kernels");
        ck(cudaDeviceSynchronize(),"group sync");
        std::swap(cur,nxt);
    }
    if(ms.size)ck(cudaMemcpy(c.hM.data(),cur,size_t(ms.size)*sizeof(Count),cudaMemcpyDeviceToHost),"D2H main");
    if(ds.size)ck(cudaMemcpy(c.hD.data(),c.dD,size_t(ds.size)*sizeof(Count),cudaMemcpyDeviceToHost),"D2H block");
    c.transferred += double(ms.size+ds.size)*sizeof(Count);
    scatter_raw(mainv,mi,c.hM);
    scatter_raw(blockv,di,c.hD);
    resume.commit_group(static_cast<uint32_t>(g),mainv,mi,blockv,di);
    c.groups_done += 1;
    c.state_slots_done += static_cast<unsigned long long>(ms.size + ds.size);
    c.active_s += std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}

int main(int argc,char**argv){
    if(const char* selftest=std::getenv("GRIDFP_PARTITION_SELFTEST")){
        if(std::strcmp(selftest,"1")==0){
            try{return partition_selftest();}
            catch(const std::exception&e){std::cerr<<"partition selftest failed: "<<e.what()<<"\n";return 7;}
        }
    }
    int n=10,target_mib=256,max_window=0,requested_gpus=0;
    Count mod=2305843009213693951ULL;
    try{
        if(argc>1)n=parse_int_arg(argv[1],"n");
        if(argc>2)mod=static_cast<Count>(parse_u64_arg(argv[2],"modulus"));
        if(argc>3)target_mib=parse_int_arg(argv[3],"target_mib");
        max_window=argc>4?parse_int_arg(argv[4],"max_window"):n;
        if(argc>5)requested_gpus=parse_int_arg(argv[5],"gpu_count");
    }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
    if(mod<2){std::cerr<<"modulus must be >= 2\n";return 1;}
    if(target_mib<1||max_window<1){std::cerr<<"target_mib and max_window must be positive\n";return 1;}
    int W=n+1;
    if(n<2||W>MAXW){std::cerr<<"multi GPU solver supports n=2..27\n";return 1;}
    build_full_dp();

    int visible=0; ck(cudaGetDeviceCount(&visible),"cudaGetDeviceCount");
    if(visible<1){std::cerr<<"no CUDA GPU\n";return 2;}
    int ngpu=requested_gpus<=0?visible:std::min(requested_gpus,visible);
    std::cerr<<"using "<<ngpu<<" / "<<visible<<" CUDA GPUs; target="<<target_mib<<" MiB/GPU\n";

    // Enable peer access wherever the platform exposes P2P/NVLink/NVSwitch.
    // The current transition-closed group scheduler does not require peer traffic,
    // but this prepares the same binary for HBM-to-HBM redistribution.
    int peer_links=0;
    for(int a=0;a<ngpu;++a){
        for(int b=0;b<ngpu;++b){
            if(a==b)continue;
            int can=0; ck(cudaDeviceCanAccessPeer(&can,a,b),"cudaDeviceCanAccessPeer");
            if(can){
                ck(cudaSetDevice(a),"cudaSetDevice peer");
                cudaError_t e=cudaDeviceEnablePeerAccess(b,0);
                if(e!=cudaSuccess && e!=cudaErrorPeerAccessAlreadyEnabled)ck(e,"cudaDeviceEnablePeerAccess");
                if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();
                ++peer_links;
            }
        }
    }
    std::cerr<<"peer-access directed links="<<peer_links<<" / "<<(ngpu*(ngpu-1))<<"\n";

    std::vector<DeviceCtx> ctx(ngpu);
    for(int d=0;d<ngpu;++d)ctx[d].init(d,mod);

    Code mainN=H_DP[W][1], blockN=H_DP[W-1][1];
    std::string store_dir=argc>6?argv[6]:".gridfp_multigpu_store";
    std::filesystem::create_directories(store_dir);
    const bool resume_enabled=env_flag("GRIDFP_RESUME",true);
    const bool force_fresh=env_flag("GRIDFP_FRESH",false);
    oneesan::mmap_resume::DirectoryLock store_lock;
    try{store_lock.acquire(store_dir);}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}
    ResumeCoordinator resume(resume_enabled,n,store_dir);
    if(force_fresh){
        std::error_code ec;
        std::filesystem::remove(resume.checkpoint_path,ec);
        std::filesystem::remove_all(resume.undo_dir,ec);
        oneesan::mmap_resume::fsync_directory(store_dir);
    }else if(!resume_enabled&&std::filesystem::exists(resume.checkpoint_path)){
        std::cerr<<"checkpoint exists but GRIDFP_RESUME=0; set GRIDFP_FRESH=1 to discard it safely\n";
        return 2;
    }
    const uint64_t executable_fingerprint=resume_enabled?oneesan::mmap_resume::self_fingerprint():0;
    try{resume.load();resume.validate_identity(mod,target_mib,max_window,executable_fingerprint,mainN,blockN);resume.validate_position(W);resume.cleanup_temps();}
    catch(const std::exception&e){std::cerr<<"resume checkpoint rejected: "<<e.what()<<"\n";return 2;}
    const bool resuming=resume.has_checkpoint;
    MappedCounts mainv,blockv;
    const std::string main_path=store_dir+"/main.bin", block_path=store_dir+"/blocked.bin";
    try{
        mainv.open_file(main_path,mainN,!resuming);
        blockv.open_file(block_path,blockN,!resuming);
    }catch(const std::exception& e){
        mainv.close_file(); blockv.close_file();
        if(!resuming){::unlink(main_path.c_str());::unlink(block_path.c_str());}
        std::cerr<<"external-store allocation failed: "<<e.what()<<"\n";
        return 2;
    }
    MateID init=MateID(R)<<(2*(W-1));
    if(!resuming){
        const Code init_rank=rank_full(init,W);
        mainv[init_rank]=1;
        std::vector<Interval> init_iv{{init_rank,0,1}};
        oneesan::mmap_resume::sync_intervals(mainv.p,init_iv);
    }else{
        std::cerr<<"resuming external store at row="<<resume.cp.row+1<<" p_hi="<<resume.cp.p_hi
                 <<" complete="<<resume.cp.complete<<"\n";
    }
    int threads=256;
    size_t target_bytes=size_t(std::max(1,target_mib))<<20;
    int total_windows=0,max_groups=0,max_window_len=0;
    uint64_t max_journal_reserve_bytes=0;
    auto wall0=std::chrono::steady_clock::now();

    if(resuming&&resume.cp.complete){
        const Count ans=mainv[rank_full(MateID(R),W)];
        std::cout<<"backend=gridfp-multigpu-mmap n="<<n<<" residue="<<ans<<" modulus="<<mod
                 <<" gpus="<<ngpu<<" resumed_complete=1 store_dir="<<store_dir<<"\n";
        for(auto& c:ctx)c.destroy();
        return 0;
    }

    const int start_row=resuming?resume.cp.row:0;
    for(int row=start_row;row<W;++row){
        int p_hi=(resuming&&row==start_row)?resume.cp.p_hi:W-1;
        while(p_hi>=1){
            WindowPlan wp; bool found=false;
            for(int p_lo=std::max(1,p_hi-max_window+1);p_lo<=p_hi;++p_lo){
                auto t=plan_window(W,p_hi,p_lo,target_bytes);
                if(t.max_bytes&&t.max_bytes<=target_bytes){wp=std::move(t);found=true;break;}
            }
            if(!found){
                wp=plan_window(W,p_hi,p_hi,target_bytes,24);
                if(!wp.max_bytes||wp.max_bytes>target_bytes){
                    std::cerr<<"cannot fit one update p="<<p_hi<<" target_mib="<<target_mib<<"\n";
                    return 3;
                }
            }
            int k=wp.fixed_pos.size();
            if(k>=31){std::cerr<<"too many group bits\n";return 4;}
            int ng=1<<k;
            try{
                resume.begin_window(row,wp,static_cast<uint32_t>(ng),mod,target_mib,max_window,
                                    executable_fingerprint,mainN,blockN);
                for(int g=0;g<ng;++g)resume.recover_group(static_cast<uint32_t>(g),W,wp,mainv.p,blockv.p);
            }catch(const std::exception&e){
                std::cerr<<"resume recovery failed: "<<e.what()<<"\n";
                return 5;
            }
            ++total_windows; max_groups=std::max(max_groups,ng);
            max_window_len=std::max(max_window_len,wp.p_hi-wp.p_lo+1);
            struct JobOrder { int g; Code work; Code journal_slots; };
            std::vector<JobOrder> jobs; jobs.reserve(ng);
            for(int g=0;g<ng;++g){
                uint32_t mf,mo,bf,bo;
                window_masks(W,wp.p_hi,wp.p_lo,wp.fixed_pos,(uint32_t)g,mf,mo,bf,bo);
                GroupSpec ms=make_spec(W,mf,mo), ds=make_spec(W-1,bf,bo);
                jobs.push_back({g,2*ms.size+ds.size,ms.size+ds.size}); // VRAM work and undo payload
            }
            if(resume.enabled){
                std::vector<Code> journal_sizes;journal_sizes.reserve(jobs.size());
                for(const auto& j:jobs)if(!resume.done(static_cast<uint32_t>(j.g)))journal_sizes.push_back(j.journal_slots);
                std::sort(journal_sizes.begin(),journal_sizes.end(),std::greater<Code>());
                uint64_t need=0;
                for(size_t i=0;i<std::min<size_t>(static_cast<size_t>(ngpu),journal_sizes.size());++i)
                    need+=static_cast<uint64_t>(journal_sizes[i])*sizeof(Count);
                max_journal_reserve_bytes=std::max(max_journal_reserve_bytes,need);
                try{
                    const auto avail=std::filesystem::space(store_dir).available;
                    constexpr uint64_t kSafety=64ULL<<20;
                    if(avail<need+kSafety){
                        std::cerr<<"insufficient space for crash-safe journals before row="<<row+1
                                 <<" p="<<wp.p_hi<<".."<<wp.p_lo<<": need="<<need
                                 <<" available="<<avail<<" (plus 64 MiB safety)\n";
                        return 6;
                    }
                }catch(const std::exception&e){
                    std::cerr<<"cannot query journal filesystem capacity: "<<e.what()<<"\n";
                    return 6;
                }
            }
            std::sort(jobs.begin(),jobs.end(),[](auto const& a,auto const& b){return a.work>b.work;});
            std::vector<int> order(ng); for(int q=0;q<ng;++q)order[q]=jobs[q].g;

            std::atomic<int> next{0};
            std::atomic<bool> worker_failed{false};
            std::mutex worker_error_mu;
            std::exception_ptr worker_error;
            std::vector<std::thread> workers;
            workers.reserve(ngpu);
            for(int d=0;d<ngpu;++d){
                workers.emplace_back([&,d]{
                    try{
                        for(;;){
                            if(worker_failed.load(std::memory_order_relaxed))break;
                            int q=next.fetch_add(1,std::memory_order_relaxed);
                            if(q>=ng)break;
                            const int g=order[q];
                            if(resume.done(static_cast<uint32_t>(g)))continue;
                            process_group(ctx[d],W,wp,g,mainv.p,blockv.p,threads,resume);
                        }
                    }catch(...){
                        worker_failed.store(true,std::memory_order_relaxed);
                        std::lock_guard<std::mutex> lock(worker_error_mu);
                        if(!worker_error)worker_error=std::current_exception();
                    }
                });
            }
            for(auto& th:workers)th.join();
            if(worker_error){
                try{std::rethrow_exception(worker_error);}
                catch(const std::exception&e){std::cerr<<"group worker failed: "<<e.what()<<"\n";}
                return 5;
            }
            if(resume.enabled&&!resume.cp.all_done()){
                std::cerr<<"internal error: window finished with incomplete checkpoint bitmap\n";
                return 5;
            }
            p_hi=wp.p_lo-1;
        }
        std::cerr<<"row "<<row+1<<"/"<<W<<" windows="<<total_windows<<"\n";
    }

    try{resume.finish();}
    catch(const std::exception&e){std::cerr<<"failed to finalize resume checkpoint: "<<e.what()<<"\n";return 5;}
    double wall_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();
    Count ans=mainv[rank_full(MateID(R),W)];
    Code maxGM=0,maxGD=0;size_t maxIntervals=0;double transferred=0,active_sum=0,active_max=0;
    for(auto const& c:ctx){
        maxGM=std::max(maxGM,c.maxGM);maxGD=std::max(maxGD,c.maxGD);maxIntervals=std::max(maxIntervals,c.maxIntervals);
        transferred+=c.transferred;active_sum+=c.active_s;active_max=std::max(active_max,c.active_s);
    }
    for(auto const& c:ctx){
        std::cerr<<"gpu_worker dev="<<c.dev
                 <<" groups_done="<<c.groups_done
                 <<" state_slots_done="<<c.state_slots_done
                 <<" active_s="<<c.active_s
                 <<" transfer_gib="<<c.transferred/(1ull<<30)<<"\n";
    }
    std::cout<<"backend=gridfp-multigpu-mmap n="<<n<<" residue="<<ans<<" modulus="<<mod
             <<" gpus="<<ngpu<<" peer_links="<<peer_links<<" main_states="<<mainN<<" blocked_states="<<blockN
             <<" external_bytes="<<size_t(mainN+blockN)*sizeof(Count)
             <<" max_group_main="<<maxGM<<" max_group_blocked="<<maxGD
             <<" max_vram_per_gpu_bytes="<<size_t(2*maxGM+maxGD)*sizeof(Count)
             <<" max_intervals="<<maxIntervals<<" windows="<<total_windows
             <<" max_groups="<<max_groups<<" max_window_len="<<max_window_len
             <<" target_mib_per_gpu="<<target_mib<<" max_window_cfg="<<max_window
             <<" transfer_gib="<<transferred/(1ull<<30)
             <<" worker_active_sum_s="<<active_sum<<" worker_active_max_s="<<active_max
             <<" max_journal_reserve_bytes="<<max_journal_reserve_bytes
             <<" wall_s="<<wall_s<<" resume="<<(resume_enabled?1:0)<<" resumed="<<(resuming?1:0)
             <<" store_dir="<<store_dir<<"\n";
    for(auto& c:ctx)c.destroy();
}
