#pragma once
#include "row8_tensor_init.cuh"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oneesan::row8gap {

static constexpr int DIMS[9]={1107,1640,1428,888,420,152,42,8,1};
static constexpr int DELTA[3]={0,-1,1};
struct SHdr {char magic[8];uint32_t version,r;uint32_t dims[9];uint64_t total_nz,fnv_hash;};
struct BHdr {uint32_t sym,h,h2,rows,cols,nnz;};
struct VHdr {uint32_t tag,sym,h,nnz;};
struct SparseVec {int h=-1;std::vector<uint16_t> nz;};
struct HostBlock {
    int h2=-1,rows=0,cols=0;
    std::vector<uint32_t> rp,cp;
    std::vector<uint16_t> ci,ri;
};
struct HostCache {
    SHdr hdr{};
    std::array<SparseVec,3> alpha;
    std::array<SparseVec,3> beta;
    std::array<std::array<HostBlock,9>,3> q;
};

inline uint64_t fnv64(uint64_t h,const void*vp,size_t n){auto p=(const unsigned char*)vp;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;}
inline std::string cache_path(){if(const char*e=std::getenv("GRIDFP_ROW8_GAP_CACHE"))return e;return "src/cuda/b300/row8_gap01.bin";}

template<class T> inline T parse_one(const std::vector<unsigned char>&b,size_t&off){if(off+sizeof(T)>b.size())throw std::runtime_error("row8 gap cache truncated");T x{};std::memcpy(&x,b.data()+off,sizeof(T));off+=sizeof(T);return x;}
template<class T> inline std::vector<T> parse_vec(const std::vector<unsigned char>&b,size_t&off,size_t n){if(off+n*sizeof(T)>b.size())throw std::runtime_error("row8 gap cache truncated vector");std::vector<T>x(n);if(n)std::memcpy(x.data(),b.data()+off,n*sizeof(T));off+=n*sizeof(T);return x;}

inline HostCache load_host_cache(){
    auto path=cache_path();std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("row8 gap cache open failed: "+path);
    in.seekg(0,std::ios::end);size_t sz=(size_t)in.tellg();in.seekg(0);std::vector<unsigned char>b(sz);in.read((char*)b.data(),b.size());if(!in)throw std::runtime_error("row8 gap cache read failed");
    size_t off=0;HostCache c;c.hdr=parse_one<SHdr>(b,off);if(std::string(c.hdr.magic,7)!="R8GAP01"||c.hdr.version!=1||c.hdr.r!=8)throw std::runtime_error("row8 gap cache header mismatch");
    for(int h=0;h<9;++h)if((int)c.hdr.dims[h]!=DIMS[h])throw std::runtime_error("row8 gap dimension mismatch");
    uint64_t got=fnv64(1469598103934665603ULL,b.data()+sizeof(SHdr),b.size()-sizeof(SHdr));if(got!=c.hdr.fnv_hash)throw std::runtime_error("row8 gap cache hash mismatch");
    for(int z=0;z<5;++z){auto vh=parse_one<VHdr>(b,off);if(vh.sym>=3||vh.h>=9||(vh.tag!=1&&vh.tag!=2))throw std::runtime_error("row8 gap vector header");auto &v=vh.tag==1?c.alpha[vh.sym]:c.beta[vh.sym];v.h=(int)vh.h;v.nz.reserve(vh.nnz);for(uint32_t i=0;i<vh.nnz;++i)v.nz.push_back(parse_one<uint16_t>(b,off));}
    uint64_t nzsum=0;
    for(int z=0;z<25;++z){auto bh=parse_one<BHdr>(b,off);if(bh.sym>=3||bh.h>=9||bh.h2>=9||bh.rows!=DIMS[bh.h]||bh.cols!=DIMS[bh.h2])throw std::runtime_error("row8 gap block header");auto &q=c.q[bh.sym][bh.h];q.h2=(int)bh.h2;q.rows=(int)bh.rows;q.cols=(int)bh.cols;q.rp=parse_vec<uint32_t>(b,off,bh.rows+1);q.ci=parse_vec<uint16_t>(b,off,bh.nnz);if(q.rp.back()!=bh.nnz)throw std::runtime_error("row8 gap csr offsets");
        q.cp.assign(q.cols+1,0);for(auto j:q.ci){if(j>=q.cols)throw std::runtime_error("row8 gap column OOB");++q.cp[(size_t)j+1];}for(int j=0;j<q.cols;++j)q.cp[j+1]+=q.cp[j];q.ri.resize(bh.nnz);auto cur=q.cp;for(int i=0;i<q.rows;++i)for(uint32_t e=q.rp[i];e<q.rp[i+1];++e){auto j=q.ci[e];auto k=cur[j]++;q.ri[k]=(uint16_t)i;}nzsum+=bh.nnz;
    }
    if(off!=b.size()||nzsum!=c.hdr.total_nz)throw std::runtime_error("row8 gap cache trailing/count mismatch");
    std::cerr<<"row8 gap cache hit path="<<path<<" bytes="<<sz<<" nz="<<nzsum<<"\n";return c;
}
inline const HostCache& host_cache(){static HostCache c=load_host_cache();return c;}

struct DevBlock {
    int rows=0,cols=0,h2=-1;
    uint32_t *rp=nullptr,*cp=nullptr;
    uint16_t *ci=nullptr,*ri=nullptr;
};
struct DeviceCache {int dev=-1;std::array<std::array<DevBlock,9>,3>q;};
inline void free_block(DevBlock&d){if(d.rp)cudaFree(d.rp);if(d.cp)cudaFree(d.cp);if(d.ci)cudaFree(d.ci);if(d.ri)cudaFree(d.ri);d={};}
inline void upload_vec(void**dst,const void*src,size_t n,const char*w){ck(cudaMalloc(dst,n),w);ck(cudaMemcpy(*dst,src,n,cudaMemcpyHostToDevice),w);}
inline DeviceCache upload_device_cache(int dev){ck(cudaSetDevice(dev),"row8 gap set device");DeviceCache d;d.dev=dev;auto const&h=host_cache();for(int a=0;a<3;++a)for(int q=0;q<9;++q){auto const&s=h.q[a][q];if(s.h2<0)continue;auto&t=d.q[a][q];t.rows=s.rows;t.cols=s.cols;t.h2=s.h2;upload_vec((void**)&t.rp,s.rp.data(),s.rp.size()*4,"row8 gap rp");upload_vec((void**)&t.cp,s.cp.data(),s.cp.size()*4,"row8 gap cp");upload_vec((void**)&t.ci,s.ci.data(),s.ci.size()*2,"row8 gap ci");upload_vec((void**)&t.ri,s.ri.data(),s.ri.size()*2,"row8 gap ri");}return d;}
inline void destroy(DeviceCache&d){ck(cudaSetDevice(d.dev),"row8 gap destroy set device");for(int a=0;a<3;++a)for(int h=0;h<9;++h)free_block(d.q[a][h]);}

__device__ __forceinline__ uint32_t gap_reduce_u64(unsigned long long sum,uint32_t mod){
    if(mod==4294967291u){
        constexpr unsigned long long P=4294967291ULL;
        unsigned long long t=(unsigned int)sum+5ULL*(sum>>32);
        if(t>=P)t-=P;
        return (uint32_t)t;
    }
    return (uint32_t)(sum%mod);
}
__global__ void gap_forward_soa(const uint32_t*x,int inRows,uint32_t*y,int outRows,int outOff,const uint32_t*cp,const uint16_t*ri,int n,uint32_t mod){unsigned long long total=(unsigned long long)n*inRows;for(unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,st=(unsigned long long)gridDim.x*blockDim.x;q<total;q+=st){int r=q%inRows,j=q/inRows;unsigned long long sum=0;for(uint32_t e=cp[j];e<cp[j+1];++e)sum+=x[(size_t)ri[e]*inRows+r];y[(size_t)j*outRows+outOff+r]=gap_reduce_u64(sum,mod);}}
__global__ void gap_forward_soa_indexed(const uint32_t*x,int inRows,const uint32_t*sel,int selRows,uint32_t*y,int outRows,int outOff,const uint32_t*cp,const uint16_t*ri,int n,uint32_t mod){unsigned long long total=(unsigned long long)n*selRows;for(unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,st=(unsigned long long)gridDim.x*blockDim.x;q<total;q+=st){int rr=q%selRows,j=q/selRows;int r=sel[rr];unsigned long long sum=0;for(uint32_t e=cp[j];e<cp[j+1];++e)sum+=x[(size_t)ri[e]*inRows+r];y[(size_t)j*outRows+outOff+rr]=gap_reduce_u64(sum,mod);}}
__global__ void gap_reverse_soa(const uint32_t* __restrict__ x,int inRows,uint32_t* __restrict__ y,int outRows,int outOff,const uint32_t* __restrict__ rp,const uint16_t* __restrict__ ci,int k,uint32_t mod){
    int r=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    int i=int(blockIdx.y);
    if(r>=inRows||i>=k)return;
    unsigned long long sum=0;
    for(uint32_t e=rp[i];e<rp[i+1];++e)sum+=x[(size_t)ci[e]*inRows+r];
    y[(size_t)i*outRows+outOff+r]=gap_reduce_u64(sum,mod);
}
__global__ void structural_soa_to_row(const uint32_t*x,uint32_t*y,int dim,int rows){unsigned long long total=(unsigned long long)dim*rows;for(unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,st=(unsigned long long)gridDim.x*blockDim.x;q<total;q+=st){int r=q%rows,j=q/rows;y[(size_t)r*dim+j]=x[q];}}
__global__ void structural_count_nz(const uint32_t*x,unsigned long long n,unsigned long long*out){unsigned long long local=0;for(unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,st=(unsigned long long)gridDim.x*blockDim.x;q<n;q+=st)local+=x[q]!=0;if(local)atomicAdd(out,local);}
__global__ void structural_value_stats(const uint32_t*x,unsigned long long n,uint32_t mod,unsigned long long*out){unsigned long long nz=0,b8=0,b16=0,b24=0,mx=0;for(unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,st=(unsigned long long)gridDim.x*blockDim.x;q<n;q+=st){uint32_t v=x[q];if(!v)continue;++nz;unsigned long long b=uint64_t(mod)-v;unsigned long long a=v<b?v:b;mx=mx>a?mx:a;b8+=a<=127;b16+=a<=32767;b24+=a<=8388607;}if(nz)atomicAdd(out,nz);if(b8)atomicAdd(out+1,b8);if(b16)atomicAdd(out+2,b16);if(b24)atomicAdd(out+3,b24);atomicMax(out+4,mx);}

struct SGroup {int h=0,rows=0,dim=0;uint32_t*d=nullptr;std::vector<uint32_t>code;};
inline void profile_groups(const std::array<SGroup,9>&g,const char*tag,uint32_t mod){unsigned long long *d=nullptr;ck(cudaMalloc(&d,5*8),"row8 gap profile alloc");for(int h=0;h<9;++h)if(g[h].rows){ck(cudaMemset(d,0,5*8),"row8 gap profile zero");unsigned long long n=(unsigned long long)g[h].rows*g[h].dim;int blocks=std::max<unsigned long long>(1,std::min<unsigned long long>(65535,(n+255)/256));structural_value_stats<<<blocks,256>>>(g[h].d,n,mod,d);unsigned long long z[5]{};ck(cudaMemcpy(z,d,sizeof(z),cudaMemcpyDeviceToHost),"row8 gap profile copy");std::cerr<<"row8 gap "<<tag<<" h="<<h<<" rows="<<g[h].rows<<" dim="<<g[h].dim<<" nz="<<z[0]<<" total="<<n<<" density="<<double(z[0])/double(n)<<" per_row="<<double(z[0])/g[h].rows<<" le7="<<z[1]<<" le15="<<z[2]<<" le23="<<z[3]<<" maxabs="<<z[4]<<"\n";}cudaFree(d);}
inline void free_groups(std::array<SGroup,9>&g){for(auto&x:g)if(x.d){cudaFree(x.d);x.d=nullptr;}}
inline SGroup vector_group(const SparseVec&v,int h,uint32_t code,uint32_t mod){SGroup g;g.h=h;g.rows=1;g.dim=DIMS[h];g.code={code};std::vector<uint32_t>x(g.dim);for(auto i:v.nz)x[i]=1;ck(cudaMalloc(&g.d,x.size()*4),"row8 gap vector alloc");ck(cudaMemcpy(g.d,x.data(),x.size()*4,cudaMemcpyHostToDevice),"row8 gap vector copy");return g;}
inline std::array<SGroup,9> initial_prefix(uint32_t mod){std::array<SGroup,9>g{};auto const&c=host_cache();for(int a=0;a<3;++a){int h=c.alpha[a].h;if(h<0)continue;g[h]=vector_group(c.alpha[a],h,(uint32_t)a,mod);}for(int h=0;h<9;++h){g[h].h=h;g[h].dim=DIMS[h];}return g;}
inline std::array<SGroup,9> initial_suffix(uint32_t mod){std::array<SGroup,9>g{};auto const&c=host_cache();for(int a=0;a<3;++a){int h=c.beta[a].h;if(h<0)continue;g[h]=vector_group(c.beta[a],h,(uint32_t)a,mod);}for(int h=0;h<9;++h){g[h].h=h;g[h].dim=DIMS[h];}return g;}

inline std::array<SGroup,9> build_levels(DeviceCache const&dc,uint32_t mod,int len,bool suffix,double&seconds){if(len<=0)throw std::runtime_error("row8 gap len must be positive");auto cur=suffix?initial_suffix(mod):initial_prefix(mod);auto wall0=std::chrono::steady_clock::now();uint64_t p3=3;for(int lev=1;lev<len;++lev){auto t0=std::chrono::steady_clock::now();std::array<int,9>nr{};for(int h=0;h<9;++h)if(cur[h].rows)for(int a=0;a<3;++a){int h2=suffix?h-DELTA[a]:h+DELTA[a];if(0<=h2&&h2<9)nr[h2]+=cur[h].rows;}std::array<SGroup,9>nxt{};for(int h=0;h<9;++h){nxt[h].h=h;nxt[h].dim=DIMS[h];nxt[h].rows=nr[h];if(nr[h]){ck(cudaMalloc(&nxt[h].d,(size_t)DIMS[h]*nr[h]*4),"row8 gap next alloc");nxt[h].code.reserve(nr[h]);}}std::array<int,9>off{};for(int h=0;h<9;++h)if(cur[h].rows)for(int a=0;a<3;++a){int h2=suffix?h-DELTA[a]:h+DELTA[a];if(h2<0||h2>=9)continue;auto const&b=suffix?dc.q[a][h2]:dc.q[a][h];int nout=suffix?b.rows:b.cols;unsigned long long work=(unsigned long long)nout*cur[h].rows;int blocks=std::max<unsigned long long>(1,std::min<unsigned long long>(65535,(work+255)/256));if(suffix){constexpr int GT=128;dim3 gdim((unsigned(cur[h].rows)+GT-1)/GT,(unsigned)b.rows);gap_reverse_soa<<<gdim,GT>>>(cur[h].d,cur[h].rows,nxt[h2].d,nxt[h2].rows,off[h2],b.rp,b.ci,b.rows,mod);}else gap_forward_soa<<<blocks,256>>>(cur[h].d,cur[h].rows,nxt[h2].d,nxt[h2].rows,off[h2],b.cp,b.ri,b.cols,mod);ck(cudaGetLastError(),"row8 gap step");for(uint32_t c:cur[h].code)nxt[h2].code.push_back(suffix?uint32_t(uint64_t(a)*p3+c):uint32_t(c*3u+a));off[h2]+=cur[h].rows;}ck(cudaDeviceSynchronize(),"row8 gap level sync");free_groups(cur);cur=std::move(nxt);if(suffix)p3*=3;seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();}seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();return cur;}


inline std::array<SGroup,9> build_prefix_levels_pruned(DeviceCache const&dc,uint32_t mod,int len,const row8tensor::PrefixPrunePlan&plan,double&seconds){
    if(len<=0||plan.len!=len||int(plan.keep.size())!=len+1)throw std::runtime_error("row8 gap prune plan mismatch");
    std::array<SGroup,9> cur{};auto const&hc=host_cache();
    for(int a=0;a<3;++a){int h=hc.alpha[a].h;if(h<0)continue;if((uint32_t)a>=plan.keep[1].size()||!plan.keep[1][a])continue;cur[h]=vector_group(hc.alpha[a],h,(uint32_t)a,mod);}
    for(int h=0;h<9;++h){cur[h].h=h;cur[h].dim=DIMS[h];}
    auto wall0=std::chrono::steady_clock::now();
    for(int lev=1;lev<len;++lev){
        std::array<std::array<std::vector<uint32_t>,3>,9> sel;std::array<int,9>nr{};
        for(int h=0;h<9;++h)if(cur[h].rows)for(int i=0;i<cur[h].rows;++i){uint32_t c=cur[h].code[i];for(int a=0;a<3;++a){int h2=h+DELTA[a];if(h2<0||h2>=9)continue;uint32_t child=c*3u+uint32_t(a);if(child<plan.keep[lev+1].size()&&plan.keep[lev+1][child]){sel[h][a].push_back((uint32_t)i);++nr[h2];}}}
        std::array<SGroup,9>nxt{};for(int h=0;h<9;++h){nxt[h].h=h;nxt[h].dim=DIMS[h];nxt[h].rows=nr[h];if(nr[h]){ck(cudaMalloc(&nxt[h].d,(size_t)DIMS[h]*nr[h]*4),"row8 gap pruned next");nxt[h].code.reserve(nr[h]);}}
        std::array<int,9>off{};
        for(int h=0;h<9;++h)if(cur[h].rows)for(int a=0;a<3;++a){auto const&sv=sel[h][a];if(sv.empty())continue;int h2=h+DELTA[a];auto const&b=dc.q[a][h];unsigned long long work=(unsigned long long)b.cols*sv.size();int blocks=std::max<unsigned long long>(1,std::min<unsigned long long>(65535,(work+255)/256));
            if(sv.size()==size_t(cur[h].rows))gap_forward_soa<<<blocks,256>>>(cur[h].d,cur[h].rows,nxt[h2].d,nxt[h2].rows,off[h2],b.cp,b.ri,b.cols,mod);
            else{uint32_t*dsel=nullptr;ck(cudaMalloc(&dsel,sv.size()*4),"row8 gap selected alloc");ck(cudaMemcpy(dsel,sv.data(),sv.size()*4,cudaMemcpyHostToDevice),"row8 gap selected copy");gap_forward_soa_indexed<<<blocks,256>>>(cur[h].d,cur[h].rows,dsel,(int)sv.size(),nxt[h2].d,nxt[h2].rows,off[h2],b.cp,b.ri,b.cols,mod);ck(cudaGetLastError(),"row8 gap selected step");cudaFree(dsel);}
            for(uint32_t i:sv)nxt[h2].code.push_back(cur[h].code[i]*3u+uint32_t(a));off[h2]+=int(sv.size());
        }
        ck(cudaDeviceSynchronize(),"row8 gap pruned level sync");free_groups(cur);cur=std::move(nxt);
        if(const char*e=std::getenv("GRIDFP_ROW8_PROFILE_PREFIX_PRUNE");e&&std::atoi(e)){size_t rows=0,bytes=0;for(int h=0;h<9;++h){rows+=cur[h].rows;bytes+=size_t(cur[h].rows)*DIMS[h]*4;}std::cerr<<"row8 gap pruned prefix level="<<lev+1<<" rows="<<rows<<" mib="<<double(bytes)/(1<<20)<<"\n";}
    }
    seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();return cur;
}

inline std::array<row8tensor::Group,9> to_row_groups(std::array<SGroup,9>&src,double&seconds){auto t0=std::chrono::steady_clock::now();std::array<row8tensor::Group,9>out{};for(int h=0;h<9;++h){auto&s=src[h];auto&o=out[h];o.h=h;o.rows=s.rows;o.dim=DIMS[h];o.code=std::move(s.code);if(!s.rows)continue;ck(cudaMalloc(&o.d,(size_t)s.rows*DIMS[h]*4),"row8 gap transpose alloc");unsigned long long n=(unsigned long long)s.rows*DIMS[h];int blocks=std::max<unsigned long long>(1,std::min<unsigned long long>(65535,(n+255)/256));structural_soa_to_row<<<blocks,256>>>(s.d,o.d,DIMS[h],s.rows);ck(cudaGetLastError(),"row8 gap transpose");cudaFree(s.d);s.d=nullptr;}ck(cudaDeviceSynchronize(),"row8 gap transpose sync");seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();return out;}

struct InitStats {double upload_s=0,prefix_s=0,suffix_s=0,transpose_s=0,join_s=0,gpu_init_s=0;};
inline InitStats init_single_gpu(int W,uint32_t mod,Count*fullMain,int dev=0){if(W!=TARGET_W)throw std::runtime_error("row8 gap width mismatch");ck(cudaSetDevice(dev),"row8 gap init set device");auto all0=std::chrono::steady_clock::now();auto u0=std::chrono::steady_clock::now();auto dc=upload_device_cache(dev);double upload=std::chrono::duration<double>(std::chrono::steady_clock::now()-u0).count();int lo=TARGET_W/2,hi=W-lo;InitStats st;st.upload_s=upload;auto ps=build_levels(dc,mod,hi,false,st.prefix_s);auto ss=build_levels(dc,mod,lo,true,st.suffix_s);if(const char*e=std::getenv("GRIDFP_ROW8_GAP_PROFILE");e&&std::atoi(e)){profile_groups(ps,"prefix",mod);profile_groups(ss,"suffix",mod);}double tp=0,ts=0;auto pg=to_row_groups(ps,tp);auto sg=to_row_groups(ss,ts);st.transpose_s=tp+ts;destroy(dc);row8tensor::Runtime dummy;dummy.mod=mod;cublasHandle_t handle=nullptr;row8tensor::cublas_ck(cublasCreate(&handle),"row8 gap join create");row8tensor::cublas_ck(cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH),"row8 gap join math");st.join_s=row8tensor::scatter_all(handle,dummy,pg,sg,hi,lo,fullMain,0,H_DP[W][1]);row8tensor::free_groups(pg);row8tensor::free_groups(sg);cublasDestroy(handle);st.gpu_init_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-all0).count();std::cerr<<"direct row8 gap mod="<<mod<<" upload_s="<<st.upload_s<<" prefix_s="<<st.prefix_s<<" suffix_s="<<st.suffix_s<<" transpose_s="<<st.transpose_s<<" join_s="<<st.join_s<<" gpu_init_s="<<st.gpu_init_s<<"\n";return st;}


inline InitStats init_gpu_shard(int W,uint32_t mod,Count*fullMain,int dev,Code rankBase,Code rankLimit,const row8tensor::PrefixPrunePlan*plan=nullptr){
    if(W!=TARGET_W)throw std::runtime_error("row8 gap width mismatch");ck(cudaSetDevice(dev),"row8 gap shard set device");auto all0=std::chrono::steady_clock::now();auto u0=std::chrono::steady_clock::now();auto dc=upload_device_cache(dev);InitStats st;st.upload_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-u0).count();int lo=TARGET_W/2,hi=W-lo;
    auto ps=plan?build_prefix_levels_pruned(dc,mod,hi,*plan,st.prefix_s):build_levels(dc,mod,hi,false,st.prefix_s);auto ss=build_levels(dc,mod,lo,true,st.suffix_s);
    double tp=0,ts=0;auto pg=to_row_groups(ps,tp);auto sg=to_row_groups(ss,ts);st.transpose_s=tp+ts;destroy(dc);row8tensor::Runtime dummy;dummy.mod=mod;cublasHandle_t handle=nullptr;row8tensor::cublas_ck(cublasCreate(&handle),"row8 gap shard join create");row8tensor::cublas_ck(cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH),"row8 gap shard join math");st.join_s=row8tensor::scatter_all(handle,dummy,pg,sg,hi,lo,fullMain,rankBase,rankLimit);row8tensor::free_groups(pg);row8tensor::free_groups(sg);cublasDestroy(handle);st.gpu_init_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-all0).count();std::cerr<<"direct row8 gap shard dev="<<dev<<" mod="<<mod<<" prefix_prune="<<(plan!=nullptr)<<" upload_s="<<st.upload_s<<" prefix_s="<<st.prefix_s<<" suffix_s="<<st.suffix_s<<" transpose_s="<<st.transpose_s<<" join_s="<<st.join_s<<" gpu_init_s="<<st.gpu_init_s<<"\n";return st;
}
inline unsigned long long compare_gpu_shard(int W,uint32_t mod,const Count*baseline,int dev,Code rankBase,Code rankLimit,const row8tensor::PrefixPrunePlan*plan=nullptr){
    if(W!=TARGET_W)throw std::runtime_error("row8 gap compare width mismatch");ck(cudaSetDevice(dev),"row8 gap compare set device");auto dc=upload_device_cache(dev);int lo=TARGET_W/2,hi=W-lo;double ps=0,ss=0;auto psg=plan?build_prefix_levels_pruned(dc,mod,hi,*plan,ps):build_levels(dc,mod,hi,false,ps);auto ssg=build_levels(dc,mod,lo,true,ss);double tp=0,ts=0;auto pg=to_row_groups(psg,tp);auto sg=to_row_groups(ssg,ts);destroy(dc);row8tensor::Runtime dummy;dummy.mod=mod;cublasHandle_t handle=nullptr;row8tensor::cublas_ck(cublasCreate(&handle),"row8 gap compare cublas create");row8tensor::cublas_ck(cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH),"row8 gap compare cublas math");unsigned long long*dMis=nullptr;ck(cudaMalloc(&dMis,8),"row8 gap mismatch alloc");ck(cudaMemset(dMis,0,8),"row8 gap mismatch zero");double js=row8tensor::scatter_all_compare(handle,dummy,pg,sg,hi,lo,baseline,rankBase,rankLimit,dMis);unsigned long long mis=0;ck(cudaMemcpy(&mis,dMis,8,cudaMemcpyDeviceToHost),"row8 gap mismatch copy");cudaFree(dMis);row8tensor::free_groups(pg);row8tensor::free_groups(sg);cublasDestroy(handle);std::cerr<<"row8 gap compare shard dev="<<dev<<" mod="<<mod<<" rank=["<<rankBase<<","<<rankLimit<<") mismatch="<<mis<<" prefix_s="<<ps<<" suffix_s="<<ss<<" join_s="<<js<<"\n";return mis;
}
inline unsigned long long compare_multi_gpu(int W,uint32_t mod,Count**baseline,const std::vector<Code>&mainLen,Code mainChunk,int ng){if(ng==1){int fake=1;if(const char*e=std::getenv("GRIDFP_ROW8_CERT_FAKE_SHARDS"))fake=std::max(1,std::atoi(e));if(fake<=1){auto z=compare_gpu_shard(W,mod,baseline[0],0,0,mainLen[0],nullptr);std::cerr<<"row8 gap compare multi mod="<<mod<<" gpus=1 mismatch="<<z<<"\n";return z;}int lo=TARGET_W/2,hi=W-lo;Code total=H_DP[W][1],chunk=(total+fake-1)/fake;std::vector<Code>lens(fake);for(int d=0;d<fake;++d){Code base=Code(d)*chunk;lens[d]=base<total?std::min<Code>(chunk,total-base):0;}auto const&pps=row8tensor::prefix_plan_set_cached(W,hi,lo,chunk,lens,fake);unsigned long long totalMis=0;for(int d=0;d<fake;++d){Code base=Code(d)*chunk;if(!lens[d])continue;totalMis+=compare_gpu_shard(W,mod,baseline[0]+base,0,base,base+lens[d],&pps.plans[d]);}std::cerr<<"row8 gap compare fake-shards="<<fake<<" mod="<<mod<<" mismatch="<<totalMis<<"\n";return totalMis;}int lo=TARGET_W/2,hi=W-lo;auto const&pps=row8tensor::prefix_plan_set_cached(W,hi,lo,mainChunk,mainLen,ng);std::vector<unsigned long long>mis(ng);std::vector<std::thread>ths;for(int d=0;d<ng;++d)ths.emplace_back([&,d]{Code base=Code(d)*mainChunk;mis[d]=compare_gpu_shard(W,mod,baseline[d],d,base,base+mainLen[d],&pps.plans[d]);});for(auto&t:ths)t.join();unsigned long long total=0;for(auto x:mis)total+=x;std::cerr<<"row8 gap compare multi mod="<<mod<<" gpus="<<ng<<" mismatch="<<total<<"\n";return total;}

inline void init_multi_gpu(int W,uint32_t mod,Count**fullMain,const std::vector<Code>&mainLen,Code mainChunk,int ng){
    if(ng<1)throw std::runtime_error("row8 gap ng");
    if(ng==1){int fake=0;if(const char*e=std::getenv("GRIDFP_ROW8_FAKE_SHARDS"))fake=std::max(0,std::atoi(e));if(fake<=1){init_single_gpu(W,mod,fullMain[0],0);return;}int lo=TARGET_W/2,hi=W-lo;Code total=H_DP[W][1],chunk=(total+fake-1)/fake;std::vector<Code>lens(fake);for(int d=0;d<fake;++d){Code base=Code(d)*chunk;lens[d]=base<total?std::min<Code>(chunk,total-base):0;}auto const&pps=row8tensor::prefix_plan_set_cached(W,hi,lo,chunk,lens,fake);auto wall0=std::chrono::steady_clock::now();double pm=0,sm=0,jm=0;for(int d=0;d<fake;++d){Code base=Code(d)*chunk;if(base>=total)break;auto st=init_gpu_shard(W,mod,fullMain[0]+base,0,base,base+lens[d],&pps.plans[d]);pm=std::max(pm,st.prefix_s);sm=std::max(sm,st.suffix_s);jm=std::max(jm,st.join_s);}double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();std::cerr<<"direct row8 gap fake-shards="<<fake<<" mod="<<mod<<" prefix_max_s="<<pm<<" suffix_max_s="<<sm<<" join_max_s="<<jm<<" wall_s="<<wall<<"\n";return;}
    int lo=TARGET_W/2,hi=W-lo;auto const&pps=row8tensor::prefix_plan_set_cached(W,hi,lo,mainChunk,mainLen,ng);std::vector<InitStats>stats(ng);std::vector<std::thread>ths;ths.reserve(ng);auto wall0=std::chrono::steady_clock::now();for(int d=0;d<ng;++d)ths.emplace_back([&,d]{Code base=Code(d)*mainChunk,lim=base+mainLen[d];stats[d]=init_gpu_shard(W,mod,fullMain[d],d,base,lim,&pps.plans[d]);});for(auto&t:ths)t.join();double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count(),pm=0,sm=0,jm=0,gm=0;for(auto const&st:stats){pm=std::max(pm,st.prefix_s);sm=std::max(sm,st.suffix_s);jm=std::max(jm,st.join_s);gm=std::max(gm,st.gpu_init_s);}std::cerr<<"direct row8 gap multi mod="<<mod<<" gpus="<<ng<<" prefix_max_s="<<pm<<" suffix_max_s="<<sm<<" join_max_s="<<jm<<" gpu_max_s="<<gm<<" wall_s="<<wall<<"\n";
}

} // namespace oneesan::row8gap
