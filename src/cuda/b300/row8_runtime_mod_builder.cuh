#pragma once
#include "row8_mod_matrix_runtime.cuh"
#include <array>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace oneesan::row8runtime {
static constexpr int DIMS[9]={1107,1640,1428,888,420,152,42,8,1};
static constexpr int DELTA[3]={0,-1,1};
static constexpr uint64_t CACHE_ABI=0x52384d4443414249ULL; // "R8MDCABI" semantic version 1
struct PivotPair{uint32_t pc=0,sc=0;};
struct PivotBlock{std::vector<PivotPair> piv;};
struct PivotSet{std::array<PivotBlock,9> b;uint64_t fingerprint=1469598103934665603ULL;};
inline uint64_t fnv_bytes(uint64_t h,const void*ptr,size_t n){auto*p=(const unsigned char*)ptr;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;}

inline std::string pivot_path(){std::string p="src/cuda/b300/row8_pivots_w19.bin";if(const char*e=std::getenv("GRIDFP_ROW8_PIVOTS"))p=e;return p;}
inline const PivotSet& pivots(){static std::string cached;static std::unique_ptr<PivotSet> ps;std::string p=pivot_path();if(ps&&cached==p)return *ps;std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("row8 pivots open failed: "+p);char magic[8];uint32_t ver=0,ds[9]{};in.read(magic,8);in.read((char*)&ver,4);in.read((char*)ds,sizeof(ds));if(std::string(magic,7)!="R8PIV19"||ver!=1)throw std::runtime_error("row8 pivots header mismatch");auto z=std::make_unique<PivotSet>();for(int h=0;h<9;++h){if((int)ds[h]!=DIMS[h])throw std::runtime_error("row8 pivot dimension mismatch");z->fingerprint=fnv_bytes(z->fingerprint,&ds[h],sizeof(ds[h]));z->b[h].piv.resize(DIMS[h]);in.read((char*)z->b[h].piv.data(),z->b[h].piv.size()*sizeof(PivotPair));z->fingerprint=fnv_bytes(z->fingerprint,z->b[h].piv.data(),z->b[h].piv.size()*sizeof(PivotPair));}if(!in)throw std::runtime_error("row8 pivots truncated");cached=p;ps=std::move(z);return *ps;}

struct Compact{int width=0;Code n=0;Code dp[MAXW+1][MAXW+2]{};std::vector<Count>a;double build_s=0,copy_s=0;};
inline Compact make_compact(int width,Count mod,int dev=0){Compact z;z.width=width;auto t0=std::chrono::steady_clock::now();Count*d=build_row8_bounded_compact_runtime_width(width,8,mod,256,z.n,z.dp,dev);auto t1=std::chrono::steady_clock::now();z.a.resize(z.n);ck(cudaMemcpy(z.a.data(),d,size_t(z.n)*sizeof(Count),cudaMemcpyDeviceToHost),"row8 runtime compact copy");cudaFree(d);auto t2=std::chrono::steady_clock::now();z.build_s=std::chrono::duration<double>(t1-t0).count();z.copy_s=std::chrono::duration<double>(t2-t1).count();if(const char*e=std::getenv("GRIDFP_ROW8_BUILD_PROFILE");e&&std::atoi(e))std::cerr<<"row8 compact mod="<<mod<<" W="<<width<<" states="<<z.n<<" build_s="<<z.build_s<<" copy_s="<<z.copy_s<<"\n";return z;}
inline void decode(uint32_t code,int len,int*d){for(int i=len-1;i>=0;--i){d[i]=code%3;code/=3;}}
inline bool full_rank_code(uint32_t code,int len,const Code dp[MAXW+1][MAXW+2],Code&rank){int d[32];decode(code,len,d);int h=1;rank=0;for(int q=0;q<len;++q){int v=d[q],pos=len-1-q;if(v>0)rank+=dp[pos][h];if(v>1&&h>0)rank+=dp[pos][h-1];if(v==1)--h;else if(v==2)++h;if(h<0||h>8)return false;}return h==0;}
inline bool prefix_base(uint32_t code,int len,int width,const Code dp[MAXW+1][MAXW+2],Code&rank,int&hout){int d[32];decode(code,len,d);int h=1;rank=0;for(int q=0;q<len;++q){int v=d[q],pos=width-1-q;if(v>0)rank+=dp[pos][h];if(v>1&&h>0)rank+=dp[pos][h-1];if(v==1)--h;else if(v==2)++h;if(h<0||h>8)return false;}hout=h;return true;}
inline bool prefix_base_sym(uint32_t code,int len,int sym,int width,const Code dp[MAXW+1][MAXW+2],Code&rank,int&hout){if(!prefix_base(code,len,width,dp,rank,hout))return false;int pos=width-1-len;if(sym>0)rank+=dp[pos][hout];if(sym>1&&hout>0)rank+=dp[pos][hout-1];if(sym==1)--hout;else if(sym==2)++hout;return hout>=0&&hout<=8;}
inline bool suffix_rank(uint32_t code,int len,int h,const Code dp[MAXW+1][MAXW+2],Code&rank){int d[32];decode(code,len,d);rank=0;for(int q=0;q<len;++q){int v=d[q],pos=len-1-q;if(v>0)rank+=dp[pos][h];if(v>1&&h>0)rank+=dp[pos][h-1];if(v==1)--h;else if(v==2)++h;if(h<0||h>8)return false;}return h==0;}
inline std::vector<uint32_t> rowmul_cpu(const std::vector<uint32_t>&x,const std::vector<uint32_t>&M,int m,int n,uint32_t mod){std::vector<uint32_t>y(n);for(int j=0;j<n;++j){unsigned __int128 z=0;for(int i=0;i<m;++i)z+=(unsigned __int128)x[i]*M[size_t(i)*n+j];y[j]=uint32_t(z%mod);}return y;}

inline row8tensor::ModData build_mod(Count mod,bool validate=false,int dev=0){auto all0=std::chrono::steady_clock::now();auto const&P=pivots();row8tensor::ModData md;
    ck(cudaSetDevice(dev),"row8 build set device");auto c9=make_compact(9,mod,dev);md.beta.resize(DIMS[0]);for(int i=0;i<DIMS[0];++i){Code r;if(!full_rank_code(P.b[0].piv[i].pc,9,c9.dp,r)||r>=c9.n)throw std::runtime_error("row8 beta rank");md.beta[i]=c9.a[r];}c9.a.clear();c9.a.shrink_to_fit();
    auto c10=make_compact(10,mod,dev);std::vector<uint32_t>init(DIMS[1]);for(int j=0;j<DIMS[1];++j){Code r;if(!full_rank_code(P.b[1].piv[j].sc,10,c10.dp,r)||r>=c10.n)throw std::runtime_error("row8 alpha residual rank");init[j]=c10.a[r];}c10.a.clear();c10.a.shrink_to_fit();
    auto c19=make_compact(19,mod,dev);std::array<std::vector<uint32_t>,9>B,I;for(int h=0;h<9;++h){int n=DIMS[h];B[h].resize(size_t(n)*n);std::vector<Code>sr(n);for(int j=0;j<n;++j)if(!suffix_rank(P.b[h].piv[j].sc,10,h,c19.dp,sr[j]))throw std::runtime_error("row8 B suffix rank");for(int i=0;i<n;++i){Code base;int hh;if(!prefix_base(P.b[h].piv[i].pc,9,19,c19.dp,base,hh)||hh!=h)throw std::runtime_error("row8 B prefix rank");for(int j=0;j<n;++j){Code r=base+sr[j];if(r>=c19.n)throw std::runtime_error("row8 B rank OOB");B[h][size_t(i)*n+j]=c19.a[r];}}}c19.a.clear();c19.a.shrink_to_fit();
    for(int h=0;h<9;++h){double s=0;I[h]=modgpu_runtime::invert(B[h],DIMS[h],mod,&s);md.inv_s+=s;}
    double as=0;md.alpha=modgpu_runtime::multiply(init,1,DIMS[1],I[1],DIMS[1],mod,&as);md.trans_s+=as;
    auto c20=make_compact(20,mod,dev);std::array<std::array<std::vector<uint32_t>,9>,3>V;for(int a=0;a<3;++a)for(int h=0;h<9;++h){int h2=h+DELTA[a];if(h2<0||h2>=9)continue;int m=DIMS[h],n=DIMS[h2];auto&v=V[a][h];v.resize(size_t(m)*n);std::vector<Code>sr(n);for(int j=0;j<n;++j)if(!suffix_rank(P.b[h2].piv[j].sc,10,h2,c20.dp,sr[j]))throw std::runtime_error("row8 V suffix rank");for(int i=0;i<m;++i){Code base;int hh;if(!prefix_base_sym(P.b[h].piv[i].pc,9,a,20,c20.dp,base,hh)||hh!=h2)throw std::runtime_error("row8 V prefix rank");for(int j=0;j<n;++j){Code r=base+sr[j];if(r>=c20.n)throw std::runtime_error("row8 V rank OOB");v[size_t(i)*n+j]=c20.a[r];}}}c20.a.clear();c20.a.shrink_to_fit();
    for(int a=0;a<3;++a)for(int h=0;h<9;++h){int h2=h+DELTA[a];if(h2<0||h2>=9)continue;double s=0;md.trans[a][h]=modgpu_runtime::multiply(V[a][h],DIMS[h],DIMS[h2],I[h2],DIMS[h2],mod,&s);md.trans_s+=s;}
    if(validate){for(int a=0;a<3;++a)for(int h=0;h<9;++h){int h2=h+DELTA[a];if(h2<0||h2>=9)continue;auto chk=modgpu_runtime::multiply(md.trans[a][h],DIMS[h],DIMS[h2],B[h2],DIMS[h2],mod);if(chk!=V[a][h])throw std::runtime_error("row8 M*B validation failed");}if(rowmul_cpu(md.alpha,B[1],DIMS[1],DIMS[1],mod)!=init)throw std::runtime_error("row8 alpha validation failed");}
    md.build_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-all0).count();std::cerr<<"row8 runtime mod build mod="<<mod<<" build_s="<<md.build_s<<" inv_gpu_s="<<md.inv_s<<" trans_gpu_s="<<md.trans_s<<" validate="<<validate<<"\n";return md;}

inline uint64_t moddata_hash(const row8tensor::ModData& md) {
    uint64_t h = 1469598103934665603ULL;
    h = fnv_bytes(h, md.alpha.data(), md.alpha.size() * sizeof(uint32_t));
    h = fnv_bytes(h, md.beta.data(), md.beta.size() * sizeof(uint32_t));
    for (int a = 0; a < 3; ++a)
        for (int q = 0; q < 9; ++q)
            h = fnv_bytes(h, md.trans[a][q].data(), md.trans[a][q].size() * sizeof(uint32_t));
    return h;
}

inline bool cache_enabled() {
    if (const char* e = std::getenv("GRIDFP_ROW8_CACHE")) return std::atoi(e) != 0;
    return true;
}

inline std::filesystem::path cache_path(Count mod, uint64_t fp) {
    std::filesystem::path dir = "work/row8_mod_cache";
    if (const char* e = std::getenv("GRIDFP_ROW8_CACHE_DIR")) dir = e;
    std::ostringstream n;
    n << "row8_mod_" << mod << "_" << std::hex << std::setw(16) << std::setfill('0') << fp
      << "_" << std::setw(16) << CACHE_ABI << ".bin";
    return dir / n.str();
}

inline bool load_cache(Count mod, const PivotSet& ps, row8tensor::ModData& md) {
    if (!cache_enabled()) return false;
    auto p = cache_path(mod, ps.fingerprint);
    std::ifstream in(p, std::ios::binary);
    if (!in) return false;
    char magic[8]; uint32_t ver=0, pm=0, ds[9]{};
    uint64_t fp=0, abi=0, an=0, bn=0, counts[27]{}, storedHash=0;
    in.read(magic,8); in.read((char*)&ver,4); in.read((char*)&pm,4); in.read((char*)&fp,8); in.read((char*)&abi,8);
    in.read((char*)ds,sizeof(ds)); in.read((char*)&an,8); in.read((char*)&bn,8);
    in.read((char*)counts,sizeof(counts)); in.read((char*)&storedHash,8);
    if (!in || std::string(magic,7)!="R8MDC02" || ver!=2 || pm!=mod || fp!=ps.fingerprint || abi!=CACHE_ABI ||
        an!=size_t(DIMS[1]) || bn!=size_t(DIMS[0])) return false;
    for (int h=0; h<9; ++h) if ((int)ds[h] != DIMS[h]) return false;
    md.alpha.resize(an); md.beta.resize(bn);
    in.read((char*)md.alpha.data(), an*4); in.read((char*)md.beta.data(), bn*4);
    int qi=0;
    for (int a=0; a<3; ++a) for (int h=0; h<9; ++h) {
        int h2=h+DELTA[a]; uint64_t want=(h2>=0&&h2<9)?uint64_t(DIMS[h])*DIMS[h2]:0;
        if (counts[qi++] != want) return false;
        md.trans[a][h].resize(want);
        if (want) in.read((char*)md.trans[a][h].data(), want*4);
    }
    char extra=0;
    if (!in || in.read(&extra,1)) return false;
    if (moddata_hash(md) != storedHash) return false;
    md.build_s = md.inv_s = md.trans_s = 0;
    std::cerr << "row8 mod cache hit mod=" << mod << " path=" << p.string() << "\n";
    return true;
}

inline void save_cache(Count mod, const PivotSet& ps, const row8tensor::ModData& md) {
    if (!cache_enabled()) return;
    auto p=cache_path(mod, ps.fingerprint);
    std::filesystem::create_directories(p.parent_path());
    auto tmp=p; tmp += std::string(".tmp.") + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::ofstream out(tmp, std::ios::binary|std::ios::trunc);
    char magic[8]={'R','8','M','D','C','0','2','\0'}; uint32_t ver=2, pm=mod, ds[9];
    for (int h=0;h<9;++h) ds[h]=DIMS[h];
    uint64_t fp=ps.fingerprint, abi=CACHE_ABI, an=md.alpha.size(), bn=md.beta.size(), counts[27]; int qi=0;
    for (int a=0;a<3;++a) for (int h=0;h<9;++h) counts[qi++]=md.trans[a][h].size();
    uint64_t hash=moddata_hash(md);
    out.write(magic,8); out.write((char*)&ver,4); out.write((char*)&pm,4); out.write((char*)&fp,8); out.write((char*)&abi,8);
    out.write((char*)ds,sizeof(ds)); out.write((char*)&an,8); out.write((char*)&bn,8);
    out.write((char*)counts,sizeof(counts)); out.write((char*)&hash,8);
    out.write((char*)md.alpha.data(),an*4); out.write((char*)md.beta.data(),bn*4);
    for (int a=0;a<3;++a) for (int h=0;h<9;++h)
        if (!md.trans[a][h].empty()) out.write((char*)md.trans[a][h].data(),md.trans[a][h].size()*4);
    out.close();
    if (!out) throw std::runtime_error("row8 cache write failed: "+tmp.string());
    std::error_code ec; std::filesystem::rename(tmp,p,ec);
    if (ec) { std::filesystem::remove(p,ec); ec.clear(); std::filesystem::rename(tmp,p,ec); }
    if (ec) throw std::runtime_error("row8 cache rename failed: "+ec.message());
    std::cerr << "row8 mod cache saved mod=" << mod << " path=" << p.string()
              << " bytes=" << std::filesystem::file_size(p) << "\n";
}

inline row8tensor::ModData load_or_build_mod(Count mod, bool validate=false, int dev=0) {
    auto const& ps=pivots(); row8tensor::ModData md;
    if (load_cache(mod,ps,md)) return md;
    md=build_mod(mod,validate,dev); save_cache(mod,ps,md); return md;
}

inline double prebuild_mod_caches(const std::vector<Count>& mods, int ng, bool validate=false) {
    if (mods.empty()) return 0;
    (void)pivots();
    auto t0=std::chrono::steady_clock::now();
    std::atomic<size_t> next{0};
    int nw=std::max(1,std::min<int>(ng,mods.size()));
    std::vector<std::thread> ths; ths.reserve(nw);
    for (int d=0; d<nw; ++d) ths.emplace_back([&,d] {
        for (;;) {
            size_t q=next.fetch_add(1,std::memory_order_relaxed);
            if (q>=mods.size()) break;
            auto md=load_or_build_mod(mods[q],validate,d);
            std::cerr << "row8 prebuild ready mod=" << mods[q] << " device=" << d
                      << " build_s=" << md.build_s << "\n";
        }
    });
    for (auto& t:ths) t.join();
    double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::cerr << "row8 prebuild complete mods=" << mods.size() << " gpus=" << nw
              << " wall_s=" << sec << "\n";
    return sec;
}
} // namespace oneesan::row8runtime
