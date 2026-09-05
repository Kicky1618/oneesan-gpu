#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
namespace oneesan::row7runtime {
using U128=unsigned __int128;
static constexpr int DIM=1778;
static constexpr int DIMS[8]={393,547,441,251,105,33,7,1};
static constexpr int DELTA[3]={0,-1,1};
struct U128LE{uint64_t lo,hi;};
struct Header{char magic[8];uint32_t version;uint32_t dims[8];uint64_t b_count;uint64_t v_count[3];uint64_t init_count,beta_count;};
struct ExactData{Header h{};std::vector<U128LE>B,V[3],init,beta;};
struct ModData{std::array<std::array<std::vector<uint32_t>,8>,3> trans;std::vector<uint32_t> alpha,beta;double build_s=0,inv_s=0,trans_s=0;};

struct ModCacheHeader{char magic[8];uint32_t version,mod;uint64_t exact_fp;uint32_t dims[8];};
inline uint64_t exact_fingerprint(const ExactData&e){uint64_t h=1469598103934665603ULL;auto mix=[&](const void*p,size_t n){auto*b=(const unsigned char*)p;for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ULL;}};mix(&e.h,sizeof(e.h));auto mv=[&](auto const&v){if(!v.empty())mix(v.data(),v.size()*sizeof(v[0]));};mv(e.B);for(auto const&v:e.V)mv(v);mv(e.init);mv(e.beta);return h;}
inline std::string default_cache_dir(){if(const char*e=std::getenv("GRIDFP_ROW7_MOD_CACHE_DIR"))return e;return "work/row7_mod_cache";}
inline std::string cache_path(uint32_t p,uint64_t fp,const std::string&dir=default_cache_dir()){char buf[128];std::snprintf(buf,sizeof(buf),"row7_mod_%u_%016llx.bin",p,(unsigned long long)fp);return (std::filesystem::path(dir)/buf).string();}
inline bool load_mod_cache(const std::string&path,uint32_t p,uint64_t fp,ModData&z){std::ifstream in(path,std::ios::binary);if(!in)return false;ModCacheHeader h{};if(!in.read((char*)&h,sizeof(h)))return false;if(std::string(h.magic,7)!="R7MODC1"||h.version!=1||h.mod!=p||h.exact_fp!=fp)return false;for(int q=0;q<8;++q)if((int)h.dims[q]!=DIMS[q])return false;auto rd=[&](std::vector<uint32_t>&v,size_t n){v.resize(n);return bool(in.read((char*)v.data(),v.size()*sizeof(v[0])));};for(int a=0;a<3;++a)for(int q=0;q<8;++q){int q2=q+DELTA[a];if(q2<0||q2>=8)continue;if(!rd(z.trans[a][q],size_t(DIMS[q])*DIMS[q2]))return false;}if(!rd(z.alpha,DIMS[1])||!rd(z.beta,DIMS[0]))return false;char extra;if(in.read(&extra,1))return false;z.build_s=z.inv_s=z.trans_s=0;return true;}
inline void save_mod_cache(const std::string&path,uint32_t p,uint64_t fp,const ModData&z){std::filesystem::path pp(path);std::filesystem::create_directories(pp.parent_path());std::string tmp=path+".tmp."+std::to_string((unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count());std::ofstream out(tmp,std::ios::binary|std::ios::trunc);if(!out)return;ModCacheHeader h{};std::memcpy(h.magic,"R7MODC1",7);h.version=1;h.mod=p;h.exact_fp=fp;for(int q=0;q<8;++q)h.dims[q]=DIMS[q];out.write((char*)&h,sizeof(h));auto wr=[&](auto const&v){out.write((const char*)v.data(),v.size()*sizeof(v[0]));};for(int a=0;a<3;++a)for(int q=0;q<8;++q){int q2=q+DELTA[a];if(q2<0||q2>=8)continue;wr(z.trans[a][q]);}wr(z.alpha);out.write((const char*)z.beta.data(),size_t(DIMS[0])*sizeof(uint32_t));out.close();if(out){std::error_code ec;std::filesystem::rename(tmp,path,ec);if(ec)std::filesystem::remove(tmp,ec);}else{std::error_code ec;std::filesystem::remove(tmp,ec);}}
inline U128 exact(U128LE x){return U128(x.lo)|(U128(x.hi)<<64);}
inline ExactData load_exact(const std::string&path){ExactData e;std::ifstream in(path,std::ios::binary);if(!in.read((char*)&e.h,sizeof(e.h)))throw std::runtime_error("row7 exact header read failed");if(std::string(e.h.magic,7)!="R7EXACT"||e.h.version!=1)throw std::runtime_error("row7 exact table format mismatch");for(int h=0;h<8;++h)if((int)e.h.dims[h]!=DIMS[h])throw std::runtime_error("row7 exact dimensions mismatch");auto rd=[&](auto&v,uint64_t n){v.resize(n);if(!in.read((char*)v.data(),v.size()*sizeof(v[0])))throw std::runtime_error("row7 exact data read failed");};rd(e.B,e.h.b_count);for(int a=0;a<3;++a)rd(e.V[a],e.h.v_count[a]);rd(e.init,e.h.init_count);rd(e.beta,e.h.beta_count);return e;}
inline uint32_t mod_pow(uint32_t a,uint32_t e,uint32_t p){uint64_t r=1,x=a;while(e){if(e&1)r=r*x%p;x=x*x%p;e>>=1;}return uint32_t(r);}
inline std::vector<uint32_t> invert(const std::vector<uint32_t>&b,int n,uint32_t p){std::vector<uint32_t>a((size_t)n*2*n);for(int i=0;i<n;++i){std::copy_n(&b[(size_t)i*n],n,&a[(size_t)i*2*n]);a[(size_t)i*2*n+n+i]=1;}for(int c=0;c<n;++c){int q=c;while(q<n&&!a[(size_t)q*2*n+c])++q;if(q==n)throw std::runtime_error("row7 basis singular for modulus "+std::to_string(p));if(q!=c)for(int j=0;j<2*n;++j)std::swap(a[(size_t)q*2*n+j],a[(size_t)c*2*n+j]);uint32_t iv=mod_pow(a[(size_t)c*2*n+c],p-2,p);for(int j=c;j<2*n;++j)a[(size_t)c*2*n+j]=uint64_t(a[(size_t)c*2*n+j])*iv%p;for(int i=0;i<n;++i)if(i!=c){uint32_t z=a[(size_t)i*2*n+c];if(!z)continue;for(int j=c;j<2*n;++j){uint32_t sub=uint64_t(z)*a[(size_t)c*2*n+j]%p;uint32_t&x=a[(size_t)i*2*n+j];x=x>=sub?x-sub:x+p-sub;}}}std::vector<uint32_t>inv((size_t)n*n);for(int i=0;i<n;++i)std::copy_n(&a[(size_t)i*2*n+n],n,&inv[(size_t)i*n]);return inv;}
inline std::vector<uint32_t> multiply(const std::vector<uint32_t>&a,int m,int k,const std::vector<uint32_t>&b,int n,uint32_t p){std::vector<uint32_t>c((size_t)m*n);for(int i=0;i<m;++i)for(int j=0;j<n;++j){U128 z=0;for(int q=0;q<k;++q)z+=U128(a[(size_t)i*k+q])*b[(size_t)q*n+j];c[(size_t)i*n+j]=uint32_t(z%p);}return c;}
inline ModData build_mod(const ExactData&e,uint32_t p){if(p<2)throw std::runtime_error("row7 modulus < 2");auto all0=std::chrono::steady_clock::now();std::array<std::vector<uint32_t>,8>b,inv;uint64_t bo=0;for(int h=0;h<8;++h){int n=DIMS[h];b[h].resize((size_t)n*n);for(size_t i=0;i<b[h].size();++i)b[h][i]=uint32_t(exact(e.B[bo+i])%p);bo+=b[h].size();}auto i0=std::chrono::steady_clock::now();for(int h=0;h<8;++h)inv[h]=invert(b[h],DIMS[h],p);auto i1=std::chrono::steady_clock::now();std::array<std::array<uint64_t,8>,3>vo{};for(int a=0;a<3;++a){uint64_t q=0;for(int h=0;h<8;++h){vo[a][h]=q;int h2=h+DELTA[a];if(0<=h2&&h2<8)q+=uint64_t(DIMS[h])*DIMS[h2];}}
 ModData z;for(int a=0;a<3;++a)for(int h=0;h<8;++h){int h2=h+DELTA[a];if(h2<0||h2>=8)continue;int m=DIMS[h],k=DIMS[h2];std::vector<uint32_t>v((size_t)m*k);uint64_t q=vo[a][h];for(size_t i=0;i<v.size();++i)v[i]=uint32_t(exact(e.V[a][q+i])%p);z.trans[a][h]=multiply(v,m,k,inv[h2],k,p);}auto t1=std::chrono::steady_clock::now();std::vector<uint32_t>iv(DIMS[1]);for(int i=0;i<DIMS[1];++i)iv[i]=uint32_t(exact(e.init[i])%p);z.alpha=multiply(iv,1,DIMS[1],inv[1],DIMS[1],p);z.beta.resize(e.beta.size());for(size_t i=0;i<z.beta.size();++i)z.beta[i]=uint32_t(exact(e.beta[i])%p);auto all1=std::chrono::steady_clock::now();z.inv_s=std::chrono::duration<double>(i1-i0).count();z.trans_s=std::chrono::duration<double>(t1-i1).count();z.build_s=std::chrono::duration<double>(all1-all0).count();return z;}
inline ModData load_or_build_mod(const ExactData&e,uint32_t p,bool*cache_hit=nullptr){uint64_t fp=exact_fingerprint(e);std::string path=cache_path(p,fp);ModData z;if(load_mod_cache(path,p,fp,z)){if(cache_hit)*cache_hit=true;return z;}if(cache_hit)*cache_hit=false;z=build_mod(e,p);save_mod_cache(path,p,fp,z);return z;}

}
