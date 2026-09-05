#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct SHdr {char magic[8];uint32_t version,r;uint32_t dims[9];uint64_t total_nz,fnv_hash;};
struct BHdr {uint32_t sym,h,h2,rows,cols,nnz;};
struct VHdr {uint32_t tag,sym,h,nnz;};
struct SparseVec {int h=-1;std::vector<std::pair<uint16_t,int8_t>> nz;};
struct Block {int h2=-1,rows=0,cols=0;std::vector<uint32_t> rp;std::vector<uint16_t> ci;std::vector<int8_t> cv;};
struct Cache {SHdr hdr{};std::array<SparseVec,3> alpha;std::array<SparseVec,3> beta;std::array<std::array<Block,9>,3> q;};

template<class T> static T get(std::ifstream&in){T x{};in.read((char*)&x,sizeof(x));if(!in)throw std::runtime_error("short cache");return x;}
static Cache load(std::string const&path){std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("open cache");Cache c;c.hdr=get<SHdr>(in);if(std::string(c.hdr.magic,7)!="R8STR01"||c.hdr.version!=1||c.hdr.r!=8)throw std::runtime_error("cache header");for(int z=0;z<5;++z){auto h=get<VHdr>(in);auto&v=h.tag==1?c.alpha[h.sym]:c.beta[h.sym];v.h=h.h;v.nz.resize(h.nnz);for(auto&[i,a]:v.nz){i=get<uint16_t>(in);a=get<int8_t>(in);}}for(int z=0;z<25;++z){auto h=get<BHdr>(in);auto&b=c.q[h.sym][h.h];b.h2=h.h2;b.rows=h.rows;b.cols=h.cols;b.rp.resize(h.rows+1);b.ci.resize(h.nnz);b.cv.resize(h.nnz);in.read((char*)b.rp.data(),b.rp.size()*4);in.read((char*)b.ci.data(),b.ci.size()*2);in.read((char*)b.cv.data(),b.cv.size());if(!in)throw std::runtime_error("cache block");}return c;}
static uint32_t valmod(int8_t x,uint32_t p){int v=x;if(v>=0)return (uint32_t)v;return p-(uint32_t)(-v);}
static std::vector<uint32_t> vecmod(SparseVec const&v,int dim,uint32_t p){std::vector<uint32_t>x(dim);for(auto[i,a]:v.nz)x[i]=valmod(a,p);return x;}
static std::vector<uint32_t> step(std::vector<uint32_t>const&x,Block const&b,uint32_t p){std::vector<uint32_t>y(b.cols);for(int i=0;i<b.rows;++i)if(x[i])for(uint32_t e=b.rp[i];e<b.rp[i+1];++e){uint32_t a=valmod(b.cv[e],p);y[b.ci[e]]=(y[b.ci[e]]+(__uint128_t)x[i]*a)%p;}return y;}
static uint32_t dot(std::vector<uint32_t>const&x,SparseVec const&b,uint32_t p){uint64_t z=0;for(auto[i,a]:b.nz){z=(z+(__uint128_t)x[i]*valmod(a,p))%p;}return z;}
static std::array<int,9> d9(uint32_t x){std::array<int,9>d{};for(int i=8;i>=0;--i){d[i]=x%3;x/=3;}return d;}
static std::array<int,10>d10(uint32_t x){std::array<int,10>d{};for(int i=9;i>=0;--i){d[i]=x%3;x/=3;}return d;}
static uint32_t eval(Cache const&c,uint32_t mod,uint32_t pc,uint32_t sc){auto p=d9(pc);auto s=d10(sc);int h=c.alpha[p[0]].h;if(h<0)return 0;auto x=vecmod(c.alpha[p[0]],c.hdr.dims[h],mod);for(int i=1;i<9;++i){auto const&b=c.q[p[i]][h];if(b.h2<0)return 0;x=step(x,b,mod);h=b.h2;}for(int i=0;i<9;++i){auto const&b=c.q[s[i]][h];if(b.h2<0)return 0;x=step(x,b,mod);h=b.h2;}auto const&bt=c.beta[s[9]];if(bt.h!=h)return 0;return dot(x,bt,mod);}
int main(int ac,char**av){std::string path="work/row8_structural_cache/row8_structural_int_v1.bin";uint32_t mod=1000000007u,pc=1,sc=0;if(ac>1)mod=std::stoul(av[1]);if(ac>2)pc=std::stoul(av[2]);if(ac>3)sc=std::stoul(av[3]);auto c=load(path);auto x=eval(c,mod,pc,sc);std::cout<<"mod="<<mod<<" pc="<<pc<<" sc="<<sc<<" value="<<x<<" cache_nz="<<c.hdr.total_nz<<"\n";}
