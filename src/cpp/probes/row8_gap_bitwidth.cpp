#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

struct FileHdr { char magic[8]; uint32_t version,r,dims[9]; uint64_t total_nnz; };
struct BlockHdr { uint32_t sym,h,h2,src_dim,dst_dim; uint64_t nnz; };
struct VecHdr { char magic[8]; uint32_t version,r,count; };
struct VecEntry { uint32_t tag,sym,h,nz; };
struct Block { int h2=-1,src=0,dst=0; std::vector<uint32_t> rp; std::vector<uint16_t> ci; };
struct BaseVec { int h=-1; std::vector<uint16_t> idx; };
static constexpr int DEL[3]={0,-1,1};
static constexpr uint32_t CAP=0xffffffffu;

static std::array<std::array<Block,9>,3> load_graph(std::array<int,9>&dims){
  std::ifstream in("work/row8_gap/row8_gap_u01_v1.bin",std::ios::binary); if(!in)throw std::runtime_error("gap graph open");
  FileHdr f{};in.read((char*)&f,sizeof(f));if(std::string(f.magic,7)!="GAP8U01"||f.version!=1||f.r!=8)throw std::runtime_error("gap graph header");for(int h=0;h<9;++h)dims[h]=f.dims[h];uint32_t nb=0;in.read((char*)&nb,4);std::array<std::array<Block,9>,3> g{};
  for(uint32_t q=0;q<nb;++q){BlockHdr b{};in.read((char*)&b,sizeof(b));auto&z=g[b.sym][b.h];z.h2=b.h2;z.src=b.src_dim;z.dst=b.dst_dim;z.rp.resize(z.src+1);z.ci.resize(b.nnz);in.read((char*)z.rp.data(),z.rp.size()*4);in.read((char*)z.ci.data(),z.ci.size()*2);if(!in)throw std::runtime_error("gap graph short");}
  return g;
}
static void load_vecs(std::array<BaseVec,3>&alpha,std::array<BaseVec,3>&beta){
  std::ifstream in("work/row8_gap/row8_gap_vectors_v1.bin",std::ios::binary);if(!in)throw std::runtime_error("gap vec open");VecHdr h{};in.read((char*)&h,sizeof(h));if(std::string(h.magic,7)!="GAP8VEC"||h.version!=1||h.r!=8)throw std::runtime_error("gap vec header");for(uint32_t q=0;q<h.count;++q){VecEntry e{};in.read((char*)&e,sizeof(e));auto&v=e.tag==1?alpha[e.sym]:beta[e.sym];v.h=e.h;v.idx.resize(e.nz);in.read((char*)v.idx.data(),v.idx.size()*2);}
}
struct Group{int rows=0,dim=0;std::vector<uint32_t>d;};
static Group base_group(BaseVec const&v,int dim){Group g;g.rows=1;g.dim=dim;g.d.assign(dim,0);for(auto i:v.idx)g.d[i]=1;return g;}
static inline uint32_t satadd(uint32_t a,uint32_t b){uint64_t s=(uint64_t)a+b;return s>CAP?CAP:(uint32_t)s;}
static void report(std::array<Group,9>const&g,int lev,const char*tag){uint32_t mx=0;uint64_t n=0,n8=0,n16=0,n24=0,n32=0;for(int h=0;h<9;++h)if(g[h].rows){for(auto x:g[h].d){mx=std::max(mx,x);n8+=x>=256;n16+=x>=65536;n24+=x>=(1u<<24);n32+=x==CAP;}n+=(uint64_t)g[h].rows*g[h].dim;}int bits=mx?32-__builtin_clz(mx):0;std::cout<<tag<<" lev="<<lev<<" max="<<mx<<" bits="<<bits<<" entries="<<n<<" ge8="<<n8<<" ge16="<<n16<<" ge24="<<n24<<" saturated32="<<n32;for(int h=0;h<9;++h)if(g[h].rows)std::cout<<" h"<<h<<"="<<g[h].rows<<"x"<<g[h].dim;std::cout<<"\n";}
static std::array<Group,9> step(std::array<Group,9>const&cur,std::array<std::array<Block,9>,3>const&G,bool suffix,std::array<int,9>const&dims){
  std::array<int,9>nr{};for(int h=0;h<9;++h)if(cur[h].rows)for(int a=0;a<3;++a){int h2=suffix?h-DEL[a]:h+DEL[a];if(0<=h2&&h2<9)nr[h2]+=cur[h].rows;}
  std::array<Group,9>nxt{};for(int h=0;h<9;++h){nxt[h].rows=nr[h];nxt[h].dim=dims[h];if(nr[h])nxt[h].d.assign((size_t)nr[h]*dims[h],0);}
  std::array<int,9>off{};
  for(int h=0;h<9;++h)if(cur[h].rows)for(int a=0;a<3;++a){int h2=suffix?h-DEL[a]:h+DEL[a];if(h2<0||h2>=9)continue;auto const&b=suffix?G[a][h2]:G[a][h];int rows=cur[h].rows;int outoff=off[h2];
#pragma omp parallel for schedule(static)
    for(int r=0;r<rows;++r){auto const*src=cur[h].d.data()+(size_t)r*cur[h].dim;auto*dst=nxt[h2].d.data()+(size_t)(outoff+r)*nxt[h2].dim;if(!suffix){for(int i=0;i<b.src;++i){uint32_t v=src[i];if(!v)continue;for(uint32_t e=b.rp[i];e<b.rp[i+1];++e){uint16_t j=b.ci[e];dst[j]=satadd(dst[j],v);}}}else{for(int i=0;i<b.src;++i){uint32_t s=0;for(uint32_t e=b.rp[i];e<b.rp[i+1];++e)s=satadd(s,src[b.ci[e]]);dst[i]=s;}}}
    off[h2]+=rows;
  }
  return nxt;
}
int main(int ac,char**av){int maxlev=ac>1?std::atoi(av[1]):11;std::array<int,9>dims{};auto G=load_graph(dims);std::array<BaseVec,3>alpha{},beta{};load_vecs(alpha,beta);std::array<Group,9>p{},s{};
  for(int a=0;a<3;++a)if(alpha[a].h>=0){int h=alpha[a].h;auto one=base_group(alpha[a],dims[h]);int old=p[h].rows++;p[h].dim=dims[h];p[h].d.resize((size_t)p[h].rows*dims[h]);std::copy(one.d.begin(),one.d.end(),p[h].d.begin()+(size_t)old*dims[h]);}
  for(int a=0;a<3;++a)if(beta[a].h>=0){int h=beta[a].h;auto one=base_group(beta[a],dims[h]);int old=s[h].rows++;s[h].dim=dims[h];s[h].d.resize((size_t)s[h].rows*dims[h]);std::copy(one.d.begin(),one.d.end(),s[h].d.begin()+(size_t)old*dims[h]);}
  report(p,1,"prefix");report(s,1,"suffix");for(int lev=2;lev<=maxlev;++lev){p=step(p,G,false,dims);report(p,lev,"prefix");s=step(s,G,true,dims);report(s,lev,"suffix");}
}
