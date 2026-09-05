#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using u32=uint32_t; using u64=uint64_t;
static constexpr u32 MOD=1000000007u;
static constexpr int D[9]={1107,1640,1428,888,420,152,42,8,1};
static constexpr int DEL[3]={0,-1,1};
struct SH{char magic[8];u32 version,r,dims[9];u64 total,hash;};
struct BH{u32 sym,h,h2,rows,cols,nnz;};
struct VH{u32 tag,sym,h,nnz;};
struct Vec{int h=-1;std::vector<std::pair<uint16_t,int8_t>> z;};
struct Block{int h2=-1,rows=0,cols=0;std::vector<u32>rp;std::vector<uint16_t>ci;std::vector<int8_t>cv;};
struct Cache{std::array<Vec,3>a,b;std::array<std::array<Block,9>,3>q;};
template<class T>static T rd(std::ifstream&f){T x{};f.read((char*)&x,sizeof(x));if(!f)throw std::runtime_error("short read");return x;}
static Cache load(const std::string&p,bool gap){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("open "+p);auto sh=rd<SH>(f);Cache c;for(int z=0;z<5;++z){auto h=rd<VH>(f);auto &v=h.tag==1?c.a[h.sym]:c.b[h.sym];v.h=h.h;v.z.reserve(h.nnz);for(u32 i=0;i<h.nnz;++i){uint16_t j=rd<uint16_t>(f);int8_t a=1;if(!gap)a=rd<int8_t>(f);v.z.push_back({j,a});}}for(int z=0;z<25;++z){auto h=rd<BH>(f);auto &b=c.q[h.sym][h.h];b.h2=h.h2;b.rows=h.rows;b.cols=h.cols;b.rp.resize((size_t)h.rows+1);b.ci.resize(h.nnz);f.read((char*)b.rp.data(),b.rp.size()*4);f.read((char*)b.ci.data(),b.ci.size()*2);if(!gap){b.cv.resize(h.nnz);f.read((char*)b.cv.data(),b.cv.size());}if(!f)throw std::runtime_error("block short");}return c;}
static u32 cm(int a){long long x=a% (long long)MOD;if(x<0)x+=MOD;return (u32)x;}
struct StateVec{int h=-1;std::vector<u32>x;};
static StateVec seed(const Vec&v){StateVec s;s.h=v.h;s.x.assign(D[s.h],0);for(auto[j,a]:v.z)s.x[j]=cm(a);return s;}
static StateVec fwd(const StateVec&s,const Block&b){if((int)s.x.size()!=b.rows)throw std::runtime_error("fwd shape");StateVec t;t.h=b.h2;t.x.assign(b.cols,0);for(int i=0;i<b.rows;++i)if(s.x[i])for(u32 e=b.rp[i];e<b.rp[i+1];++e){u32 a=b.cv.empty()?1:cm(b.cv[e]);t.x[b.ci[e]]=(t.x[b.ci[e]]+(u64)s.x[i]*a)%MOD;}return t;}
static StateVec rev(const StateVec&s,const Block&b){if((int)s.x.size()!=b.cols)throw std::runtime_error("rev shape");StateVec t;t.h=-1;t.x.assign(b.rows,0);for(int i=0;i<b.rows;++i){u64 z=0;for(u32 e=b.rp[i];e<b.rp[i+1];++e){u32 a=b.cv.empty()?1:cm(b.cv[e]);z+=(u64)s.x[b.ci[e]]*a;z%=MOD;}t.x[i]=z;}return t;}
static u32 dot(const StateVec&a,const StateVec&b){if(a.x.size()!=b.x.size())throw std::runtime_error("dot shape");u64 z=0;for(size_t i=0;i<a.x.size();++i){z+=(u64)a.x[i]*b.x[i];z%=MOD;}return z;}
// Build prefix from symbols [0,p), suffix dual from symbols [p,n), and dot.
static bool eval_split(const Cache&c,const std::vector<int>&w,int p,u32&out){if(p<=0||p>=(int)w.size())return false;auto const&av=c.a[w[0]];if(av.h<0)return false;StateVec L=seed(av);for(int i=1;i<p;++i){auto const&b=c.q[w[i]][L.h];if(b.h2<0)return false;L=fwd(L,b);}auto const&bv=c.b[w.back()];if(bv.h<0)return false;StateVec R=seed(bv);for(int i=(int)w.size()-2;i>=p;--i){int hs=R.h-DEL[w[i]];if(hs<0||hs>8)return false;auto const&b=c.q[w[i]][hs];if(b.h2!=R.h)return false;R=rev(R,b);R.h=hs;}if(L.h!=R.h)return false;out=dot(L,R);return true;}
static std::string ws(const std::vector<int>&w){static char const*c="NRL";std::string s;for(int a:w)s+=c[a];return s;}
int main(int ac,char**av){int n=ac>1?std::atoi(av[1]):10;int trials=ac>2?std::atoi(av[2]):20000;auto S=load("work/row8_structural_cache/row8_structural_int_v1.bin",false),G=load("work/row8_gap_cache/row8_gap01.bin",true);std::mt19937_64 rng(0x8badf00dULL);size_t checked=0,skipped=0,bad=0;for(int t=0;t<trials;++t){std::vector<int>w(n);for(auto&a:w)a=rng()%3;int p=1+rng()%(n-1);u32 a=0,b=0;bool oa=eval_split(S,w,p,a),ob=eval_split(G,w,p,b);if(oa!=ob){if(bad<8)std::cerr<<"validity mismatch w="<<ws(w)<<" p="<<p<<" S="<<oa<<" G="<<ob<<"\n";++bad;continue;}if(!oa){++skipped;continue;}++checked;if(a!=b){if(bad<8)std::cerr<<"value mismatch w="<<ws(w)<<" p="<<p<<" S="<<a<<" G="<<b<<"\n";++bad;}}
std::cout<<"gap_mitm n="<<n<<" trials="<<trials<<" checked="<<checked<<" skipped="<<skipped<<" bad="<<bad<<" exact="<<(bad==0)<<"\n";return bad?1:0;}
