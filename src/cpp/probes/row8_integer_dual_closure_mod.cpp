#define ROW8_H2_PARTITION_NO_MAIN 1
#include "row8_h2_relation_partition.cpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <omp.h>
using i64=int64_t; using Edge=std::pair<int,uint32_t>; using Trans=std::vector<std::vector<Edge>>;
static uint32_t P;
static uint32_t invp(uint32_t a){uint64_t e=(uint64_t)P-2,x=a,r=1;while(e){if(e&1)r=(__uint128_t)r*x%P;x=(__uint128_t)x*x%P;e>>=1;}return (uint32_t)r;}
static uint32_t mod_i64(i64 x){i64 q=x%(i64)P;if(q<0)q+=P;return (uint32_t)q;}
struct RR{int p;std::vector<std::pair<int,uint32_t>>z;};
struct Red{int n;std::vector<int>at;std::vector<RR>b;Red(int n):n(n),at(n,-1){}bool add(std::vector<uint32_t>y){reduce(y);int p=-1;for(int i=0;i<n;++i)if(y[i]){p=i;break;}if(p<0)return false;uint32_t iv=invp(y[p]);RR r{p,{}};for(int i=p;i<n;++i)if(y[i])r.z.push_back({i,(uint32_t)((__uint128_t)y[i]*iv%P)});at[p]=b.size();b.push_back(std::move(r));return true;}void reduce(std::vector<uint32_t>&y)const{for(int p=0;p<n;++p)if(y[p]&&at[p]>=0){uint32_t f=y[p];for(auto[k,v]:b[at[p]].z){uint32_t s=(uint32_t)((__uint128_t)f*v%P);y[k]=y[k]>=s?y[k]-s:y[k]+P-s;}}}bool inSpan(std::vector<uint32_t>y)const{reduce(y);for(auto x:y)if(x)return false;return true;}};
struct IH{char m[8];uint32_t ver,h,rows,states;uint64_t maxabs;};
struct DenseBasis{int h,rows,states;std::vector<i64>x;std::vector<uint32_t>m;Red red;DenseBasis(std::string path):red(1){std::ifstream in(path,std::ios::binary);IH q{};in.read((char*)&q,sizeof(q));if(!in)throw std::runtime_error("dense hdr");h=q.h;rows=q.rows;states=q.states;x.resize((size_t)rows*states);in.read((char*)x.data(),x.size()*8);if(!in)throw std::runtime_error("dense read");m.resize(x.size());for(size_t i=0;i<x.size();++i)m[i]=mod_i64(x[i]);red=Red(states);for(int r=0;r<rows;++r){std::vector<uint32_t>v(m.begin()+(size_t)r*states,m.begin()+(size_t)(r+1)*states);if(!red.add(std::move(v)))throw std::runtime_error("basis rank drop h="+std::to_string(h));}}};
struct Recipe{int kind,source,parent,depth;};
static std::vector<Recipe> load_recipe(std::string path,int rows){std::ifstream in(path);std::vector<Recipe>r(rows);for(int i=0;i<rows;++i){int id;if(!(in>>id>>r[i].kind>>r[i].source>>r[i].parent>>r[i].depth)||id!=i)throw std::runtime_error("recipe");}return r;}
static Trans make_trans(std::vector<Packed>const&s,std::vector<Packed>const&t,int sym){Trans T(s.size());size_t ne=0;for(int i=0;i<(int)s.size();++i){WVec v{{s[i],1}};auto z=wcolumn(std::move(v),8,false,sym);for(auto&e:z){auto it=std::lower_bound(t.begin(),t.end(),e.p);if(it==t.end()||!(*it==e.p))throw std::runtime_error("target missing");T[i].push_back({int(it-t.begin()),e.v});++ne;}}std::cerr<<"T sym="<<sym<<" "<<s.size()<<"->"<<t.size()<<" edges="<<ne<<"\n";return T;}
static std::vector<std::vector<Edge>> incoming(Trans const&T,int nt){std::vector<std::vector<Edge>>I(nt);for(int s=0;s<(int)T.size();++s)for(auto[t,c]:T[s])I[t].push_back({s,c});return I;}
struct SF{std::vector<std::pair<int,i64>>z;};
static std::vector<SF> phi2_sparse(std::vector<Packed>const&h2,std::vector<int>&goodPos,std::vector<int>&non,Red&extraRed){auto ws=words2();std::set<int>bad;{std::ifstream q("work/formal-probes/canonical-matrix/h2_actual_bad_indices.txt");int x;while(q>>x)bad.insert(x);}std::vector<SF>F(1428);std::vector<char>good(h2.size());int r=0;for(int j=0;j<(int)ws.size();++j)if(!bad.count(j)){int i=std::lower_bound(h2.begin(),h2.end(),ws[j].first)-h2.begin();F[r++].z.push_back({i,1});good[i]=1;goodPos.push_back(i);}if(r!=1308)throw std::runtime_error("good1308");for(int i=0;i<(int)h2.size();++i)if(!good[i])non.push_back(i);std::vector<int>nid(h2.size(),-1);for(int k=0;k<(int)non.size();++k)nid[non[k]]=k;extraRed=Red(non.size());struct HH{char m[8];uint32_t ver,mod,h,rows,states;}hh{};std::ifstream in("work/formal-probes/dual-basis/Phi_h2_extra_mod1000000007.bin",std::ios::binary);in.read((char*)&hh,sizeof(hh));for(int q=0;q<120;++q){std::vector<uint32_t>row(h2.size()),er(non.size());in.read((char*)row.data(),row.size()*4);for(int i=0;i<(int)h2.size();++i)if(row[i]){i64 v=row[i]<=500000003u?(i64)row[i]:(i64)row[i]-1000000007LL;F[1308+q].z.push_back({i,v});if(nid[i]>=0)er[nid[i]]=mod_i64(v);}if(!extraRed.add(std::move(er)))throw std::runtime_error("phi2 extra rank drop");}return F;}
static bool h3hard(std::string const&w){int b=0,n=0;auto fl=[&](){bool z=n>=2;b=n=0;return z;};for(char c:w){if(c=='T'){if(fl())return true;continue;}if(c=='N')continue;int d=c=='U'?1:-1;if(b==0&&d<0)++n;b+=d;}return fl();}
static Packed h3mix(std::string const&w){State s{};s.n=8;int q=1,sk=0;std::array<int,4>z{};bool got=false;int st=0;while(st<8){int en=st;while(en<8&&w[en]!='T')++en;int b=0,ns=-1,c=0;for(int i=st;i<en;++i){if(w[i]=='N')continue;int d=w[i]=='U'?1:-1;if(b==0&&d<0)ns=i;b+=d;if(ns>=0&&b==0){if(c<2){z[2*c]=ns;z[2*c+1]=i;}++c;ns=-1;}}if(c>=2){got=true;break;}st=en+1;}if(!got)throw std::runtime_error("h3 mix");for(int i=0;i<8;++i)if(w[i]=='T'){int c=q++;s.deg[i]=1;s.comp[i]=c;s.stack[sk++]=c;}auto add=[&](int a,int b,int stt){int c=q++;s.deg[a]=s.deg[b]=1;s.comp[a]=s.comp[b]=c;s.status[c]=stt;};add(z[0],z[3],1);add(z[1],z[2],0);s.sp=sk;s.ns=q;return pack(s);}
static std::vector<SF> phi3_sparse(std::vector<Packed>const&h3){std::string w(8,'N');std::vector<std::pair<Packed,std::string>>ws;rw2(0,3,0,w,ws);std::sort(ws.begin(),ws.end(),[](auto&a,auto&b){return a.first<b.first;});ws.erase(std::unique(ws.begin(),ws.end(),[](auto&a,auto&b){return a.first==b.first;}),ws.end());if(ws.size()!=888)throw std::runtime_error("h3 words");std::vector<SF>F(888);int hp=0;for(int j=0;j<888;++j){int i=std::lower_bound(h3.begin(),h3.end(),ws[j].first)-h3.begin();F[j].z.push_back({i,1});if(h3hard(ws[j].second)){Packed m=h3mix(ws[j].second);int k=std::lower_bound(h3.begin(),h3.end(),m)-h3.begin();F[j].z.push_back({k,1});++hp;}}if(hp!=32)throw std::runtime_error("h3 hard32");return F;}
static std::vector<uint32_t> pull_sparse(std::vector<std::vector<Edge>>const&I,SF const&f,int ns){std::vector<uint32_t>y(ns);for(auto[t,v0]:f.z){uint32_t v=mod_i64(v0);for(auto[s,c]:I[t]){uint32_t a=(uint32_t)((uint64_t)c*v%P);uint32_t z=y[s]+a;if(z>=P||z<y[s])z-=P;y[s]=z;}}return y;}
static std::vector<uint32_t> pull_dense(Trans const&T,uint32_t const*f,int nt){std::vector<uint32_t>y(T.size());for(int i=0;i<(int)T.size();++i){uint64_t z=0;for(auto[t,c]:T[i])z+=(uint64_t)c*f[t];y[i]=z%P;}return y;}
static int check_dense_targets(std::string name,Trans const&T,DenseBasis const&target,Red const&src,std::vector<char>const&skip){std::atomic<int>bad{0};double t0=omp_get_wtime();
#pragma omp parallel for schedule(dynamic,8)
 for(int f=0;f<target.rows;++f){if(!skip.empty()&&skip[f])continue;auto y=pull_dense(T,target.m.data()+(size_t)f*target.states,target.states);if(!src.inSpan(std::move(y)))bad.fetch_add(1,std::memory_order_relaxed);}
 int checks=target.rows-(skip.empty()?0:std::count(skip.begin(),skip.end(),1));std::cout<<name<<" checks="<<checks<<" bad="<<bad.load()<<" sec="<<(omp_get_wtime()-t0)<<"\n";return bad.load();}
static int check_sparse_targets(std::string name,std::vector<std::vector<Edge>>const&I,std::vector<SF>const&F,Red const&src,std::vector<char>const&skip={}){std::atomic<int>bad{0};double t0=omp_get_wtime();
#pragma omp parallel for schedule(dynamic,16)
 for(int f=0;f<(int)F.size();++f){if(!skip.empty()&&skip[f])continue;auto y=pull_sparse(I,F[f],src.n);if(!src.inSpan(std::move(y)))bad.fetch_add(1,std::memory_order_relaxed);}
 int checks=F.size()-(skip.empty()?0:std::count(skip.begin(),skip.end(),1));std::cout<<name<<" checks="<<checks<<" bad="<<bad.load()<<" sec="<<(omp_get_wtime()-t0)<<"\n";return bad.load();}
static int check_phi2_source(std::string name,Trans const&T,DenseBasis const&target,std::vector<int>const&non,Red const&ex){std::atomic<int>bad{0};double t0=omp_get_wtime();
#pragma omp parallel for schedule(dynamic,8)
 for(int f=0;f<target.rows;++f){auto y=pull_dense(T,target.m.data()+(size_t)f*target.states,target.states);std::vector<uint32_t>yn(non.size());for(int k=0;k<(int)non.size();++k)yn[k]=y[non[k]];if(!ex.inSpan(std::move(yn)))bad.fetch_add(1,std::memory_order_relaxed);}
 std::cout<<name<<" checks="<<target.rows<<" bad="<<bad.load()<<" sec="<<(omp_get_wtime()-t0)<<"\n";return bad.load();}
static int check_phi2_source_sparse(std::string name,std::vector<std::vector<Edge>>const&I,std::vector<SF>const&F,std::vector<int>const&non,Red const&ex){std::atomic<int>bad{0};double t0=omp_get_wtime();
#pragma omp parallel for schedule(dynamic,16)
 for(int f=0;f<(int)F.size();++f){auto y=pull_sparse(I,F[f],I.empty()?0:0); // replaced below
  std::vector<uint32_t> full(I.size()?0:0);}
 return bad.load();}
int main(int argc,char**argv){P=argc>1?std::strtoul(argv[1],nullptr,10):4294967291u;MODP=4294967291u;omp_set_num_threads(std::max(1,omp_get_max_threads()));Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::array<std::vector<Packed>,4>H;for(auto&p:all){int h=unpack(p).sp;if(h<=3)H[h].push_back(p);}DenseBasis B0("work/formal-probes/dual-basis/Phi_h0_integer.bin"),B1("work/formal-probes/dual-basis/Phi_h1_integer.bin");std::vector<int>good2,non2;Red ex2(1);auto F2=phi2_sparse(H[2],good2,non2,ex2);auto F3=phi3_sparse(H[3]);auto R0=load_recipe("work/formal-probes/dual-basis/Phi_h0_integer.recipe",1107),R1=load_recipe("work/formal-probes/dual-basis/Phi_h1_integer.recipe",1640);
 std::vector<char>h0L(1640),h0N(1107),h1L(1428),h1N(1640);for(int i=0;i<1107;++i){if(R0[i].kind==0)h0L[R0[i].source]=1;else h0N[R0[i].parent]=1;}for(int i=0;i<1640;++i){if(R1[i].kind==0)h1L[R1[i].source]=1;else h1N[R1[i].parent]=1;}
 auto T00=make_trans(H[0],H[0],0),T01=make_trans(H[0],H[1],2),T11=make_trans(H[1],H[1],0),T10=make_trans(H[1],H[0],1),T12=make_trans(H[1],H[2],2),T22=make_trans(H[2],H[2],0),T21=make_trans(H[2],H[1],1),T23=make_trans(H[2],H[3],2);
 auto I12=incoming(T12,H[2].size()),I22=incoming(T22,H[2].size()),I23=incoming(T23,H[3].size());int bad=0;
 bad+=check_dense_targets("h0_N",T00,B0,B0.red,h0N);bad+=check_dense_targets("h0_L",T01,B1,B0.red,h0L);
 bad+=check_dense_targets("h1_N",T11,B1,B1.red,h1N);bad+=check_dense_targets("h1_R",T10,B0,B1.red,{});bad+=check_sparse_targets("h1_L",I12,F2,B1.red,h1L);
 // h2 source: good singleton coordinates are free; only the non-good restriction must lie in the 120-extra span.
 auto h2checkSparse=[&](std::string name,std::vector<std::vector<Edge>>const&I,std::vector<SF>const&F){std::atomic<int>b{0};double t0=omp_get_wtime();
#pragma omp parallel for schedule(dynamic,16)
   for(int f=0;f<(int)F.size();++f){std::vector<uint32_t>y(H[2].size());for(auto[t,v0]:F[f].z){uint32_t v=mod_i64(v0);for(auto[s,c]:I[t]){uint32_t a=(uint32_t)((uint64_t)c*v%P);uint64_t z=(uint64_t)y[s]+a;y[s]=z%P;}}std::vector<uint32_t>yn(non2.size());for(int k=0;k<(int)non2.size();++k)yn[k]=y[non2[k]];if(!ex2.inSpan(std::move(yn)))b.fetch_add(1,std::memory_order_relaxed);}
   std::cout<<name<<" checks="<<F.size()<<" bad="<<b.load()<<" sec="<<(omp_get_wtime()-t0)<<"\n";return b.load();};
 bad+=h2checkSparse("h2_N",I22,F2);bad+=check_phi2_source("h2_R",T21,B1,non2,ex2);bad+=h2checkSparse("h2_L",I23,F3);
 std::cout<<"integer_dual_closure_mod p="<<P<<" bad="<<bad<<" exact="<<(bad==0)<<"\n";return bad?1:0;}
