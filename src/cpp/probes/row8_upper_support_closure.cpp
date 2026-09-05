#define ROW8_CANONICAL_TRIE_NO_MAIN 1
#include "row8_canonical_trie_verify.cpp"
#include <atomic>
#include <functional>
#include <iostream>
#include <set>
#include <unordered_map>
#include <omp.h>
struct PHU{size_t operator()(Packed const&p)const noexcept{uint64_t h=0;for(auto x:p.w)h^=x+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);return h;}};
static void gwr(int p,int nt,int bal,std::string&w,std::vector<std::pair<Packed,std::string>>&o){if(p==8){if(!nt&&!bal){Packed x;if(enc(w,x))o.push_back({x,w});}return;}if(nt>8-p)return;w[p]='N';gwr(p+1,nt,bal,w,o);w[p]='U';gwr(p+1,nt,bal+1,w,o);w[p]='D';gwr(p+1,nt,bal-1,w,o);if(nt&&bal==0){w[p]='T';gwr(p+1,nt-1,0,w,o);}}
static std::vector<std::pair<Packed,std::string>> wds(int h){std::string w(8,'N');std::vector<std::pair<Packed,std::string>>v;gwr(0,h,0,w,v);std::sort(v.begin(),v.end(),[](auto&a,auto&b){return a.first<b.first;});v.erase(std::unique(v.begin(),v.end(),[](auto&a,auto&b){return a.first==b.first;}),v.end());return v;}
static bool hardu(std::string const&w){int b=0,n=0;auto fl=[&](){bool z=n>=2;b=n=0;return z;};for(char c:w){if(c=='T'){if(fl())return true;continue;}if(c=='N')continue;int d=c=='U'?1:-1;if(b==0&&d<0)++n;b+=d;}return fl();}
static Packed mixu(std::string const&w){State s{};s.n=8;int q=1,sk=0;std::array<int,4>z{};bool got=false;int st=0;while(st<8){int en=st;while(en<8&&w[en]!='T')++en;int b=0,ns=-1,c=0;for(int i=st;i<en;++i){if(w[i]=='N')continue;int d=w[i]=='U'?1:-1;if(b==0&&d<0)ns=i;b+=d;if(ns>=0&&b==0){if(c<2){z[2*c]=ns;z[2*c+1]=i;}++c;ns=-1;}}if(c>=2){got=true;break;}st=en+1;}if(!got)throw std::runtime_error("mix");for(int i=0;i<8;++i)if(w[i]=='T'){int c=q++;s.deg[i]=1;s.comp[i]=c;s.stack[sk++]=c;}auto add=[&](int a,int b,int stt){int c=q++;s.deg[a]=s.deg[b]=1;s.comp[a]=s.comp[b]=c;s.status[c]=stt;};add(z[0],z[3],1);add(z[1],z[2],0);s.sp=sk;s.ns=q;return pack(s);}
struct Rel{std::unordered_map<Packed,int,PHU> coord;std::vector<std::array<Packed,2>>pairs;int dim=0;};
static Rel rel(int h){auto w=wds(h);Rel r;r.dim=w.size();for(int j=0;j<(int)w.size();++j){r.coord[w[j].first]=j;if((h==3||h==4)&&hardu(w[j].second)){Packed m=mixu(w[j].second);r.coord[m]=j;r.pairs.push_back({w[j].first,m});}}return r;}
static std::vector<std::pair<int,uint32_t>> prow(Packed p,int sym,Rel const&t){WVec v{{p,1}};auto z=wcolumn(std::move(v),8,false,sym);std::vector<std::pair<int,uint32_t>>r;for(auto&e:z){auto it=t.coord.find(e.p);if(it!=t.coord.end())r.push_back({it->second,e.v});}std::sort(r.begin(),r.end());std::vector<std::pair<int,uint32_t>>o;for(auto[q,c]:r){if(!o.empty()&&o.back().first==q)o.back().second+=c;else o.push_back({q,c});}return o;}
int main(int ac,char**av){MODP=1000000007u;int h=ac>1?atoi(av[1]):4,a=ac>2?atoi(av[2]):0;int dh[3]={0,-1,1};int h2=h+dh[a];if(h<3||h>8||h2<3||h2>8)return 2;Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::vector<Packed>src;for(auto&p:all)if(unpack(p).sp==h)src.push_back(p);auto S=rel(h),T=rel(h2);std::atomic<size_t> outside{0};double t0=omp_get_wtime();
#pragma omp parallel for schedule(dynamic,256)
 for(long long i=0;i<(long long)src.size();++i){if(S.coord.find(src[i])!=S.coord.end())continue;WVec v{{src[i],1}};auto z=wcolumn(std::move(v),8,false,a);bool hit=false;for(auto&e:z)if(T.coord.find(e.p)!=T.coord.end()){hit=true;break;}if(hit)outside.fetch_add(1,std::memory_order_relaxed);}
 size_t pbad=0;for(auto const&p:S.pairs){auto x=prow(p[0],a,T),y=prow(p[1],a,T);if(x!=y){if(pbad<5)std::cerr<<"pair mismatch sizes="<<x.size()<<','<<y.size()<<"\n";++pbad;}}
 double sec=omp_get_wtime()-t0;std::cout<<"h="<<h<<" sym="<<a<<" h2="<<h2<<" raw="<<src.size()<<" src_support="<<S.coord.size()<<" src_pairs="<<S.pairs.size()<<" target_support="<<T.coord.size()<<" outside_hit="<<outside.load()<<" pair_bad="<<pbad<<" sec="<<sec<<" exact="<<(outside.load()==0&&pbad==0)<<"\n";return outside.load()||pbad?1:0;}
