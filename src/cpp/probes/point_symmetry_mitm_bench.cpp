#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;
static double sec(Clock::time_point a, Clock::time_point b){return std::chrono::duration<double>(b-a).count();}
struct Rec{uint32_t mask,rmask;int k;std::string top,bot;uint64_t val;};
static std::vector<Mate> code_states(MateCodec const&mc){std::vector<Mate>o(mc.codeSize());for(Code bi=0;bi<mc.codeSizeL();++bi){auto const&b=mc.codeTable(bi);for(Code i=0;i<b.size;++i)o[b.base+i]=b.mateL|b.mateR[i];}return o;}
static uint32_t occ(Mate m,int W){uint32_t z=0;for(int p=0;p<W;++p)if(m.get(p)!=N)z|=1u<<p;return z;}
static uint32_t revbits(uint32_t x,int W){uint32_t z=0;for(int p=0;p<W;++p)if(x>>p&1)z|=1u<<(W-1-p);return z;}
static std::vector<std::pair<int,int>> raw_pairing(Mate m,int W){std::vector<int>st{W};std::vector<std::pair<int,int>>e;for(int p=W-1;p>=0;--p){auto v=m.get(p);if(v==L)st.push_back(p);else if(v==R){int q=st.back();st.pop_back();e.push_back({q,p});}}return e;}
static std::string sig_top(Mate m,int W){uint32_t om=occ(m,W);std::vector<int>idx(W,-1);int n=1;for(int p=W-1;p>=0;--p)if(om>>p&1)idx[p]=n++;std::string s(n,'\0');for(auto [u,v]:raw_pairing(m,W)){int a=u==W?0:idx[u],b=v==W?0:idx[v];s[a]=char(b);s[b]=char(a);}return s;}
static std::string sig_bottom_reflected(Mate m,int W){uint32_t om=revbits(occ(m,W),W);std::vector<int>idx(W,-1);int n=1;for(int p=W-1;p>=0;--p)if(om>>p&1)idx[p]=n++;std::string s(n,'\0');for(auto [u,v]:raw_pairing(m,W)){int ru=u==W?-1:W-1-u,rv=v==W?-1:W-1-v;int a=ru<0?0:idx[ru],b=rv<0?0:idx[rv];s[a]=char(b);s[b]=char(a);}return s;}
static bool compat(std::string const&a,std::string const&b){int n=a.size();std::vector<uint8_t>seen(n);std::vector<int>st{0};seen[0]=1;int got=0;while(!st.empty()){int u=st.back();st.pop_back();++got;int vs[2]={(unsigned char)a[u],(unsigned char)b[u]};for(int v:vs)if(!seen[v]){seen[v]=1;st.push_back(v);}}return got==n;}
static void init(PathCounter<Modnum<uint64_t>>&pc){for(Code i=0;i<pc.mc.codeSize();++i)pc.value[i]=0;for(Code i=0;i<pc.wc.codeSize();++i)pc.deferred[i]=0;pc.value[pc.mc.encode(Mate(pc.cols-1,R))]=1;}
static void run_rows(PathCounter<Modnum<uint64_t>>&pc,int nr){for(int i=0;i<nr;++i){for(int j=0;j<pc.cols-2;++j)pc.update(j,false);pc.update(pc.cols-2,false);}}
static uint64_t addm(uint64_t a,uint64_t b){a+=b;if(a>=modulus||a<b)a-=modulus;return a;}
static uint64_t mulm(uint64_t a,uint64_t b){return (__uint128_t)a*b%modulus;}

int main(int argc,char**argv){
    msg=NONE; modulus=4294967291ULL; int W=argc>1?std::atoi(argv[1]):16;
    auto t0=Clock::now(); PathCounter<Modnum<uint64_t>> half(W,W,false,false); auto t1=Clock::now();
    init(half); run_rows(half,W/2); auto t2=Clock::now();
    auto states=code_states(half.mc); std::vector<Rec> recs; recs.reserve(states.size());
    std::vector<std::unordered_map<std::string,int>> ids(W+1); std::vector<std::vector<std::string>> pats(W+1);
    uint64_t nonzero=0;
    for(Code i=0;i<half.mc.codeSize();++i){uint64_t v=half.value[i];if(!v)continue;++nonzero;Mate m=states[i];uint32_t mask=occ(m,W);int k=__builtin_popcount(mask);auto a=sig_top(m,W),b=sig_bottom_reflected(m,W);for(auto const&s:{a,b})if(!ids[k].count(s)){int id=pats[k].size();ids[k][s]=id;pats[k].push_back(s);}recs.push_back({mask,revbits(mask,W),k,a,b,v});}
    struct G{std::vector<uint64_t>x,y;}; std::unordered_map<uint32_t,G> groups;
    for(auto const&r:recs){auto&gx=groups[r.mask];if(gx.x.empty()){gx.x.assign(pats[r.k].size(),0);gx.y.assign(pats[r.k].size(),0);}gx.x[ids[r.k][r.top]]=addm(gx.x[ids[r.k][r.top]],r.val);auto&gy=groups[r.rmask];if(gy.x.empty()){gy.x.assign(pats[r.k].size(),0);gy.y.assign(pats[r.k].size(),0);}gy.y[ids[r.k][r.bot]]=addm(gy.y[ids[r.k][r.bot]],r.val);} auto t3=Clock::now();
    std::vector<std::vector<std::vector<int>>> adj(W+1); uint64_t medges=0;
    for(int k=1;k<=W;k+=2){int n=pats[k].size();adj[k].resize(n);for(int i=0;i<n;++i)for(int j=0;j<n;++j)if(compat(pats[k][i],pats[k][j])){adj[k][i].push_back(j);++medges;}} auto t4=Clock::now();
    uint64_t ans=0,used=0;
    for(auto const&kv:groups){int k=__builtin_popcount(kv.first);auto const&g=kv.second;for(size_t i=0;i<g.x.size();++i)if(g.x[i])for(int j:adj[k][i])if(g.y[j]){ans=addm(ans,mulm(g.x[i],g.y[j]));++used;}} auto t5=Clock::now();
    PathCounter<Modnum<uint64_t>> full(W,W,false,false); auto t6=Clock::now(); uint64_t exact=full.count(); auto t7=Clock::now();
    std::cout<<"W="<<W<<" states="<<half.mc.codeSize()<<" nonzero_half="<<nonzero<<" masks="<<groups.size()<<" meander_edges="<<medges<<" used_edges="<<used<<"\n";
    std::cout<<"half_ctor_s="<<sec(t0,t1)<<" half_dp_s="<<sec(t1,t2)<<" encode_s="<<sec(t2,t3)<<" compat_build_s="<<sec(t3,t4)<<" join_s="<<sec(t4,t5)<<" full_ctor_s="<<sec(t5,t6)<<" full_dp_s="<<sec(t6,t7)<<"\n";
    std::cout<<"mitm="<<ans<<" full="<<exact<<" "<<(ans==exact?"OK":"MISMATCH")<<"\n";
}
