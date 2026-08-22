#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

static std::vector<Mate> code_states(MateCodec const& mc){
    std::vector<Mate> out(mc.codeSize());
    for(Code bi=0;bi<mc.codeSizeL();++bi){auto const& b=mc.codeTable(bi);for(Code i=0;i<b.size;++i)out[b.base+i]=b.mateL|b.mateR[i];}
    return out;
}

static uint32_t occ_mask(Mate m,int W){uint32_t z=0;for(int p=0;p<W;++p)if(m.get(p)!=N)z|=1u<<p;return z;}
static uint32_t reverse_bits(uint32_t x,int W){uint32_t z=0;for(int p=0;p<W;++p)if(x&(1u<<p))z|=1u<<(W-1-p);return z;}

// Convert a frontier Mate state to the noncrossing matching formed by its open
// path fragments.  Vertex W is the exterior defect (source/target).  Scan in
// GGCount's rank order: high column to low column, with an initial open defect.
static std::vector<std::pair<int,int>> pairing(Mate m,int W){
    std::vector<int> st{W};std::vector<std::pair<int,int>> e;
    for(int p=W-1;p>=0;--p){auto v=m.get(p);if(v==L)st.push_back(p);else if(v==R){if(st.empty())throw std::runtime_error("bad Mate");int q=st.back();st.pop_back();e.push_back({q,p});}}
    if(!st.empty())throw std::runtime_error("unclosed Mate");return e;
}

static bool compatible(Mate a,Mate b,int W){
    uint32_t oa=occ_mask(a,W), ob=reverse_bits(occ_mask(b,W),W);if(oa!=ob)return false;
    const int S=W,T=W+1,NV=W+2;std::vector<std::vector<int>> g(NV);
    auto ea=pairing(a,W);for(auto [u,v]:ea){g[u].push_back(v);g[v].push_back(u);} // ext=W is S
    auto eb=pairing(b,W);for(auto [u,v]:eb){if(u==W)u=T;else u=W-1-u;if(v==W)v=T;else v=W-1-v;g[u].push_back(v);g[v].push_back(u);}
    for(int p=0;p<W;++p){int want=(oa>>p)&1?2:0;if((int)g[p].size()!=want)return false;}
    if(g[S].size()!=1||g[T].size()!=1)return false;
    std::vector<uint8_t> seen(NV);std::vector<int> st{S};seen[S]=1;
    while(!st.empty()){int u=st.back();st.pop_back();for(int v:g[u])if(!seen[v]){seen[v]=1;st.push_back(v);}}
    if(!seen[T])return false;for(int p=0;p<W;++p)if((oa>>p)&1&&!seen[p])return false;return true;
}

static void init(PathCounter<uint64_t>&pc){for(Code i=0;i<pc.mc.codeSize();++i)pc.value[i]=0;for(Code i=0;i<pc.wc.codeSize();++i)pc.deferred[i]=0;pc.value[pc.mc.encode(Mate(pc.cols-1,R))]=1;}
static void run_rows(PathCounter<uint64_t>&pc,int nrows){for(int i=0;i<nrows;++i){for(int j=0;j<pc.cols-2;++j)pc.update(j,false);pc.update(pc.cols-2,false);}}

int main(int argc,char**argv){msg=NONE;int maxW=argc>1?std::atoi(argv[1]):8;for(int W=2;W<=maxW;++W){if(W&1)continue;PathCounter<uint64_t> half(W,W,false,false);init(half);run_rows(half,W/2);auto st=code_states(half.mc);uint64_t mitm=0,edges=0;for(Code i=0;i<half.mc.codeSize();++i){uint64_t a=half.value[i];if(!a)continue;for(Code j=0;j<half.mc.codeSize();++j){uint64_t b=half.value[j];if(!b)continue;if(compatible(st[i],st[j],W)){mitm+=a*b;++edges;}}}
 PathCounter<uint64_t> full(W,W,false,false);uint64_t exact=full.count();std::cout<<"W="<<W<<" states="<<half.mc.codeSize()<<" nonzero_matching_pairs="<<edges<<" mitm="<<mitm<<" full="<<exact<<" "<<(mitm==exact?"OK":"MISMATCH")<<"\n";}
}
