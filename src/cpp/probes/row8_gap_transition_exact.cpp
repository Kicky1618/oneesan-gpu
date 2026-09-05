#define ROW8_STRUCTURAL_UPPER_NO_MAIN 1
#include "row8_structural_upper_sparsity.cpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

static bool gap8(State const& s) {
    constexpr int R=8;
    int h=s.sp;
    std::array<int,MAXC> nf{},ns{},lo{},hi{};
    lo.fill(99); hi.fill(-1);
    for(int i=0;i<R;++i) if(s.comp[i]) {int q=s.comp[i];++nf[q];lo[q]=std::min(lo[q],i);hi[q]=std::max(hi[q],i);}
    for(int i=0;i<h;++i) if(s.stack[i]) ++ns[s.stack[i]];
    for(int q=1;q<s.ns;++q) {
        if(ns[q]) { if(ns[q]!=1||nf[q]!=1||(s.status[q]&1)) return false; }
        else if(nf[q]&&nf[q]!=2) return false;
    }
    for(int a=1;a<s.ns;++a) if(!ns[a]&&nf[a]==2)
        for(int b=1;b<s.ns;++b) if(!ns[b]&&nf[b]==2)
            if(lo[a]<lo[b]&&hi[b]<hi[a]&&((s.status[a]^s.status[b])&1)) return false;
    return true;
}

static std::vector<std::vector<uint32_t>> inverse_square(std::vector<std::vector<uint32_t>> A) {
    const int n=A.size();
    std::vector<std::vector<uint32_t>> B(n,std::vector<uint32_t>(n));
    for(int i=0;i<n;++i) B[i][i]=1;
    for(int c=0;c<n;++c){
        int p=c; while(p<n&&!A[p][c])++p;
        if(p==n) throw std::runtime_error("singular gap block");
        std::swap(A[p],A[c]); std::swap(B[p],B[c]);
        uint32_t iv=invq(A[c][c]);
        for(int j=0;j<n;++j){A[c][j]=(uint64_t)A[c][j]*iv%Q;B[c][j]=(uint64_t)B[c][j]*iv%Q;}
        for(int i=0;i<n;++i) if(i!=c&&A[i][c]){
            uint32_t f=A[i][c];
            for(int j=0;j<n;++j){
                A[i][j]=(A[i][j]+Q-(uint64_t)f*A[c][j]%Q)%Q;
                B[i][j]=(B[i][j]+Q-(uint64_t)f*B[c][j]%Q)%Q;
            }
        }
    }
    return B;
}

struct GBlock {
    int off=0;
    std::vector<int> gap_local;       // gap coordinate -> raw local column
    std::vector<std::vector<uint32_t>> inv; // gap coordinate x quotient row
};
struct GSpace {
    Space* S=nullptr;
    std::vector<GBlock> b;
    std::vector<int> raw_gap_global;  // raw state index -> gap global index, -1 if non-gap
};

static GSpace make_gspace(Space& S){
    GSpace G; G.S=&S; G.b.resize(S.blocks.size()); G.raw_gap_global.assign(S.states.size(),-1);
    int off=0;
    for(int bi=0;bi<(int)S.blocks.size();++bi){
        auto const& b=S.blocks[bi]; auto& gb=G.b[bi]; gb.off=off;
        for(int c=0;c<(int)b.idx.size();++c) if(gap8(unpack(S.states[b.idx[c]]))) gb.gap_local.push_back(c);
        if(gb.gap_local.size()!=b.rr.size()) throw std::runtime_error("gap block dimension mismatch");
        int n=b.rr.size(); std::vector<std::vector<uint32_t>> M(n,std::vector<uint32_t>(n));
        for(int r=0;r<n;++r) for(int j=0;j<n;++j) M[r][j]=b.rr[r][gb.gap_local[j]];
        gb.inv=inverse_square(std::move(M));
        for(int j=0;j<n;++j) G.raw_gap_global[b.idx[gb.gap_local[j]]]=off+j;
        off+=n;
    }
    if(off!=S.dim) throw std::runtime_error("gap total mismatch");
    return G;
}

static std::vector<std::pair<int,uint32_t>> project_raw_combo(
    GSpace const& G, WVec const& z) {
    // Accumulate existing quotient coordinates separately in each occupancy block.
    std::map<int,std::vector<uint32_t>> q;
    for(auto const&e:z){
        auto it=std::lower_bound(G.S->states.begin(),G.S->states.end(),e.p);
        if(it==G.S->states.end()||!(*it==e.p)) throw std::runtime_error("target raw state missing");
        int si=it-G.S->states.begin(); int bi=G.S->stateBlock[si], lc=G.S->stateLocal[si];
        auto const& b=G.S->blocks[bi];
        auto [jt,ins]=q.try_emplace(bi,b.rr.size(),0);
        auto& v=jt->second;
        for(int r=0;r<(int)b.rr.size();++r) if(b.rr[r][lc])
            v[r]=(v[r]+(uint64_t)e.v*b.rr[r][lc])%Q;
    }
    std::vector<std::pair<int,uint32_t>> out;
    for(auto& [bi,v]:q){
        auto const& gb=G.b[bi]; int n=v.size();
        for(int j=0;j<n;++j){
            uint64_t sum=0;
            // Reduce periodically to avoid overflow, though n is small for r=8 blocks.
            for(int r=0;r<n;++r) if(gb.inv[j][r]&&v[r]) sum=(sum+(uint64_t)gb.inv[j][r]*v[r])%Q;
            uint32_t x=sum%Q; if(x) out.push_back({gb.off+j,x});
        }
    }
    std::sort(out.begin(),out.end());
    return out;
}

#ifndef ROW8_GAP_TRANSITION_NO_MAIN
int main(){
    MODP=Q;
    Vec all; int col=0;
    if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all)) return 2;
    std::array<std::vector<Packed>,9> H;
    for(auto const&p:all) H[unpack(p).sp].push_back(p);
    std::array<std::unique_ptr<Space>,9> S;
    S[0]=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));
    S[1]=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));
    S[2]=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));
    for(int h=3;h<=8;++h) S[h]=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));
    std::array<std::unique_ptr<GSpace>,9> G;
    for(int h=0;h<=8;++h){G[h]=std::make_unique<GSpace>(make_gspace(*S[h]));std::cout<<"space h="<<h<<" dim="<<S[h]->dim<<"\n";}

    static constexpr int DEL[3]={0,-1,1};
    uint64_t grand_nnz=0, nonunit=0; int global_maxout=0;
    for(int a=0;a<3;++a){
        uint64_t annz=0; int amax=0;
        for(int h=0;h<=8;++h){
            int h2=h+DEL[a]; if(h2<0||h2>8) continue;
            uint64_t nnz=0; int maxout=0; std::map<int,uint64_t> hist;
            // Iterate gap basis raw states in global gap order.
            std::vector<std::pair<int,int>> src; // (gap global, raw state index)
            for(int si=0;si<(int)S[h]->states.size();++si) if(G[h]->raw_gap_global[si]>=0)
                src.push_back({G[h]->raw_gap_global[si],si});
            std::sort(src.begin(),src.end());
            for(auto [gi,si]:src){
                WVec v{{S[h]->states[si],1}};
                auto z=wcolumn(std::move(v),8,false,a);
                auto o=project_raw_combo(*G[h2],z);
                int od=o.size(); maxout=std::max(maxout,od); nnz+=od; ++hist[od];
                for(auto [j,x]:o) if(x!=1) ++nonunit;
            }
            std::cout<<"TRANS a="<<a<<" h="<<h<<" h2="<<h2<<" src="<<S[h]->dim
                     <<" dst="<<S[h2]->dim<<" nnz="<<nnz<<" maxout="<<maxout<<" hist";
            for(auto[k,c]:hist) std::cout<<' '<<k<<':'<<c;
            std::cout<<"\n";
            annz+=nnz; amax=std::max(amax,maxout);
        }
        std::cout<<"SYMBOL a="<<a<<" nnz="<<annz<<" maxout="<<amax<<"\n";
        grand_nnz+=annz; global_maxout=std::max(global_maxout,amax);
    }
    std::cout<<"TOTAL nnz="<<grand_nnz<<" nonunit="<<nonunit<<" maxout="<<global_maxout<<"\n";
    return nonunit?1:0;
}
#endif
