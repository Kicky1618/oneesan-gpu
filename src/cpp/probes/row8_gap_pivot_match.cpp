#define ROW8_STRUCTURAL_UPPER_NO_MAIN 1
#include "row8_structural_upper_sparsity.cpp"
#include <array>
#include <iostream>
#include <memory>
#include <set>

static bool is_gap_canonical8(State const& s) {
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

int main(){
    MODP=4294967291u;
    Vec all; int col=0;
    if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all)) return 2;
    std::array<std::vector<Packed>,9> H;
    for(auto const&p:all) H[unpack(p).sp].push_back(p);
    std::array<std::unique_ptr<Space>,9> S;
    S[0]=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));
    S[1]=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));
    S[2]=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));
    for(int h=3;h<=8;++h) S[h]=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));

    bool global=true;
    for(int h=0;h<=8;++h){
        std::set<Packed> piv,gap;
        for(auto const&b:S[h]->blocks) for(int r=0;r<(int)b.rr.size();++r) piv.insert(S[h]->states[b.idx[b.piv[r]]]);
        for(auto const&p:H[h]) if(is_gap_canonical8(unpack(p))) gap.insert(p);
        size_t missing=0,extra=0;
        for(auto const&p:gap) if(!piv.count(p)) ++missing;
        for(auto const&p:piv) if(!gap.count(p)) ++extra;
        bool ok=piv==gap;
        global &= ok;
        std::cout<<"h="<<h<<" dim="<<S[h]->dim<<" piv="<<piv.size()<<" gap="<<gap.size()
                 <<" gap_not_pivot="<<missing<<" pivot_not_gap="<<extra<<" exact="<<ok<<"\n";
    }
    std::cout<<"gap_pivot_match_exact="<<global<<"\n";
    return global?0:1;
}
