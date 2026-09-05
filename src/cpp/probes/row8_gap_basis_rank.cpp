#define ROW8_STRUCTURAL_UPPER_NO_MAIN 1
#include "row8_structural_upper_sparsity.cpp"
#include <array>
#include <iostream>
#include <memory>
#include <vector>

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

static int rank_mod(std::vector<std::vector<uint32_t>> a){
    int R=a.size(), C=R?int(a[0].size()):0, rk=0;
    for(int c=0;c<C&&rk<R;++c){
        int p=rk; while(p<R&&!a[p][c])++p;
        if(p==R) continue;
        std::swap(a[p],a[rk]);
        uint32_t iv=invq(a[rk][c]);
        for(int j=c;j<C;++j) a[rk][j]=(uint64_t)a[rk][j]*iv%Q;
        for(int i=rk+1;i<R;++i) if(a[i][c]){
            uint32_t f=a[i][c];
            for(int j=c;j<C;++j) a[i][j]=(a[i][j]+Q-(uint64_t)f*a[rk][j]%Q)%Q;
        }
        ++rk;
    }
    return rk;
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
        size_t gapTotal=0; int rankTotal=0; bool blockCounts=true, blockRanks=true;
        for(auto const&b:S[h]->blocks){
            std::vector<int> gc;
            for(int c=0;c<(int)b.idx.size();++c) if(is_gap_canonical8(unpack(S[h]->states[b.idx[c]]))) gc.push_back(c);
            gapTotal += gc.size();
            if(gc.size()!=b.rr.size()) blockCounts=false;
            std::vector<std::vector<uint32_t>> M(b.rr.size(),std::vector<uint32_t>(gc.size()));
            for(int r=0;r<(int)b.rr.size();++r) for(int j=0;j<(int)gc.size();++j) M[r][j]=b.rr[r][gc[j]];
            int rk=rank_mod(std::move(M)); rankTotal+=rk;
            if(rk!=(int)b.rr.size()) blockRanks=false;
        }
        bool ok=gapTotal==(size_t)S[h]->dim && rankTotal==S[h]->dim && blockCounts && blockRanks;
        global &= ok;
        std::cout<<"h="<<h<<" dim="<<S[h]->dim<<" gap="<<gapTotal<<" rank="<<rankTotal
                 <<" block_counts="<<blockCounts<<" block_fullrank="<<blockRanks<<" exact_basis="<<ok<<"\n";
    }
    std::cout<<"gap_basis_fullrank_exact="<<global<<"\n";
    return global?0:1;
}
