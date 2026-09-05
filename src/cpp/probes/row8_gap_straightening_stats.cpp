#define ROW8_STRUCTURAL_UPPER_NO_MAIN 1
#include "row8_structural_upper_sparsity.cpp"
#include <array>
#include <iostream>
#include <map>
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

static std::vector<std::vector<uint32_t>> inverse_square(std::vector<std::vector<uint32_t>> A){
    int n=A.size();
    std::vector<std::vector<uint32_t>> B(n,std::vector<uint32_t>(2*n));
    for(int i=0;i<n;++i){for(int j=0;j<n;++j)B[i][j]=A[i][j];B[i][n+i]=1;}
    for(int c=0;c<n;++c){
        int p=c;while(p<n&&!B[p][c])++p;if(p==n)throw std::runtime_error("singular");
        std::swap(B[p],B[c]);uint32_t iv=invq(B[c][c]);
        for(int j=c;j<2*n;++j)B[c][j]=(uint64_t)B[c][j]*iv%Q;
        for(int i=0;i<n;++i)if(i!=c&&B[i][c]){uint32_t f=B[i][c];for(int j=c;j<2*n;++j)B[i][j]=(B[i][j]+Q-(uint64_t)f*B[c][j]%Q)%Q;}
    }
    std::vector<std::vector<uint32_t>> I(n,std::vector<uint32_t>(n));
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)I[i][j]=B[i][n+j];
    return I;
}

static int64_t signedq(uint32_t x){return x<=Q/2?(int64_t)x:(int64_t)x-(int64_t)Q;}

int main(int argc,char**argv){
    int only=argc>1?std::atoi(argv[1]):-1;
    MODP=4294967291u; Vec all; int col=0;
    if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all)) return 2;
    std::array<std::vector<Packed>,9> H;for(auto const&p:all)H[unpack(p).sp].push_back(p);
    std::array<std::unique_ptr<Space>,9>S;
    S[0]=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));
    S[1]=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));
    S[2]=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));
    for(int h=3;h<=8;++h)S[h]=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));

    for(int h=0;h<=8;++h){if(only>=0&&h!=only)continue;
        size_t totalStates=0,totalGap=0,totalExprNz=0,maxExprNz=0,invNz=0,invSize=0;
        std::map<int64_t,size_t> hist;
        std::map<size_t,size_t> supportHist;
        for(auto const&b:S[h]->blocks){
            int d=b.rr.size(); if(!d)continue;
            std::vector<int> gapc;for(int c=0;c<(int)b.idx.size();++c)if(is_gap_canonical8(unpack(S[h]->states[b.idx[c]])))gapc.push_back(c);
            if((int)gapc.size()!=d)throw std::runtime_error("gap count");
            // G[row coordinate][gap basis column]
            std::vector<std::vector<uint32_t>>G(d,std::vector<uint32_t>(d));
            for(int r=0;r<d;++r)for(int j=0;j<d;++j)G[r][j]=b.rr[r][gapc[j]];
            auto GI=inverse_square(G); // gap coefficients = GI * coordinate vector
            for(auto const&row:GI)for(auto x:row){++invSize;if(x)++invNz;}
            totalGap+=d; totalStates+=b.idx.size();
            for(int c=0;c<(int)b.idx.size();++c){
                size_t nz=0;
                for(int j=0;j<d;++j){uint64_t z=0;for(int r=0;r<d;++r)z=(z+(uint64_t)GI[j][r]*b.rr[r][c])%Q;uint32_t x=z;if(x){++nz;++hist[signedq(x)];}}
                totalExprNz+=nz;maxExprNz=std::max(maxExprNz,nz);++supportHist[nz];
            }
        }
        std::cout<<"h="<<h<<" states="<<totalStates<<" gapdim="<<totalGap
                 <<" avg_expr_nz="<<(double)totalExprNz/totalStates<<" max_expr_nz="<<maxExprNz
                 <<" inv_density="<<(invSize?double(invNz)/invSize:0)<<" support";
        int shown=0;for(auto const&[k,v]:supportHist)if(shown++<16)std::cout<<' '<<k<<':'<<v;
        std::cout<<" coeff";shown=0;for(auto const&[k,v]:hist)if(shown++<24)std::cout<<' '<<k<<':'<<v;
        std::cout<<'\n';
    }
}
