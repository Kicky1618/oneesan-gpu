#define ROW8_STRUCTURAL_UPPER_NO_MAIN 1
#include "row8_structural_upper_sparsity.cpp"
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

static bool is_gap_canonical8(State const& s) {
    constexpr int R = 8;
    int h = s.sp;
    std::array<int,MAXC> nf{}, ns{}, lo{}, hi{};
    lo.fill(99); hi.fill(-1);
    for (int i=0;i<R;++i) if (s.comp[i]) {
        int q=s.comp[i]; ++nf[q]; lo[q]=std::min(lo[q],i); hi[q]=std::max(hi[q],i);
    }
    for (int i=0;i<h;++i) if (s.stack[i]) ++ns[s.stack[i]];
    for (int q=1;q<s.ns;++q) {
        if (ns[q]) {
            if (ns[q]!=1 || nf[q]!=1 || (s.status[q]&1)) return false;
        } else if (nf[q] && nf[q]!=2) return false;
    }
    for (int a=1;a<s.ns;++a) if (!ns[a] && nf[a]==2)
        for (int b=1;b<s.ns;++b) if (!ns[b] && nf[b]==2)
            if (lo[a]<lo[b] && hi[b]<hi[a] && ((s.status[a]^s.status[b])&1)) return false;
    return true;
}

static std::vector<std::vector<uint32_t>> inverse_square(std::vector<std::vector<uint32_t>> A) {
    int n=A.size();
    std::vector<std::vector<uint32_t>> B(n,std::vector<uint32_t>(2*n));
    for (int i=0;i<n;++i) { for (int j=0;j<n;++j) B[i][j]=A[i][j]; B[i][n+i]=1; }
    for (int c=0;c<n;++c) {
        int p=c; while (p<n && !B[p][c]) ++p;
        if (p==n) throw std::runtime_error("singular gap basis");
        std::swap(B[p],B[c]);
        uint32_t iv=invq(B[c][c]);
        for (int j=c;j<2*n;++j) B[c][j]=(uint64_t)B[c][j]*iv%Q;
        for (int i=0;i<n;++i) if (i!=c && B[i][c]) {
            uint32_t f=B[i][c];
            for (int j=c;j<2*n;++j) B[i][j]=(B[i][j]+Q-(uint64_t)f*B[c][j]%Q)%Q;
        }
    }
    std::vector<std::vector<uint32_t>> I(n,std::vector<uint32_t>(n));
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) I[i][j]=B[i][n+j];
    return I;
}

struct GapSpace {
    Space const* s{};
    std::vector<int> basis_raw; // gap coordinate -> raw state index
    std::vector<std::vector<std::pair<int,uint32_t>>> straight; // raw state -> gap coordinates
};

static GapSpace make_gap_space(Space const& S) {
    GapSpace G; G.s=&S; G.basis_raw.resize(S.dim,-1); G.straight.resize(S.states.size());
    for (auto const& b:S.blocks) {
        int d=b.rr.size(); if (!d) continue;
        std::vector<int> gapc;
        for (int c=0;c<(int)b.idx.size();++c)
            if (is_gap_canonical8(unpack(S.states[b.idx[c]]))) gapc.push_back(c);
        if ((int)gapc.size()!=d) throw std::runtime_error("gap count != block rank");

        std::vector<std::vector<uint32_t>> M(d,std::vector<uint32_t>(d));
        for (int r=0;r<d;++r) for (int j=0;j<d;++j) M[r][j]=b.rr[r][gapc[j]];
        auto MI=inverse_square(std::move(M));

        for (int j=0;j<d;++j) G.basis_raw[b.off+j]=b.idx[gapc[j]];
        for (int c=0;c<(int)b.idx.size();++c) {
            auto& out=G.straight[b.idx[c]];
            for (int j=0;j<d;++j) {
                uint64_t z=0;
                for (int r=0;r<d;++r) z=(z+(uint64_t)MI[j][r]*b.rr[r][c])%Q;
                if (z) out.push_back({b.off+j,(uint32_t)z});
            }
        }
    }
    for (int i=0;i<S.dim;++i) if (G.basis_raw[i]<0) throw std::runtime_error("missing gap basis raw state");
    return G;
}

static int64_t signedq(uint32_t x) { return x<=Q/2 ? (int64_t)x : (int64_t)x-(int64_t)Q; }

static void measure(GapSpace const& S, GapSpace const& T, int sym) {
    std::map<int,size_t> out_hist, in_hist;
    std::map<int64_t,size_t> coeff_hist;
    std::vector<size_t> indeg(T.s->dim);
    size_t nnz=0, raw_edges=0, max_raw=0, max_out=0;
    std::vector<uint32_t> acc(T.s->dim);
    std::vector<int> touched; touched.reserve(T.s->dim);

    for (int sc=0;sc<S.s->dim;++sc) {
        std::fill(acc.begin(),acc.end(),0);
        touched.clear();
        WVec v{{S.s->states[S.basis_raw[sc]],1}};
        auto z=wcolumn(std::move(v),8,false,sym);
        raw_edges+=z.size(); max_raw=std::max(max_raw,z.size());
        for (auto const& e:z) {
            auto it=std::lower_bound(T.s->states.begin(),T.s->states.end(),e.p);
            if (it==T.s->states.end() || !(*it==e.p)) throw std::runtime_error("transition target missing");
            int ti=it-T.s->states.begin();
            for (auto [tc,c]:T.straight[ti]) {
                if (!acc[tc]) touched.push_back(tc);
                acc[tc]=(acc[tc]+(uint64_t)e.v*c)%Q;
            }
        }
        size_t od=0;
        for (int tc:touched) if (acc[tc]) {
            ++od; ++nnz; ++indeg[tc]; ++coeff_hist[signedq(acc[tc])];
        }
        ++out_hist[(int)od]; max_out=std::max(max_out,od);
    }
    size_t max_in=0;
    for (auto x:indeg) { ++in_hist[(int)x]; max_in=std::max(max_in,x); }
    static char const* names[3]={"N","R","L"};
    std::cout << "GAPTRANS h="<<S.s->h<<" sym="<<names[sym]<<" h2="<<T.s->h
              <<" dims="<<S.s->dim<<"x"<<T.s->dim
              <<" raw_edges="<<raw_edges<<" max_raw="<<max_raw
              <<" nnz="<<nnz<<" density="<<(double)nnz/((size_t)S.s->dim*T.s->dim)
              <<" avg_out="<<(double)nnz/S.s->dim<<" max_out="<<max_out<<" max_in="<<max_in;
    std::cout << " out"; for (auto [k,v]:out_hist) std::cout<<' '<<k<<':'<<v;
    std::cout << " in"; int shown=0; for (auto [k,v]:in_hist) { if (shown++>=24) break; std::cout<<' '<<k<<':'<<v; }
    std::cout << " coeff"; shown=0; for (auto [k,v]:coeff_hist) { if (shown++>=32) break; std::cout<<' '<<k<<':'<<v; }
    std::cout << '\n';
}

int main(int argc,char**argv) {
    int only_h=argc>1?std::atoi(argv[1]):-1;
    int only_sym=argc>2?std::atoi(argv[2]):-1;
    MODP=4294967291u;
    Vec all; int col=0;
    if (!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all)) return 2;
    std::array<std::vector<Packed>,9> H;
    for (auto const&p:all) H[unpack(p).sp].push_back(p);
    std::array<std::unique_ptr<Space>,9> S;
    S[0]=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));
    S[1]=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));
    S[2]=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));
    for (int h=3;h<=8;++h) S[h]=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));

    std::array<std::unique_ptr<GapSpace>,9> G;
    for (int h=0;h<=8;++h) G[h]=std::make_unique<GapSpace>(make_gap_space(*S[h]));

    for (int h=0;h<=8;++h) {
        if (only_h>=0 && h!=only_h) continue;
        for (int a=0;a<3;++a) {
            if (only_sym>=0 && a!=only_sym) continue;
            int h2=h+DEL_[a]; if (h2<0 || h2>8) continue;
            measure(*G[h],*G[h2],a);
        }
    }
}
