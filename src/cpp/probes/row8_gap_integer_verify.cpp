#define ROW8_STRUCTURAL_INTEGER_NO_MAIN 1
#include "row8_structural_integer_closure.cpp"
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
    for(int i=0;i<R;++i) if(s.comp[i]) { int q=s.comp[i]; ++nf[q]; lo[q]=std::min(lo[q],i); hi[q]=std::max(hi[q],i); }
    for(int i=0;i<h;++i) if(s.stack[i]) ++ns[s.stack[i]];
    for(int q=1;q<s.ns;++q) {
        if(ns[q]) { if(ns[q]!=1 || nf[q]!=1 || (s.status[q]&1)) return false; }
        else if(nf[q] && nf[q]!=2) return false;
    }
    for(int a=1;a<s.ns;++a) if(!ns[a] && nf[a]==2)
        for(int b=1;b<s.ns;++b) if(!ns[b] && nf[b]==2)
            if(lo[a]<lo[b] && hi[b]<hi[a] && ((s.status[a]^s.status[b])&1)) return false;
    return true;
}

static std::vector<std::vector<uint32_t>> invsq(std::vector<std::vector<uint32_t>> A) {
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

struct GapExact {
    Space const* s{};
    std::vector<int> basis_raw;
    std::vector<std::vector<int>> support; // exact: raw structural coordinate = sum of these gap columns
};

static GapExact build_gap_exact(Space const& S) {
    GapExact G; G.s=&S; G.basis_raw.resize(S.dim,-1); G.support.resize(S.states.size());
    size_t bad=0, non01=0, maxsup=0;
    for(auto const&b:S.blocks){
        int d=b.rr.size(); if(!d) continue;
        std::vector<int> gc;
        for(int c=0;c<(int)b.idx.size();++c) if(is_gap_canonical8(unpack(S.states[b.idx[c]]))) gc.push_back(c);
        if((int)gc.size()!=d) throw std::runtime_error("gap count");
        std::vector<std::vector<uint32_t>> M(d,std::vector<uint32_t>(d));
        for(int r=0;r<d;++r)for(int j=0;j<d;++j)M[r][j]=b.rr[r][gc[j]];
        auto MI=invsq(std::move(M));
        for(int j=0;j<d;++j)G.basis_raw[b.off+j]=b.idx[gc[j]];

        for(int c=0;c<(int)b.idx.size();++c){
            auto& sp=G.support[b.idx[c]];
            for(int j=0;j<d;++j){
                uint64_t z=0;for(int r=0;r<d;++r)z=(z+(uint64_t)MI[j][r]*b.rr[r][c])%Q;
                if(z){ if(z!=1) ++non01; sp.push_back(b.off+j); }
            }
            maxsup=std::max(maxsup,sp.size());
            // Exact integer reconstruction in the existing structural Z-coordinate system.
            for(int r=0;r<d;++r){
                int64_t got=0;
                for(int gj:sp){ int j=gj-b.off; got += si(b.rr[r][gc[j]]); }
                int64_t want=si(b.rr[r][c]);
                if(got!=want){ if(bad<4)std::cerr<<"reconstruct bad h="<<S.h<<" blockoff="<<b.off<<" c="<<c<<" r="<<r<<" got="<<got<<" want="<<want<<"\n"; ++bad; break; }
            }
        }
    }
    std::cout<<"GAP_INTEGER h="<<S.h<<" dim="<<S.dim<<" states="<<S.states.size()<<" non01_mod="<<non01<<" reconstruct_bad="<<bad<<" max_support="<<maxsup<<" exact="<<(non01==0&&bad==0)<<"\n";
    if(non01||bad) throw std::runtime_error("gap integer reconstruction failed");
    return G;
}

static bool verify_transition(GapExact const&S,GapExact const&T,int sym){
    size_t bad=0, nnz=0, maxout=0; std::map<int64_t,size_t> hist;
    std::vector<int64_t> acc(T.s->dim);
    for(int sc=0;sc<S.s->dim;++sc){
        std::fill(acc.begin(),acc.end(),0);
        WVec v{{S.s->states[S.basis_raw[sc]],1}}; auto z=wcolumn(std::move(v),8,false,sym);
        for(auto const&e:z){
            auto it=std::lower_bound(T.s->states.begin(),T.s->states.end(),e.p);
            if(it==T.s->states.end()||!(*it==e.p))throw std::runtime_error("target missing");
            int ti=it-T.s->states.begin();
            for(int tc:T.support[ti]) acc[tc]+=(int64_t)e.v;
        }
        size_t od=0;for(auto x:acc)if(x){++od;++nnz;++hist[x];if(x!=1)++bad;}maxout=std::max(maxout,od);
    }
    static char const*n[3]={"N","R","L"};
    std::cout<<"GAP_INTEGER_TRANS h="<<S.s->h<<" sym="<<n[sym]<<" h2="<<T.s->h<<" nnz="<<nnz<<" maxout="<<maxout<<" nonunit="<<bad<<" coeff";for(auto[x,c]:hist)std::cout<<' '<<x<<':'<<c;std::cout<<" exact01="<<(bad==0)<<"\n";
    return bad==0;
}

static bool verify_initial(std::array<GapExact,9> const&G,int sym){
    int h=1+DEL_[sym];State z{};z.n=8;WVec v{{pack(z),1}};auto w=wcolumn(std::move(v),8,true,sym);std::vector<int64_t>a(G[h].s->dim);
    for(auto const&e:w){auto it=std::lower_bound(G[h].s->states.begin(),G[h].s->states.end(),e.p);if(it==G[h].s->states.end()||!(*it==e.p))throw std::runtime_error("initial target");int ti=it-G[h].s->states.begin();for(int q:G[h].support[ti])a[q]+=(int64_t)e.v;}
    size_t nz=0,bad=0;for(auto x:a)if(x){++nz;bad+=x!=1;}std::cout<<"GAP_INITIAL sym="<<sym<<" h="<<h<<" nz="<<nz<<" nonunit="<<bad<<" exact01="<<(bad==0)<<"\n";return bad==0;
}
static bool verify_final(GapExact const&G,int sym){size_t nz=0,bad=0;std::map<int64_t,size_t>hist;for(int q=0;q<G.s->dim;++q){WVec v{{G.s->states[G.basis_raw[q]],1}};int64_t x=wlast_value(std::move(v),8,sym);if(x){++nz;++hist[x];bad+=x!=1;}}std::cout<<"GAP_FINAL h="<<G.s->h<<" sym="<<sym<<" nz="<<nz<<" nonunit="<<bad<<" coeff";for(auto[x,c]:hist)std::cout<<' '<<x<<':'<<c;std::cout<<" exact01="<<(bad==0)<<"\n";return bad==0;}

#ifndef ROW8_GAP_INTEGER_VERIFY_NO_MAIN
int main(){
    MODP=4294967291u;Vec all;int col=0;if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all))return 2;
    std::array<std::vector<Packed>,9>H;for(auto const&p:all)H[unpack(p).sp].push_back(p);
    std::array<std::unique_ptr<Space>,9>S;
    S[0]=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));
    S[1]=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));
    S[2]=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));
    for(int h=3;h<=8;++h)S[h]=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));
    std::array<GapExact,9>G;for(int h=0;h<=8;++h)G[h]=build_gap_exact(*S[h]);
    bool ok=true;for(int h=0;h<=8;++h)for(int a=0;a<3;++a){int h2=h+DEL_[a];if(h2>=0&&h2<=8)ok&=verify_transition(G[h],G[h2],a);}for(int a=0;a<3;++a)ok&=verify_initial(G,a);ok&=verify_final(G[0],0);ok&=verify_final(G[1],1);std::cout<<"gap_integer_automaton_exact01="<<ok<<"\n";return ok?0:1;
}
#endif
