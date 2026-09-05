#define main rowr_batch_embedded_main
#include "rowr_column_batch_wfa_checkpoint.cpp"
#undef main
#include <array>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <vector>

// Canonical r=8 basis suggested by GapInterfaceCode.
// Token per frontier row:
//   T = through strand to one stack endpoint
//   N = vacancy
//   U/D = the two oriented ends of a grand-Motzkin bridge arc.
// Between consecutive T tokens (and at the two ends), U/D/N forms a
// grand-Motzkin bridge: its signed height returns to zero at the gap boundary.

static constexpr int R = 8;

static bool is_gap_canonical(State const& s) {
    int h = s.sp;
    std::array<int, MAXC> nf{}, ns{}, lo{}, hi{};
    lo.fill(99); hi.fill(-1);
    for (int i = 0; i < R; ++i) if (s.comp[i]) {
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
            if (lo[a]<lo[b] && hi[b]<hi[a] && ((s.status[a]^s.status[b])&1))
                return false;
    return true;
}

static bool decode_tokens(Packed pk, std::string& tok) {
    State s=unpack(pk);
    if (!is_gap_canonical(s)) return false;
    tok.assign(R,'?');
    std::array<int,MAXC> ns{},lo{},hi{};
    lo.fill(99);hi.fill(-1);
    for(int i=0;i<s.sp;++i) if(s.stack[i]) ++ns[s.stack[i]];
    for(int i=0;i<R;++i) {
        int q=s.comp[i];
        if(!q){tok[i]='N';continue;}
        lo[q]=std::min(lo[q],i);hi[q]=std::max(hi[q],i);
    }
    for(int i=0;i<R;++i) if(s.comp[i]) {
        int q=s.comp[i];
        if(ns[q]) tok[i]='T';
    }
    for(int q=1;q<s.ns;++q) if(!ns[q] && hi[q]>=0) {
        int a=lo[q],b=hi[q];
        if(s.status[q]&1){tok[a]='D';tok[b]='U';}
        else {tok[a]='U';tok[b]='D';}
    }
    int bal=0, nt=0;
    for(char c:tok){
        if(c=='T'){if(bal!=0)return false; ++nt;}
        else if(c=='U') ++bal;
        else if(c=='D') --bal;
        else if(c!='N') return false;
    }
    return bal==0 && nt==s.sp;
}

static bool encode_tokens(std::string const& tok, Packed& out) {
    if ((int)tok.size()!=R) return false;
    int h=0,bal=0;
    for(char c:tok){
        if(c=='T'){if(bal!=0)return false;++h;}
        else if(c=='U')++bal;
        else if(c=='D')--bal;
        else if(c!='N')return false;
    }
    if(bal!=0 || h>R) return false;
    State s{};s.n=R;s.sp=h;
    // Reserve through component ids first. pack() canonicalizes labels anyway.
    std::vector<int> throughPos; throughPos.reserve(h);
    for(int i=0;i<R;++i)if(tok[i]=='T')throughPos.push_back(i);
    int next=1;
    for(int k=0;k<h;++k){int q=next++;int i=throughPos[k];s.deg[i]=1;s.comp[i]=q;s.stack[k]=q;s.status[q]=0;}
    // Pair U/D inside each gap by distance from zero. Moving away from zero opens;
    // moving toward zero closes. This is the standard excursion pairing for a
    // grand-Motzkin bridge, including negative excursions.
    std::vector<std::pair<int,int>> open; // (position, sign +1/-1)
    bal=0;
    for(int i=0;i<R;++i){char c=tok[i];
        if(c=='T') { if(bal!=0 || !open.empty()) return false; continue; }
        if(c=='N') continue;
        int step=(c=='U'?1:-1);
        bool opens = (bal==0) || (bal>0 && step>0) || (bal<0 && step<0);
        if(opens){open.push_back({i,step});bal+=step;}
        else {
            if(open.empty())return false;
            auto [j,sgn]=open.back();open.pop_back();
            if(sgn!=-step)return false;
            int q=next++;s.deg[j]=s.deg[i]=1;s.comp[j]=s.comp[i]=q;s.status[q]=(sgn<0?1:0);
            bal+=step;
        }
    }
    if(bal!=0||!open.empty())return false;
    s.ns=next;out=pack(s);return true;
}

static void gen_words_rec(int pos,int needT,int bal,std::string& w,std::vector<std::string>& out){
    if(pos==R){if(needT==0&&bal==0)out.push_back(w);return;}
    int rem=R-pos;
    if(needT>rem)return;
    // N
    w[pos]='N';gen_words_rec(pos+1,needT,bal,w,out);
    // U/D stay inside the current gap; balance may be negative (grand Motzkin).
    w[pos]='U';gen_words_rec(pos+1,needT,bal+1,w,out);
    w[pos]='D';gen_words_rec(pos+1,needT,bal-1,w,out);
    // T only at gap balance zero.
    if(needT>0&&bal==0){w[pos]='T';gen_words_rec(pos+1,needT-1,0,w,out);}
}

static std::vector<std::string> gen_words(int h){std::string w(R,'N');std::vector<std::string> out;gen_words_rec(0,h,0,w,out);return out;}

int main(){
    Vec all;int col=0;if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all))return 2;
    int D[9]={1107,1640,1428,888,420,152,42,8,1};
    bool global=true;
    for(int h=0;h<=8;++h){
        auto words=gen_words(h);
        std::set<Packed> enc;
        size_t badEncode=0,badFixed=0,badRound=0;
        for(auto const&w:words){Packed p;if(!encode_tokens(w,p)){++badEncode;continue;}enc.insert(p);if(!std::binary_search(all.begin(),all.end(),p))++badFixed;std::string d;if(!decode_tokens(p,d)||d!=w)++badRound;}
        std::set<Packed> filt;
        for(auto const&p:all){State s=unpack(p);if(s.sp==h&&is_gap_canonical(s))filt.insert(p);}
        size_t missing=0,extra=0;
        for(auto const&p:filt)if(!enc.count(p))++missing;
        for(auto const&p:enc)if(!filt.count(p))++extra;
        bool ok=words.size()==(size_t)D[h]&&enc.size()==(size_t)D[h]&&filt.size()==(size_t)D[h]&&!badEncode&&!badFixed&&!badRound&&!missing&&!extra;
        global&=ok;
        std::cout<<"h="<<h<<" words="<<words.size()<<" encoded="<<enc.size()<<" filtered="<<filt.size()<<" target="<<D[h]
                 <<" badEncode="<<badEncode<<" badFixed="<<badFixed<<" badRound="<<badRound<<" missing="<<missing<<" extra="<<extra<<" exact="<<ok<<"\n";
    }
    std::cout<<"gap_basis_bijection_exact="<<global<<"\n";
    return global?0:1;
}
