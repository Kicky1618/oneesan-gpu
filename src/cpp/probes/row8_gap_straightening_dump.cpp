#define ROW8_STRUCTURAL_UPPER_NO_MAIN 1
#include "row8_structural_upper_sparsity.cpp"
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

static bool is_gap_canonical8(State const& s) {
    constexpr int R=8; int h=s.sp;
    std::array<int,MAXC> nf{},ns{},lo{},hi{}; lo.fill(99);hi.fill(-1);
    for(int i=0;i<R;++i)if(s.comp[i]){int q=s.comp[i];++nf[q];lo[q]=std::min(lo[q],i);hi[q]=std::max(hi[q],i);}
    for(int i=0;i<h;++i)if(s.stack[i])++ns[s.stack[i]];
    for(int q=1;q<s.ns;++q){if(ns[q]){if(ns[q]!=1||nf[q]!=1||(s.status[q]&1))return false;}else if(nf[q]&&nf[q]!=2)return false;}
    for(int a=1;a<s.ns;++a)if(!ns[a]&&nf[a]==2)for(int b=1;b<s.ns;++b)if(!ns[b]&&nf[b]==2)
        if(lo[a]<lo[b]&&hi[b]<hi[a]&&((s.status[a]^s.status[b])&1))return false;
    return true;
}
static std::string token(State const&s){
    std::string t(8,'?');std::array<int,MAXC> ns{},lo{},hi{};lo.fill(99);hi.fill(-1);
    for(int i=0;i<s.sp;++i)if(s.stack[i])++ns[s.stack[i]];
    for(int i=0;i<8;++i){if(!s.comp[i]){t[i]='N';continue;}int q=s.comp[i];lo[q]=std::min(lo[q],i);hi[q]=std::max(hi[q],i);if(ns[q])t[i]='T';}
    for(int q=1;q<s.ns;++q)if(!ns[q]&&hi[q]>=0){int a=lo[q],b=hi[q];if(s.status[q]&1){t[a]='D';t[b]='U';}else{t[a]='U';t[b]='D';}}
    return t;
}
static std::string sig(State const&s){std::ostringstream o;o<<"deg=";for(int i=0;i<8;++i)o<<int(s.deg[i]);o<<" comp=";for(int i=0;i<8;++i)o<<int(s.comp[i])<<(i==7?' ':',');o<<"stack=";for(int i=0;i<s.sp;++i)o<<int(s.stack[i])<<(i+1==s.sp?' ':',');o<<"status=";for(int q=1;q<s.ns;++q)o<<q<<':'<<int(s.status[q])<<',';return o.str();}
static std::vector<std::vector<uint32_t>> invsq(std::vector<std::vector<uint32_t>>A){int n=A.size();std::vector<std::vector<uint32_t>>B(n,std::vector<uint32_t>(2*n));for(int i=0;i<n;++i){for(int j=0;j<n;++j)B[i][j]=A[i][j];B[i][n+i]=1;}for(int c=0;c<n;++c){int p=c;while(p<n&&!B[p][c])++p;if(p==n)throw std::runtime_error("sing");std::swap(B[p],B[c]);uint32_t iv=invq(B[c][c]);for(int j=c;j<2*n;++j)B[c][j]=(uint64_t)B[c][j]*iv%Q;for(int i=0;i<n;++i)if(i!=c&&B[i][c]){uint32_t f=B[i][c];for(int j=c;j<2*n;++j)B[i][j]=(B[i][j]+Q-(uint64_t)f*B[c][j]%Q)%Q;}}std::vector<std::vector<uint32_t>>I(n,std::vector<uint32_t>(n));for(int i=0;i<n;++i)for(int j=0;j<n;++j)I[i][j]=B[i][n+j];return I;}
int main(int ac,char**av){int h=ac>1?std::atoi(av[1]):0, want=ac>2?std::atoi(av[2]):2,lim=ac>3?std::atoi(av[3]):20;MODP=4294967291u;Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::array<std::vector<Packed>,9>H;for(auto&p:all)H[unpack(p).sp].push_back(p);std::unique_ptr<Space>S;if(h==0)S=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));else if(h==1)S=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));else if(h==2)S=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));else S=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));int shown=0;
for(auto const&b:S->blocks){int d=b.rr.size();if(!d)continue;std::vector<int>gc;for(int c=0;c<(int)b.idx.size();++c)if(is_gap_canonical8(unpack(S->states[b.idx[c]])))gc.push_back(c);std::vector<std::vector<uint32_t>>G(d,std::vector<uint32_t>(d));for(int r=0;r<d;++r)for(int j=0;j<d;++j)G[r][j]=b.rr[r][gc[j]];auto GI=invsq(G);for(int c=0;c<(int)b.idx.size();++c){std::vector<int>out;for(int j=0;j<d;++j){uint64_t z=0;for(int r=0;r<d;++r)z=(z+(uint64_t)GI[j][r]*b.rr[r][c])%Q;if(z)out.push_back(j);}if((int)out.size()!=want)continue;State s=unpack(S->states[b.idx[c]]);std::cout<<"RAW "<<sig(s)<<" ->";for(int j:out){State g=unpack(S->states[b.idx[gc[j]]]);std::cout<<' '<<token(g);}std::cout<<'\n';if(++shown>=lim)return 0;}}
}
