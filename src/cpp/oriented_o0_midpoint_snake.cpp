#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::uint32_t MOD = 65521;
constexpr std::uint32_t ZETA = 61640;
constexpr std::uint32_t ZETA_INV = 19685;

std::uint32_t addm(std::uint32_t a, std::uint32_t b) {
    std::uint32_t x=a+b; if(x>=MOD)x-=MOD; return x;
}
std::uint32_t mulm(std::uint32_t a, std::uint32_t b) {
    return static_cast<std::uint32_t>(std::uint64_t{a}*b%MOD);
}
std::uint32_t subm(std::uint32_t a, std::uint32_t b) {
    return a>=b?a-b:a+MOD-b;
}
std::uint32_t powm(std::uint32_t a,std::uint32_t e){std::uint64_t r=1;while(e){if(e&1)r=r*a%MOD;a=std::uint64_t{a}*a%MOD;e>>=1;}return static_cast<std::uint32_t>(r);}
std::uint32_t invm(std::uint32_t a){return powm(a,MOD-2);}

int qcharge(unsigned c){return c==1?1:c==2?-1:0;}
unsigned flip(unsigned c){return c==1?2:c==2?1:0;}

struct EdgeFlow { bool used=false,in=false; int dir=0; };
EdgeFlow L(unsigned c){if(c==1)return{true,true,0};if(c==2)return{true,false,2};return{};}
EdgeFlow U(unsigned c){if(c==1)return{true,true,3};if(c==2)return{true,false,1};return{};}
EdgeFlow D(unsigned c){if(c==1)return{true,false,3};if(c==2)return{true,true,1};return{};}
EdgeFlow R(unsigned c){if(c==1)return{true,false,0};if(c==2)return{true,true,2};return{};}
std::uint32_t turn_weight(int in,int out){int d=(out-in+4)&3;if(d==0)return 1;if(d==1)return ZETA;if(d==3)return ZETA_INV;return 0;}
std::uint32_t vertex(unsigned l,unsigned u,unsigned d,unsigned r){
    std::array<EdgeFlow,4> es={L(l),U(u),D(d),R(r)};int used=0,ni=0,no=0,id=0,od=0;
    for(auto e:es)if(e.used){++used;if(e.in){++ni;id=e.dir;}else{++no;od=e.dir;}}
    if(used==0)return 1;if(used==2&&ni==1&&no==1)return turn_weight(id,od);return 0;
}

using Mat9=std::array<std::array<std::uint32_t,9>,9>;
Mat9 forward_gate(){
    Mat9 m{};
    for(unsigned l=0;l<3;++l)for(unsigned u=0;u<3;++u)for(unsigned d=0;d<3;++d)for(unsigned r=0;r<3;++r)
        m[d+3*r][l+3*u]=vertex(l,u,d,r);
    return m;
}
Mat9 reverse_gate(){
    Mat9 m{};
    // Logical reverse carry is sign-flipped physical horizontal orientation.
    for(unsigned c=0;c<3;++c)for(unsigned u=0;u<3;++u)for(unsigned d=0;d<3;++d)for(unsigned cn=0;cn<3;++cn){
        unsigned rphys=flip(c), lphys=flip(cn);
        m[d+3*cn][c+3*u]=vertex(lphys,u,d,rphys);
    }
    return m;
}

struct Factor {
    std::array<std::array<std::uint32_t,9>,4> A{}; // k,input
    std::array<std::array<std::uint32_t,4>,9> B{}; // output,k
    std::array<int,4> charge{};
};

int rank_small(std::vector<std::vector<std::uint32_t>> a){
    if(a.empty())return 0;int m=a.size(),n=a[0].size(),r=0;
    for(int c=0;c<n&&r<m;++c){int p=r;while(p<m&&!a[p][c])++p;if(p==m)continue;std::swap(a[p],a[r]);auto iv=invm(a[r][c]);for(int j=c;j<n;++j)a[r][j]=mulm(a[r][j],iv);for(int i=0;i<m;++i)if(i!=r&&a[i][c]){auto f=a[i][c];for(int j=c;j<n;++j)a[i][j]=subm(a[i][j],mulm(f,a[r][j]));}++r;}return r;
}

std::vector<std::uint32_t> solve_basis(const std::vector<std::vector<std::uint32_t>>& B,const std::vector<std::uint32_t>& y){
    const int m=B.size(),r=B.empty()?0:B[0].size();
    std::vector<std::vector<std::uint32_t>> a(m,std::vector<std::uint32_t>(r+1));
    for(int i=0;i<m;++i){for(int j=0;j<r;++j)a[i][j]=B[i][j];a[i][r]=y[i];}
    int row=0;std::vector<int> piv(r,-1);
    for(int c=0;c<r&&row<m;++c){int p=row;while(p<m&&!a[p][c])++p;if(p==m)continue;std::swap(a[p],a[row]);auto iv=invm(a[row][c]);for(int j=c;j<=r;++j)a[row][j]=mulm(a[row][j],iv);for(int i=0;i<m;++i)if(i!=row&&a[i][c]){auto f=a[i][c];for(int j=c;j<=r;++j)a[i][j]=subm(a[i][j],mulm(f,a[row][j]));}piv[c]=row++;}
    std::vector<std::uint32_t>x(r);for(int c=0;c<r;++c){if(piv[c]<0)throw std::runtime_error("singular basis");x[c]=a[piv[c]][r];}return x;
}

Factor factor_gate(const Mat9& M){
    Factor f;int kbase=0;
    for(int Q=-1;Q<=1;++Q){
        std::vector<int> ins,outs;
        for(unsigned a=0;a<3;++a)for(unsigned b=0;b<3;++b)if(qcharge(a)+qcharge(b)==Q)ins.push_back(a+3*b);
        outs=ins;
        std::vector<std::vector<std::uint32_t>> block(outs.size(),std::vector<std::uint32_t>(ins.size()));
        for(size_t i=0;i<outs.size();++i)for(size_t j=0;j<ins.size();++j)block[i][j]=M[outs[i]][ins[j]];
        std::vector<int> pivcols;std::vector<std::vector<std::uint32_t>> basis_cols(outs.size());
        int oldrank=0;
        for(size_t j=0;j<ins.size();++j){
            auto trial=basis_cols;for(size_t i=0;i<outs.size();++i)trial[i].push_back(block[i][j]);int nr=rank_small(trial);
            if(nr>oldrank){pivcols.push_back(j);basis_cols.swap(trial);oldrank=nr;}
        }
        const int r=oldrank;
        for(int t=0;t<r;++t){int k=kbase+t;f.charge[k]=Q;for(size_t i=0;i<outs.size();++i)f.B[outs[i]][k]=basis_cols[i][t];}
        for(size_t j=0;j<ins.size();++j){std::vector<std::uint32_t> col(outs.size());for(size_t i=0;i<outs.size();++i)col[i]=block[i][j];auto x=solve_basis(basis_cols,col);for(int t=0;t<r;++t)f.A[kbase+t][ins[j]]=x[t];}
        kbase+=r;
    }
    if(kbase!=4)throw std::runtime_error("local gate rank != 4");
    // Verify B*A=M.
    for(int o=0;o<9;++o)for(int i=0;i<9;++i){std::uint32_t z=0;for(int k=0;k<4;++k)z=addm(z,mulm(f.B[o][k],f.A[k][i]));if(z!=M[o][i])throw std::runtime_error("factorization mismatch");}
    return f;
}

using Mat12=std::array<std::array<std::uint32_t,12>,12>;
using Mat4=std::array<std::array<std::uint32_t,4>,4>;

Mat12 forward_mid(const Factor& f){
    Mat12 C{};
    for(int k=0;k<4;++k)for(unsigned u=0;u<3;++u)for(unsigned d=0;d<3;++d)for(int kp=0;kp<4;++kp){
        std::uint32_t z=0;for(unsigned r=0;r<3;++r)z=addm(z,mulm(f.B[d+3*r][k],f.A[kp][r+3*u]));
        C[d+3*kp][k+4*u]=z;
    }return C;
}
Mat12 reverse_mid(const Factor& f){
    Mat12 C{};
    for(int k=0;k<4;++k)for(unsigned u=0;u<3;++u)for(unsigned d=0;d<3;++d)for(int kp=0;kp<4;++kp){
        std::uint32_t z=0;for(unsigned c=0;c<3;++c)z=addm(z,mulm(f.B[d+3*c][k],f.A[kp][c+3*u]));
        C[kp+4*d][u+3*k]=z;
    }return C;
}
Mat4 turn_fr(const Factor& ff,const Factor& fr){
    Mat4 J{};for(int ko=0;ko<4;++ko)for(int ki=0;ki<4;++ki){std::uint32_t z=0;for(unsigned d=0;d<3;++d)z=addm(z,mulm(ff.B[d][ki],fr.A[ko][3*d]));J[ko][ki]=z;}return J;
}
Mat4 turn_rf(const Factor& fr,const Factor& ff){
    Mat4 J{};for(int ko=0;ko<4;++ko)for(int ki=0;ki<4;++ki){std::uint32_t z=0;for(unsigned d=0;d<3;++d)z=addm(z,mulm(fr.B[d][ki],ff.A[ko][3*d]));J[ko][ki]=z;}return J;
}

std::uint64_t pow3(unsigned e){std::uint64_t x=1;while(e--)x*=3;return x;}

void apply12(std::vector<std::uint32_t>&v,std::uint64_t stride,const Mat12&M){
    const std::uint64_t total=v.size(), span=12*stride;
    std::array<std::uint32_t,12> in{},out{};
    for(std::uint64_t base0=0;base0<total;base0+=span)for(std::uint64_t lo=0;lo<stride;++lo){
        const auto base=base0+lo;for(int i=0;i<12;++i)in[i]=v[base+std::uint64_t(i)*stride];out.fill(0);
        for(int o=0;o<12;++o)for(int i=0;i<12;++i)if(M[o][i]&&in[i])out[o]=addm(out[o],mulm(M[o][i],in[i]));
        for(int o=0;o<12;++o)v[base+std::uint64_t(o)*stride]=out[o];
    }
}
void apply4(std::vector<std::uint32_t>&v,std::uint64_t stride,const Mat4&M){
    const std::uint64_t total=v.size(),span=4*stride;std::array<std::uint32_t,4>in{},out{};
    for(std::uint64_t base0=0;base0<total;base0+=span)for(std::uint64_t lo=0;lo<stride;++lo){const auto base=base0+lo;for(int i=0;i<4;++i)in[i]=v[base+std::uint64_t(i)*stride];out.fill(0);for(int o=0;o<4;++o)for(int i=0;i<4;++i)if(M[o][i]&&in[i])out[o]=addm(out[o],mulm(M[o][i],in[i]));for(int o=0;o<4;++o)v[base+std::uint64_t(o)*stride]=out[o];}
}

struct Result{std::uint32_t residue;std::uint64_t states;double sec;};
Result solve(unsigned n){
    if(n<3||!(n&1)||n>13)throw std::runtime_error("prototype requires odd n in [3,13]");
    auto ff=factor_gate(forward_gate()),fr=factor_gate(reverse_gate());
    auto Cf=forward_mid(ff); auto Cr=reverse_mid(fr); auto Jfr=turn_fr(ff,fr); auto Jrf=turn_rf(fr,ff);
    Mat12 CfBottom=Cf;for(int o=0;o<12;++o)if((o%3)!=0)for(int i=0;i<12;++i)CfBottom[o][i]=0; // output d must be 0

    const std::uint64_t total=4*pow3(n-1);std::vector<std::uint32_t> v(total,0);
    // Source at column 0, then A of normal vertex 1. Factor order: d0(qutrit), K1, u2...
    for(unsigned d0=0;d0<2;++d0){unsigned r=(d0==0?1:0);std::uint32_t sw=(d0==0?1:ZETA_INV);for(int k=0;k<4;++k){auto w=mulm(sw,ff.A[k][r]);if(w)v[d0+3*k]=addm(v[d0+3*k],w);}}

    const auto t0=std::chrono::steady_clock::now();
    // Finish top row: K starts at position 1 and moves right.
    for(unsigned p=1;p+1<n;++p)apply12(v,pow3(p),Cf);
    // K is now at right endpoint position n-1.
    bool forward=true;
    for(unsigned y=1;y+1<n;++y){
        if(forward){ // turn F->R, process right endpoint of this row to midpoint
            apply4(v,pow3(n-1),Jfr);forward=false;
            for(int p=int(n)-1;p>0;--p)apply12(v,pow3(p-1),Cr);
        }else{ // turn R->F
            apply4(v,1,Jrf);forward=true;
            for(unsigned p=0;p+1<n;++p)apply12(v,pow3(p),Cf);
        }
    }
    // n odd => after row n-2 (reverse) K sits at left. Turn into bottom row forward.
    if(forward)throw std::runtime_error("unexpected orientation before bottom");
    apply4(v,1,Jrf);forward=true; // midpoint at bottom-left vertex
    // Move only through normal bottom vertices 1..n-2, emitting d=0.
    for(unsigned p=0;p+2<n;++p)apply12(v,pow3(p),CfBottom);
    // Now K at position n-2, target up qutrit at n-1. Lower qutrits must all be 0.
    const std::uint64_t stride=pow3(n-2);
    std::uint32_t ans=0;
    // local layout before target is K(low within pair) then u(high): index k + 4*u, with low prefix stride.
    for(int k=0;k<4;++k)for(unsigned u=0;u<3;++u){std::uint32_t tw=0;for(unsigned r=0;r<3;++r){std::uint32_t ew=0;if(r==1&&u==0)ew=1;else if(r==0&&u==1)ew=ZETA;if(ew)tw=addm(tw,mulm(ff.B[3*r][k],ew));}if(!tw)continue;const std::uint64_t idx=(std::uint64_t(k)+4*u)*stride;ans=addm(ans,mulm(v[idx],tw));}
    const auto t1=std::chrono::steady_clock::now();
    return{ans,total,std::chrono::duration<double>(t1-t0).count()};
}

} // namespace

int main(int argc,char**argv){unsigned first=argc>1?std::strtoul(argv[1],nullptr,10):3,last=argc>2?std::strtoul(argv[2],nullptr,10):first;for(unsigned n=first;n<=last;n+=2){auto r=solve(n);std::cout<<"n="<<n<<" residue="<<r.residue<<" midpoint_dense="<<r.states<<" sec="<<std::fixed<<std::setprecision(6)<<r.sec<<'\n';}}
