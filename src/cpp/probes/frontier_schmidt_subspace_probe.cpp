#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

// Test whether the low Schmidt rank seen for the physical source vector comes
// from fixed left/right separator subspaces, or only from one special trajectory.
//
// For several independent initial frontier vectors we apply the same completed-row
// transfer and build the balanced HIGH|LOW block V_h.  We then measure
//
//   rank([V_h^(0) V_h^(1) ...])      -- union of column spaces
//   rank([V_h^(0); V_h^(1); ...])    -- union of row spaces
//
// If both union ranks remain equal to the single-trajectory A111960 rank, then a
// fixed channel basis independent of the initial vector is plausible.  If they
// grow rapidly, the small rank is trajectory-specific and a scalar channel DP is
// not justified by rank alone.

using U32 = std::uint32_t;
using U64 = std::uint64_t;
static constexpr U64 P = 4294967291ULL;

static U64 addp(U64 a,U64 b){ U64 z=a+b; if(z>=P)z-=P; return z; }
static U64 subp(U64 a,U64 b){ return a>=b?a-b:a+P-b; }
static U64 mulp(U64 a,U64 b){ return U64((__uint128_t(a)*b)%P); }
static U64 powp(U64 a,U64 e){ U64 r=1; while(e){ if(e&1)r=mulp(r,a); a=mulp(a,a); e>>=1; } return r; }
static U64 invp(U64 a){ assert(a); return powp(a,P-2); }

static int rank_dense(std::vector<U64> a,int R,int C){
    int r=0;
    for(int c=0;c<C && r<R;++c){
        int p=r; while(p<R && !a[size_t(p)*C+c])++p;
        if(p==R)continue;
        if(p!=r) for(int j=c;j<C;++j) std::swap(a[size_t(p)*C+j],a[size_t(r)*C+j]);
        U64 iv=invp(a[size_t(r)*C+c]);
        for(int i=r+1;i<R;++i){
            U64 x=a[size_t(i)*C+c]; if(!x)continue;
            U64 f=mulp(x,iv); a[size_t(i)*C+c]=0;
            for(int j=c+1;j<C;++j) a[size_t(i)*C+j]=subp(a[size_t(i)*C+j],mulp(f,a[size_t(r)*C+j]));
        }
        ++r;
    }
    return r;
}

static std::vector<Mate> code_states(MateCodec const&mc){
    std::vector<Mate> out(mc.codeSize());
    for(Code bi=0;bi<mc.codeSizeL();++bi){ auto const&b=mc.codeTable(bi); for(Code i=0;i<b.size;++i) out[b.base+i]=b.mateL|b.mateR[i]; }
    return out;
}
static U32 seg(Mate m,int first,int len){ U32 z=0; for(int i=0;i<len;++i) z|=U32(m.get(first+i))<<(2*i); return z; }
static int cut_height(Mate m,int W){ int k=W/2,h=1; for(int p=W-1;p>=k;--p){ auto v=m.get(p); if(v==R)--h; else if(v==L)++h; } return h; }
static void run_row(PathCounter<Modnum<U64>>&pc){ for(int j=0;j<pc.cols-2;++j)pc.update(j,false); pc.update(pc.cols-2,false); }

struct SampleBlock { std::vector<std::tuple<U32,U32,U64>> e; };

static std::vector<SampleBlock> collect(PathCounter<Modnum<U64>>&pc,std::vector<Mate>const&states){
    int W=pc.cols,k=W/2;
    std::vector<SampleBlock> out(k+2);
    for(Code i=0;i<pc.mc.codeSize();++i){
        U64 v=U64(pc.value[i])%P; if(!v)continue;
        Mate m=states[i]; int h=cut_height(m,W); if(h<0||h>=int(out.size()))continue;
        out[h].e.emplace_back(seg(m,k,k),seg(m,0,k),v);
    }
    return out;
}

struct DenseTrials {
    int R=0,C=0,T=0;
    std::vector<std::vector<U64>> M;
};
static DenseTrials densify(std::vector<SampleBlock>const&bs){
    std::map<U32,int> rm,cm;
    for(auto const&b:bs) for(auto const&[r,c,v]:b.e){ (void)v; rm.emplace(r,0); cm.emplace(c,0); }
    int q=0; for(auto &kv:rm)kv.second=q++;
    q=0; for(auto &kv:cm)kv.second=q++;
    DenseTrials d; d.R=rm.size(); d.C=cm.size(); d.T=bs.size(); d.M.resize(bs.size(),std::vector<U64>(size_t(d.R)*d.C));
    for(size_t t=0;t<bs.size();++t) for(auto const&[r,c,v]:bs[t].e){ U64&z=d.M[t][size_t(rm[r])*d.C+cm[c]]; z=addp(z,v); }
    return d;
}

static int union_col_rank(DenseTrials const&d){
    if(!d.R||!d.C)return 0;
    int C=d.C*d.T;
    std::vector<U64> A(size_t(d.R)*C,0);
    for(int t=0;t<d.T;++t)for(int i=0;i<d.R;++i)for(int j=0;j<d.C;++j) A[size_t(i)*C+t*d.C+j]=d.M[t][size_t(i)*d.C+j];
    return rank_dense(std::move(A),d.R,C);
}
static int union_row_rank(DenseTrials const&d){
    if(!d.R||!d.C)return 0;
    int R=d.R*d.T;
    std::vector<U64> A(size_t(R)*d.C,0);
    for(int t=0;t<d.T;++t)for(int i=0;i<d.R;++i)for(int j=0;j<d.C;++j) A[size_t(t*d.R+i)*d.C+j]=d.M[t][size_t(i)*d.C+j];
    return rank_dense(std::move(A),R,d.C);
}

static U64 channel_formula(int r,int h){
    if(h<0||h>r)return 0; int n=r-h;
    __uint128_t am1=0,a=1;
    for(int j=0;j<n;++j){ __uint128_t rhs=__uint128_t(2*j+h+1)*a; if(j>0)rhs+=__uint128_t(3*(j+h))*am1; assert(rhs%(j+1)==0); __uint128_t an=rhs/(j+1); am1=a; a=an; }
    return U64(a);
}

int main(int argc,char**argv){
    msg=NONE; modulus=P;
    int W=argc>1?std::atoi(argv[1]):8;
    int rows=argc>2?std::atoi(argv[2]):W/2;
    int trials=argc>3?std::atoi(argv[3]):4;
    if(W<4||W>10||W%2||rows<0||rows>W||trials<1||trials>8){ std::cerr<<"usage: W_even(4..10) completed_rows trials(1..8)\n"; return 2; }

    PathCounter<Modnum<U64>> proto(W,W,false,false); auto states=code_states(proto.mc);
    std::vector<std::vector<SampleBlock>> all;
    std::mt19937_64 rng(0x51b5e7ULL);
    for(int t=0;t<trials;++t){
        PathCounter<Modnum<U64>> pc(W,W,false,false);
        for(Code i=0;i<pc.wc.codeSize();++i)pc.deferred[i]=0;
        if(t==0){
            for(Code i=0;i<pc.mc.codeSize();++i)pc.value[i]=0;
            pc.value[pc.mc.encode(Mate(W-1,R))]=1;
        } else {
            for(Code i=0;i<pc.mc.codeSize();++i)pc.value[i]=U64(rng()%P);
        }
        for(int r=0;r<rows;++r)run_row(pc);
        all.push_back(collect(pc,states));
    }

    std::cout<<"W="<<W<<" completed_rows="<<rows<<" trials="<<trials<<" states="<<states.size()<<"\n";
    bool fixed=true;
    for(int h=0;h<=W/2+1;++h){
        std::vector<SampleBlock> bs; for(int t=0;t<trials;++t)bs.push_back(all[t][h]);
        DenseTrials d=densify(bs); if(!d.R||!d.C)continue;
        int base=rank_dense(d.M[0],d.R,d.C);
        int uc=union_col_rank(d),ur=union_row_rank(d);
        U64 pred=channel_formula(rows,h);
        std::cout<<"h="<<h<<" R="<<d.R<<" C="<<d.C<<" physical_rank="<<base<<" predicted="<<pred
                 <<" union_col="<<uc<<" union_row="<<ur
                 <<((uc==base&&ur==base)?" FIXED":" GROWS")<<"\n";
        if(uc!=base||ur!=base)fixed=false;
    }
    std::cout<<(fixed?"FIXED_SUBSPACES":"TRAJECTORY_DEPENDENT_SUBSPACES")<<"\n";
    return 0;
}
