#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

// Construct an exact minimal-rank MPS for the actual half-DP frontier vector at
// small width, using rank factorizations of every prefix|suffix flattening.
// The decisive measurement is the sparsity of the physical-symbol transition
// matrices A[N], A[R], A[L].  Small Schmidt rank alone is not algorithmically
// useful if these matrices become dense.

using U32 = std::uint32_t;
using U64 = std::uint64_t;
static constexpr U64 P = 4294967291ULL;

static U64 addp(U64 a,U64 b){ U64 z=a+b; if(z>=P)z-=P; return z; }
static U64 subp(U64 a,U64 b){ return a>=b?a-b:a+P-b; }
static U64 mulp(U64 a,U64 b){ return U64((__uint128_t(a)*b)%P); }
static U64 powp(U64 a,U64 e){ U64 r=1; while(e){ if(e&1)r=mulp(r,a); a=mulp(a,a); e>>=1; } return r; }
static U64 invp(U64 a){ assert(a); return powp(a,P-2); }

static std::vector<Mate> code_states(MateCodec const&mc){
    std::vector<Mate> out(mc.codeSize());
    for(Code bi=0;bi<mc.codeSizeL();++bi){ auto const&b=mc.codeTable(bi); for(Code i=0;i<b.size;++i) out[b.base+i]=b.mateL|b.mateR[i]; }
    return out;
}

static U32 prefix_code(Mate m,int W,int a){
    U32 z=0;
    for(int i=0;i<a;++i) z |= U32(m.get(W-1-i))<<(2*i);
    return z;
}
static U32 suffix_code(Mate m,int W,int a){
    U32 z=0; int n=W-a;
    for(int i=0;i<n;++i) z |= U32(m.get(i))<<(2*i);
    return z;
}

struct Flat {
    int a=0;
    std::vector<U32> row_code,col_code;
    std::unordered_map<U32,int> row_id,col_id;
    std::vector<U64> M;
    int R=0,C=0,rank=0;
    std::vector<int> piv_rows,piv_cols;
    std::vector<U64> Binv; // rank x rank
};

static std::vector<U64> invert_square(std::vector<U64> A,int n){
    std::vector<U64> aug(size_t(n)*2*n,0);
    for(int i=0;i<n;++i){ for(int j=0;j<n;++j) aug[size_t(i)*2*n+j]=A[size_t(i)*n+j]; aug[size_t(i)*2*n+n+i]=1; }
    for(int c=0;c<n;++c){
        int p=c; while(p<n && !aug[size_t(p)*2*n+c]) ++p; assert(p<n);
        if(p!=c) for(int j=0;j<2*n;++j) std::swap(aug[size_t(p)*2*n+j],aug[size_t(c)*2*n+j]);
        U64 iv=invp(aug[size_t(c)*2*n+c]);
        for(int j=0;j<2*n;++j) aug[size_t(c)*2*n+j]=mulp(aug[size_t(c)*2*n+j],iv);
        for(int i=0;i<n;++i) if(i!=c){ U64 f=aug[size_t(i)*2*n+c]; if(!f)continue; for(int j=0;j<2*n;++j) aug[size_t(i)*2*n+j]=subp(aug[size_t(i)*2*n+j],mulp(f,aug[size_t(c)*2*n+j])); }
    }
    std::vector<U64> I(size_t(n)*n);
    for(int i=0;i<n;++i)for(int j=0;j<n;++j) I[size_t(i)*n+j]=aug[size_t(i)*2*n+n+j];
    return I;
}

static void factor_flat(Flat& f){
    std::vector<U64> A=f.M;
    std::vector<int> orig(f.R); for(int i=0;i<f.R;++i)orig[i]=i;
    int rr=0;
    for(int c=0;c<f.C && rr<f.R;++c){
        int p=rr; while(p<f.R && !A[size_t(p)*f.C+c])++p;
        if(p==f.R)continue;
        if(p!=rr){
            for(int j=0;j<f.C;++j)std::swap(A[size_t(p)*f.C+j],A[size_t(rr)*f.C+j]);
            std::swap(orig[p],orig[rr]);
        }
        U64 iv=invp(A[size_t(rr)*f.C+c]);
        for(int i=rr+1;i<f.R;++i){ U64 x=A[size_t(i)*f.C+c]; if(!x)continue; U64 q=mulp(x,iv); A[size_t(i)*f.C+c]=0; for(int j=c+1;j<f.C;++j) A[size_t(i)*f.C+j]=subp(A[size_t(i)*f.C+j],mulp(q,A[size_t(rr)*f.C+j])); }
        f.piv_rows.push_back(orig[rr]); f.piv_cols.push_back(c); ++rr;
    }
    f.rank=rr;
    std::vector<U64> B(size_t(rr)*rr);
    for(int i=0;i<rr;++i)for(int j=0;j<rr;++j) B[size_t(i)*rr+j]=f.M[size_t(f.piv_rows[i])*f.C+f.piv_cols[j]];
    f.Binv=invert_square(B,rr);
}

static std::vector<U64> coords(Flat const&f,int row){
    std::vector<U64> z(f.rank,0);
    if(row<0)return z;
    std::vector<U64> x(f.rank);
    for(int j=0;j<f.rank;++j)x[j]=f.M[size_t(row)*f.C+f.piv_cols[j]];
    // row = coeff * pivot_rows; x = coeff * B => coeff=x*B^{-1}
    for(int j=0;j<f.rank;++j)for(int k=0;k<f.rank;++k) z[j]=addp(z[j],mulp(x[k],f.Binv[size_t(k)*f.rank+j]));
    return z;
}

static Flat build_flat(std::vector<Mate>const&states,std::vector<U64>const&val,int W,int a){
    Flat f; f.a=a;
    std::map<U32,int> rm,cm;
    for(size_t i=0;i<states.size();++i) if(val[i]) { rm.emplace(prefix_code(states[i],W,a),0); cm.emplace(suffix_code(states[i],W,a),0); }
    for(auto &kv:rm){ kv.second=f.row_code.size(); f.row_id[kv.first]=kv.second; f.row_code.push_back(kv.first); }
    for(auto &kv:cm){ kv.second=f.col_code.size(); f.col_id[kv.first]=kv.second; f.col_code.push_back(kv.first); }
    f.R=f.row_code.size(); f.C=f.col_code.size(); f.M.assign(size_t(f.R)*f.C,0);
    for(size_t i=0;i<states.size();++i) if(val[i]){
        int r=f.row_id[prefix_code(states[i],W,a)], c=f.col_id[suffix_code(states[i],W,a)];
        f.M[size_t(r)*f.C+c]=addp(f.M[size_t(r)*f.C+c],val[i]);
    }
    factor_flat(f); return f;
}

static void init_pc(PathCounter<Modnum<U64>>&pc){
    for(Code i=0;i<pc.mc.codeSize();++i)pc.value[i]=0;
    for(Code i=0;i<pc.wc.codeSize();++i)pc.deferred[i]=0;
    pc.value[pc.mc.encode(Mate(pc.cols-1,R))]=1;
}
static void run_row(PathCounter<Modnum<U64>>&pc){ for(int j=0;j<pc.cols-2;++j)pc.update(j,false); pc.update(pc.cols-2,false); }

int main(int argc,char**argv){
    msg=NONE; modulus=P;
    int W=argc>1?std::atoi(argv[1]):10;
    int rows=argc>2?std::atoi(argv[2]):W/2;
    if(W<2||W>12||rows<0||rows>W){ std::cerr<<"usage: W(2..12) rows(0..W)\n"; return 2; }
    PathCounter<Modnum<U64>> pc(W,W,false,false); init_pc(pc); for(int r=0;r<rows;++r)run_row(pc);
    auto states=code_states(pc.mc); std::vector<U64> val(states.size()); for(size_t i=0;i<val.size();++i)val[i]=U64(pc.value[i])%P;
    std::vector<Flat> F; F.reserve(W+1);
    for(int a=0;a<=W;++a){ F.push_back(build_flat(states,val,W,a)); std::cout<<"cut="<<a<<" rows="<<F.back().R<<" cols="<<F.back().C<<" rank="<<F.back().rank<<"\n"; }

    U64 total_nnz=0,total_slots=0; int global_max_row=0;
    for(int a=0;a<W;++a){
        auto const&A=F[a]; auto const&B=F[a+1];
        U64 nnz_sym[3]={0,0,0}, rows_nonzero[3]={0,0,0}; int max_row_sym[3]={0,0,0};
        for(int i=0;i<A.rank;++i){
            U32 pcode=A.row_code[A.piv_rows[i]];
            for(int sv=0;sv<3;++sv){
                U32 qcode=pcode|(U32(sv)<<(2*a));
                auto it=B.row_id.find(qcode); int qr=it==B.row_id.end()?-1:it->second;
                auto z=coords(B,qr); int rn=0; for(U64 x:z)if(x){++rn;++nnz_sym[sv];}
                if(rn)++rows_nonzero[sv]; max_row_sym[sv]=std::max(max_row_sym[sv],rn); global_max_row=std::max(global_max_row,rn);
            }
        }
        U64 nnz=nnz_sym[0]+nnz_sym[1]+nnz_sym[2], slots=U64(3)*A.rank*B.rank; total_nnz+=nnz; total_slots+=slots;
        std::cout<<"EDGE cut="<<a<<"->"<<a+1<<" D="<<A.rank<<"->"<<B.rank
                 <<" nnz="<<nnz<<" avg_per_source="<<(A.rank?double(nnz)/A.rank:0.0)
                 <<" density="<<(slots?double(nnz)/double(slots):0.0)
                 <<" N="<<nnz_sym[0]<<" R="<<nnz_sym[1]<<" L="<<nnz_sym[2]
                 <<" maxrow="<<std::max({max_row_sym[0],max_row_sym[1],max_row_sym[2]})<<"\n";
    }
    std::cout<<"SUMMARY W="<<W<<" completed_rows="<<rows<<" total_nnz="<<total_nnz
             <<" dense_slots="<<total_slots<<" density="<<(total_slots?double(total_nnz)/double(total_slots):0.0)
             <<" global_max_row="<<global_max_row<<"\n";
    return 0;
}
