#define main odd_tl_probe_original_main
#include "odd_tl_gram_factorization_probe.cpp"
#undef main

#include <chrono>
#include <iomanip>
#include <set>
#include <unordered_map>

using Sparse = std::unordered_map<int, std::uint32_t>;

static void sadd(Sparse& z, int i, std::uint32_t x) {
    if (!x) return;
    auto it = z.find(i);
    if (it == z.end()) {
        z.emplace(i, x);
    } else {
        auto y = addm(it->second, x);
        if (y) it->second = y;
        else z.erase(it);
    }
}
static void sadd_scaled(Sparse& z, int i, std::uint32_t c, std::uint32_t x) {
    sadd(z, i, mulm(c, x));
}
static void ssub_scaled(Sparse& z, int i, std::uint32_t c, std::uint32_t x) {
    sadd(z, i, negm(mulm(c, x)));
}

static void sparse_partial(int n, int d, Sparse const& src, Sparse& dst, bool subtract=false) {
    if (d < 3) return;
    auto const& B = basis(n, d);
    int r = (d - 3) / 2;
    for (auto const& [idx, x] : src) {
        for (int j = 0; j <= (d - 3) / 2; ++j) {
            int to = map_cup_word(B.words[idx], n, d, 2*j, 2*j+1);
            assert(to >= 0);
            std::uint32_t c = ((r-j)&1) ? MOD-1 : 1;
            if (subtract) ssub_scaled(dst, to, c, x);
            else sadd_scaled(dst, to, c, x);
        }
    }
}

static void sparse_J(int n, int d, Sparse const& src, Sparse& dst, bool subtract=false) {
    if (d < 2) return;
    auto const& B = basis(n, d);
    for (auto const& [idx, x] : src) {
        int to = map_cup_word(B.words[idx], n, d, d-2, d-1);
        assert(to >= 0);
        if (subtract) ssub_scaled(dst, to, 1, x);
        else sadd(dst, to, x);
    }
}

static void sparse_Q(int n, int d, Sparse const& src, Sparse& dst, bool subtract=false) {
    int din = d + 2;
    int r = (d - 1) / 2;
    if (r == 0) return;
    auto const& B = basis(n, din);
    std::uint32_t den_inv = invm(std::uint32_t(r+1));
    auto emit = [&](int to, std::uint32_t c, std::uint32_t x) {
        if (subtract) ssub_scaled(dst, to, c, x);
        else sadd_scaled(dst, to, c, x);
    };
    for (auto const& [idx, x] : src) {
        for (int s = 0; s < 2*r; ++s) {
            int to = map_two_cups(B.words[idx], n, din,
                                  {s,s+3}, {s+1,s+2});
            assert(to >= 0);
            int num = s/2 + 1;
            std::uint32_t c = mulm(std::uint32_t(num), den_inv);
            if (s&1) c = negm(c);
            emit(to,c,x);
        }
        for (int i = 0; i <= r; ++i) for (int j = i+1; j <= r; ++j) {
            int to = map_two_cups(B.words[idx], n, din,
                                  {2*i,2*i+1}, {2*j+1,2*j+2});
            assert(to >= 0);
            int num=1, sign_exp=j-i;
            if (j==r) { num=r; sign_exp=r-1-i; }
            std::uint32_t c = mulm(std::uint32_t(num), den_inv);
            if (sign_exp&1) c=negm(c);
            emit(to,c,x);
        }
    }
}

static void split4(int n, int d, Sparse const& v,
                   Sparse& A, Sparse& B, Sparse& C, Sparse& D) {
    int da=int(basis(n-2,d-2).words.size());
    int d0=int(basis(n-2,d).words.size());
    int oB=da, oC=da+d0, oD=da+2*d0;
    for (auto const& [i,x]:v) {
        if (i<oB) A.emplace(i,x);
        else if (i<oC) B.emplace(i-oB,x);
        else if (i<oD) C.emplace(i-oC,x);
        else D.emplace(i-oD,x);
    }
}
static void join4(int n,int d,Sparse& v,
                  Sparse const&A,Sparse const&B,Sparse const&C,Sparse const&D) {
    int da=int(basis(n-2,d-2).words.size());
    int d0=int(basis(n-2,d).words.size());
    int oB=da,oC=da+d0,oD=da+2*d0;
    v.clear();
    v.reserve(A.size()+B.size()+C.size()+D.size());
    for(auto const&[i,x]:A)sadd(v,i,x);
    for(auto const&[i,x]:B)sadd(v,oB+i,x);
    for(auto const&[i,x]:C)sadd(v,oC+i,x);
    for(auto const&[i,x]:D)sadd(v,oD+i,x);
}

static void sparse_transform(int n,int d,Sparse&v){
    if(n<=1)return;
    Sparse A,B,C,D; split4(n,d,v,A,B,C,D);
    if(!A.empty() || !basis(n-2,d-2).words.empty()) {
        sparse_partial(n-2,d,C,A,false);
        sparse_Q(n-2,d,D,A,false);
    }
    if(!basis(n-2,d).words.empty() && !D.empty()) {
        sparse_partial(n-2,d+2,D,B,false);
        sparse_J(n-2,d+2,D,C,false);
    }
    if(!A.empty())sparse_transform(n-2,d-2,A);
    if(!B.empty())sparse_transform(n-2,d,B);
    if(!C.empty())sparse_transform(n-2,d,C);
    if(!D.empty())sparse_transform(n-2,d+2,D);
    join4(n,d,v,A,B,C,D);
}

static void sparse_inverse_transform(int n,int d,Sparse&v){
    if(n<=1)return;
    Sparse A,B,C,D; split4(n,d,v,A,B,C,D);
    if(!A.empty())sparse_inverse_transform(n-2,d-2,A);
    if(!B.empty())sparse_inverse_transform(n-2,d,B);
    if(!C.empty())sparse_inverse_transform(n-2,d,C);
    if(!D.empty())sparse_inverse_transform(n-2,d+2,D);
    if(!D.empty()) {
        sparse_J(n-2,d+2,D,C,true);
        sparse_partial(n-2,d+2,D,B,true);
    }
    sparse_partial(n-2,d,C,A,true);
    sparse_Q(n-2,d,D,A,true);
    join4(n,d,v,A,B,C,D);
}

static void sparse_apply_D(int n,int d,Sparse&v){
    if(n<=1)return;
    Sparse A,B,C,D; split4(n,d,v,A,B,C,D);
    if(!A.empty())sparse_apply_D(n-2,d-2,A);
    std::swap(B,C);
    if(!B.empty())sparse_apply_D(n-2,d,B);
    if(!C.empty())sparse_apply_D(n-2,d,C);
    if(!D.empty()){
        sparse_apply_D(n-2,d+2,D);
        std::uint32_t c=negm(mulm(std::uint32_t(d+3),invm(std::uint32_t(d+1))));
        for(auto&[i,x]:D)x=mulm(c,x);
    }
    join4(n,d,v,A,B,C,D);
}

static std::uint32_t reflect_word_sparse(std::uint32_t w,int n){
    auto m=mates(w,n); std::uint32_t z=0;
    for(int i=0;i<n;++i){
        int ri=n-1-i;
        if(m[i]<0) z|=std::uint32_t(1)<<ri;
        else if(i<m[i]) z|=std::uint32_t(1)<<(n-1-m[i]);
    }
    return z;
}
static void sparse_reflect(int n,int d,Sparse&v){
    auto const&B=basis(n,d); Sparse z; z.reserve(v.size());
    for(auto const&[i,x]:v){
        auto rw=reflect_word_sparse(B.words[i],n);
        auto it=B.rank.find(rw); assert(it!=B.rank.end());
        sadd(z,it->second,x);
    }
    v.swap(z);
}
static void sparse_apply_K(int n,Sparse&v){
    sparse_inverse_transform(n,1,v);
    sparse_reflect(n,1,v);
    sparse_transform(n,1,v);
    sparse_apply_D(n,1,v);
}

static std::vector<unsigned long long> a004148(int nmax){
    std::vector<unsigned long long>a(nmax+1); a[0]=1; if(nmax)a[1]=1;
    for(int n=1;n<nmax;++n){
        unsigned long long z=a[n];
        for(int k=1;k<n;++k)z+=a[k]*a[n-1-k];
        a[n+1]=z;
    }
    return a;
}

int main(int argc,char**argv){
    int maxn=argc>1?std::atoi(argv[1]):23;
    if(maxn>23)maxn=23;
    if(!(maxn&1))--maxn;
    auto seq=a004148(28);
    for(int n=1;n<=maxn;n+=2){
        auto t0=std::chrono::steady_clock::now();
        int dim=int(basis(n,1).words.size());
        unsigned long long nnz=0; int maxcol=0;
        std::set<std::uint32_t> coeffs;
        for(int j=0;j<dim;++j){
            Sparse v;v.emplace(j,1);sparse_apply_K(n,v);
            nnz+=v.size();maxcol=std::max(maxcol,int(v.size()));
            for(auto const&[i,x]:v){(void)i;coeffs.insert(x);}
        }
        double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        std::cout<<"n="<<std::setw(2)<<n
                 <<" dim="<<std::setw(8)<<dim
                 <<" nnz="<<std::setw(12)<<nnz
                 <<" expected="<<std::setw(12)<<seq[n]
                 <<" avg="<<std::fixed<<std::setprecision(4)<<double(nnz)/double(dim)
                 <<" maxcol="<<maxcol
                 <<" coeff_mod_values="<<coeffs.size()
                 <<" sec="<<std::setprecision(3)<<sec
                 <<(nnz==seq[n]?" OK":" MISMATCH")<<"\n";
        if(nnz!=seq[n])return 1;
    }
    return 0;
}
