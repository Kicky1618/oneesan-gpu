#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
static constexpr u32 P = 998244353u;

static u32 addp(u32 a,u32 b){u32 z=a+b;return z>=P?z-P:z;}
static u32 mulp(u32 a,u32 b){return u32((u64(a)*b)%P);} 
static u32 powp(u32 a,u64 e){u32 r=1;while(e){if(e&1)r=mulp(r,a);a=mulp(a,a);e>>=1;}return r;}
static u32 invp(u32 a){return powp(a,P-2);} 
static u32 negp(u32 a){return a?P-a:0;}

static u64 BC[32][32];
static void build_binom(){
    for(int n=0;n<32;++n){BC[n][0]=BC[n][n]=1;for(int k=1;k<n;++k)BC[n][k]=BC[n-1][k-1]+BC[n-1][k];}
}
static u64 cat(int n){return BC[2*n][n]/u64(n+1);} 

static int dh(int s){return s==0?-1:s==3?1:0;} // A,B,C,D = 0,1,2,3
static u64 WAYS[16][18];
static void build_ways(){
    WAYS[0][0]=1;
    for(int len=1;len<16;++len)for(int h=0;h<17;++h){
        u64 z=2*WAYS[len-1][h];if(h)z+=WAYS[len-1][h-1];z+=WAYS[len-1][h+1];WAYS[len][h]=z;
    }
}

static u32 sym(u32 code,int k){return (code>>(2*k))&3u;}
static int height_before(u32 code,int k){int h=0;for(int i=0;i<k;++i)h+=dh(sym(code,i));return h;}
static bool valid_code(u32 code,int r){int h=0;for(int i=0;i<r;++i){h+=dh(sym(code,i));if(h<0)return false;}return h==0;}

static u32 fusion_rank(u32 code,int r){
    u64 rank=0;int h=0;
    for(int pos=0;pos<r;++pos){int s=sym(code,pos),rem=r-pos-1;for(int t=0;t<s;++t){int hh=h+dh(t);if(hh>=0)rank+=WAYS[rem][hh];}h+=dh(s);assert(h>=0);}
    assert(h==0&&rank<cat(r+1));return u32(rank);
}

static std::vector<std::vector<u32>> UNRANK;
static void gen_rec(int r,int pos,int h,u32 code,std::vector<u32>&out){
    if(pos==r){if(h==0)out.push_back(code);return;}
    if(h>0)gen_rec(r,pos+1,h-1,code,out);                       // A=0
    gen_rec(r,pos+1,h,code|(1u<<(2*pos)),out);                 // B=1
    gen_rec(r,pos+1,h,code|(2u<<(2*pos)),out);                 // C=2
    gen_rec(r,pos+1,h+1,code|(3u<<(2*pos)),out);               // D=3
}
static void build_unrank(int maxr){
    UNRANK.resize(maxr+1);
    for(int r=0;r<=maxr;++r){gen_rec(r,0,0,0,UNRANK[r]);assert(UNRANK[r].size()==cat(r+1));for(u32 i=0;i<UNRANK[r].size();++i)assert(fusion_rank(UNRANK[r][i],r)==i);}
}

static u64 mask_rank(u32 mask,int W){u64 r=0;int j=0;for(int p=0;p<W;++p)if((mask>>p)&1u){++j;r+=BC[p][j];}return r;}

struct Codec{
    int W=0;std::vector<u64> base;u64 size=0;
    explicit Codec(int w=0):W(w),base(w+2){
        u64 z=0;for(int m=1;m<=W;m+=2){base[m]=z;z+=BC[W][m]*cat((m+1)/2);}size=z;
    }
    u64 id(u32 mask,u32 path)const{
        int m=__builtin_popcount(mask);assert(m&1);int r=(m-1)/2;u64 d=cat(r+1);u64 fr=fusion_rank(path,r);return base[m]+mask_rank(mask,W)*d+fr;
    }
};

static u32 low_mask_bits(int k){return k?((1u<<(2*k))-1u):0u;}
static u32 insert_sym(u32 code,int r,int k,int s){assert(0<=k&&k<=r);u32 lo=code&low_mask_bits(k),hi=code>>(2*k);return lo|(u32(s)<<(2*k))|(hi<<(2*(k+1)));}
static u32 remove_sym(u32 code,int r,int k){assert(0<=k&&k<r);u32 lo=code&low_mask_bits(k),hi=code>>(2*(k+1));return lo|(hi<<(2*k));}
static u32 replace1with2(u32 code,int r,int k,int a,int b){u32 t=remove_sym(code,r,k);t=insert_sym(t,r-1,k,a);return insert_sym(t,r,k+1,b);}
static u32 replace2with1(u32 code,int r,int k,int a){u32 t=remove_sym(code,r,k+1);t=remove_sym(t,r-1,k);return insert_sym(t,r-2,k,a);}

struct Term{u32 path,coeff;};
static std::vector<Term> arc_insert(u32 code,int r,int i){
    if(i==0){u32 t=insert_sym(code,r,r,2);assert(valid_code(t,r+1));return{{t,1}};}
    if(i&1){int j=(i-1)/2,k=r-j;u32 t=insert_sym(code,r,k,1);assert(valid_code(t,r+1));return{{t,1}};}
    int j=i/2,k=r-j;assert(0<=k&&k<r);int h=height_before(code,k),x=sym(code,k);std::vector<Term>out;
    auto add=[&](int a,int b,u32 c){u32 t=replace1with2(code,r,k,a,b);assert(valid_code(t,r+1));out.push_back({t,c});};
    if(x==0){add(0,2,1);add(2,0,1);} // A -> AC+CA
    else if(x==1){add(1,2,1);add(2,1,1);add(3,0,1);if(h)add(0,3,negp(mulp(u32(h),invp(u32(h+1)))));}
    else if(x==2)add(2,2,1);
    else{add(2,3,1);add(3,2,1);}
    return out;
}

static std::vector<Term> arc_cap(u32 code,int r,int i){
    assert(r>=1);
    if(i==0){if(sym(code,r-1)!=1)return{};u32 t=remove_sym(code,r,r-1);assert(valid_code(t,r-1));return{{t,1}};}
    if(i&1){int j=(i-1)/2,k=(r-1)-j;if(sym(code,k)!=2)return{};u32 t=remove_sym(code,r,k);assert(valid_code(t,r-1));return{{t,1}};}
    int j=i/2,k=(r-1)-j;assert(0<=k&&k+1<r);int h=height_before(code,k);int a=sym(code,k),b=sym(code,k+1),y=-1;u32 c=1;
    if((a==0&&b==1)||(a==1&&b==0))y=0;               // AB,BA -> A
    else if(a==1&&b==1)y=1;                            // BB -> B
    else if((a==0&&b==3)||(a==1&&b==2)||(a==2&&b==1))y=2; // AD,BC,CB -> C
    else if(a==3&&b==0){y=2;c=negp(mulp(u32(h+2),invp(u32(h+1))));} // DA
    else if((a==1&&b==3)||(a==3&&b==1))y=3;            // BD,DB -> D
    else return{};
    u32 t=replace2with1(code,r,k,y);assert(valid_code(t,r-1));return{{t,c}};
}

static u32 remove_bit(u32 m,int p){u32 lo=m&((1u<<p)-1u),hi=m>>(p+1);return lo|(hi<<p);} 
static u32 insert_zero(u32 m,int p){u32 lo=m&((1u<<p)-1u),hi=m>>p;return lo|(hi<<(p+1));}

static void add_at(std::vector<u32>&v,u64 id,u32 x){if(!x)return;v[id]=addp(v[id],x);} 

static void step_dense(std::vector<u32>const&main,std::vector<u32>const&blocked,int W,int p,Codec const&MC,Codec const&DC,std::vector<u32>&nm,std::vector<u32>&nb){
    nm=main;std::fill(nb.begin(),nb.end(),0);

    // Blocked excluded branch.
    for(u32 mask=0;mask<(1u<<(W-1));++mask){int m=__builtin_popcount(mask);if(!(m&1))continue;int r=(m-1)/2;for(u32 fr=0;fr<UNRANK[r].size();++fr){u32 path=UNRANK[r][fr];u32 x=blocked[DC.id(mask,path)];if(x)add_at(nm,MC.id(insert_zero(mask,p),path),x);}}

    // Main included branches.
    for(u32 mask=0;mask<(1u<<W);++mask){int m=__builtin_popcount(mask);if(!(m&1))continue;int r=(m-1)/2;for(u32 fr=0;fr<UNRANK[r].size();++fr){u32 path=UNRANK[r][fr];u32 x=main[MC.id(mask,path)];if(!x)continue;int lo=(mask>>(p-1))&1,hi=(mask>>p)&1;u32 below=mask&((1u<<(p-1))-1u);int i=__builtin_popcount(below);
        if(!lo&&!hi){u32 m2=mask|(1u<<(p-1))|(1u<<p);for(auto t:arc_insert(path,r,i))add_at(nm,MC.id(m2,t.path),mulp(x,t.coeff));}
        else if(lo&&!hi){if(p==1){u32 m2=(mask&~(1u<<(p-1)))|(1u<<p);add_at(nm,MC.id(m2,path),x);}else add_at(nb,DC.id(remove_bit(mask,p),path),x);}
        else if(!lo&&hi){u32 m2=(mask&~(1u<<p))|(1u<<(p-1));add_at(nm,MC.id(m2,path),x);}
        else{u32 cleared=mask&~((1u<<(p-1))|(1u<<p));for(auto t:arc_cap(path,r,i)){u32 z=mulp(x,t.coeff);if(p==1)add_at(nm,MC.id(cleared,t.path),z);else add_at(nb,DC.id(remove_bit(cleared,p-1),t.path),z);}}
    }}
}

static u32 dense_count(int W,bool verbose){
    Codec MC(W),DC(W-1);std::vector<u32>main(MC.size),blocked(DC.size),nm(MC.size),nb(DC.size);main[MC.id(1u<<(W-1),0)]=1;
    for(int row=0;row<W;++row){for(int p=W-1;p>=1;--p){step_dense(main,blocked,W,p,MC,DC,nm,nb);main.swap(nm);blocked.swap(nb);}if(verbose)std::cerr<<"dense row="<<row+1<<"\n";}
    return main[MC.id(1,0)];
}

int main(int argc,char**argv){
    build_binom();build_ways();build_unrank(14);msg=NONE;modulus=P;int maxW=argc>1?std::atoi(argv[1]):9;if(maxW>10)maxW=10;bool verbose=argc>2;
    for(int W=2;W<=maxW;++W){Codec mc(W),dc(W-1);PathCounter<Modnum<u64>> pc(W,W,false,false);assert(mc.size==pc.mc.codeSize());assert(dc.size==pc.wc.codeSize());u32 f=dense_count(W,verbose);u64 g=pc.count();std::cout<<"W="<<W<<" dense_fusion="<<f<<" gridfp="<<g<<" main_states="<<mc.size<<" blocked_states="<<dc.size<<" "<<(f==u32(g)?"OK":"MISMATCH")<<"\n";if(f!=u32(g))return 1;}
    return 0;
}
