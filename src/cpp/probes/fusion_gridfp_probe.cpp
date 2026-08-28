#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
static constexpr u32 P = 998244353u;

static u32 mulp(u32 a,u32 b){return u32((u64(a)*b)%P);} 
static u32 powp(u32 a,u64 e){u32 r=1;while(e){if(e&1)r=mulp(r,a);a=mulp(a,a);e>>=1;}return r;}
static u32 invp(u32 a){return powp(a,P-2);} 
static u32 negp(u32 a){return a?P-a:0;}

struct Key {
    u32 mask = 0;
    std::string path; // outer -> inner, alphabet A,B,C,D
    bool operator==(Key const&o)const{return mask==o.mask&&path==o.path;}
};
struct KeyHash {
    size_t operator()(Key const&k)const{
        size_t h=std::hash<std::string>{}(k.path);
        return h^(size_t(k.mask)*0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2));
    }
};
using Map=std::unordered_map<Key,u32,KeyHash>;

static void add(Map&z,Key k,u32 v){
    if(!v)return;auto it=z.find(k);if(it==z.end())z.emplace(std::move(k),v);else{u32 x=it->second+v;if(x>=P)x-=P;if(x)it->second=x;else z.erase(it);}
}
static int dh(char c){return c=='A'?-1:c=='D'?1:0;}
static int height_before(std::string const&s,int k){int h=0;for(int i=0;i<k;++i)h+=dh(s[i]);return h;}
static bool valid(std::string const&s){int h=0;for(char c:s){h+=dh(c);if(h<0)return false;}return h==0;}

struct Term { std::string path; u32 coeff; };

// Insert a new adjacent dense arc UD at dense terminal position i.  This is
// T_{m+2} I_i T_m^{-1} in the finite beta=0 odd-TL fusion basis.
static std::vector<Term> arc_insert(std::string const&s,int i){
    int r=int(s.size());std::vector<Term>out;
    if(i==0){auto t=s+"C";assert(valid(t));return {{t,1}};}
    if(i&1){int j=(i-1)/2,k=r-j;auto t=s.substr(0,k)+"B"+s.substr(k);assert(valid(t));return {{t,1}};}
    int j=i/2,k=r-j;assert(0<=k&&k<r);int h=height_before(s,k);char x=s[k];
    std::vector<std::pair<std::string,u32>> q;
    if(x=='A')q={{"AC",1},{"CA",1}};
    else if(x=='B'){
        q={{"BC",1},{"CB",1},{"DA",1}};
        if(h)q.push_back({"AD",negp(mulp(u32(h),invp(u32(h+1))))});
    }
    else if(x=='C')q={{"CC",1}};
    else if(x=='D')q={{"CD",1},{"DC",1}};
    for(auto const&[rep,c]:q){auto t=s.substr(0,k)+rep+s.substr(k+1);assert(valid(t));out.push_back({std::move(t),c});}
    return out;
}

// Contract/delete two adjacent occupied dense terminals.  This is the metric
// adjoint of arc_insert.  Every input has at most one output.
static std::vector<Term> arc_cap(std::string const&s,int i){
    int r=int(s.size());assert(r>=1);
    if(i==0){if(s.back()!='B')return {};return {{s.substr(0,r-1),1}};}
    if(i&1){int j=(i-1)/2,k=(r-1)-j;assert(0<=k&&k<r);if(s[k]!='C')return {};auto t=s.substr(0,k)+s.substr(k+1);assert(valid(t));return {{t,1}};}
    int j=i/2,k=(r-1)-j;assert(0<=k&&k+1<r);int h=height_before(s,k);std::string p=s.substr(k,2);char y=0;u32 c=1;
    if(p=="AB"||p=="BA")y='A';
    else if(p=="BB")y='B';
    else if(p=="AD"||p=="BC"||p=="CB")y='C';
    else if(p=="DA"){y='C';c=negp(mulp(u32(h+2),invp(u32(h+1))));}
    else if(p=="BD"||p=="DB")y='D';
    else return {};
    auto t=s.substr(0,k)+std::string(1,y)+s.substr(k+2);assert(valid(t));return {{t,c}};
}

static u32 remove_bit(u32 m,int p){u32 lo=m&((1u<<p)-1u),hi=m>>(p+1);return lo|(hi<<p);} 
static u32 insert_zero(u32 m,int p){u32 lo=m&((1u<<p)-1u),hi=m>>p;return lo|(hi<<(p+1));}

static void step(Map const&main,Map const&blocked,int W,int p,Map&nm,Map&nb){
    nm=main;nb.clear();
    // A blocked state has no included branch; its excluded branch reinserts N.
    for(auto const&[k,v]:blocked)add(nm,{insert_zero(k.mask,p),k.path},v);

    for(auto const&[k,v]:main){
        u32 mask=k.mask;int lo=(mask>>(p-1))&1,hi=(mask>>p)&1;
        u32 below=mask&((1u<<(p-1))-1u);int i=__builtin_popcount(below);
        if(!lo&&!hi){
            u32 m2=mask|(1u<<(p-1))|(1u<<p);
            for(auto const&t:arc_insert(k.path,i))add(nm,{m2,t.path},mulp(v,t.coeff));
        }
        else if(lo&&!hi){
            if(p==1){
                u32 m2=(mask&~(1u<<(p-1)))|(1u<<p);
                add(nm,{m2,k.path},v);
            }else{
                // Production NR/NL branch: shrink the vacant high position.
                add(nb,{remove_bit(mask,p),k.path},v);
            }
        }
        else if(!lo&&hi){
            // Move the dense terminal through the adjacent vacancy; topology is unchanged.
            u32 m2=(mask&~(1u<<p))|(1u<<(p-1));
            add(nm,{m2,k.path},v);
        }
        else{
            u32 cleared=mask&~((1u<<(p-1))|(1u<<p));
            for(auto const&t:arc_cap(k.path,i)){
                u32 z=mulp(v,t.coeff);
                if(p==1)add(nm,{cleared,t.path},z);
                else add(nb,{remove_bit(cleared,p-1),t.path},z);
            }
        }
    }
}

static u32 fusion_count(int W,bool verbose){
    Map main,blocked,nm,nb;main.emplace(Key{1u<<(W-1),""},1);
    for(int row=0;row<W;++row){
        for(int p=W-1;p>=1;--p){step(main,blocked,W,p,nm,nb);main.swap(nm);blocked.swap(nb);}
        if(verbose)std::cerr<<"fusion row="<<row+1<<" main="<<main.size()<<" blocked="<<blocked.size()<<"\n";
    }
    auto it=main.find(Key{1,""});return it==main.end()?0:it->second;
}

int main(int argc,char**argv){
    msg=NONE;modulus=P;int maxW=argc>1?std::atoi(argv[1]):9;if(maxW>11)maxW=11;bool verbose=argc>2;
    for(int W=2;W<=maxW;++W){
        u32 f=fusion_count(W,verbose);
        PathCounter<Modnum<u64>> pc(W,W,false,false);u64 g=pc.count();
        std::cout<<"W="<<W<<" fusion="<<f<<" gridfp="<<g<<" "<<(f==u32(g)?"OK":"MISMATCH")<<"\n";
        if(f!=u32(g))return 1;
    }
    return 0;
}
