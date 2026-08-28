#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main

#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
static constexpr u32 P = 998244353u;

static u32 mulp(u32 a,u32 b){return u32((u64(a)*b)%P);} 
static u32 powp(u32 a,u64 e){u32 r=1;while(e){if(e&1)r=mulp(r,a);a=mulp(a,a);e>>=1;}return r;}
static u32 addp(u32 a,u32 b){u32 z=a+b;return z>=P?z-P:z;}

// Internal signed edge states are -1,0,+1.  Horizontal +1 is east and
// vertical +1 is south.  Ternary storage digits are 0=empty, 1=+1, 2=-1 so
// the all-empty frontier is code zero.
struct Opt { int down=0,right=0; u32 w=0; };
static std::array<std::vector<Opt>,9> OPT;
static u32 ZETA=0,ZETA_INV=0;

static int dir_for(char edge,int s){
    assert(s!=0);
    if(edge=='L'||edge=='R') return s>0?0:2; // E/W
    return s>0?1:3;                          // S/N
}
static bool incoming(char edge,int s){
    if(edge=='L') return s>0;
    if(edge=='R') return s<0;
    if(edge=='U') return s>0;
    return s<0; // D
}
static u32 vertex_weight(int up,int left,int down,int right){
    const char E[4]={'U','L','D','R'};
    const int S[4]={up,left,down,right};
    int nin=0,nout=0,di=-1,doo=-1,nz=0;
    for(int k=0;k<4;++k) if(S[k]){
        ++nz;
        if(incoming(E[k],S[k])){++nin;di=dir_for(E[k],S[k]);}
        else{++nout;doo=dir_for(E[k],S[k]);}
    }
    if(nz==0) return 1;
    if(nz!=2||nin!=1||nout!=1) return 0;
    int t=(doo-di+4)&3;
    if(t==0) return 1;
    if(t==1) return ZETA;      // clockwise quarter turn
    if(t==3) return ZETA_INV;  // counter-clockwise quarter turn
    return 0;                  // U-turn is not an allowed degree-two vertex
}

static void build_opts(){
    ZETA=powp(3,(P-1)/16);
    ZETA_INV=powp(ZETA,P-2);
    assert(powp(ZETA,8)==P-1); // primitive 16th root
    for(int u=-1;u<=1;++u) for(int l=-1;l<=1;++l){
        auto &v=OPT[(u+1)*3+(l+1)];
        for(int d=-1;d<=1;++d) for(int r=-1;r<=1;++r){
            u32 w=vertex_weight(u,l,d,r);
            if(w)v.push_back({d,r,w});
        }
        assert(v.size()<=3);
    }
}

static int decode_digit(int d){return d==0?0:(d==1?1:-1);}
static int encode_digit(int s){return s==0?0:(s>0?1:2);}
static int trit(u64 code,int p){for(int i=0;i<p;++i)code/=3;return decode_digit(int(code%3));}
static u64 put_trit(u64 code,int p,int s){
    u64 q=1;for(int i=0;i<p;++i)q*=3;
    return code+u64(encode_digit(s))*q;
}

static u32 oriented_count(int n,std::vector<size_t>* row_support=nullptr){
    // Boundary condition closes the desired source->sink path by a fixed exterior
    // arc.  The west half-edge at the source is an east-pointing incoming arrow;
    // the east half-edge at the sink is an east-pointing outgoing arrow.  The grid
    // portion of every distinguished simple loop then has total signed turn zero.
    std::unordered_map<u64,u32> V,NV;
    V.emplace(0,1);
    for(int y=0;y<n;++y){
        NV.clear();
        for(auto [upcode,v0]:V){
            struct PState{int h;u64 down;u32 v;};
            std::vector<PState> cur{{y==0?1:0,0,v0}},next;
            for(int x=0;x<n;++x){
                next.clear();
                int up=trit(upcode,x);
                for(auto const&s:cur){
                    auto const& os=OPT[(up+1)*3+(s.h+1)];
                    for(auto const&o:os){
                        u64 dc=put_trit(s.down,x,o.down);
                        next.push_back({o.right,dc,mulp(s.v,o.w)});
                    }
                }
                // Merge identical (horizontal carry, down-prefix) states.
                std::unordered_map<u64,u32> tmp;
                for(auto const&s:next){
                    u64 key=(s.down<<2)|u64(s.h+1);
                    auto it=tmp.find(key);
                    if(it==tmp.end())tmp.emplace(key,s.v);else it->second=addp(it->second,s.v);
                }
                cur.clear();cur.reserve(tmp.size());
                for(auto [key,v]:tmp)cur.push_back({int(key&3)-1,key>>2,v});
            }
            int want=y==n-1?1:0;
            for(auto const&s:cur)if(s.h==want){
                auto it=NV.find(s.down);
                if(it==NV.end())NV.emplace(s.down,s.v);else it->second=addp(it->second,s.v);
            }
        }
        V.swap(NV);
        if(row_support)row_support->push_back(V.size());
    }
    auto it=V.find(0);return it==V.end()?0:it->second;
}

static u64 charge1_dim(int n){
    std::unordered_map<int,u64>d{{0,1}},nd;
    for(int i=0;i<n;++i){nd.clear();for(auto [q,v]:d)for(int s=-1;s<=1;++s)nd[q+s]+=v;d.swap(nd);}return d[1];
}

int main(int argc,char**argv){
    build_opts();
    msg=NONE;modulus=P;
    int maxn=argc>1?std::atoi(argv[1]):9;
    if(maxn>11)maxn=11;
    for(int n=2;n<=maxn;++n){
        std::vector<size_t> support;
        u32 z=oriented_count(n,&support);
        PathCounter<Modnum<u64>> pc(n,n,false,false);
        u64 exact=pc.count();
        size_t peak=0;for(auto x:support)peak=std::max(peak,x);
        u64 charge=charge1_dim(n);
        std::cout<<"n="<<n<<" oriented="<<z<<" gridfp="<<exact
                 <<" peak_support="<<peak<<" charge1_dim="<<charge
                 <<" "<<(z==exact%P?"OK":"MISMATCH")<<"\n";
        if(z!=exact%P)return 1;
    }
    return 0;
}
